import logging
from typing import Any, Dict, List, Optional, Union, Tuple
import json
import uuid 
import datetime 
import asyncio
import functools
import decimal
import re

from langchain.tools import BaseTool
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from pydantic import BaseModel, Field
import sqlparse

from app.core.config import settings
from app.db.schema_definitions import SCHEMA_DEFINITIONS
from app.utils import json_default

logger = logging.getLogger(__name__)

class SQLExecutionTool(BaseTool):
    """Tool for executing pre-generated SQL queries against databases, ensuring results are scoped to the user's organization."""
    
    name: str = "execute_sql"
    description: str = """Executes a provided SQL query against the specified database (typically 'report_management').
    This tool ONLY executes SQL - the query must be provided as an argument.
    The query MUST include organization filtering (e.g., WHERE "organizationId" = :organization_id).
    Always provide parameters as a dictionary, including the mandatory organization_id."""
    
    organization_id: str
    
    # Define the expected arguments schema
    class SQLExecutionArgs(BaseModel):
        sql: str = Field(description="The complete SQL query to execute with parameter placeholders.")
        params: Dict[str, Any] = Field(description="The parameters dictionary for the query. MUST include organization_id.")
        db_name: str = Field(default="report_management", description="The database name to execute against.")

    args_schema: type[BaseModel] = SQLExecutionArgs
    
    async def _execute_sql(self, sql: str, parameters: Dict[str, Any], db_name: str) -> Dict:
        """Execute SQL with parameters and return results (asynchronous version)."""
        log_prefix = f"[SQLExecutionTool] "
        logger.debug(f"{log_prefix}Executing SQL on DB '{db_name}'...")
        MAX_ROWS = 50
        original_sql = sql
        
        # --- SQL Syntax Validation and Organization Check ---
        try:
            # Removed try/except for sqlparse import - it's now mandatory
            logger.debug(f"{log_prefix}Validating SQL syntax using sqlparse...") # ADDED LOG

            # Store SQL elements for security verification
            cte_blocks = []
            subqueries = []
            main_query = sql
            
            # Parse with sqlparse for syntax validation
            parsed_statements = sqlparse.parse(sql)
            if not parsed_statements:
                raise ValueError("Provided SQL is empty or could not be parsed.")
            
            first_statement_type = parsed_statements[0].get_type()
            if first_statement_type not in ('SELECT', 'UNKNOWN'):
                 # Allow UNKNOWN type as complex statements might be parsed as such initially
                 logger.warning(f"{log_prefix}Provided SQL might not be a SELECT statement (Type: {first_statement_type}). Proceeding cautiously.")
            
            # Extract CTEs using sqlparse
            for stmt in parsed_statements:
                # Look for WITH keyword to identify CTEs
                for token in stmt.tokens:
                    if hasattr(token, 'ttype') and token.ttype is sqlparse.tokens.Keyword and token.value.upper() == 'WITH':
                        # Find the CTE block (from WITH to the main SELECT)
                        cte_start = sql.upper().find('WITH')
                        if cte_start >= 0:
                            # Find the main SELECT after the CTE
                            # More robustly find the main statement body after CTE block
                            # This is simplified; a full parser would handle complex cases better
                            paren_level = 0
                            body_start = -1
                            in_string = False
                            for i, char in enumerate(sql[cte_start:]):
                                if char == "'" and (i == 0 or sql[cte_start + i - 1] != '\\'): # Handle basic escapes
                                    in_string = not in_string
                                if not in_string:
                                    if char == '(': paren_level += 1
                                    elif char == ')': paren_level -= 1
                                    # Found start of main query body after CTEs closed
                                    if paren_level == 0 and sql[cte_start + i:].lstrip().upper().startswith(('SELECT', 'INSERT', 'UPDATE', 'DELETE')):
                                        body_start = cte_start + i
                                        break
                            
                            if body_start > 0:
                                cte_part = sql[cte_start:body_start].strip()
                                main_query = sql[body_start:].strip()
                                cte_blocks.append(cte_part)
                                logger.debug(f"{log_prefix}Extracted CTE block. Remaining main query starts with: {main_query[:50]}...")
                            else:
                                logger.warning(f"{log_prefix}Could not accurately determine end of CTE block. Main query extraction might be inaccurate.")
                                main_query = sql # Fallback to full sql if CTE parsing fails
                        break # Assume only one WITH clause block at the start

                # Extract subqueries using sqlparse (simplified)
                for stmt in parsed_statements:
                    for token in stmt.flatten():
                        # Check if token is a Parenthesis type and contains a SELECT
                        if isinstance(token, sqlparse.sql.Parenthesis) and 'SELECT' in str(token).upper():
                             subquery_text = str(token).strip()
                             # Basic check to avoid adding the entire outer query if it's wrapped
                             if subquery_text.upper() != main_query.upper():
                                 subqueries.append(subquery_text)

            # Remove duplicates from subqueries
            subqueries = list(dict.fromkeys(subqueries))
            logger.debug(f"{log_prefix}SQL parsing complete. Found {len(cte_blocks)} CTE blocks, {len(subqueries)} unique subqueries.")
            
        except ValueError as ve:
            logger.error(f"{log_prefix}SQL validation failed: {ve}. SQL: {sql[:200]}...", exc_info=False)
            raise
        except ImportError as ie:
            # This shouldn't happen now, but good practice
            logger.critical(f"{log_prefix}CRITICAL ERROR: sqlparse library is required but not found. Please install it. Error: {ie}")
            raise RuntimeError("SQL validation library 'sqlparse' is missing.") from ie
        except Exception as validation_err:
            logger.error(f"{log_prefix}Unexpected error during SQL validation: {validation_err}. SQL: {sql}", exc_info=True)
            raise ValueError(f"Provided SQL failed validation: {validation_err}")
        
        # --- Security Check: Verify organization_id parameter usage in all SQL components ---
        try:
            # Define valid org identifier column names from schema conventions
            # (We check if the :organization_id param is used with ANY of these quoted cols)
            valid_org_id_cols = {"organizationId", "parentId", "id"} # Based on SCHEMA_DEFINITIONS structure
            log_prefix = f"[SQLExecutionTool]"

            # Function to check if a SQL block has the required filter pattern
            def check_block_for_org_filter(sql_block: str, block_type: str) -> bool:
                lower_sql_block = sql_block.lower()
                if ":organization_id" not in lower_sql_block:
                    logger.debug(f"{log_prefix}Security Check: :organization_id param missing in {block_type}.")
                    return False # Parameter missing

                # Simplified check: Ensure at least one valid org id column name exists (quoted) in the block
                found_col_presence = False
                for col in valid_org_id_cols:
                    quoted_col = f'"{col}"' # e.g., "organizationId"
                    if quoted_col.lower() in lower_sql_block:
                        found_col_presence = True
                        logger.debug(f"{log_prefix}Security Check: Found valid column '{quoted_col}' in {block_type} containing :organization_id param.")
                        break # Found one valid column mentioned

                if not found_col_presence:
                    # Pre-format the list string to avoid complex nesting in logger f-string
                    valid_cols_str = ', '.join([f'"{c}"' for c in valid_org_id_cols])
                    logger.warning(f"{log_prefix}Security Check: :organization_id param found in {block_type}, but no valid org identifier columns ({valid_cols_str}) were detected nearby. Check query logic.")
                    # Allow potentially valid but complex cases for now, but log a warning
                    # return False # Stricter check would fail here
                return True # Return True if param is present (and optionally log if no col found)

            # Check main query
            if not check_block_for_org_filter(main_query, "Main Query"):
                valid_cols_str = ', '.join([f'"{c}"' for c in valid_org_id_cols]) # Pre-format the list
                error_msg = f"SECURITY CHECK FAILED: Main SQL query MUST include the :organization_id parameter."
                # Simplified error message as detailed column check is now a warning
                # error_msg = f"SECURITY CHECK FAILED: Main SQL query MUST use the :organization_id parameter in conjunction with a valid organization identifier column ({valid_cols_str})."
                logger.error(f"{log_prefix}{error_msg} SQL: {main_query[:200]}...")
                raise ValueError(error_msg)

            # Check CTE blocks
            for i, cte_block in enumerate(cte_blocks):
                 if not check_block_for_org_filter(cte_block, f"CTE #{i+1}"):
                    valid_cols_str = ', '.join([f'"{c}"' for c in valid_org_id_cols]) # Pre-format the list
                    error_msg = f"SECURITY CHECK FAILED: CTE block #{i+1} MUST include the :organization_id parameter."
                    # error_msg = f"SECURITY CHECK FAILED: CTE block #{i+1} MUST use the :organization_id parameter with a valid organization identifier column ({valid_cols_str})."
                    logger.error(f"{log_prefix}{error_msg} CTE Block: {cte_block[:200]}...")
                    raise ValueError(error_msg)

            # Check subqueries
            for i, subquery in enumerate(subqueries):
                 # Only check substantive subqueries
                 if len(subquery) > 100:
                     if not check_block_for_org_filter(subquery, f"Subquery #{i+1}"):
                        valid_cols_str = ', '.join([f'"{c}"' for c in valid_org_id_cols]) # Pre-format the list
                        error_msg = f"SECURITY CHECK FAILED: Complex subquery #{i+1} MUST include the :organization_id parameter."
                        # error_msg = f"SECURITY CHECK FAILED: Complex subquery #{i+1} MUST use the :organization_id parameter with a valid organization identifier column ({valid_cols_str})."
                        logger.error(f"{log_prefix}{error_msg} Subquery: {subquery[:200]}...")
                        raise ValueError(error_msg)

            logger.debug(f"{log_prefix}Security check passed: :organization_id parameter found in all required SQL components.")
            # Removed the more complex check message as the check itself is simplified
            # logger.debug(f"{log_prefix}Security check passed: :organization_id parameter found and appears associated with valid identifier columns in all required SQL components.")

        except ValueError as filter_check_err:
            # Re-raise any ValueError from the check
            raise
        except Exception as filter_check_err:
            logger.error(f"{log_prefix}Error during SQL organization filter check: {filter_check_err}", exc_info=True)
            # Re-raise as a ValueError to be handled like other validation errors
            raise ValueError(f"Error validating SQL organization filter: {filter_check_err}")
        # --- End Security Check for organization_id usage ---

        # --- Security Check: organization_id in parameters dictionary --- 
        if 'organization_id' not in parameters:
             error_msg = f"SECURITY CHECK FAILED: organization_id missing from parameters. Aborting."
             logger.error(f"{log_prefix}{error_msg}")
             raise ValueError(error_msg)
        if parameters['organization_id'] != self.organization_id:
            error_msg = f"SECURITY CHECK FAILED: organization_id mismatch in parameters. Expected {self.organization_id}, got {parameters.get('organization_id')}. Aborting."
            logger.error(f"[SQLExecutionTool] {error_msg}")
            raise ValueError(error_msg)
        
        # Log execution details
        logger.debug(f"{log_prefix}Executing SQL: {sql[:500]}...")
        logger.debug(f"{log_prefix}With Parameters: {parameters}")
        
        try:
            from app.db.connection import get_async_db_connection
            
            async with get_async_db_connection(db_name) as conn:
                # Apply execution timeout using asyncio.wait_for
                timeout_seconds = settings.SQL_EXECUTION_TIMEOUT_SECONDS
                logger.debug(f"{log_prefix}Applying asyncio timeout: {timeout_seconds} seconds.")

                # Execute within the timeout window
                result = await asyncio.wait_for(
                    conn.execute(text(sql), parameters),
                    timeout=timeout_seconds
                )

                columns = list(result.keys())
                raw_rows = result.fetchall()
                
                truncated = False
                total_count = len(raw_rows)
                is_count_query = original_sql.strip().upper().startswith("SELECT COUNT")
                
                if not is_count_query and total_count > MAX_ROWS:
                    truncated = True
                    raw_rows = raw_rows[:MAX_ROWS]
                    logger.warning(f"{log_prefix}Query results exceeded {MAX_ROWS} rows. Truncating.")
                
                rows = [list(row) for row in raw_rows]
                
                response_data = {"columns": columns, "rows": rows}
                if truncated:
                    response_data["metadata"] = {"truncated": True, "total_rows_returned": total_count, "rows_shown": MAX_ROWS}
                    
                return response_data
                
        except asyncio.TimeoutError as te:
            # Specific handling for asyncio timeouts
            error_msg = f"Database query exceeded the timeout limit ({settings.SQL_EXECUTION_TIMEOUT_SECONDS} seconds)."
            logger.error(f"{log_prefix}{error_msg} SQL: {sql[:200]}...", exc_info=False)
            # Raise a generic ValueError
            raise ValueError("The database query took too long to complete.") from te
        except SQLAlchemyError as e:
            # Log the full error internally
            logger.error(f"[SQLExecutionTool] SQL execution error for org {self.organization_id}. Error: {str(e)}", exc_info=True)
            logger.debug(f"{log_prefix}Failed SQL: {sql[:500]}... | Params: {parameters}")
            # Raise a generic error message
            raise ValueError("An error occurred while executing the database query.") from e
        except Exception as e:
            # Log the full error internally
            logger.error(f"[SQLExecutionTool] Unexpected error during SQL execution for org {self.organization_id}: {e}", exc_info=True)
            # Raise a generic error message
            raise ValueError("An unexpected error occurred during query execution.") from e
    
    async def _run(
        self, sql: str, params: Dict[str, Any], db_name: str = "report_management", run_manager=None
    ) -> str:
        """Execute the SQL query against the database."""
        log_prefix = f"[SQLExecutionTool] "
        logger.info(f"{log_prefix}Executing provided SQL query on DB '{db_name}'.")
        
        try:
            results = await self._execute_sql(sql, params, db_name)
            row_count = len(results.get('rows', []))
            logger.debug(f"{log_prefix}Completed successfully. Returned {row_count} rows.")
            
            # Return the entire result dictionary (containing columns/rows/metadata) serialized as JSON string
            # Use the imported json_default from utils
            return json.dumps(results, default=json_default)
            
        except ValueError as ve:
             logger.error(f"{log_prefix}Failed execution: {ve}", exc_info=False)
             fallback_output = {
                 "table": {"columns": ["Error"], "rows": [[f"Failed to execute query: {ve}"]]},
                 "text": f"Error executing provided SQL: {ve}"
             }
             return json.dumps(fallback_output, default=json_default)
        except Exception as e:
            logger.exception(f"[SQLExecutionTool] Unexpected critical error during SQL execution for org {self.organization_id}, SQL: '{sql[:100]}...': {e}", exc_info=True)
            fallback_output = {
                 "table": {"columns": ["Error"], "rows": [["An unexpected critical error occurred during execution."]]},
                 "text": f"An unexpected critical error occurred while executing the query."
             }
            return json.dumps(fallback_output, default=json_default)

    # BaseTool handles ainvoke implementation for us when _run is async
