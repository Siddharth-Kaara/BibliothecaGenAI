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
        # Use configurable MAX_ROWS from settings
        MAX_ROWS = settings.MAX_SQL_RESULT_ROWS
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
            # --- DYNAMICALLY DERIVE ORG ID COLS --- 
            valid_org_id_cols = set()
            if db_name in SCHEMA_DEFINITIONS:
                # Access the 'tables' dictionary within the db definition
                tables_dict = SCHEMA_DEFINITIONS[db_name].get("tables", {})
                for table_name, table_def in tables_dict.items():
                    # table_def is the dictionary for a single table
                    # table_def['columns'] should be a LIST of column dictionaries
                    column_list = table_def.get("columns", [])
                    if not isinstance(column_list, list):
                        logger.warning(f"{log_prefix}Unexpected structure for columns in table '{table_name}'. Expected list, got {type(column_list)}. Skipping table.")
                        continue

                    # Iterate through the list of column dictionaries
                    for col_def in column_list:
                        if not isinstance(col_def, dict) or "name" not in col_def:
                            # Log unexpected column definition format
                            logger.warning(f"{log_prefix}Unexpected column definition format found in table '{table_name}': {col_def}. Skipping column.")
                            continue 
                        
                        col_name = col_def["name"]
                        # Heuristic: Add columns commonly used for organization linkage
                        if col_name == "organizationId":
                            valid_org_id_cols.add("organizationId")
                        elif col_name == "parentId":
                            valid_org_id_cols.add("parentId")
                        # Add 'id' only if the table is hierarchyCaches (specific known case)
                        elif col_name == "id" and table_name == "hierarchyCaches":
                            valid_org_id_cols.add("id")
            else:
                logger.warning(f"{log_prefix}Schema definition for db '{db_name}' not found. Falling back to default org ID columns.")
            
            if not valid_org_id_cols:
                 logger.error(f"{log_prefix}Could not determine any valid organization ID columns for schema '{db_name}'. Using defaults.")
                 valid_org_id_cols = {"organizationId", "parentId", "id"}
            logger.debug(f"{log_prefix}Dynamically determined valid org ID columns for '{db_name}': {valid_org_id_cols}")
            # --- END DYNAMIC DERIVATION ---
            # Removed old hardcoded set: valid_org_id_cols = {"organizationId", "parentId", "id"} 
            # <<< log_prefix redefinition removed >>>

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

        # We still trust self.organization_id was set correctly during tool init
        if not self.organization_id:
            # This check remains as a safeguard in case the tool was misconfigured
            logger.error(f"{log_prefix}CRITICAL: SQLExecutionTool instance has no self.organization_id set during init. Cannot execute query.")
            raise ValueError("Tool is missing the required organization context.")

        # --- START Query Fingerprint Logging ---
        try:
            # Normalize query: lowercase, replace literals, remove comments
            normalized_sql = sqlparse.format(original_sql, strip_comments=True, reindent=False, keyword_case='lower', identifier_case='lower', use_space_around_operators=False)
            # Replace common literal types for better fingerprinting
            normalized_sql = re.sub(r'\'.*?\'', '?', normalized_sql) # Replace string literals
            normalized_sql = re.sub(r'\b\d+\.?\d*\b', '?', normalized_sql) # Replace numeric literals
            # Generate a simple hash (fingerprint)
            fingerprint = hex(hash(normalized_sql) & 0xffffffffffffffff) # Use 64-bit hash hex
            logger.info(f"{log_prefix}Executing SQL (Fingerprint: {fingerprint}) on DB '{db_name}'.")
        except Exception as fp_err:
            logger.warning(f"{log_prefix}Failed to generate SQL fingerprint: {fp_err}. Proceeding without fingerprint.")
            logger.info(f"{log_prefix}Executing SQL on DB '{db_name}'.") # Log without fingerprint on error
        # --- END Query Fingerprint Logging ---
        
        # <<< START Audit Logging >>>
        try:
             # Use default=str for non-serializable params like UUIDs
            params_str = json.dumps(parameters, default=str)
        except TypeError:
             params_str = "Error serializing parameters"
        logger.info(
            f"SQL_SECURITY_AUDIT|org:{self.organization_id}|"
            f"db:{db_name}|fingerprint:{fingerprint if 'fingerprint' in locals() else 'N/A'}|"
            f"params:{params_str}"
        )
        # <<< END Audit Logging >>>

        # Log parameters *after* fingerprinting/main exec log, before the actual execution block
        logger.debug(f"{log_prefix}With Parameters: {parameters}")

        # <<< START SQL Comment Injection >>>
        # Prepend comment for tracing/auditing in DB logs
        sql_with_comment = f"/* org_id={self.organization_id} tool={self.name} */\n{sql}"
        # <<< END SQL Comment Injection >>>

        try:
            from app.db.connection import get_async_db_connection
            
            async with get_async_db_connection(db_name) as conn:
                # Apply execution timeout using asyncio.wait_for
                timeout_seconds = settings.SQL_EXECUTION_TIMEOUT_SECONDS
                logger.debug(f"{log_prefix}Applying asyncio timeout: {timeout_seconds} seconds.")

                # Execute within the timeout window (use sql_with_comment)
                result = await asyncio.wait_for(
                    conn.execute(text(sql_with_comment), parameters),
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
            logger.error(f"{log_prefix}{error_msg} SQL: {sql[:200]}...", exc_info=False) # exc_info=False for cleaner log for timeout
            # Add original error to the message for _arun to potentially parse
            raise ValueError(f"{error_msg} Original error: {str(te)}") from te

        except SQLAlchemyError as e:
            # Log the full error internally
            logger.error(f"[SQLExecutionTool] SQL execution error for org {self.organization_id}. Error: {str(e)}", exc_info=True)
            logger.debug(f"{log_prefix}Failed SQL: {sql[:500]}... | Params: {parameters}")
            
            # Attempt to get a more specific message from the original exception if available
            specific_detail = ""
            if hasattr(e, 'orig') and e.orig is not None and hasattr(e.orig, 'pgerror') and e.orig.pgerror: # For asyncpg specific errors
                specific_detail = str(e.orig.pgerror).strip()
            elif hasattr(e, 'orig') and e.orig is not None: # General original error
                 specific_detail = str(e.orig).strip()
            elif e.__cause__: # Fallback to cause
                specific_detail = str(e.__cause__).strip()
            else: # Fallback to the error itself
                specific_detail = str(e).strip()
            
            # Clean up common prefixes from SQLAlchemy/DB driver messages for clarity
            specific_detail = re.sub(r"^\(<class '[^']*'>\)\\s*", "", specific_detail) # Removes "(<class 'package.module.ErrorName'>) "
            specific_detail = re.sub(r"^[A-Za-z_]*Error:\\s*", "", specific_detail)    # Removes "ErrorName: "
            specific_detail = specific_detail.replace('\\n', ' ').replace('\n', ' ') # Replace newlines for a cleaner single line message

            final_error_message = f"An error occurred while executing the database query. Details: {specific_detail}"
            raise ValueError(final_error_message) from e

        except Exception as e: # Generic catch-all for other unexpected errors
            # Log the full error internally
            logger.error(f"[SQLExecutionTool] Unexpected error during SQL execution for org {self.organization_id}: {e}", exc_info=True)
            # Raise a generic error message, including details from the original exception
            raise ValueError(f"An unexpected error occurred during query execution. Details: {str(e)}") from e
    
    async def _arun(
        self, sql: str, params: Dict[str, Any], db_name: str = "report_management", run_manager=None
    ) -> str:
        """Execute SQL query and return result or error message."""
        log_prefix = f"[SQLExecutionTool] ({self.name}) "
        logger.info(f"{log_prefix}Executing provided SQL query on DB '{db_name}'.")
        logger.debug(f"{log_prefix}LLM-provided SQL: {sql[:300]}... LLM-provided Params: {params}")

        corrected_params = params.copy() if params is not None else {}
        
        if self.organization_id:
            original_llm_org_id = corrected_params.get('organization_id')
            corrected_params['organization_id'] = self.organization_id
            if original_llm_org_id != self.organization_id:
                logger.info(f"{log_prefix}Overwriting/Correcting params['organization_id']. LLM provided: '{original_llm_org_id}', Corrected to: '{self.organization_id}'.")
            else:
                logger.debug(f"{log_prefix}params['organization_id'] from LLM ('{original_llm_org_id}') matches trusted ID. No correction needed.")
        else:
            logger.error(f"{log_prefix}CRITICAL: SQLExecutionTool instance has no self.organization_id set. Cannot enforce organization scope robustly. Proceeding with LLM-provided params.")

        try:
            result_dict = await self._execute_sql(sql, corrected_params, db_name)
            
            if "error" in result_dict and result_dict["error"]:
                error_type = result_dict["error"].get("type", "UNKNOWN_ERROR")
                error_message = result_dict["error"].get("message", "An unknown error occurred during SQL execution.")
                logger.warning(f"{log_prefix}SQL execution resulted in error. Type: {error_type}, Message: {error_message}")
                return json.dumps({"error": result_dict["error"], "table": None, "text": result_dict.get("text", error_message)}, default=json_default)
            
            final_result_package = {
                "columns": result_dict.get("columns", []),
                "rows": result_dict.get("rows", []),
                "text": result_dict.get("text")
            }
            logger.info(f"{log_prefix}SQL execution successful. Returning {len(final_result_package['rows'])} rows.")
            return json.dumps(final_result_package, default=json_default)

        except ValueError as ve:
            logger.error(f"{log_prefix}ValueError during SQL execution: {ve}", exc_info=True) # exc_info=True to see the full chain
            error_message = str(ve)
            error_type = "VALIDATION_ERROR" # Default, can be overridden by more specific checks

            lower_error_message = error_message.lower()
            if "column" in lower_error_message and "does not exist" in lower_error_message:
                error_type = "DATABASE_UNDEFINED_COLUMN_ERROR"
            elif "syntax error" in lower_error_message or ("near" in lower_error_message and any(kw in lower_error_message for kw in ["select", "from", "where", "group by", "order by"])):
                error_type = "DATABASE_SYNTAX_ERROR"
            elif "timeout" in lower_error_message and "exceeded" in lower_error_message:
                error_type = "QUERY_TIMEOUT_ERROR"
            elif "unique constraint" in lower_error_message or "duplicate key" in lower_error_message:
                 error_type = "DATABASE_INTEGRITY_ERROR"
            elif "division by zero" in lower_error_message:
                 error_type = "DATABASE_NUMERIC_ERROR"
            elif "invalid input" in lower_error_message and ("type" in lower_error_message or "syntax for type" in lower_error_message or "value too long for type" in lower_error_message):
                 error_type = "DATABASE_TYPE_CONVERSION_ERROR"
            elif "an error occurred while executing the database query" in lower_error_message: # Generic DB execution error
                error_type = "DATABASE_EXECUTION_ERROR"
            # If "permission denied" or "access denied"
            elif "permission denied" in lower_error_message or "access denied" in lower_error_message:
                error_type = "DATABASE_PERMISSION_ERROR"
            # If "deadlock"
            elif "deadlock" in lower_error_message:
                error_type = "DATABASE_DEADLOCK_ERROR"
            # If "connection" error (but not timeout, which is handled above)
            elif "connection" in lower_error_message and not ("timeout" in lower_error_message):
                error_type = "DATABASE_CONNECTION_ERROR"

            return json.dumps({"error": {"type": error_type, "message": error_message}, "table": None, "text": f"Error during SQL operation: {error_message}"}, default=json_default)
        
        except SQLAlchemyError as db_err: # This might be hit if _execute_sql directly re-raises an SQLAlchemyError not wrapped in ValueError
            logger.error(f"{log_prefix}Database error during SQL execution: {db_err}", exc_info=True)
            
            specific_detail = ""
            if hasattr(db_err, 'orig') and db_err.orig is not None and hasattr(db_err.orig, 'pgerror') and db_err.orig.pgerror:
                specific_detail = str(db_err.orig.pgerror).strip()
            elif hasattr(db_err, 'orig') and db_err.orig is not None:
                 specific_detail = str(db_err.orig).strip()
            elif db_err.__cause__:
                specific_detail = str(db_err.__cause__).strip()
            else:
                specific_detail = str(db_err).strip()
            
            specific_detail = re.sub(r"^\(<class '[^']*'>\)\\s*", "", specific_detail)
            specific_detail = re.sub(r"^[A-Za-z_]*Error:\\s*", "", specific_detail)
            specific_detail = specific_detail.replace('\\n', ' ').replace('\n', ' ')

            error_message = f"Database execution error. Details: {specific_detail}"
            error_type = "DATABASE_ERROR" 

            lower_specific_detail = specific_detail.lower()
            if "column" in lower_specific_detail and "does not exist" in lower_specific_detail:
                error_type = "DATABASE_UNDEFINED_COLUMN_ERROR"
            elif "syntax error" in lower_specific_detail:
                error_type = "DATABASE_SYNTAX_ERROR"
            elif "unique constraint" in lower_specific_detail or "duplicate key" in lower_specific_detail:
                 error_type = "DATABASE_INTEGRITY_ERROR"
            elif "division by zero" in lower_specific_detail:
                 error_type = "DATABASE_NUMERIC_ERROR"
            elif "invalid input" in lower_specific_detail and ("type" in lower_specific_detail or "syntax for type" in lower_specific_detail or "value too long for type" in lower_specific_detail):
                 error_type = "DATABASE_TYPE_CONVERSION_ERROR"
            elif "permission denied" in lower_specific_detail or "access denied" in lower_specific_detail:
                error_type = "DATABASE_PERMISSION_ERROR"
            elif "deadlock" in lower_specific_detail:
                error_type = "DATABASE_DEADLOCK_ERROR"
            elif "connection" in lower_specific_detail and not ("timeout" in lower_specific_detail): # Check not timeout
                error_type = "DATABASE_CONNECTION_ERROR"
                
            return json.dumps({"error": {"type": error_type, "message": error_message}, "table": None, "text": f"Error executing SQL: {error_message}"}, default=json_default)
        
        except Exception as e:
            logger.error(f"{log_prefix}Unexpected error in _arun: {e}", exc_info=True)
            error_message = f"An unexpected error occurred in the SQL tool: {str(e)}"
            return json.dumps({"error": {"type": "TOOL_INTERNAL_ERROR", "message": error_message}, "table": None, "text": f"Unexpected tool error: {error_message}"}, default=json_default)

    # Synchronous _run for completeness, if BaseTool doesn't auto-generate from _arun
    # or if some other part of the system might try to call it synchronously.
    # For robust async operation, _arun is the primary method.
    def _run(
        self, sql: str, params: Dict[str, Any], db_name: str = "report_management", run_manager=None
    ) -> str:
        logger.warning("[SQLExecutionTool] Synchronous _run called. Consider using async interface if possible.")
        try:
            # This is a blocking call for the async event loop.
            # Not ideal for production async servers if called frequently.
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If an event loop is running, we cannot simply run_until_complete.
                # This path might lead to issues or require a more sophisticated async-to-sync bridge.
                # For simplicity, we raise an error or log a strong warning if called in an async context.
                logger.error("[SQLExecutionTool] Synchronous _run was called from within an existing asyncio event loop. This is not supported and may block the loop. Returning error.")
                return json.dumps({"error": {"type": "ASYNC_CONTEXT_ERROR", "message": "Synchronous SQL tool execution attempted from async context."}})
            else:
                return loop.run_until_complete(self._arun(sql, params, db_name, run_manager))
        except RuntimeError as re:
             if "There is no current event loop in thread" in str(re):
                 # No event loop, create a new one to run the async method
                 new_loop = asyncio.new_event_loop()
                 asyncio.set_event_loop(new_loop)
                 result = new_loop.run_until_complete(self._arun(sql, params, db_name, run_manager))
                 new_loop.close()
                 return result
             else:
                 raise # Re-raise other RuntimeErrors
        except Exception as e:
            logger.error(f"[SQLExecutionTool] Error in synchronous _run wrapper: {e}", exc_info=True)
            return json.dumps({"error": {"type": "SYNC_WRAPPER_ERROR", "message": f"Error in sync execution: {str(e)}"}})
