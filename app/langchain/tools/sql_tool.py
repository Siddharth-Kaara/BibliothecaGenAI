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
from langchain_core.runnables import RunnableConfig 
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from pydantic import BaseModel, Field, ValidationError

from app.core.config import settings
from app.db.schema_definitions import SCHEMA_DEFINITIONS

logger = logging.getLogger(__name__)

# Helper function for JSON serialization
def json_default(obj):
    if isinstance(obj, uuid.UUID):
        # Convert UUID to string
        return str(obj)
    if isinstance(obj, (datetime.datetime, datetime.date, datetime.time)):
        # Keep this for potential future use or if params contain dates
        return obj.isoformat()
    if isinstance(obj, decimal.Decimal):
        # Convert Decimal to float for JSON serialization
        # Use str(obj) if exact decimal representation as string is needed
        return float(obj)
    # Let the base class default method raise the TypeError
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

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
            # First try to validate using sqlparse if available
            sqlparse_available = False
            try:
                import sqlparse
                sqlparse_available = True
            except ImportError:
                logger.warning(f"{log_prefix}sqlparse library not installed. Using regex-based validation as fallback.")
            
            # Store SQL elements for security verification
            cte_blocks = []
            subqueries = []
            main_query = sql
            
            if sqlparse_available:
                # Parse with sqlparse for syntax validation
                parsed_statements = sqlparse.parse(sql)
                if not parsed_statements:
                    raise ValueError("Provided SQL is empty or could not be parsed.")
                
                first_statement_type = parsed_statements[0].get_type()
                if first_statement_type not in ('SELECT', 'UNKNOWN'):
                    if first_statement_type != 'UNKNOWN':
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
                                select_pos = sql.upper().find('SELECT', cte_start + 4)
                                if select_pos > 0:
                                    # Extract the CTE part
                                    cte_part = sql[cte_start:select_pos].strip()
                                    cte_blocks.append(cte_part)
                                    # Main query starts at SELECT
                                    main_query = sql[select_pos:]
                            break
                    
                    # Extract subqueries using sqlparse
                    for token in stmt.flatten():
                        if isinstance(token, sqlparse.sql.Parenthesis):
                            subquery_text = str(token).strip()
                            if 'SELECT' in subquery_text.upper():
                                subqueries.append(subquery_text)
            else:
                # Fallback to regex-based SQL parsing
                # Extract CTE blocks
                cte_match = re.search(r'WITH\s+(.+?)(?=SELECT)', sql, re.IGNORECASE | re.DOTALL)
                if cte_match:
                    cte_blocks.append(f"WITH {cte_match.group(1)}")
                    # Main query starts at SELECT
                    select_pos = sql.upper().find('SELECT', cte_match.end())
                    if select_pos > 0:
                        main_query = sql[select_pos:]
                
                # Extract subqueries using regex
                # This is a simplified approach - won't catch all cases
                subquery_matches = re.finditer(r'\(\s*SELECT.*?\)', sql, re.IGNORECASE | re.DOTALL)
                for match in subquery_matches:
                    subqueries.append(match.group(0))
                
            logger.debug(f"{log_prefix}SQL parsing complete. Found {len(cte_blocks)} CTE blocks, {len(subqueries)} subqueries.")
            
        except ValueError as ve:
            logger.error(f"{log_prefix}SQL validation failed: {ve}. SQL: {sql[:200]}...", exc_info=False)
            raise
        except Exception as validation_err:
            logger.error(f"{log_prefix}Unexpected error during SQL validation: {validation_err}. SQL: {sql}", exc_info=True)
            raise ValueError(f"Provided SQL failed validation: {validation_err}")
        
        # --- Security Check: Verify organization_id parameter usage in all SQL components ---
        try:
            # Function to check for organization_id filter using regex
            def check_filter(sql_segment: str) -> bool:
                # SIMPLIFIED REGEX: Checks if :organization_id is used in a comparison
                # with organizationId, parentId, or id (case-insensitive, allows quotes).
                # Looks for: "column_name" [operator] :organization_id
                # Does NOT strictly validate the operator list anymore, focuses on linkage.
                pattern = r'["\']?(?:organizationid|parentid|id)["\']?\s*(?:[=<>]|!=|\bIN\b)\s*:organization_id\b'
                # Alternative more robust pattern if needed:
                # pattern = r'(?i)["\']?(?:organizationid|parentId|id)["\']?\s*(?:=|>=|<=|<>|!=|\bIN\b)\s*:\s*organization_id\b'
                return bool(re.search(pattern, sql_segment, re.IGNORECASE))

            # Check main query for organization_id filter
            if not check_filter(main_query):
                error_msg = "SECURITY CHECK FAILED: Main SQL query MUST contain a WHERE clause filtering by :organization_id on 'organizationId', 'parentId', or 'id'."
                logger.error(f"{log_prefix}{error_msg} SQL: {main_query[:200]}...")
                raise ValueError(error_msg)
            
            # Check CTE blocks for organization_id filter
            for i, cte_block in enumerate(cte_blocks):
                if not check_filter(cte_block):
                    error_msg = f"SECURITY CHECK FAILED: CTE block #{i+1} is missing required organization_id filter. Every CTE must include a filter on 'organizationId', 'parentId', or 'id' using the :organization_id parameter."
                    logger.error(f"{log_prefix}{error_msg} CTE Block: {cte_block[:200]}...")
                    raise ValueError(error_msg)
            
            # Check subqueries for organization_id filter (more complex ones)
            for i, subquery in enumerate(subqueries):
                # Only check substantive subqueries (larger than a simple lookup)
                if len(subquery) > 100:  # Arbitrary threshold to avoid checking simple subqueries
                     if not check_filter(subquery):
                        error_msg = f"SECURITY CHECK FAILED: Complex subquery #{i+1} is missing required organization_id filter. Each substantive subquery must include a filter on 'organizationId', 'parentId', or 'id' using the :organization_id parameter."
                        logger.error(f"{log_prefix}{error_msg} Subquery: {subquery[:200]}...")
                        raise ValueError(error_msg)
            
            logger.debug(f"{log_prefix}Security check passed: organization_id parameter found in all required SQL components.")
                
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
                result = await conn.execute(text(sql), parameters)
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
                
        except SQLAlchemyError as e:
            logger.error(f"[SQLExecutionTool] SQL execution error for org {self.organization_id}. Error: {str(e)}", exc_info=True)
            logger.debug(f"{log_prefix}Failed SQL: {sql[:500]}... | Params: {parameters}")
            raise ValueError(f"Database error executing query. Details: {str(e)}")
        except Exception as e:
            logger.error(f"[SQLExecutionTool] Unexpected error during SQL execution for org {self.organization_id}: {e}", exc_info=True)
            raise ValueError(f"An unexpected error occurred during query execution: {str(e)}")
    
    async def _run(
        self, sql: str, params: Dict[str, Any], db_name: str = "report_management", run_manager=None
    ) -> str:
        """Execute the SQL query against the database."""
        log_prefix = f"[SQLExecutionTool] "
        logger.info(f"{log_prefix}Executing provided SQL query on DB '{db_name}'.")
        
        try:
            results = await self._execute_sql(sql, params, db_name)
            row_count = len(results.get('rows', []))
            logger.info(f"{log_prefix}Completed successfully. Returned {row_count} rows.")
            
            text_summary = f"Executed query successfully, retrieved {row_count} rows."
            if results.get("metadata", {}).get("truncated"):
                text_summary += f" (Results truncated to {results['metadata']['rows_shown']} rows from {results['metadata']['total_rows_returned']} total)."

            output_dict = {
                "table": results,
                "text": text_summary
            }
            return json.dumps(output_dict, default=json_default)
            
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

    # Explicitly define ainvoke for clarity and consistency
    async def ainvoke(self, tool_input: Union[str, Dict], config: Optional[RunnableConfig] = None, **kwargs: Any) -> str:
        """Execute the SQL query asynchronously."""
        # Validate and parse the input using the tool's args_schema
        try:
            # Ensure input is a dictionary before passing to schema
            if isinstance(tool_input, str):
                # Attempt to parse if it's a JSON string, otherwise raise error
                try:
                    input_dict = json.loads(tool_input)
                except json.JSONDecodeError:
                    raise ValueError(f"Tool input must be a valid JSON string or a dictionary, received string: {tool_input[:100]}...")
            elif isinstance(tool_input, dict):
                input_dict = tool_input
            else:
                raise ValueError(f"Tool input must be a dictionary or JSON string, received {type(tool_input)}")
                
            # Use the args_schema for validation
            validated_args = self.args_schema(**input_dict)
            
            # Extract arguments for _run from the validated model
            sql = validated_args.sql
            params = validated_args.params
            db_name = validated_args.db_name # Takes default if not provided
                
        except ValidationError as e:
            raise ValueError(f"Input validation failed for {self.name}: {e}") from e
        except Exception as e:
             # Catch other potential errors during parsing/validation
            raise ValueError(f"Error processing input for {self.name}: {e}") from e
        
        # Directly await the _run method with validated arguments
        # No need to pass run_manager explicitly, BaseTool handles callbacks via config
        return await self._run(sql=sql, params=params, db_name=db_name)
