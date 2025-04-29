import logging
from typing import Any, Dict, List, Optional, Union, Tuple
import json
import uuid 
import datetime 
import asyncio
import functools
import decimal

from langchain.tools import BaseTool
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from pydantic import BaseModel, Field

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
        log_prefix = f"[Org: {self.organization_id}] [SQLExecutionTool] "
        logger.debug(f"{log_prefix}Executing SQL on DB '{db_name}'...")
        MAX_ROWS = 50
        original_sql = sql
        
        # --- SQL Syntax Validation using sqlparse ---
        try:
            import sqlparse
            parsed_statements = sqlparse.parse(sql)
            if not parsed_statements:
                raise ValueError("Provided SQL is empty or could not be parsed.")
            
            first_statement_type = parsed_statements[0].get_type()
            if first_statement_type not in ('SELECT', 'UNKNOWN'):
                 if first_statement_type != 'UNKNOWN':
                     logger.warning(f"{log_prefix}Provided SQL might not be a SELECT statement (Type: {first_statement_type}). Proceeding cautiously.")
            
            logger.debug(f"{log_prefix}SQL syntax parsed successfully.")
            
        except ImportError:
            logger.warning(f"{log_prefix}sqlparse library not installed. Skipping SQL syntax validation.")
        except ValueError as ve:
             logger.error(f"{log_prefix}SQL validation failed: {ve}. SQL: {sql[:200]}...", exc_info=False)
             raise
        except Exception as validation_err:
            logger.error(f"{log_prefix}Unexpected error during SQL validation: {validation_err}. SQL: {sql}", exc_info=True)
            raise ValueError(f"Provided SQL failed validation: {validation_err}")
        
        # --- Security Check: Verify :organization_id parameter usage in WHERE clause --- #
        try:
            # Use the parsed statements from the syntax validation above
            org_filter_found = False
            for stmt in parsed_statements:
                where_clause = None
                # Find the WHERE clause in the statement
                for token in stmt.tokens:
                    if isinstance(token, sqlparse.sql.Where):
                        where_clause = token
                        break
                
                if where_clause:
                    # Check if the WHERE clause contains the required parameter :organization_id
                    # And references an appropriate column ("organizationId", "parentId", "id")
                    where_content = str(where_clause).upper() # Check content case-insensitively
                    param_present = ":ORGANIZATION_ID" in where_content
                    column_present = any(col in where_content for col in ['"ORGANIZATIONID" = ', '"PARENTID" = ', '"ID" = '])
                    
                    if param_present and column_present:
                        org_filter_found = True
                        logger.debug(f"{log_prefix}Security check passed: :organization_id parameter found in WHERE clause.")
                        break # Found in this statement, no need to check others
            
            if not org_filter_found:
                error_msg = "SECURITY CHECK FAILED: SQL query MUST contain a WHERE clause filtering by :organization_id on 'organizationId', 'parentId', or 'id'."
                logger.error(f"{log_prefix}{error_msg} SQL: {sql[:200]}...")
                raise ValueError(error_msg)
                
        except Exception as filter_check_err: # Catch potential errors during this specific check
            logger.error(f"{log_prefix}Error during SQL organization filter check: {filter_check_err}", exc_info=True)
            # Re-raise as a ValueError to be handled like other validation errors
            raise ValueError(f"Error validating SQL organization filter: {filter_check_err}")
        # --- End Security Check for :organization_id usage --- #

        # --- Security Check: organization_id in parameters dictionary --- 
        if 'organization_id' not in parameters:
             error_msg = f"SECURITY CHECK FAILED: organization_id missing from parameters. Aborting."
             logger.error(f"{log_prefix}{error_msg}")
             raise ValueError(error_msg)
        if parameters['organization_id'] != self.organization_id:
            error_msg = f"SECURITY CHECK FAILED: organization_id mismatch in parameters. Expected {self.organization_id}, got {parameters.get('organization_id')}. Aborting."
            logger.error(f"{log_prefix}{error_msg}")
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
            logger.error(f"{log_prefix}SQL execution error for org {self.organization_id}. Error: {str(e)}", exc_info=True)
            logger.debug(f"{log_prefix}Failed SQL: {sql[:500]}... | Params: {parameters}")
            raise ValueError(f"Database error executing query. Details: {str(e)}")
        except Exception as e:
            logger.error(f"{log_prefix}Unexpected error during SQL execution for org {self.organization_id}: {e}", exc_info=True)
            raise ValueError(f"An unexpected error occurred during query execution: {str(e)}")
    
    async def _run(
        self, sql: str, params: Dict[str, Any], db_name: str = "report_management", run_manager=None
    ) -> str:
        """Execute the SQL query against the database."""
        log_prefix = f"[Org: {self.organization_id}] [SQLExecutionTool] "
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
            logger.exception(f"{log_prefix}Unexpected critical error during SQL execution for org {self.organization_id}, SQL: '{sql[:100]}...': {e}", exc_info=True)
            fallback_output = {
                 "table": {"columns": ["Error"], "rows": [["An unexpected critical error occurred during execution."]]},
                 "text": f"An unexpected critical error occurred while executing the query."
             }
            return json.dumps(fallback_output, default=json_default)

    # BaseTool handles ainvoke implementation for us when _run is async
