import logging
from typing import Any, Dict, List, Optional, Union, Tuple
import json
import uuid 
import datetime 
import asyncio
import functools
import decimal

from langchain.tools import BaseTool
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import AzureChatOpenAI
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from pydantic import BaseModel, Field

from app.core.config import settings
from app.db.connection import get_db_engine
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

class SQLQueryTool(BaseTool):
    """Tool for querying SQL databases, ensuring results are scoped to the user's organization."""
    
    name: str = "sql_query"
    description: str = """
    Executes organization-scoped SQL queries against PostgreSQL databases.
    Use this tool when you need to fetch specific data from the database.
    The tool handles query generation and execution, automatically filtering by the user's organization.
    Input should be a description of the data needed (e.g., 'total borrows last week').
    DO NOT include organization filtering in the description; the tool adds it automatically.
    """
    
    organization_id: str
    selected_db: Optional[str] = None
    
    def _get_schema_info(self, db_name: Optional[str] = None) -> str:
        """Get schema information from predefined schema definitions."""
        if not db_name and not self.selected_db:
            all_schemas = []
            for db_name, db_info in SCHEMA_DEFINITIONS.items():
                all_schemas.append(f"Database: {db_name}")
                all_schemas.append(f"Description: {db_info['description']}")
                
                for table_name, table_info in db_info['tables'].items():
                    all_schemas.append(f"  Table: {table_name}")
                    all_schemas.append(f"  Description: {table_info['description']}")
                    
                    all_schemas.append("  Columns:")
                    for column in table_info['columns']:
                        primary_key = " (PRIMARY KEY)" if column.get('primary_key') else ""
                        foreign_key = f" (FOREIGN KEY -> {column.get('foreign_key')})" if column.get('foreign_key') else ""
                        timestamp_note = " (Timestamp for filtering)" if 'timestamp' in column['type'].lower() else ""
                        all_schemas.append(f"    {column['name']} ({column['type']}){primary_key}{foreign_key} - {column['description']}{timestamp_note}")
                    
                    if 'example_queries' in table_info:
                        all_schemas.append("  Example queries:")
                        for query in table_info['example_queries']:
                            all_schemas.append(f"    {query}")
                    
                    all_schemas.append("")  # Empty line between tables
            
            return "\n".join(all_schemas)
        
        target_db = db_name or self.selected_db
        if target_db not in SCHEMA_DEFINITIONS:
            return f"No schema definition found for database {target_db}."
        
        db_info = SCHEMA_DEFINITIONS[target_db]
        schema_info = [
            f"Database: {target_db}",
            f"Description: {db_info['description']}",
            ""
        ]
        
        for table_name, table_info in db_info['tables'].items():
            schema_info.append(f"Table: {table_name}")
            schema_info.append(f"Description: {table_info['description']}")
            
            schema_info.append("Columns:")
            for column in table_info['columns']:
                primary_key = " (PRIMARY KEY)" if column.get('primary_key') else ""
                foreign_key = f" (FOREIGN KEY -> {column.get('foreign_key')})" if column.get('foreign_key') else ""
                timestamp_note = " (Timestamp for filtering)" if 'timestamp' in column['type'].lower() else ""
                schema_info.append(f"  {column['name']} ({column['type']}){primary_key}{foreign_key} - {column['description']}{timestamp_note}")
            
            if 'example_queries' in table_info:
                schema_info.append("Example queries:")
                for query in table_info['example_queries']:
                    schema_info.append(f"  {query}")
            
            schema_info.append("")  # Empty line between tables
        
        return "\n".join(schema_info)
    
    def _generate_sql(
        self, 
        query_description: str, 
        db_name: str
    ) -> Tuple[str, Dict[str, Any]]:
        """Generate SQL with placeholders and parameters from a natural language query description using LCEL, enforcing organization filtering."""
        schema_info = self._get_schema_info(db_name)
        
        llm = AzureChatOpenAI(
            openai_api_key=settings.AZURE_OPENAI_API_KEY,
            azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
            openai_api_version=settings.AZURE_OPENAI_API_VERSION,
            deployment_name=settings.AZURE_OPENAI_DEPLOYMENT_NAME,
            model_name=settings.LLM_MODEL_NAME,
            temperature=0,
        )
        
        # Integrating org filtering into the existing prompt structure
        template = """You are a SQL expert. Given the following database schema and a query description,
generate a PostgreSQL SQL query and its corresponding parameters dictionary.

Schema:
{schema}

Query description: {query_description}

Important Guidelines:
1. Use parameter placeholders (e.g., :filter_value, :hierarchy_id) for ALL dynamic values derived from the query description (like names, IDs, specific filter values) EXCEPT for the mandatory :organization_id and time-related values. DO NOT use parameters for date/time calculations.
2. Generate a valid JSON dictionary mapping placeholder names (without the colon) to their actual values. This MUST include `organization_id`.
3. Quote table and column names with double quotes (e.g., "hierarcyCaches", "createdAt"). Use the actual table names from the schema (e.g., '5', '8') when writing the SQL, even though logical names ('events', 'footfall') are used in the schema description itself.
4. **Mandatory Organization Filtering:** ALWAYS filter results by the organization ID. Use the parameter `:organization_id`. Add the appropriate WHERE clause:
    *   If querying table '5' (event data), add `"5"."organizationId" = :organization_id` to your WHERE clause (using AND if other conditions exist).
    *   If querying `hierarchyCaches` directly for the organization's details, filter using `"id" = :organization_id`.
    *   If querying `hierarchyCaches` for specific locations *within* an organization, ensure the data relates back to the `:organization_id` (e.g., via JOIN or direct filter on `parentId` if appropriate).
    *   If querying table '8' (footfall data), add `"8"."organizationId" = :organization_id` to your WHERE clause.
    *   You MUST include `:organization_id` as a key in the `params` dictionary with the correct value. **Use the exact `organization_id` value provided to you in the context (e.g., '{organization_id}'), do NOT use example UUIDs or placeholders like 'your-organization-uuid'.**
5. **JOINs for Related Data:** When joining table '5' or '8' and `hierarchyCaches`, use appropriate keys like `"5"."hierarchyId" = hc."id"` or `"8"."hierarchyId" = hc."id"`. Remember to apply the organization filter (Guideline #4).
6. **Case Sensitivity:** PostgreSQL is case-sensitive; respect exact table/column capitalization.
7. **Column Selection:** Use specific column selection instead of SELECT *.
8. **Sorting:** Add ORDER BY clauses for meaningful sorting, especially when LIMIT is used.
9. **LIMIT Clause:**
    *   For standard SELECT queries retrieving multiple rows, ALWAYS include `LIMIT 50` at the end.
    *   **DO NOT** add `LIMIT` for aggregate queries (like COUNT(*), SUM(...)) expected to return a single summary row.
10. **Aggregations (COUNT vs SUM):**
    *   Use `COUNT(*)` for "how many records/items".
    *   Use `SUM("column_name")` for "total number/sum" based on a specific value column (e.g., total logins from column "5").
    *   Ensure `GROUP BY` includes all non-aggregated selected columns.
11. **User-Friendly Aliases:**
    *   When selecting columns or using aggregate functions (SUM, COUNT, etc.), ALWAYS use descriptive, user-friendly aliases with title casing using the `AS` keyword.
    *   Examples: `SELECT hc."hierarchyId" AS "Hierarchy ID"`, `SELECT COUNT(*) AS "Total Records"`, `SELECT SUM("39") AS "Total Entries"`.
    *   Do NOT use code-style aliases like `total_entries` or `hierarchyId`.
12. **Benchmarking for Analytical Queries:**
    *   If the `query_description` asks for analysis or comparison regarding a specific entity (e.g., "is branch X busy?", "compare borrows for branch Y"), *in addition* to selecting the specific metric(s) for that entity, try to include a simple benchmark for comparison in the same query.
    *   **Use CTEs for Benchmarks:** The preferred way to calculate an organization-wide average (or similar benchmark) alongside a specific entity's value is using a Common Table Expression (CTE).
        *   First, define a CTE (e.g., `WITH EntityMetrics AS (...)`) that calculates the metric (e.g., `SUM("metric_column")`) grouped by the relevant entity ID (`hierarchyId`). Apply necessary time/organization filters within the CTE.
        *   In the final `SELECT` statement, query the CTE (`FROM EntityMetrics em`), filter for the specific entity ID (`WHERE em."hierarchyId" = :target_entity_id`), and select the entity's metric.
        *   Calculate the overall average using a subquery on the CTE (e.g., `(SELECT AVG(metric_sum) FROM EntityMetrics) AS "Org Average Metric"`).
    *   **Avoid nested aggregates:** Do NOT use invalid nested aggregate/window functions like `AVG(SUM(...)) OVER ()`.
    *   Only include this benchmark if it can be done efficiently. The CTE approach is generally efficient.
    *   Ensure both the specific value and the benchmark value have clear, user-friendly aliases.
13. **Time Filtering (Generate SQL Directly):**
    *   If the `query_description` includes time references (e.g., "last week", "yesterday", "past 3 months", "since June 1st", "before 2024"), you MUST generate the appropriate SQL `WHERE` clause condition directly.
    *   Use relevant SQL functions like `NOW()`, `CURRENT_DATE`, `INTERVAL`, `DATE_TRUNC`, `EXTRACT`, and comparison operators (`>=`, `<`, `BETWEEN`).
    *   **Relative Time Interpretation:** For simple relative terms like "last week", "last month", prioritize using straightforward intervals like `NOW() - INTERVAL '7 days'` or `NOW() - INTERVAL '1 month'`, respectively. Use `DATE_TRUNC` or specific date ranges only if the user query explicitly demands calendar alignment (e.g., "the week starting Monday", "the calendar month of March").
    *   **Relative Months/Years:** For month names (e.g., "March", "in June") without a specified year, **ALWAYS** assume the **current year** in your date logic. For years alone (e.g., "in 2024"), query the whole year. **Critically, incorporate the current year directly into your date comparisons using `NOW()` or `CURRENT_DATE` where appropriate, don't just extract the year separately and then use a hardcoded year in the comparison.**
    *   Identify the correct timestamp column for filtering (e.g., `"eventTimestamp"` for table `"5"` and `"8"`, `"createdAt"` for others - check schema).
    *   Example for "last week": `WHERE "eventTimestamp" >= NOW() - INTERVAL '7 days'` # Prefer this
    *   Example for "yesterday": `WHERE DATE_TRUNC('day', "eventTimestamp") = CURRENT_DATE - INTERVAL '1 day'` # DATE_TRUNC makes sense here
    *   Example for "March" (current year): `WHERE EXTRACT(MONTH FROM "eventTimestamp") = 3 AND EXTRACT(YEAR FROM "eventTimestamp") = EXTRACT(YEAR FROM NOW())` # Check month AND current year
    *   Example for "first week of February" (current year): `WHERE "eventTimestamp" >= DATE_TRUNC('year', NOW()) + INTERVAL '1 month' AND "eventTimestamp" < DATE_TRUNC('year', NOW()) + INTERVAL '1 month' + INTERVAL '7 days'` 
    *   Example for "June 2024": `WHERE "eventTimestamp" >= '2024-06-01' AND "eventTimestamp" < '2024-07-01'`
    *   **DO NOT** use parameters like `:start_date` or `:end_date` for these time calculations.
14. **Footfall Queries (Table "8"):**
    *   If the query asks generally about "footfall", "visitors", "people entering/leaving", or "how many people visited", calculate **both** the sum of entries (`SUM("39")`) and the sum of exits (`SUM("40")`).
    *   Alias them clearly (e.g., `AS "Total Entries"`, `AS "Total Exits"`).
    *   If the query specifically asks *only* for entries (e.g., "people came in") or *only* for exits (e.g., "people went out"), then only sum the corresponding column ("39" or "40").

Output Format:
Return ONLY a JSON object with two keys:
- "sql": A string containing the generated SQL query (potentially with direct time logic) and placeholders for non-time values.
- "params": A JSON dictionary mapping placeholder names (e.g., filter_value, organization_id) to their corresponding values. This dictionary should NOT contain start/end dates.
**Do not copy parameter values directly from the examples; use the actual values relevant to the query description and the provided Organization ID.**

Example (Query hierarchy details for the Org):
{{
  "sql": "SELECT \"name\", \"shortName\" FROM \"hierarchyCaches\" hc WHERE hc.\"id\" = :organization_id LIMIT 50",
  "params": {{ "organization_id": "{organization_id}" }}
}}

Example (Aggregate Query for the Org):
{{
   "sql": "SELECT COUNT(*) FROM \"5\" WHERE \"organizationId\" = :organization_id AND \"eventTimestamp\" >= NOW() - INTERVAL '1 month'",
   "params": {{ "organization_id": "{organization_id}" }}
}}
"""
        
        prompt = PromptTemplate(
            input_variables=["schema", "organization_id", "query_description"],
            template=template,
        )
        
        class SQLOutput(BaseModel):
            sql: str = Field(description="SQL query with placeholders")
            params: Dict[str, Any] = Field(description="Dictionary of parameters")
        
        sql_chain = prompt | llm | JsonOutputParser(pydantic_object=SQLOutput)
        
        logger.debug(f"Invoking SQL generation chain for org {self.organization_id} with query: {query_description}")
            
        try:
            invoke_payload = {
                "schema": schema_info,
                "organization_id": self.organization_id,
                "query_description": query_description,
            }
            
            structured_output = sql_chain.invoke(invoke_payload)
            logger.debug(f"Raw LLM Output for SQL generation: {structured_output}") 
            
            sql_query = structured_output.get('sql', '')
            parameters = structured_output.get('params', {})
            
            if not isinstance(sql_query, str) or not isinstance(parameters, dict):
                 logger.error(f"LLM returned unexpected types. SQL: {type(sql_query)}, Params: {type(parameters)}")
                 raise ValueError("LLM failed to return the expected SQL/parameter structure.")
            
            if not sql_query:
                raise ValueError("LLM failed to generate an SQL query string.")

            if 'organization_id' not in parameters:
                logger.warning(f":organization_id missing from LLM params. Manually adding {self.organization_id}.")
                parameters['organization_id'] = self.organization_id
            elif parameters['organization_id'] != self.organization_id:
                logger.warning(f"LLM parameter :organization_id ({parameters['organization_id']}) != tool's ({self.organization_id}). Overwriting.")
                parameters['organization_id'] = self.organization_id
            else:
                logger.debug(f":organization_id ({self.organization_id}) present and correct in LLM params.")
                
            # Remove any user_id parameter if LLM included it erroneously
            if 'user_id' in parameters:
                logger.warning("LLM included :user_id parameter erroneously. Removing.")
                del parameters['user_id']
            
            logger.debug(f"Generated SQL: {sql_query}, Params: {parameters}")
            return sql_query, parameters
        except Exception as e:
            logger.error(f"Error generating SQL: {e}", exc_info=True)
            raise
    
    def _execute_sql(self, sql: str, parameters: Dict[str, Any], db_name: str) -> Dict:
        """Execute SQL with parameters and return results."""
        MAX_ROWS = 50 # Used for Python-level truncation check if needed (e.g., if LLM forgets LIMIT)
        original_sql = sql # Keep for logging/checks if needed
        
        # --- SQL Syntax Validation using sqlparse ---
        try:
            import sqlparse
            # Attempt to parse the SQL
            parsed_statements = sqlparse.parse(sql)
            if not parsed_statements:
                raise ValueError("Generated SQL is empty or could not be parsed.")
            
            # Basic check: Ensure it's a SELECT statement (allowing CTEs)
            first_statement_type = parsed_statements[0].get_type()
            if first_statement_type not in ('SELECT', 'UNKNOWN'): # UNKNOWN can be CTEs (WITH ... SELECT)
                 if first_statement_type != 'UNKNOWN':
                     logger.warning(f"Generated SQL might not be a SELECT statement (Type: {first_statement_type}). Proceeding cautiously.")
                     # Optionally raise stricter error:
                     # raise ValueError(f"Generated SQL is not a SELECT statement (Type: {first_statement_type}). Only SELECT is allowed.")
            
            logger.debug("SQL syntax parsed successfully.")
            
        except ImportError:
            logger.warning("sqlparse library not installed or found. Skipping SQL syntax validation.")
        except ValueError as ve:
             logger.error(f"SQL validation failed: {ve}. SQL: {sql}", exc_info=True)
             raise # Re-raise the ValueError to be caught by _run
        except Exception as validation_err:
            logger.error(f"Unexpected error during SQL validation: {validation_err}. SQL: {sql}", exc_info=True)
            raise ValueError(f"Generated SQL failed validation: {validation_err}")
        # --- End Validation ---
        
        # <<< ADDED: CRITICAL Check for organization_id before execution >>>
        if 'organization_id' not in parameters or parameters['organization_id'] != self.organization_id:
             error_msg = f"SECURITY CHECK FAILED: organization_id mismatch/missing in parameters before execution. Expected {self.organization_id}, got {parameters.get('organization_id')}. Aborting."
             logger.error(error_msg)
             raise ValueError(error_msg)
        
        engine = get_db_engine(db_name)
        if not engine:
            raise ValueError(f"Database engine for '{db_name}' not found")
        
        # Log execution details - be mindful of sensitive data in parameters in production
        logger.debug(f"Executing SQL for org {self.organization_id}: {sql}")
        logger.debug(f"With Parameters: {parameters}") # Ensure datetime objects are handled correctly by logger/SQLAlchemy
        
        try:
            with engine.connect() as conn:
                # Execute with parameters for safety
                result = conn.execute(text(sql), parameters)
                columns = list(result.keys())
                
                # Fetch rows
                raw_rows = result.fetchall() 
                
                truncated = False
                total_count = len(raw_rows)
                
                # Use original_sql check for COUNT queries because `sql` might have placeholders
                is_count_query = original_sql.strip().upper().startswith("SELECT COUNT")
                
                if not is_count_query and total_count > MAX_ROWS:
                    # This acts as a safeguard if the LLM generates a large LIMIT or no LIMIT.
                    truncated = True
                    raw_rows = raw_rows[:MAX_ROWS]
                    logger.warning(f"Query results exceeded {MAX_ROWS} rows. Truncating.")
                
                rows = [list(row) for row in raw_rows]
                
                if is_count_query:
                     if len(rows) == 1 and len(columns) == 1:
                         return {"columns": columns, "rows": rows}
                     else:
                         logger.warning(f"COUNT query returned unexpected structure: {columns}, {rows}")
                         # Fall through to return structure anyway
                 
                # Include truncation info if applicable
                response_data = {"columns": columns, "rows": rows}
                if truncated:
                    response_data["metadata"] = {"truncated": True, "total_rows_returned": total_count, "rows_shown": MAX_ROWS}
                    
                return response_data
                
        except SQLAlchemyError as e:
            # Log the specific SQL and params that caused the error
            logger.error(f"SQL execution error for org {self.organization_id}, query: {sql}, params: {parameters}. Error: {str(e)}", exc_info=True)
            # Provide a more informative error message
            raise ValueError(f"Database error executing query. Please check query syntax and parameters. Details: {str(e)}")
        except Exception as e: # Catch other potential errors (like connection issues)
            logger.error(f"Unexpected error during SQL execution for org {self.organization_id}: {e}", exc_info=True)
            raise ValueError(f"An unexpected error occurred during query execution: {str(e)}")
    
    def _run(
        self, query_description: str, db_name: Optional[str] = None
    ) -> str:
        """Run the tool: generate parameterized SQL, execute, format results."""
        logger.info(f"Executing SQL query tool for org {self.organization_id} with description: '{query_description}'")
        
        target_db = db_name or self.selected_db
        if not target_db:
            if not SCHEMA_DEFINITIONS:
                logger.error("No database schemas defined in SCHEMA_DEFINITIONS.")
                raise ValueError("No database schemas defined.")
            # Fallback to the first defined database schema if none is specified
            try:
                target_db = next(iter(SCHEMA_DEFINITIONS.keys()))
                logger.info(f"No database specified, using first defined schema: {target_db}")
            except StopIteration: # Handle empty SCHEMA_DEFINITIONS case
                 logger.error("SCHEMA_DEFINITIONS is empty. Cannot select a default database.")
                 raise ValueError("No database schemas available to select a default.")

        self.selected_db = target_db # Store for potential future calls within the same agent run

        # Execute the main SQL generation and execution logic
        try:
            # Generate SQL and parameters (no dates passed)
            sql, parameters = self._generate_sql(query_description, target_db)
            
            results = self._execute_sql(sql, parameters, target_db)
            row_count = len(results.get('rows', []))
            logger.info(f"SQL query returned {row_count} rows for org {self.organization_id}, description: '{query_description}'")
            
            text_summary = f"Retrieved {row_count} rows of data matching your query."
            if results.get("metadata", {}).get("truncated"):
                text_summary += f" (Results truncated to {results['metadata']['rows_shown']} rows from {results['metadata']['total_rows_returned']} total)."

            output_dict = {
                "table": results, # Includes potential metadata key
                "text": text_summary
            }
            return json.dumps(output_dict, default=json_default)
            
        except ValueError as ve: # Catch generation/execution ValueErrors
             logger.error(f"SQL Tool failed for org {self.organization_id}, description '{query_description}': {ve}", exc_info=False) # Keep log cleaner
             # Return structured error message to the agent/user
             fallback_output = {
                 "table": {"columns": ["Error"], "rows": [[f"Failed to process query: {ve}"]]},
                 "text": f"Error processing your query: {ve}"
             }
             return json.dumps(fallback_output, default=json_default)
        except Exception as e: # Catch other unexpected errors
            logger.exception(f"Unexpected critical error in SQL Tool for org {self.organization_id}, description '{query_description}': {e}", exc_info=True)
            fallback_output = {
                 "table": {"columns": ["Error"], "rows": [["An unexpected critical error occurred."]]},
                 "text": f"An unexpected critical error occurred while processing your query."
             }
            return json.dumps(fallback_output, default=json_default)
    
    async def _arun(
        self, query_description: str, db_name: Optional[str] = None
    ) -> str:
        """Run the tool asynchronously."""
        # Still uses run_in_executor for now as core logic is synchronous.
        # True async requires asyncpg/aiopg integration and async LLM calls.
        loop = asyncio.get_event_loop()
        # Ensure _run has access to self correctly
        db_to_use = db_name or self.selected_db
        func = functools.partial(self._run, query_description=query_description, db_name=db_to_use)
        return await loop.run_in_executor(None, func)
