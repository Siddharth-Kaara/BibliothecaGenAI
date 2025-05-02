import json
import re
import logging
import asyncio
import statistics
from typing import Dict, List, Any, Optional, Tuple
import inspect

from pydantic import BaseModel, Field

from langchain.tools import BaseTool
from langchain.prompts import PromptTemplate
from langchain_openai import AzureChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import SystemMessage, HumanMessage
from openai import APIConnectionError, APITimeoutError, RateLimitError
from sqlalchemy import text

from app.core.config import settings
from app.db.connection import get_async_db_connection
from app.db.schema_definitions import SCHEMA_DEFINITIONS
from app.langchain.tools.sql_tool import SQLExecutionTool
from app.langchain.tools.hierarchy_resolver_tool import HierarchyNameResolverTool

logger = logging.getLogger(__name__)

class SQLQueryParams(BaseModel):
    """Parameters for a SQL query."""
    organization_id: str = Field(description="Organization ID parameter value")
    
    class Config:
        extra = "allow"  # Allow additional parameters beyond organization_id

class SQLGenerationResult(BaseModel):
    """Result of SQL generation from LLM."""
    sql: str = Field(description="Generated SQL query string with placeholders")
    params: SQLQueryParams = Field(description="Parameters for the SQL query")

class SubqueryResult(BaseModel):
    """Result of executing a subquery."""
    query: str = Field(description="The original subquery description")
    result: Dict[str, Any] = Field(description="The result data")
    error: Optional[str] = Field(default=None, description="Error message if query failed")
    
    @property
    def successful(self) -> bool:
        """Check if the subquery was successful."""
        return self.error is None and "table" in self.result and self.result.get("table", {}).get("rows", [])

class TrendInfo(BaseModel):
    """Information about a detected trend."""
    metric: str = Field(description="The metric showing a trend")
    direction: str = Field(description="Direction of the trend (increasing/decreasing)")
    percent_change: float = Field(description="Percent change in the metric")
    confidence: str = Field(description="Confidence level in the trend (high/medium/low)")

class AnomalyInfo(BaseModel):
    """Information about a detected anomaly."""
    entity: str = Field(description="The entity showing an anomaly")
    metric: str = Field(description="The metric with the anomaly")
    difference_from_avg: str = Field(description="Difference from average")
    severity: str = Field(description="Severity of the anomaly (high/medium/low)")

class OrganizationalComparison(BaseModel):
    """Information about an organizational comparison."""
    entity: str = Field(description="The entity being compared")
    metric: str = Field(description="The metric being compared")
    percent_difference: float = Field(description="Percent difference from org average")
    performance: str = Field(description="Performance assessment (above/below)")
    value: float = Field(description="Actual value")
    org_average: float = Field(description="Organization average")

class InsightsResult(BaseModel):
    """Automated insights detected from data."""
    trends: List[TrendInfo] = Field(default_factory=list, description="List of detected trends")
    anomalies: List[AnomalyInfo] = Field(default_factory=list, description="List of detected anomalies") 
    organizational_comparisons: List[OrganizationalComparison] = Field(default_factory=list, description="List of organizational comparisons")

class CompositeMetricsResult(BaseModel):
    """Composite metrics calculated from data."""
    success_rates: List[Dict[str, Any]] = Field(default_factory=list, description="List of success rates")

class SummarySynthesizerTool(BaseTool):
    """
    Tool for synthesizing high-level summaries from library data using concurrent query execution.
    
    This tool implements an enterprise-grade approach to data analysis and synthesis by:
    1. Decomposing complex queries into subqueries using LLM
    2. Executing subqueries concurrently for optimized performance
    3. Processing results to detect trends, anomalies, and statistical patterns
    4. Providing contextual location information by resolving hierarchy IDs
    5. Generating human-readable summaries with actionable insights
    
    Key architectural patterns:
    - Concurrent execution through asyncio for scalable performance
    - Layered error handling with graceful degradation
    - Clear separation of concerns between data fetching, analysis, and synthesis
    - Robust LLM interaction with retry mechanisms and content validation
    - Dynamic SQL generation with secure parameter handling
    
    This tool specializes in qualitative, open-ended data summaries where multiple data points
    need to be analyzed together to extract meaningful insights beyond simple metric retrieval.
    """
    
    name: str = "summary_synthesizer"
    description: str = """\
    **WARNING:** Use this tool ONLY for open-ended, qualitative summaries (e.g., 'summarize activity'). 
    For specific, quantifiable metric comparisons or retrievals (e.g., 'compare borrows for Branch A and B', 'get total renewals last month'), 
    you MUST use the 'execute_sql' tool instead.
    
    Creates high-level summaries and insights from data by intelligently decomposing the request and fetching data concurrently.
    This tool provides specialized capabilities for trend detection, anomaly identification, and pattern recognition across 
    multiple library metrics, delivering insights beyond raw SQL query results.
    
    Use this tool for complex queries requiring analysis across multiple dimensions or when a narrative summary with 
    automated trend detection is preferred.
    
    Input should be a description of the summary or analysis needed.
    """
    
    organization_id: str
    
    def _get_schema_info(self) -> str:
        """Get database schema information for SQL generation."""
        logger.debug(f"[Org: {self.organization_id}] [SummaryTool] Fetching schema information")
        db_name = "report_management"
        
        if db_name not in SCHEMA_DEFINITIONS:
            logger.warning(f"[Org: {self.organization_id}] [SummaryTool] No schema definition found for database {db_name}")
            return "Schema information not available."
        
        db_info = SCHEMA_DEFINITIONS[db_name]
        schema_info = [
            f"Database: {db_name}",
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
        
        logger.debug(f"[Org: {self.organization_id}] [SummaryTool] Successfully retrieved schema information")
        return "\n".join(schema_info)
    
    def _decompose_query(self, query: str) -> List[Dict[str, Any]]:
        """Decompose a complex query into subqueries using LLM, identifying location names."""
        log_prefix = f"[Org: {self.organization_id}] [SummaryTool] "
        
        try:
            # Get schema information
            schema_info = self._get_schema_info()
            
            llm = AzureChatOpenAI(
                openai_api_key=settings.AZURE_OPENAI_API_KEY,
                azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
                openai_api_version=settings.AZURE_OPENAI_API_VERSION,
                deployment_name=settings.AZURE_OPENAI_DEPLOYMENT_NAME,
                model_name=settings.LLM_MODEL_NAME,
                temperature=0.1,
                # Rely on AzureChatOpenAI's default retries
                max_retries=settings.LLM_MAX_RETRIES,
            )
            
            # Updated template to extract descriptions and location names
            template = """
            You are a data analyst. Given the high-level query below, break it down into atomic subqueries.
            Identify any specific location names (like "Main Library", "Argyle Branch") mentioned in relation to the data needed for each subquery.
            **Aim for an efficient plan** respecting concurrency limits (max {max_concurrent_tools} subqueries).

            HIGH-LEVEL QUERY:
            {query}

            DATABASE SCHEMA INFORMATION:
            {schema_info}

            CRITICAL REQUIREMENTS:
            - For each subquery needed, provide a clear natural language description.
            - For each subquery description, also list any specific location names (e.g., "Main Library", "Downtown Branch (DTB)") that the subquery relates to. Use the exact names as mentioned in the original query.
            - If a subquery is organizational-wide or doesn't refer to a specific location, provide an empty list for location_names.
            - Focus on the core data needed. Comparisons or complex calculations will be handled later.
            - **Do NOT include UUIDs or parameter placeholders** in the descriptions or location names.
            - Handle time-based queries appropriately by mentioning grouping periods (e.g., "monthly", "daily") in the description if trends are requested.

            OUTPUT FORMAT:
            Return ONLY a valid JSON array of objects. Each object must have two keys:
            1. "description": A string containing the natural language description of the subquery.
            2. "location_names": An array of strings, containing the exact location names relevant to this subquery (or an empty array [] if none).

            EXAMPLE OUTPUT for query "Compare borrows for Main Library and Downtown Branch (DTB) last month":
            ```json
            [
              {{
                "description": "Retrieve total successful borrows (column \\"1\\" in events table \\"5\\") for Main Library last month",
                "location_names": ["Main Library"]
              }},
              {{
                "description": "Retrieve total successful borrows (column \\"1\\" in events table \\"5\\") for Downtown Branch (DTB) last month",
                "location_names": ["Downtown Branch (DTB)"]
              }}
            ]
            ```
            
            EXAMPLE OUTPUT for query "Summarize total renewals across the organization last week":
            ```json
            [
              {{
                "description": "Calculate the total number of renewals across the entire organization last week",
                "location_names": []
              }}
            ]
            ```
            
            Ensure the output is ONLY the JSON array, without any preamble or explanation.
            """
            
            prompt = PromptTemplate(
                input_variables=["query", "schema_info", "max_concurrent_tools"],
                template=template
            )
            
            logger.debug(f"{log_prefix}Decomposing query with schema information...")
            decompose_chain = prompt | llm | StrOutputParser()
            
            subqueries_str = decompose_chain.invoke({
                "query": query, 
                "schema_info": schema_info,  # Use the actual schema information
                "max_concurrent_tools": settings.MAX_CONCURRENT_TOOLS
            })
                
            # Clean and parse the JSON
            subqueries_str = self._clean_json_response(subqueries_str)
            
            try:
                subquery_data = json.loads(subqueries_str)
                # Validate the structure
                if not isinstance(subquery_data, list):
                    raise ValueError("Expected a list of subquery objects")
                for item in subquery_data:
                    if not isinstance(item, dict) or "description" not in item or "location_names" not in item:
                        raise ValueError("Each item must be a dict with 'description' and 'location_names' keys")
                    if not isinstance(item["description"], str):
                        raise ValueError("'description' must be a string")
                    if not isinstance(item["location_names"], list) or not all(isinstance(name, str) for name in item["location_names"]):
                        raise ValueError("'location_names' must be a list of strings")

                # Cap the number of subqueries
                if len(subquery_data) > settings.MAX_CONCURRENT_TOOLS:
                    logger.warning(f"{log_prefix}Too many subqueries ({len(subquery_data)}), limiting to {settings.MAX_CONCURRENT_TOOLS}")
                    subquery_data = subquery_data[:settings.MAX_CONCURRENT_TOOLS]
                
                logger.info(f"{log_prefix}Query decomposed into {len(subquery_data)} subqueries with location extraction.")
                return subquery_data # Return list of dictionaries
                
            except json.JSONDecodeError as e:
                logger.error(f"{log_prefix}Error parsing subqueries JSON: {e}. Raw response: {subqueries_str}")
                # Fallback: Treat original query as one subquery with no specific locations
                return [{"description": query, "location_names": []}]
            except ValueError as e:
                logger.error(f"{log_prefix}Error validating subquery structure: {e}. Raw response: {subqueries_str}")
                return [{"description": query, "location_names": []}]
                
        except (APIConnectionError, APITimeoutError, RateLimitError) as e:
            logger.error(f"{log_prefix}OpenAI API error during query decomposition: {e}", exc_info=False)
            # Fallback on API error
            return [{"description": query, "location_names": []}]
        except KeyError as e:
             logger.error(f"{log_prefix}Missing key during decomposition invoke (likely 'schema_info'): {e}")
             # Fallback if schema_info wasn't provided
             return [{"description": query, "location_names": []}] 
        except Exception as e:
            logger.error(f"{log_prefix}Unexpected error during query decomposition: {e}", exc_info=True)
            # General fallback
            return [{"description": query, "location_names": []}]
    
    def _clean_json_response(self, response: str) -> str:
        """Clean JSON response from LLM output."""
        response = response.strip()
        if response.startswith("```json"):
            response = response[7:]
        elif response.startswith("```"):
            response = response[3:]
            
        if response.endswith("```"):
            response = response[:-3]
            
        return response.strip()
    
    async def _resolve_locations(self, subquery_data: List[Dict[str, Any]]) -> Dict[str, str]:
        """Resolve all unique location names found across subqueries.
        
        Args:
            subquery_data: List of dicts from _decompose_query, 
                           each with 'description' and 'location_names'.
                           
        Returns:
            A dictionary mapping original location names (lowercase) to their resolved UUIDs.
            Returns an empty dict if no locations were found or resolution failed.
            Raises ValueError if any required location name could not be resolved.
        """
        log_prefix = f"[Org: {self.organization_id}] [SummaryTool] "
        all_location_names = set()
        required_location_names = set() # Track names explicitly mentioned in subqueries

        for item in subquery_data:
            locations = item.get("location_names", [])
            if locations:
                for name in locations:
                    if name:
                        all_location_names.add(name)
                        required_location_names.add(name) # Assume names from decomposition are required

        if not all_location_names:
            logger.debug(f"{log_prefix}No location names found in subqueries to resolve.")
            return {}

        logger.info(f"{log_prefix}Attempting to resolve {len(all_location_names)} unique location names: {list(all_location_names)}")
        resolved_name_to_id_map = {}
        failed_to_resolve = []
        
        try:
            resolver = HierarchyNameResolverTool(organization_id=self.organization_id)
            # Resolve all unique names found
            resolution_result = await resolver.ainvoke({"name_candidates": list(all_location_names)})
            resolution_data = resolution_result.get("resolution_results", {})

            successfully_resolved_count = 0
            # Create map from original name (lowercase) to ID
            for original_name in all_location_names:
                result = resolution_data.get(original_name, {})
                if result.get("status") == "found" and result.get("id"):
                    resolved_id = str(result.get("id"))
                    resolved_name_for_log = result.get("resolved_name", original_name)
                    resolved_name_to_id_map[original_name.lower()] = resolved_id
                    logger.debug(f"{log_prefix}Resolved '{original_name}' -> '{resolved_name_for_log}' (ID: {resolved_id})")
                    successfully_resolved_count += 1
                else:
                    # If this name was required (i.e., explicitly extracted by decomposition)
                    if original_name in required_location_names:
                         failed_to_resolve.append(original_name)
                    logger.warning(f"{log_prefix}Failed to resolve location name: '{original_name}' (Status: {result.get('status', 'unknown')})")
            
            logger.info(f"{log_prefix}Location resolution finished. Successfully resolved {successfully_resolved_count}/{len(all_location_names)} names.")
            
            # If any required names failed to resolve, raise an error
            if failed_to_resolve:
                raise ValueError(f"Could not resolve the required location(s): {', '.join(set(failed_to_resolve))}")
                
            return resolved_name_to_id_map

        except Exception as e:
            logger.error(f"{log_prefix}Error during location name resolution: {e}", exc_info=True)
            # If resolution fails catastrophically, raise the error to stop processing
            raise ValueError(f"Failed to resolve location names due to an error: {str(e)}") from e

    async def _generate_sql_and_params(
        self,
        query_description: str,
        resolved_location_map: Dict[str, str],
        schema_info: str
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Generate SQL from a query description with proper parameter bindings.
        Uses Azure OpenAI API with retry logic for reliability.
        Returns: (sql, params dictionary with organization_id and location IDs properly set)
        """
        log_prefix = f"[Org: {self.organization_id}] [SummaryTool] "
        logger.debug(f"{log_prefix}Generating SQL for: {query_description}")
        
        # --- Preprocessing: Map original names to standardized param names and UUIDs ---
        location_param_context = []
        param_name_to_uuid_map = {}
        if resolved_location_map:
            for original_name, resolved_uuid in resolved_location_map.items():
                # Create a safe parameter name (lowercase, underscore spaces, add _id)
                safe_name_part = re.sub(r'[^a-z0-9_]', '', original_name.lower().replace(' ', '_'))
                param_name_for_llm = f"{safe_name_part}_id"
                
                # Store mapping for later substitution
                param_name_to_uuid_map[param_name_for_llm] = resolved_uuid
                
                # Create context string for the prompt
                location_param_context.append(f"- For location '{original_name}', use the parameter name ':{param_name_for_llm}' in your SQL WHERE clause.")
        
        location_context_str = "\n".join(location_param_context) if location_param_context else "No specific locations identified for this query."
        # --- End Preprocessing ---
        
        # --- System Message Updated ---
        system_message = f"""
        You are an expert SQL query generation assistant. Convert the natural language query description into a PostgreSQL compatible SQL query.

        **CRITICAL Instructions:**
        1.  **Use ONLY the provided schema.** Do not assume tables/columns exist.
        2.  **Table Specificity:** 
            - Footfall data (columns \"39\", \"40\") is ONLY in table \"8\".
            - Event metrics (borrows, returns, logins, etc.) are ONLY in table \"5\".
            - DO NOT mix columns between these tables.
        3.  **Quoting:** Double-quote ALL table and column names (e.g., \"5\", \"organizationId\").
        4.  **Mandatory Filtering:** 
            - Your query MUST ALWAYS filter by organization ID using `:organization_id`.
            - Add `WHERE \"tableName\".\"organizationId\" = :organization_id`.
            - If using CTEs or subqueries, EACH component MUST include its own independent `:organization_id` filter.
        5.  **Location Filtering (Use Provided Parameter Names):**
            {location_context_str}
            - Ensure you use these exact parameter names (e.g., `:{param_name_for_llm}`) in your SQL `WHERE` clauses when filtering by location.
        6.  **Parameters:** Use parameter placeholders (e.g., `:parameter_name`) for all dynamic values EXCEPT date/time functions.
        7.  **SELECT Clause:** Select specific columns with descriptive aliases (e.g., `SUM(\"1\") AS \"Total Borrows\"`). Avoid `SELECT *`.
        8.  **Performance:** Use appropriate JOINs, aggregations, and date functions. Add `LIMIT 50` to queries expected to return multiple rows.

        **Database Schema:**
        {schema_info}

        **Output Format (JSON):**
        Return ONLY a valid JSON object with 'sql' and 'params' keys.
        ```json
        {{
          "sql": "Your SQL query using the specified parameter names (e.g., :organization_id, :{param_name_for_llm})",
          "params": {{
            "organization_id": "SECURITY_PARAM_ORG_ID",
            "{param_name_for_llm}": "placeholder"  // The exact value here doesn't matter, it will be replaced
            // Include other necessary parameter keys with placeholder values if needed
          }}
        }}
        ```
        **IMPORTANT:** In the `params` dictionary you return, include keys for `organization_id` and *all* the required location parameter names (e.g., `{param_name_for_llm}`). The *values* for these keys in your returned JSON can be simple placeholders like \"placeholder\" or \"value\"; they will be replaced correctly later.
        """
        # --- End System Message Update ---
        
        logger.debug(f"{log_prefix}LLM System Message for SQL Generation:\n{system_message}") # Log updated message
        logger.debug(f"{log_prefix}LLM Human Message (Query Description): {query_description}")
        
        try:
            llm = AzureChatOpenAI(
                openai_api_key=settings.AZURE_OPENAI_API_KEY,
                azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
                openai_api_version=settings.AZURE_OPENAI_API_VERSION,
                deployment_name=settings.AZURE_OPENAI_DEPLOYMENT_NAME,
                model_name=settings.LLM_MODEL_NAME,
                temperature=0.0,
                max_retries=settings.LLM_MAX_RETRIES,
            )
            
            messages = [
                SystemMessage(content=system_message),
                HumanMessage(content=f"Generate SQL for this query description: {query_description}")
            ]
            
            sql_result = await llm.ainvoke(messages)
            sql_text = sql_result.content
            
            logger.debug(f"{log_prefix}Raw LLM Output (SQL Generation):\n{sql_text}")
            
            sql_text = self._clean_json_response(sql_text)
            
            try:
                result_json = json.loads(sql_text)
                if not isinstance(result_json, dict) or not all(key in result_json for key in ["sql", "params"]):
                    raise ValueError("Invalid SQL generation output format. Expected 'sql' and 'params' keys.")
                
                sql = result_json["sql"]
                params_from_llm = result_json.get("params", {})
                
                # --- Postprocessing: Construct final params using precomputed map ---
                params_for_execution = {
                    "organization_id": self.organization_id # Set the correct org ID
                }
                
                # Substitute actual UUIDs for location parameters based on the map
                # created during preprocessing, ignoring placeholder values from LLM.
                for param_name, resolved_uuid in param_name_to_uuid_map.items():
                    # Check if the LLM included the expected parameter key
                    if param_name in params_from_llm:
                        params_for_execution[param_name] = resolved_uuid
                    else:
                        # Log if LLM forgot a parameter it was asked to include
                        logger.warning(f"{log_prefix}LLM forgot to include expected parameter key '{param_name}' in its generated params dictionary.")
                        # Attempt to add it anyway, hoping the SQL uses it.
                        params_for_execution[param_name] = resolved_uuid 
                        
                # Add any other non-location, non-org parameters the LLM might have generated
                for key, value in params_from_llm.items():
                    if key != "organization_id" and key not in param_name_to_uuid_map:
                         params_for_execution[key] = value # Trust LLM's value for non-location params
                # --- End Postprocessing ---
                
                logger.debug(f"{log_prefix}Generated SQL (Parsed): {sql}")
                logger.debug(f"{log_prefix}Parameters (Final for Execution): {params_for_execution}")
                
                return sql, params_for_execution # Return the final parameters
                
            except json.JSONDecodeError as e:
                logger.error(f"{log_prefix}Failed to parse SQL generation output as JSON: {e}", exc_info=False)
                logger.debug(f"{log_prefix}Raw SQL generation output (at error): {sql_text}")
                raise ValueError(f"Failed to parse SQL generation output: {e}")
                
        except (APIConnectionError, APITimeoutError, RateLimitError) as e:
            logger.error(f"{log_prefix}OpenAI API error during SQL generation: {e}", exc_info=False)
            raise ValueError(f"Service unavailable during SQL generation: {e.__class__.__name__}")
        except ValueError as ve:
            raise ve
        except Exception as e:
            logger.error(f"{log_prefix}Unexpected error during SQL generation: {e}", exc_info=True)
            raise ValueError(f"Unexpected error during SQL generation: {e.__class__.__name__}: {str(e)}")
    
    async def _execute_subqueries_concurrently(
        self, 
        subquery_data: List[Dict[str, Any]], 
        resolved_location_map: Dict[str, str],
        schema_info: str
    ) -> List[SubqueryResult]:
        """Execute multiple subqueries concurrently and return their results."""
        log_prefix = f"[Org: {self.organization_id}] [SummaryTool] "
        logger.info(f"{log_prefix}Executing {len(subquery_data)} subqueries concurrently")
        
        if not subquery_data:
            return []
            
        sql_tool = SQLExecutionTool(organization_id=self.organization_id)
        
        tasks = []
        for i, subquery in enumerate(subquery_data):
            description = subquery.get("description", f"Subquery {i+1}")
            task = asyncio.create_task(
                self._generate_and_execute_single(
                    sql_tool, 
                    description, 
                    resolved_location_map,
                    schema_info
                )
            )
            tasks.append((description, task))
        
        # Gather results as they complete
        results = []
        for description, task in tasks:
            try:
                result = await task
                
                # --- Check if the result dictionary indicates an error from SQLExecutionTool --- 
                is_error_structure = (
                    isinstance(result, dict) and 
                    isinstance(result.get("table"), dict) and
                    result["table"].get("columns") == ["Error"]
                )
                
                if is_error_structure:
                    # This indicates _execute_sql caught an error (like security check or DB error)
                    error_message = result.get("text", "Unknown error from SQL execution")
                    logger.warning(f"{log_prefix}Subquery '{description}' failed during execution: {error_message}")
                    results.append(SubqueryResult(query=description, result={}, error=error_message))
                elif isinstance(result, dict):
                    # If await task SUCCEEDS (returns a valid data dict), append a successful result
                    results.append(SubqueryResult(query=description, result=result, error=None))
                else:
                    # Handle unexpected return types from _generate_and_execute_single
                    logger.error(f"{log_prefix}Subquery '{description}' returned unexpected type: {type(result)}")
                    results.append(SubqueryResult(query=description, result={}, error=f"Unexpected result type: {type(result)}"))

            except ValueError as ve:
                # Handle SQL generation/validation errors (expected errors from _generate_and_execute_single)
                logger.warning(f"{log_prefix}Subquery '{description}' failed: {ve}")
                results.append(SubqueryResult(query=description, result={}, error=str(ve)))
            except asyncio.TimeoutError:
                logger.error(f"{log_prefix}Subquery '{description}' timed out.")
                results.append(SubqueryResult(query=description, result={}, error="Query execution timed out"))
            except Exception as e:
                # Catch any other unexpected errors during the await task or processing
                logger.error(f"{log_prefix}Unexpected error processing subquery '{description}': {e}", exc_info=True)
                results.append(SubqueryResult(query=description, result={}, error=f"Unexpected error: {str(e)}"))

        # Log a summary of results
        successful_queries = sum(1 for r in results if r.successful)
        logger.info(f"{log_prefix}Completed {len(results)} subqueries. Success rate: {successful_queries}/{len(results)}")
        return results

    async def _generate_and_execute_single(
        self, 
        sql_tool: SQLExecutionTool, 
        description: str, 
        resolved_location_map: Dict[str, str],
        schema_info: str
    ) -> Dict[str, Any]:
        """Generate and execute a single SQL query using SQLExecutionTool."""
        log_prefix = f"[Org: {self.organization_id}] [SummaryTool] "
        
        try:
            # Generate SQL query with parameter placeholders
            sql, params = await self._generate_sql_and_params(description, resolved_location_map, schema_info)
            
            # Execute SQL using SQLExecutionTool
            # ainvoke now guarantees returning the awaited result due to explicit implementation
            result_json_str = await sql_tool.ainvoke({
                "sql": sql,
                "params": params
            })
            
            # Parse JSON string result into dictionary
            if not isinstance(result_json_str, str):
                 # If it's already a dict (shouldn't happen if _run returns str, but check)
                if isinstance(result_json_str, dict):
                    logger.warning(f"{log_prefix}SQL tool returned dict directly, expected JSON string.")
                    result = result_json_str 
                else:
                    # Handle unexpected non-string, non-dict types
                    raise ValueError(f"Expected JSON string from sql_tool, got {type(result_json_str)}")
            else:
                # Parse the JSON string returned by the tool
                try:
                    result = json.loads(result_json_str)
                except json.JSONDecodeError as json_err:
                    logger.error(f"{log_prefix}Failed to parse result from sql_tool: {json_err}. Raw: {result_json_str[:200]}...")
                    raise ValueError(f"Failed to parse result from SQL tool: {json_err}")
            
            # Process and return the result dictionary
            if not isinstance(result, dict):
                raise ValueError(f"Parsed result from sql_tool is not a dict: {type(result)}")
                
            return result
            
        except ValueError as ve:
            # For expected errors like SQL generation/validation issues
            logger.warning(f"{log_prefix}Error generating/executing query for '{description}': {ve}")
            raise ValueError(f"Generation/Validation Error: {str(ve)}")
        except (APIConnectionError, APITimeoutError) as api_e:
            # For API service availability issues
            logger.warning(f"{log_prefix}Service error for '{description}': {api_e}")
            raise ValueError(f"Service Unavailable: {api_e.__class__.__name__}")
        except Exception as e:
            # Catch unexpected errors during the process
            logger.error(f"{log_prefix}Unexpected error in '{description}': {e}", exc_info=True)
            raise ValueError(f"Unexpected Error: {e.__class__.__name__}: {str(e)}")
    
    
    async def _resolve_and_inject_names(self, subquery_results: List[SubqueryResult]) -> List[SubqueryResult]:
        """
        Resolves hierarchy IDs found in successful subquery results to names using HierarchyNameResolverTool.
        Efficiently injects location names into results for better readability.
        """
        log_prefix = f"[Org: {self.organization_id}] [SummaryTool] "
        logger.debug(f"{log_prefix}Resolving hierarchy IDs to names.")
        
        ids_to_resolve = set()
        results_with_id_col = [] # Store indices of results containing 'hierarchyId'
        
        # Step 1: Gather all unique IDs from successful results that have a 'hierarchyId' column
        for idx, result in enumerate(subquery_results):
            if not result.successful or "table" not in result.result:
                continue
                
            columns = result.result["table"].get("columns", [])
            rows = result.result["table"].get("rows", [])
            
            # Find the index of the 'hierarchyId' column
            hierarchy_id_index = -1
            if "hierarchyId" in columns:
                hierarchy_id_index = columns.index("hierarchyId")
            
            # Only process if the column exists and we have rows
            if hierarchy_id_index != -1 and rows:
                results_with_id_col.append(idx)
                for row in rows:
                    if hierarchy_id_index < len(row): # Check index bounds
                        hierarchy_id = row[hierarchy_id_index] # Access by index
                        if hierarchy_id:
                            ids_to_resolve.add(str(hierarchy_id))
        
        # If no IDs to resolve, return the original results unchanged
        if not ids_to_resolve:
            logger.debug(f"{log_prefix}No hierarchy IDs found to resolve.")
            return subquery_results
            
        logger.debug(f"{log_prefix}Found {len(ids_to_resolve)} unique hierarchy IDs to resolve.")
        
        # Step 2: Resolve IDs to names in a single batch call
        resolver = HierarchyNameResolverTool(organization_id=self.organization_id)
        id_name_map = {}
        
        try:
            resolution_result = await resolver.ainvoke({"name_candidates": list(ids_to_resolve)})
            resolution_data = resolution_result.get("resolution_results", {})
            
            # Build lookup map of ID → Name information
            for hierarchy_id, result_info in resolution_data.items():
                if result_info.get("status") == "found":
                    id_name_map[hierarchy_id] = {
                        "displayName": result_info.get("name"),
                        "parentName": result_info.get("parent_name")
                    }
        except Exception as e:
            logger.warning(f"{log_prefix}Error resolving hierarchy IDs: {e}", exc_info=False)
            # Continue with what we have (might be empty)
        
        # If no names were resolved, return original results
        if not id_name_map:
            logger.debug(f"{log_prefix}No names could be resolved for the hierarchy IDs.")
            return subquery_results
            
        # Step 3: Inject resolved names into result tables
        updated_results = list(subquery_results) # Create a copy
        
        # Column names for location info
        name_col = "Location Name"
        parent_col = "Parent Location"
        
        for idx in results_with_id_col:
            original_result = updated_results[idx]
            if not original_result.successful:
                continue
                
            original_table = original_result.result.get("table", {})
            original_columns = original_table.get("columns", [])
            original_rows = original_table.get("rows", [])
            
            # Skip if no rows to process
            if not original_rows:
                continue
                
            # Define updated columns list - add name columns if not present
            need_name_col = name_col not in original_columns
            need_parent_col = parent_col not in original_columns
            
            # If neither column needs to be added, skip this result
            if not need_name_col and not need_parent_col:
                continue
                
            new_columns = list(original_columns)
            if need_name_col:
                new_columns.append(name_col)
            if need_parent_col:
                new_columns.append(parent_col)
            
            # Process rows to add name information
            new_rows = []
            for row in original_rows:
                # Find index of hierarchyId again for this specific result
                hierarchy_id_index = -1
                if "hierarchyId" in original_columns:
                    hierarchy_id_index = original_columns.index("hierarchyId")
                
                # Get the hierarchy ID using the index
                hierarchy_id_str = ""
                if hierarchy_id_index != -1 and hierarchy_id_index < len(row):
                    hierarchy_id_str = str(row[hierarchy_id_index])
                
                # Convert the list row to a dict for easier manipulation/adding columns
                new_row_dict = dict(zip(original_columns, row))
                
                # Only process if we have this ID in our resolved map
                if hierarchy_id_str and hierarchy_id_str in id_name_map:
                    location_info = id_name_map[hierarchy_id_str]
                    
                    # Add location name if needed column exists and isn't already filled
                    if need_name_col and not new_row_dict.get(name_col):
                        new_row_dict[name_col] = location_info.get("displayName")
                    
                    # Add parent location info if needed column exists
                    if need_parent_col:
                        new_row_dict[parent_col] = location_info.get("parentName")
                
                # Convert back to list in the new column order
                final_row_list = [new_row_dict.get(col_name) for col_name in new_columns]
                new_rows.append(final_row_list)
            
            # Update the result with the enhanced data
            updated_results[idx] = SubqueryResult(
                query=original_result.query,
                result={"table": {"columns": new_columns, "rows": new_rows}},
                error=original_result.error
            )
            
            logger.debug(f"{log_prefix}Enhanced result {idx} with location names for {len(new_rows)} rows.")

        return updated_results

    def _calculate_composite_metrics(self, subquery_results: List[SubqueryResult]) -> CompositeMetricsResult:
        """
        Calculate derived metrics like success rates or ratios from aggregated data.
        Focuses on common patterns like success/failure pairs.
        Handles results where rows are lists of values.
        """
        log_prefix = f"[Org: {self.organization_id}] [SummaryTool] "
        logger.info(f"{log_prefix}Calculating composite metrics.")
        
        composite_metrics = CompositeMetricsResult()
        aggregated_data = {}
        entity_key_preference = ["Location Name", "hierarchyId"] # Preference for entity grouping

        for result in subquery_results:
            if result.successful and "table" in result.result:
                table = result.result["table"]
                columns = table.get("columns", [])
                rows = table.get("rows", [])
                if not rows or not columns:
                    continue

                # Find the best entity key column present
                entity_col = next((col for col in entity_key_preference if col in columns), None)
                entity_col_index = columns.index(entity_col) if entity_col else -1
                
                # Find the index of 'Location Name' if different from entity_col, for display purposes
                display_name_col_index = -1
                if "Location Name" in columns:
                    display_name_col_index = columns.index("Location Name")

                # Identify numeric columns and their indices
                numeric_col_indices = {}
                sample_row = rows[0] # First row to check types
                for idx, col_name in enumerate(columns):
                    if idx != entity_col_index and idx < len(sample_row):
                        sample_val = sample_row[idx]
                        if isinstance(sample_val, (int, float)):
                            numeric_col_indices[col_name] = idx
                
                if not numeric_col_indices: continue # Skip if no numeric data

                # Aggregate data using indices
                for row in rows:
                    if len(row) < len(columns): continue # Skip malformed rows
                    
                    # Determine Entity ID and Display Name
                    entity_id = "_org_wide_"
                    entity_display_name = "Organization Wide"
                    if entity_col_index != -1:
                        entity_id = str(row[entity_col_index]) # Use string representation for safety
                        # Use Location Name for display if available, otherwise use the entity ID itself
                        if display_name_col_index != -1:
                            entity_display_name = str(row[display_name_col_index])
                        else:
                            entity_display_name = entity_id 
                    
                    if entity_id not in aggregated_data:
                        aggregated_data[entity_id] = {"_entity_display_name_": entity_display_name}
                             
                    # Aggregate numeric columns by index
                    for num_col, num_idx in numeric_col_indices.items():
                        if num_idx < len(row): # Check index bounds again for safety
                            value = row[num_idx]
                            if isinstance(value, (int, float)): # Ensure it's numeric
                                aggregated_data[entity_id][num_col] = aggregated_data[entity_id].get(num_col, 0) + value
                            else:
                                # Log unexpected non-numeric value
                                logger.debug(f"{log_prefix}Skipping non-numeric value '{value}' for column '{num_col}' in entity '{entity_id}'")
        
        # Define potential success/failure pairs (using common aliases)
        success_failure_pairs = [
            (("Total Borrows", "Total Successful Borrows"), ("Total Unsuccessful Borrows",), "Borrow Success Rate"),
            (("Total Successful Returns",), ("Total Unsuccessful Returns",), "Return Success Rate"),
            # Add more pairs as needed
        ]

        # Calculate success rates where data is available
        for entity_id, metrics in aggregated_data.items():
            entity_display_name = metrics.get("_entity_display_name_", str(entity_id))
            for success_aliases, failure_aliases, rate_name in success_failure_pairs:
                success_col = next((alias for alias in success_aliases if alias in metrics), None)
                failure_col = next((alias for alias in failure_aliases if alias in metrics), None)
                
                if success_col and failure_col:
                    success_val = metrics.get(success_col, 0)
                    failure_val = metrics.get(failure_col, 0)
                    total = success_val + failure_val
                    if total > 0:
                        success_rate = (success_val / total) * 100
                        composite_metrics.success_rates.append({
                            "metric": rate_name,
                            "entity": entity_display_name,
                            "success_count": success_val,
                            "failure_count": failure_val,
                            "success_rate": round(success_rate, 1),
                        })

        logger.info(f"{log_prefix}Composite metrics calculated. Found {len(composite_metrics.success_rates)} success rates.")
        return composite_metrics
    
    def _detect_trends(self, subquery_results: List[SubqueryResult]) -> InsightsResult:
        """
        Analyze subquery results to detect trends, anomalies, and comparisons.
        Focuses on time series trends and deviations from averages.
        Handles results where rows are lists of values.
        """
        log_prefix = f"[Org: {self.organization_id}] [SummaryTool] "
        insights = InsightsResult()
        all_data_by_entity = {}

        # First pass: Aggregate data by entity and identify time series
        for result in subquery_results:
            if not result.successful or "table" not in result.result:
                continue
                
            table = result.result["table"]
            columns = table.get("columns", [])
            rows = table.get("rows", [])
            
            if not rows or not columns:
                continue
            
            # Look for entity identifier column (preference order)
            entity_col_index = -1
            entity_col = None
            for candidate in ["Location Name", "hierarchyId", "name"]:
                if candidate in columns:
                    entity_col_index = columns.index(candidate)
                    entity_col = candidate
                    break
            
            # Identify potential time/date column index
            time_col_index = -1
            time_col = None
            time_col_keywords = {"date", "time", "day", "month", "year", "week"}
            for idx, col_name in enumerate(columns):
                if any(keyword in col_name.lower() for keyword in time_col_keywords):
                    time_col_index = idx
                    time_col = col_name
                    break
            
            # Identify numeric columns and their indices (skip entity and time columns)
            numeric_col_indices = {}
            sample_row = rows[0] # Use first row to check types
            for idx, col_name in enumerate(columns):
                if idx != entity_col_index and idx != time_col_index and idx < len(sample_row):
                    sample_val = sample_row[idx]
                    if isinstance(sample_val, (int, float)):
                        numeric_col_indices[col_name] = idx
            
            # Skip if no numeric columns to analyze
            if not numeric_col_indices:
                continue
                
            # Process rows using indices
            for row in rows:
                if len(row) < len(columns): continue # Skip malformed rows
                
                # Get entity ID/name using index
                entity_id = str(row[entity_col_index]) if entity_col_index != -1 else "_org_wide_"
                if entity_id not in all_data_by_entity:
                    all_data_by_entity[entity_id] = {
                        "metrics": {},
                        "time_series": {} if time_col_index != -1 else None
                    }
                
                # Store numeric metrics using indices
                for metric_col_name, metric_col_idx in numeric_col_indices.items():
                    if metric_col_idx < len(row):
                        metric_val = row[metric_col_idx]
                        if not isinstance(metric_val, (int, float)):
                            continue
                            
                        # Add to aggregated metrics
                        if metric_col_name not in all_data_by_entity[entity_id]["metrics"]:
                            all_data_by_entity[entity_id]["metrics"][metric_col_name] = []
                        all_data_by_entity[entity_id]["metrics"][metric_col_name].append(metric_val)
                        
                        # Add to time series if applicable using indices
                        if time_col_index != -1 and all_data_by_entity[entity_id]["time_series" ] is not None:
                            if time_col_index < len(row):
                                time_val = row[time_col_index]
                                if time_val:
                                    if metric_col_name not in all_data_by_entity[entity_id]["time_series"]:
                                        all_data_by_entity[entity_id]["time_series"][metric_col_name] = []
                                    all_data_by_entity[entity_id]["time_series"][metric_col_name].append((time_val, metric_val))

        # Calculate organization-wide averages for each metric
        org_avg = {}
        for metric in set().union(*(entity_data["metrics"].keys() for entity_data in all_data_by_entity.values())):
            all_values = []
            for entity_data in all_data_by_entity.values():
                all_values.extend(entity_data["metrics"].get(metric, []))
            
            if all_values:
                try:
                    org_avg[metric] = sum(all_values) / len(all_values)
                except (TypeError, ValueError):
                    continue
        
        # Second pass: Analyze the aggregated data for insights
        for entity, data in all_data_by_entity.items():
            # Skip organization-wide data for comparison insights
            if entity == "_org_wide_":
                continue
                
            # Analyze metrics for anomalies relative to org averages
            for metric, values in data["metrics"].items():
                if not values or metric not in org_avg:
                    continue
                    
                try:
                    # Calculate statistics
                    avg_val = sum(values) / len(values)
                    entity_metric_avg = avg_val
                    
                    # Compare to organization average
                    org_metric_avg = org_avg[metric]
                    if abs(org_metric_avg) < 0.0001:  # Avoid division by zero
                        continue
                        
                    percent_diff = ((entity_metric_avg - org_metric_avg) / org_metric_avg) * 100
                    
                    # Only report significant differences
                    if abs(percent_diff) >= 20:  # 20% threshold
                        performance = "above average" if percent_diff > 0 else "below average"
                        
                        # Format the entity name for display (strip org_wide marker)
                        entity_display = entity if entity != "_org_wide_" else "Organization-wide"
                        
                        # Create proper OrganizationalComparison object directly
                        insights.organizational_comparisons.append(OrganizationalComparison(
                            entity=entity_display,
                            metric=metric,
                            percent_difference=round(percent_diff, 1),
                            performance=performance,
                            value=entity_metric_avg,
                            org_average=org_metric_avg
                        ))
                except (TypeError, ValueError, ZeroDivisionError):
                    continue
            
            # Analyze time series data for trends
            if data["time_series"]:
                for metric, time_points in data["time_series"].items():
                    if len(time_points) < 3:  # Need at least 3 points for trend
                        continue
                        
                    try:
                        # Sort by time value
                        sorted_points = sorted(time_points, key=lambda x: x[0])
                        values = [point[1] for point in sorted_points]
                        
                        # Simple trend detection - compare first and last values
                        first_val = values[0]
                        last_val = values[-1]
                        
                        if abs(first_val) < 0.0001:  # Avoid division by zero
                            continue
                            
                        change_pct = ((last_val - first_val) / first_val) * 100
                        
                        # Only report significant trends
                        if abs(change_pct) >= 10:  # 10% threshold
                            direction = "increased" if change_pct > 0 else "decreased"
                            
                            # Format the entity name for display
                            entity_display = entity if entity != "_org_wide_" else "Organization-wide"
                            
                            trend = f"{entity_display} {metric} has {direction} by {abs(change_pct):.1f}% over the period"
                            insights.trends.append(TrendInfo(
                                metric=f"{metric} for {entity_display}",
                                direction=direction,
                                percent_change=round(abs(change_pct), 1),
                                confidence="high" if len(sorted_points) >= 5 else "medium"
                            ))
                    except (TypeError, ValueError, ZeroDivisionError):
                        continue
            
            # Analyze for anomalies/outliers among peer entities
            for metric, values in data["metrics"].items():
                if len(values) <= 1:
                    continue
                    
                try:
                    # Calculate entity's average for this metric
                    entity_avg = sum(values) / len(values)
                    
                    # Need to gather peer values for this metric
                    peer_metrics = []
                    for peer_entity, peer_data in all_data_by_entity.items():
                        if peer_entity != entity and peer_entity != "_org_wide_":
                            if metric in peer_data["metrics"] and peer_data["metrics"][metric]:
                                peer_avg = sum(peer_data["metrics"][metric]) / len(peer_data["metrics"][metric])
                                peer_metrics.append(peer_avg)
                    
                    # Only detect anomalies if we have enough peer entities
                    if len(peer_metrics) >= 2:
                        peer_avg = sum(peer_metrics) / len(peer_metrics)
                        
                        # Calculate standard deviation
                        stdev_val = statistics.stdev(peer_metrics)
                        
                        if stdev_val > 0:
                            # Calculate z-score
                            z_score = (entity_avg - peer_avg) / stdev_val
                            
                            # Anomaly if z-score is significant
                            if abs(z_score) >= 2.0:
                                # Format the entity name for display
                                entity_display = entity if entity != "_org_wide_" else "Organization-wide"
                                
                                diff_percent = round(((entity_avg - peer_avg) / peer_avg) * 100 if peer_avg != 0 else 0, 1)
                                diff_description = f"{diff_percent}% {'above' if entity_avg > peer_avg else 'below'} peer average"
                                
                                anomaly = f"{entity_display} {metric} is unusually {'high' if z_score > 0 else 'low'} compared to peers (z-score: {abs(z_score):.1f})"
                                insights.anomalies.append(AnomalyInfo(
                                    entity=entity_display,
                                    metric=metric,
                                    difference_from_avg=diff_description,
                                    severity="high" if abs(z_score) > 3 else "medium"
                                ))
                except statistics.StatisticsError:
                    continue
                except (TypeError, ValueError, ZeroDivisionError):
                    continue
        
        # Convert organizational comparisons to proper objects
        for i, comparison in enumerate(insights.organizational_comparisons):
            if not isinstance(comparison, OrganizationalComparison):
                parts = comparison.split(" is ")
                if len(parts) < 2:
                    continue
                    
                entity_metric = parts[0].split(" ", 1)
                if len(entity_metric) < 2:
                    continue
                    
                entity_display = entity_metric[0]
                metric = entity_metric[1]
                
                # Extract percentage
                pct_match = re.search(r"(\d+\.\d+)%", comparison)
                pct_diff = float(pct_match.group(1)) if pct_match else 0
                
                # Determine performance
                performance = "above average" if "above average" in comparison else "below average"
                
                # Extract values
                val_match = re.search(r"\((\d+\.\d+) vs. org avg (\d+\.\d+)\)", comparison)
                if val_match:
                    value = float(val_match.group(1))
                    org_avg_val = float(val_match.group(2))
                else:
                    value = 0
                    org_avg_val = 0
                
                insights.organizational_comparisons[i] = OrganizationalComparison(
                    entity=entity_display,
                    metric=metric,
                    percent_difference=pct_diff,
                    performance=performance,
                    value=value,
                    org_average=org_avg_val
                )
                
        logger.info(f"{log_prefix}Analysis complete. Found: {len(insights.trends)} trends, "
                   f"{len(insights.anomalies)} anomalies, {len(insights.organizational_comparisons)} org comparisons")
        return insights
    
    def _synthesize_results(self, query: str, subquery_results: List[SubqueryResult]) -> str:
        """Synthesize subquery results into a coherent summary with automated insights."""
        log_prefix = f"[Org: {self.organization_id}] [SummaryTool] "
        try:
            # Check if all subqueries failed - using proper check that accounts for successful field
            all_failed = all(not result.successful for result in subquery_results)
                    
            if all_failed:
                return self._handle_all_subqueries_failed(subquery_results)
                
            # Calculate metrics and detect trends/insights
            composite_metrics = self._calculate_composite_metrics(subquery_results)
            insights = self._detect_trends(subquery_results)
            
            # Prepare the synthesizer input
            context_str = f"Organization ID: {self.organization_id}\nQuery: {query}\n"
            
            # Setup LLM for synthesis
            llm = AzureChatOpenAI(
                openai_api_key=settings.AZURE_OPENAI_API_KEY,
                azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
                openai_api_version=settings.AZURE_OPENAI_API_VERSION,
                deployment_name=settings.AZURE_OPENAI_DEPLOYMENT_NAME,
                model_name=settings.LLM_MODEL_NAME,
                temperature=0.2,
                max_retries=settings.LLM_MAX_RETRIES,
            )
            
            # Prepare the prompt template
            template = """
            You are an expert data analyst for a library system. Your job is to synthesize query results into a clear, concise summary.
            
            USER QUERY:
            {query}
            
            CONTEXT:
            {context}
            
            QUERY RESULTS:
            {results}
            
            AUTOMATED INSIGHTS:
            {insights}
            
            Your task is to create a helpful, informative summary that addresses the user's original query and highlights key findings.
            
            Guidelines:
            - Be concise but thorough
            - Highlight notable patterns or anomalies
            - Include statistical context where relevant (e.g., changes over time, comparisons to averages)
            - Use precise numbers rather than vague terms
            - Structure your response for readability
            - Do not invent data not present in the results
            - Maintain a professional, objective tone
            
            SUMMARY:
            """
            
            # Track success/error counts
            success_count = sum(1 for r in subquery_results if r.successful)
            error_count = len(subquery_results) - success_count
            
            # Prepare results string for the prompt
            results_str = f"Success Rate: {success_count}/{len(subquery_results)} subqueries successful\n\n"
            
            for subquery_result in subquery_results:
                results_str += f"Subquery: {subquery_result.query}\n"
                if subquery_result.error:
                     results_str += f"Result: Error - {subquery_result.error}\n\n"
                     continue 
                table_data = subquery_result.result.get("table", {})
                if not isinstance(table_data, dict):
                     results_str += f"Result: Invalid table data format received.\n\n"
                     continue
                
                limited_rows = table_data.get("rows", [])[:5]
                columns = table_data.get("columns", [])
                results_str += f"Results (showing up to 5 rows): {json.dumps({"columns": columns, "rows": limited_rows}, indent=2)}\n"
                results_str += f"Total rows in original result: {len(table_data.get('rows', []))}\n\n"
            
            # Add warning if partial data
            warning_prefix = ""
            if error_count > 0 and success_count > 0:
                warning_prefix = "**Note: Some data could not be retrieved due to database issues. This analysis is based on partial data.**\n\n"
            
            # Add automated insights section to the synthesizer input
            insights_str = "Automated Insights Detected:\n"
            
            # Add trends if any were detected
            if insights.trends:
                insights_str += "Trends:\n"
                for trend in insights.trends:
                    insights_str += f"- {trend}\n"
            
            # Add anomalies if any were detected
            if insights.anomalies:
                insights_str += "Anomalies:\n"
                for anomaly in insights.anomalies:
                    insights_str += f"- {anomaly}\n"
            
            # Add organizational comparisons if any
            if insights.organizational_comparisons:
                insights_str += "Comparisons to Organizational Averages:\n"
                for comparison in insights.organizational_comparisons:
                    insights_str += f"- {comparison}\n"
            
            # Add success/failure rate metrics
            if composite_metrics.success_rates:
                insights_str += "Success Rate Metrics:\n"
                for metric in composite_metrics.success_rates:
                    insights_str += f"- {metric['metric']}: {metric['success_rate']}%\n"
            
            # If no insights were detected, note that
            if (not insights.trends and not insights.anomalies and 
                not insights.organizational_comparisons and not composite_metrics.success_rates):
                insights_str += "No specific trends, anomalies, or comparisons were automatically detected in the data.\n"
            
            prompt = PromptTemplate(input_variables=["query", "context", "results", "insights"], template=template)
            synthesis_chain = prompt | llm | StrOutputParser()
            
            summary = synthesis_chain.invoke({
                "query": query, 
                "context": context_str,
                "results": results_str,
                "insights": insights_str
            })
            
            # Prepend warning if needed
            if warning_prefix:
                summary = warning_prefix + summary
                
            return summary.strip()
        except APIConnectionError as e:
            logger.error(f"{log_prefix}OpenAI connection error during summary synthesis: {e}", exc_info=False)
            return f"Unable to synthesize results due to a service connection issue. The data was collected but the synthesis service is currently unavailable."
        except APITimeoutError as e:
            logger.error(f"{log_prefix}OpenAI timeout during summary synthesis: {e}", exc_info=False)
            return f"The synthesis service timed out while processing your request. The data was collected but the summary could not be generated in time."
        except Exception as e:
            logger.error(f"{log_prefix}Error in _synthesize_results: {str(e)}", exc_info=True)
            return "I encountered an unexpected error while processing your request. Please try again or contact support if the issue persists."
            
    def _handle_all_subqueries_failed(self, subquery_results: List[SubqueryResult]) -> str:
        """Extract patterns from error messages when all subqueries fail and provide a meaningful response."""
        # Extract and analyze error messages
        error_messages = [result.error for result in subquery_results if result.error]
        unique_errors = set(error_messages)
        
        # Check for specific error patterns
        table_not_found_errors = []
        for err in unique_errors:
            if not isinstance(err, str):
                continue
                
            # Pattern matching for table not found errors
            if "Table '" in err and "does not exist in the database" in err:
                match = re.search(r"Table '([^']+)'", err)
                if match:
                    table_not_found_errors.append(match.group(1))
        
        # Generate appropriate error message based on patterns
        if table_not_found_errors:
            tables_str = ", ".join(f"'{t}'" for t in table_not_found_errors)
            return (f"I couldn't retrieve the requested data because of a schema mismatch issue. "
                   f"The following tables couldn't be found in the database: {tables_str}. "
                   "This suggests there's a discrepancy between the expected schema and the actual database structure.")
        
        # Look for permission/security errors
        if any("permission denied" in str(err).lower() or "access denied" in str(err).lower() for err in unique_errors):
            return "I couldn't access the requested data due to database permission restrictions. Please verify your access rights or contact your administrator."
        
        # Look for timeouts
        if any("timeout" in str(err).lower() or "timed out" in str(err).lower() for err in unique_errors):
            return "The database queries timed out while retrieving your data. This might indicate the query was too complex or the database is under heavy load. Try simplifying your request or try again later."
        
        # Handle syntax errors
        if any("syntax error" in str(err).lower() for err in unique_errors):
            return "There was a problem with the database query syntax. This is likely an internal issue with how I'm translating your request. Please try rephrasing your question or contact support."
        
        # Generic fallback for other errors
        if unique_errors:
            # Provide a user-friendly version of the first error
            sample_error = next(iter(unique_errors))
            sanitized_error = str(sample_error).replace(self.organization_id, "[ORGANIZATION_ID]")
            return f"I encountered a database error while trying to retrieve your data: {sanitized_error}. Please try again or contact support."
        
        # Ultra fallback
        return "I couldn't retrieve the requested data due to database errors. Please try again or contact support if the problem persists."
    
    async def ainvoke(self, input_data: Dict[str, Any], **kwargs: Any) -> Any:
        """Invoke the tool with the given input data."""
        log_prefix = f"[Org: {self.organization_id}] [SummaryTool] "
        
        # Extract the query from input_data
        query = input_data.get("query", "")
        
        if not query:
            logger.warning(f"{log_prefix}No query provided in input_data")
            return {
                "message": "Please provide a query for summarization"
            }
        
        try:
            # Get schema information for SQL generation
            schema_info = self._get_schema_info()
            
            # Decompose the query into subqueries
            subqueries = self._decompose_query(query)
            logger.info(f"{log_prefix}Query decomposed into {len(subqueries)} subqueries")
            
            # Check if any subquery contains potential hierarchy ID placeholders
            contains_hierarchy_references = any(
                "hierarchy" in sq["description"].lower() or 
                "branch" in sq["description"].lower() or 
                "location" in sq["description"].lower() or
                sq.get("location_names", [])
                for sq in subqueries
            )
            
            # If location references exist, resolve them first
            resolved_location_map = {}
            if contains_hierarchy_references:
                logger.info(f"{log_prefix}Detected potential location references, resolving IDs")
                try:
                    resolved_location_map = await self._resolve_locations(subqueries)
                except ValueError as e:
                    # If location resolution fails, we consider this a critical error and stop
                    logger.error(f"{log_prefix}Location resolution failed: {e}")
                    return {"message": str(e)}
            
            # Execute all subqueries concurrently
            results = await self._execute_subqueries_concurrently(subqueries, resolved_location_map, schema_info)
            
            # Log execution results summary
            successful_queries = sum(1 for r in results if r.successful)
            logger.info(f"{log_prefix}Subquery execution complete. {successful_queries}/{len(results)} successful.")
            
            # If all subqueries failed, return early with the error message
            if all(not result.successful for result in results):
                first_error = next((result.error for result in results if result.error), "Unknown error")
                return {"message": f"All subqueries failed. Error: {first_error}"}
            
            # Resolve hierarchy IDs to names
            results_with_names = await self._resolve_and_inject_names(results)
            
            # Generate a coherent summary
            summary = self._synthesize_results(query, results_with_names)
            
            return {"message": summary}
            
        except Exception as e:
            logger.error(f"{log_prefix}Unexpected error processing query: {e}", exc_info=True)
            return {"message": f"An unexpected error occurred: {str(e)}"}

    def _run(self, query: str) -> Dict[str, str]:
        """
        Synchronous run method required by BaseTool.
        This tool only supports async operation, so this method raises NotImplementedError.
        """
        raise NotImplementedError("This tool only supports async operation. Use ainvoke instead.")
