import json
import re
import logging
import asyncio
import statistics
import datetime
from typing import Dict, List, Any, Optional, Tuple
import inspect
from functools import lru_cache

from pydantic import BaseModel, Field, PrivateAttr

from langchain.tools import BaseTool
from langchain.prompts import PromptTemplate
from langchain_openai import AzureChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import SystemMessage, HumanMessage
from openai import APIConnectionError, APITimeoutError, RateLimitError
from sqlalchemy import text
from aiolimiter import AsyncLimiter

from app.core.config import settings
from app.db.connection import get_async_db_connection
from app.db.schema_definitions import SCHEMA_DEFINITIONS
from app.langchain.tools.sql_tool import SQLExecutionTool
from app.langchain.tools.hierarchy_resolver_tool import HierarchyNameResolverTool
from app.utils import clean_json_response, json_default
from app.prompts import SUMMARY_SQL_GENERATION_PROMPT, SUMMARY_SYNTHESIS_TEMPLATE, SUMMARY_DECOMPOSITION_TEMPLATE

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
        """Check if the subquery's SQL execution was successful and yielded a structured result."""
        # 1. If an error message was explicitly set during SubqueryResult creation, it's a definite failure.
        if self.error is not None:
            return False
        
        # 2. If no explicit error string (self.error is None), then SQLExecutionTool did not report
        #    a top-level error. Now, we must validate that self.result (which should directly be
        #    the table dictionary like {"columns": [...], "rows": [...]}) has the expected structure.
        if not isinstance(self.result, dict):
            logger.warning(
                f"SubqueryResult for '{self.query}': self.result is not a dict (type: {type(self.result)}), "
                f"but self.error was None. This indicates a malformed success response."
            )
            return False

        # 3. Final structural check for a valid table: 'columns' and 'rows' must exist directly in self.result.
        if "columns" not in self.result or "rows" not in self.result:
            logger.warning(
                f"SubqueryResult for '{self.query}': self.result is missing 'columns' or 'rows' "
                f"keys, but self.error was None. Malformed success response. Result: {str(self.result)[:200]}"
            )
            return False
            
        # If all checks pass (no explicit error, and result structure is a valid table),
        # then the subquery SQL execution is considered successful.
        return True

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
    # Declare injected tools as private attributes
    _sql_tool: SQLExecutionTool = PrivateAttr()
    _hierarchy_resolver: HierarchyNameResolverTool = PrivateAttr()
    # Declare limiter as a private attribute for internal state
    _limiter: AsyncLimiter = PrivateAttr()
    
    def __init__(self, *, sql_tool: SQLExecutionTool, hierarchy_resolver: HierarchyNameResolverTool, **data):
        """Initialize the tool, injecting dependencies."""
        super().__init__(**data)
        # Store injected tools
        self._sql_tool = sql_tool
        self._hierarchy_resolver = hierarchy_resolver
        # Initialize rate limiter based on settings, using the private attribute name
        self._limiter = AsyncLimiter(settings.LLM_SUMMARY_MAX_RATE, settings.LLM_SUMMARY_TIME_PERIOD)

    @staticmethod
    @lru_cache(maxsize=1)
    def _get_schema_info_static() -> str:
        """Get database schema information for SQL generation. Static and cached."""
        logger.debug(f"[SummaryTool] Fetching global schema information for report_management") # Generic log
        db_name = "report_management"
        
        if db_name not in SCHEMA_DEFINITIONS:
            logger.warning(f"[SummaryTool] No schema definition found for database {db_name}")
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
        
        logger.debug(f"[SummaryTool] Successfully retrieved global schema information") # Generic log
        return "\n".join(schema_info)
    
    def _decompose_query(self, query: str) -> List[Dict[str, Any]]:
        """Decompose a complex query into subqueries using LLM, identifying location names."""
        log_prefix = f"[Org: {self.organization_id}] [SummaryTool] "
        
        try:
            # Get schema information
            schema_info = SummarySynthesizerTool._get_schema_info_static()
            
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
            
            # Updated template to extract descriptions and location names (Use imported template)
            # template = """...""" # Removed original inline definition
            
            prompt = PromptTemplate(
                input_variables=["query", "schema_info", "max_concurrent_tools"],
                template=SUMMARY_DECOMPOSITION_TEMPLATE # Use imported constant
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
            # Use the injected hierarchy resolver instance
            # resolver = HierarchyNameResolverTool(organization_id=self.organization_id) # REMOVED
            # Resolve all unique names found
            resolution_result = await self._hierarchy_resolver.ainvoke({"name_candidates": list(all_location_names)})
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
        resolved_location_map: Dict[str, str]
    ) -> Tuple[str, Dict[str, Any]]:
        """Generate SQL and parameters for a subquery using LLM."""
        log_prefix = f"[Org: {self.organization_id}] [SummaryTool] "
        schema_info = SummarySynthesizerTool._get_schema_info_static() # Obtain schema_info using the static method

        # Determine the database name (assuming one DB for now)
        db_name = "report_management"
        
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
        
        # --- System Message (Uses Imported Prompt) ---
        # Use imported prompt constant, formatted with dynamic info
        system_message = SUMMARY_SQL_GENERATION_PROMPT.format(
            location_context_str=location_context_str,
            param_name_for_llm="{param_name_for_llm}", # Keep placeholder for LLM guidance
            schema_info=schema_info
        )
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
            
            sql_text = clean_json_response(sql_text)
            
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
        resolved_location_map: Dict[str, str]
    ) -> List[SubqueryResult]:
        """Execute multiple subqueries concurrently and return their results, with timeouts."""
        log_prefix = f"[Org: {self.organization_id}] [SummaryTool] "
        logger.info(f"{log_prefix}Executing {len(subquery_data)} subqueries concurrently with timeout {settings.SUBQUERY_TIMEOUT_SECONDS}s")
        
        if not subquery_data:
            return []
            
        # Use the injected sql_tool instance
        # sql_tool = SQLExecutionTool(organization_id=self.organization_id) # REMOVED
        
        tasks = []
        for i, subquery in enumerate(subquery_data):
            description = subquery.get("description", f"Subquery {i+1}")
            # Create task with timeout
            task = asyncio.create_task(
                asyncio.wait_for(
                    self._generate_and_execute_single(
                        description, 
                        resolved_location_map
                    ),
                    timeout=settings.SUBQUERY_TIMEOUT_SECONDS # Apply timeout
                )
            )
            tasks.append((description, task))
        
        # Gather results as they complete, handling timeouts
        results = []
        for description, task in tasks:
            try:
                # Await the task (which includes the wait_for)
                result_dict = await task # result_dict is the parsed JSON output of SQLExecutionTool
                
                # Check for the new error structure from SQLExecutionTool
                if isinstance(result_dict, dict) and result_dict.get("error") is not None:
                    error_obj = result_dict["error"] # error_obj is typically a dict
                    error_message = "Unknown error from SQL execution tool."
                    if isinstance(error_obj, dict) and "message" in error_obj:
                        error_message = error_obj["message"]
                    elif isinstance(error_obj, str): # Simple error string
                        error_message = error_obj
                        
                    logger.warning(f"{log_prefix}Subquery '{description}' failed (reported by SQL tool): {error_message}")
                    results.append(SubqueryResult(query=description, result={}, error=error_message))
                # Check for the legacy error structure as a fallback (less likely with new sql_tool)
                elif isinstance(result_dict, dict) and isinstance(result_dict.get("table"), dict) and result_dict["table"].get("columns") == ["Error"]:
                    error_message = result_dict.get("text", "Unknown error from SQL execution tool (legacy format).")
                    if isinstance(result_dict["table"].get("rows"), list) and len(result_dict["table"]["rows"]) > 0 and isinstance(result_dict["table"]["rows"][0], list) and len(result_dict["table"]["rows"][0]) > 0:
                        error_message = result_dict["table"]["rows"][0][0] # Get message from legacy error table
                    logger.warning(f"{log_prefix}Subquery '{description}' failed (legacy SQL tool error format): {error_message}")
                    results.append(SubqueryResult(query=description, result={}, error=error_message))
                # Check for a valid success structure (columns and rows directly in result_dict)
                elif isinstance(result_dict, dict) and \
                     "columns" in result_dict and \
                     "rows" in result_dict:
                    # This is a successful result from SQLExecutionTool.
                    # The result_dict *is* the table data, e.g., {"columns": [...], "rows": [...]}.
                    logger.debug(f"{log_prefix}Subquery '{description}' completed successfully. Result keys: {list(result_dict.keys())}")
                    results.append(SubqueryResult(query=description, result=result_dict, error=None))
                else:
                    # If it's none of the above, it's an unexpected/malformed structure
                    error_message = f"Unexpected or malformed result structure from subquery execution. Type: {type(result_dict)}."
                    logger.error(f"{log_prefix}Subquery '{description}': {error_message} Result: {str(result_dict)[:200]}")
                    results.append(SubqueryResult(query=description, result={}, error=error_message))

            except asyncio.TimeoutError:
                # Handle timeout specifically
                error_message = f"Subquery timed out after {settings.SUBQUERY_TIMEOUT_SECONDS} seconds."
                logger.warning(f"{log_prefix}Subquery '{description}': {error_message}")
                results.append(SubqueryResult(query=description, result={}, error=error_message))
            except Exception as e:
                # Handle other exceptions during task execution
                error_message = f"Unexpected error executing subquery: {str(e)}"
                logger.error(f"{log_prefix}Subquery '{description}': {error_message}", exc_info=True)
                results.append(SubqueryResult(query=description, result={}, error=error_message))
                
        return results

    async def _generate_and_execute_single(
        self, 
        description: str, 
        resolved_location_map: Dict[str, str]
    ) -> Dict[str, Any]:
        """Generate and execute a single SQL query using the injected SQLExecutionTool."""
        log_prefix = f"[Org: {self.organization_id}] [SummaryTool] "
        
        try:
            # Generate SQL query with parameter placeholders
            sql, params = await self._generate_sql_and_params(description, resolved_location_map)
            
            # Execute SQL using the injected SQLExecutionTool instance
            # Use self._sql_tool instead of the passed parameter
            result_json_str = await self._sql_tool.ainvoke({
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
            if not result.successful or not isinstance(result.result, dict): # result.result is now the table dict
                continue
                
            columns = result.result.get("columns", []) # Access columns directly from result.result
            rows = result.result.get("rows", [])       # Access rows directly from result.result
            
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
        # Use the injected hierarchy resolver instance
        # resolver = HierarchyNameResolverTool(organization_id=self.organization_id) # REMOVED
        id_name_map = {}
        
        try:
            resolution_result = await self._hierarchy_resolver.ainvoke({"name_candidates": list(ids_to_resolve)})
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
            if not original_result.successful or not isinstance(original_result.result, dict):
                continue
                
            original_table_data = original_result.result # This is now the table dictionary
            original_columns = original_table_data.get("columns", [])
            original_rows = original_table_data.get("rows", [])
            
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
                result={"columns": new_columns, "rows": new_rows}, # Store as a table dict
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
            if result.successful and isinstance(result.result, dict): # result.result is the table dict
                table_data = result.result # Access table data directly
                columns = table_data.get("columns", [])
                rows = table_data.get("rows", [])
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
            if not result.successful or not isinstance(result.result, dict): # result.result is the table dict
                continue
                
            table_data = result.result # Access table data directly
            columns = table_data.get("columns", [])
            rows = table_data.get("rows", [])
            
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
            
            # Updated template to extract descriptions and location names (Use imported template)
            # template = """...""" # Removed original inline definition
            
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
                # subquery_result.result should directly be the table data dict {"columns": ..., "rows": ...}
                if not isinstance(subquery_result.result, dict) or \
                   "columns" not in subquery_result.result or \
                   "rows" not in subquery_result.result:
                     results_str += f"Result: Invalid table data format received in subquery_result.result.\n\n"
                     continue
                
                table_data = subquery_result.result # This is the table dict
                limited_rows = table_data.get("rows", [])[:5]
                columns = table_data.get("columns", [])
                json_output = json.dumps({"columns": columns, "rows": limited_rows}, indent=2)
                results_str += f"Results (showing up to 5 rows): {json_output}\n"
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
            
            # Use imported template constant directly
            prompt = PromptTemplate(input_variables=["query", "context", "results", "insights"], template=SUMMARY_SYNTHESIS_TEMPLATE)
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
            # Decompose the query into subqueries
            subqueries = self._decompose_query(query)
            
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
            results = await self._execute_subqueries_concurrently(subqueries, resolved_location_map)
            
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
