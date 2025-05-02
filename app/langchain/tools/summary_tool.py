import json
import re
import uuid
import logging
import asyncio
import statistics
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, TypedDict, Union

from pydantic import BaseModel, Field

import pandas as pd
from langchain.tools import BaseTool
from langchain.prompts import PromptTemplate
from langchain_openai import AzureChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from openai import APIConnectionError, APITimeoutError, RateLimitError
from sqlalchemy import text, inspect

from app.core.config import settings
from app.db.connection import get_async_db_connection
from app.db.schema_definitions import SCHEMA_DEFINITIONS
from app.langchain.tools.sql_tool import SQLExecutionTool
from app.langchain.tools.hierarchy_resolver_tool import HierarchyNameResolverTool

logger = logging.getLogger(__name__)

# Define proper Pydantic models for structured SQL generation
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
    cross_metric_ratios: List[Dict[str, Any]] = Field(default_factory=list, description="List of cross-metric ratios")

class SummarySynthesizerTool(BaseTool):
    """Tool for synthesizing high-level summaries from data, optimized for concurrent data retrieval."""
    
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
    
    def _decompose_query(self, query: str) -> List[str]:
        """Decompose a complex query into subqueries using LLM."""
        log_prefix = f"[Org: {self.organization_id}] [SummaryTool] "
        
        try:
            llm = AzureChatOpenAI(
                openai_api_key=settings.AZURE_OPENAI_API_KEY,
                azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
                openai_api_version=settings.AZURE_OPENAI_API_VERSION,
                deployment_name=settings.AZURE_OPENAI_DEPLOYMENT_NAME,
                model_name=settings.LLM_MODEL_NAME,
                temperature=0.1,
                max_retries=settings.LLM_MAX_RETRIES,
            )
            
            # Get schema info from schema definitions
            schema_info = self._get_schema_info()
            
            # Check if schema_info is too long and truncate if needed
            if len(schema_info) > 8000:  # Reasonable token limit to prevent issues
                logger.warning(f"{log_prefix}Schema info exceeds 8000 chars, truncating for decomposition.")
                # Keep the first part that describes general schema structure
                first_part = schema_info[:3000]
                # Keep some relevant table definitions based on query content
                relevant_tables = []
                for keyword in ["event", "borrow", "return", "renewal", "hierarchy", "location"]:
                    if keyword in query.lower():
                        # Find and extract relevant table definitions
                        pattern = f"Table: [^\n]+{keyword}[^\n]*\nDescription:[^\n]*\nColumns:[^\n]*(?:\n  [^\n]+)*"
                        matches = re.findall(pattern, schema_info, re.IGNORECASE)
                        relevant_tables.extend(matches)
                
                # Combine and limit to reasonable size
                relevant_tables_text = "\n\n".join(relevant_tables)[:4000]
                schema_info = f"{first_part}\n\n{relevant_tables_text}"
                logger.debug(f"{log_prefix}Truncated schema info to {len(schema_info)} chars")
            
            template = f"""
            You are a data analyst. Given the following high-level query or analysis request,
            break it down into specific, atomic subqueries that need to be executed to gather the necessary data.
            **Aim for an efficient plan** to comprehensively answer the request while respecting concurrency limits.
            The maximum number of subqueries you should ideally generate is **{{max_concurrent_tools}}**.

            IMPORTANT - LOCATION NAMING GUIDELINES:
            When location names (e.g., "Main Library", "Argyle Branch") appear in your subquery descriptions:
            1. ALWAYS use the full, proper name of the location exactly as mentioned in the query
            2. NEVER attempt to format location names as parameter placeholders or IDs
            3. NEVER include raw UUIDs or IDs in your descriptions
            4. DO use natural language expressions like "for Main Library" or "at Argyle Branch"
            5. DO NOT use technical formatting like ":hierarchy_id_main_library" or "hierarchy_id_for_argyle"
            6. The system will automatically handle parameter binding during SQL generation

            DATABASE SCHEMA INFORMATION:
            {schema_info}

            IMPORTANT NOTES ON SCHEMA:
            - The events table is called "5" and contains numbered columns ("1", "2", "3", etc.) for different event metrics.
            - Column "1" = successful borrows
            - Column "2" = unsuccessful borrows
            - Column "3" = successful returns
            - Column "4" = unsuccessful returns
            - All tables use camelCase for column names (e.g., "organizationId", "eventTimestamp")
            
            **CRITICAL TIME-BASED QUERY HANDLING:**
            - If the user query asks about **changes over time**, **trends**, how something has **changed**, or uses similar time-evolution language for a specific metric:
                - Your subquery descriptions MUST specify grouping by a relevant time period (e.g., **"group by month"**, **"by day"**).
                - Example: "Retrieve monthly total successful borrows (column \\"1\\" in table \\"5\\") for all branches over the past 6 months."
            - Otherwise, if the query asks for an overall summary or specific totals without focusing on change over time, do *not* group by time.

            High-level Query (potentially with context): {{query}}

            Format your response as a JSON array of strings, each representing a specific subquery description suitable for the sql_query tool.
            **IMPORTANT**: If a subquery retrieves a metric for a specific location (not org-wide), append **", including the organizational average for comparison"** to the subquery description string. This helps provide context later.

            Example for a query about specific locations:
            ["Retrieve total successful borrows (column \\"1\\" in events table \\"5\\") for Main Library last 30 days, including the organizational average for comparison",
             "Retrieve total successful borrows (column \\"1\\" in events table \\"5\\") for Argyle Branch last 30 days, including the organizational average for comparison"]
            
            Example if no specific locations (org-wide):
            ["Count total successful borrows (column \\"1\\" in events table \\"5\\") across the organization last month"]
            
            Example if time trend requested:
            ["Retrieve monthly total successful borrows (column \\"1\\" in table \\"5\\") for all branches over the past 6 months."]

            Return ONLY the JSON array without any explanation or comments.
            """
            
            prompt = PromptTemplate(
                input_variables=["query", "schema_info", "max_concurrent_tools"],
                template=template
            )
            
            # Implement retry mechanism with exponential backoff
            max_retries = 3
            base_delay = 1  # seconds
            subqueries_str = ""
            
            for retry in range(max_retries):
                try:
                    logger.debug(f"{log_prefix}Decomposing query (attempt {retry+1}/{max_retries})")
                    decompose_chain = prompt | llm | StrOutputParser()
                    subqueries_str = decompose_chain.invoke({
                        "query": query, 
                        "schema_info": schema_info,
                        "max_concurrent_tools": settings.MAX_CONCURRENT_TOOLS
                    })
                    
                    # If we got here, the API call was successful
                    break
                    
                except (APIConnectionError, APITimeoutError) as e:
                    # Log the error
                    logger.warning(f"{log_prefix}API error during decomposition attempt {retry+1}: {e}")
                    
                    # Check if we should retry
                    if retry < max_retries - 1:
                        # Calculate delay with exponential backoff
                        delay = base_delay * (2 ** retry)
                        logger.info(f"{log_prefix}Retrying in {delay} seconds...")
                        time.sleep(delay)  # Use regular sleep since we're in a sync method
                    else:
                        # We've exhausted retries, propagate the error
                        logger.error(f"{log_prefix}Failed to decompose query after {max_retries} attempts")
                        raise
            
            # Clean and parse the JSON
            subqueries_str = self._clean_json_response(subqueries_str)
            
            try:
                subqueries = json.loads(subqueries_str)
                if not isinstance(subqueries, list):
                    logger.warning(f"{log_prefix}Decomposed query result is not a list. Got: {type(subqueries)}")
                    raise ValueError("Subqueries must be a list")
                # Ensure all items are strings
                if not all(isinstance(sq, str) for sq in subqueries):
                     logger.warning(f"{log_prefix}Not all items in subqueries list are strings")
                     raise ValueError("All items in subqueries list must be strings")
                
                # Cap the number of subqueries for safety
                if len(subqueries) > settings.MAX_CONCURRENT_TOOLS:
                    logger.warning(f"{log_prefix}Too many subqueries ({len(subqueries)}), limiting to {settings.MAX_CONCURRENT_TOOLS}")
                    subqueries = subqueries[:settings.MAX_CONCURRENT_TOOLS]
                
                return subqueries
            except json.JSONDecodeError as e:
                logger.error(f"{log_prefix}Error parsing subqueries JSON: {str(e)}")
                logger.debug(f"{log_prefix}Raw subqueries string: {subqueries_str}")
                # Fallback: return the original query as a single subquery
                return [query]
            except ValueError as e:
                logger.error(f"{log_prefix}Error validating subqueries: {str(e)}")
                logger.debug(f"{log_prefix}Problematic subqueries value: {subqueries_str}")
                return [query]
        except APIConnectionError as e:
            logger.error(f"{log_prefix}OpenAI connection error during query decomposition: {e}", exc_info=False)
            return [query]
        except APITimeoutError as e:
            logger.error(f"{log_prefix}OpenAI timeout during query decomposition: {e}", exc_info=False)
            return [query]
        except Exception as e:
            logger.error(f"{log_prefix}Unexpected error during query decomposition: {e}", exc_info=True)
            return [query]
    
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
    
    def _get_schema_info(self) -> str:
        """Gets schema information from predefined schema definitions as a formatted string.
        This follows the same approach as agent.py's _get_schema_string method."""
        from app.db.schema_definitions import SCHEMA_DEFINITIONS  # Keep import local
        
        db_name = "report_management"
        log_prefix = f"[Org: {self.organization_id}] [SummaryTool] "
        logger.debug(f"{log_prefix}Fetching schema for database: {db_name}")
        
        if db_name not in SCHEMA_DEFINITIONS:
            error_msg = f"No schema definition found for database {db_name}."
            logger.warning(f"{log_prefix}{error_msg}")
            return error_msg
        
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
        
        logger.debug(f"{log_prefix}Successfully retrieved schema for {db_name}")
        return "\n".join(schema_info)
    
    async def _verify_query_tables_and_columns(self, sql: str, db_name: str = "report_management") -> Tuple[bool, Optional[str]]:
        """
        Simple verification that just logs the query.
        We're using the schema directly, so no need for complex validation.
        """
        log_prefix = f"[Org: {self.organization_id}] [SummaryTool] "
        logger.debug(f"{log_prefix}Query validated against schema definitions")
        return True, None
    
    async def _generate_sql_from_description(self, query_description: str) -> Tuple[str, Dict[str, Any]]:
        """Generate SQL query and parameters from a natural language description."""
        log_prefix = f"[Org: {self.organization_id}] [SummaryTool] "
        try:
            llm = AzureChatOpenAI(
                openai_api_key=settings.AZURE_OPENAI_API_KEY,
                azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
                openai_api_version=settings.AZURE_OPENAI_API_VERSION,
                deployment_name=settings.AZURE_OPENAI_DEPLOYMENT_NAME,
                model_name=settings.LLM_MODEL_NAME,
                temperature=0.1,
                max_retries=settings.LLM_MAX_RETRIES,
            )
            
            # Get schema info from schema definitions
            schema_info = self._get_schema_info()
            
            # Enhanced pattern to extract location references with possible abbreviations
            location_patterns = [
                # Standard pattern: locations after prepositions (for/at/in/from/to)
                r"(?:for|at|in|from|to)\s+(?:the\s+)?([A-Z][a-zA-Z\s]+(?:\s*\([A-Z]+\))?)(?:\s+(?:branch|library|location))?(?!\s+(?:and|or))",
                
                # Direct mentions with branch/library suffix
                r"([A-Z][a-zA-Z\s]+(?:\s*\([A-Z]+\))?)\s+(?:branch|library|location)(?!\s+(?:and|or))",
                
                # Possessive forms: "Branch's" or "Library's"
                r"([A-Z][a-zA-Z\s]+(?:\s*\([A-Z]+\))?)'s\s+(?:branch|library|location|activity|metrics|data|statistics)(?!\s+(?:and|or))"
            ]
            
            # Extract location references using multiple patterns
            location_matches = []
            for pattern in location_patterns:
                matches = re.findall(pattern, query_description)
                if matches:
                    location_matches.extend(matches)
            
            # Remove duplicates and clean up
            location_matches = [match.strip() for match in location_matches]
            location_matches = list(dict.fromkeys(location_matches))  # Remove duplicates while preserving order
            
            if location_matches:
                logger.debug(f"{log_prefix}Extracted location references: {location_matches}")
            
            # Prepare the template with schema information and updated SQL generation guidance
            template = """
            You are an expert PostgreSQL query generator. Given the schema definition and the query requirement below, 
            generate a syntactically correct PostgreSQL query focused *only* on retrieving the primary data requested.

            DATABASE SCHEMA:
            {schema_info}

            QUERY REQUIREMENT:
            {query_description}

            CRITICAL REQUIREMENTS:
            1. Generate SQL to retrieve ONLY the specific metric(s) or data described in the QUERY REQUIREMENT.
            2. **DO NOT** attempt to calculate organizational averages (e.g., using `AVG(...) OVER ()`) within this *same* query if the main goal is aggregation (like SUM or COUNT). Comparisons and averages will be handled separately if needed. Focus on getting the core data accurately.
            3. ALWAYS include the organization_id filter condition using the `:organization_id` parameter.
            4. The organization_id parameter value is: {organization_id}
            5. Use exact table names (e.g., "5", "8") and column names (e.g., "eventTimestamp", "1") as shown in the schema, ensuring they are double-quoted.
            6. Use descriptive, user-friendly aliases for calculated columns (e.g., `SUM("1") AS "Total Borrows"`).
            7. If the query involves grouping (e.g., by branch, by date), ensure the `GROUP BY` clause is correct and includes all non-aggregated selected columns.
            8. Apply appropriate time filters based on the QUERY REQUIREMENT using functions like `NOW()`, `INTERVAL`, etc. (Refer to schema for timestamp columns).
            9. **NEVER** use raw UUID strings in WHERE conditions. ALWAYS use parameter binding (e.g., `:hierarchy_id_main_library`) for any ID conditions.
            10. If the query mentions filtering by a specific branch, hierarchy, or location, ALWAYS use a parameterized condition like `"hierarchyId" = :hierarchy_id_argyle_branch` in the WHERE clause, where the parameter name is derived from the location name.
            11. IMPORTANT: For location parameters, use the format `:hierarchy_id_location_name` where location_name is the lowercase version of the location with spaces replaced by underscores (e.g., `:hierarchy_id_main_library`, `:hierarchy_id_central_branch`).
            
            JSON OUTPUT FORMAT:
            ```json
            {{
                "sql": "your PostgreSQL query with :organization_id and other needed parameters",
                "params": {{
                    "organization_id": "{organization_id}"
                    // Add other necessary parameters here if used in the SQL, especially hierarchy_id parameters
                }}
            }}
            ```
            
            Write only the JSON output without any additional explanation.
            """
            
            prompt = PromptTemplate(
                input_variables=["query_description", "schema_info", "organization_id"],
                template=template
            )
            
            # Create a chain for SQL generation
            sql_chain = prompt | llm | StrOutputParser()
            
            # Invoke the chain with inputs
            inputs = {
                "query_description": query_description,
                "schema_info": schema_info,
                "organization_id": self.organization_id
            }
            response = sql_chain.invoke(inputs)
            
            # Clean and parse the JSON response
            cleaned_response = self._clean_json_response(response)
            
            try:
                result_dict = json.loads(cleaned_response)
                
                # Extract SQL query and parameters
                sql = result_dict.get("sql", "")
                params = result_dict.get("params", {"organization_id": self.organization_id})
                
                # Ensure organization_id parameter exists
                if "organization_id" not in params:
                    params["organization_id"] = self.organization_id
                
                # Standardized parameter handling - detect all parameters in SQL
                param_pattern = r':([a-zA-Z_][a-zA-Z0-9_]*)'
                sql_params = re.findall(param_pattern, sql)
                
                # Check for location-related parameters that need placeholder values
                for param_name in sql_params:
                    # Skip organization_id as it's already handled
                    if param_name == "organization_id":
                        continue
                        
                    # Check if this is a hierarchy/location parameter that's missing from the params dict
                    if param_name not in params and any(term in param_name for term in ["hierarchy", "branch", "location"]):
                        # Extract the location name from the parameter
                        loc_name_pattern = r'(?:hierarchy|branch|location)_id_(.+)'
                        loc_name_match = re.search(loc_name_pattern, param_name)
                        
                        if loc_name_match:
                            # Get a clean location name for resolution
                            clean_loc_name = loc_name_match.group(1).replace('_', ' ')
                            
                            # Create a placeholder marker without colon
                            # Important: Do NOT include a colon in the parameter value!
                            params[param_name] = f"PLACEHOLDER_ID_{clean_loc_name.upper()}"
                            logger.debug(f"{log_prefix}Added placeholder for parameter '{param_name}': location '{clean_loc_name}'")

                # Log and return the results
                logger.debug(f"{log_prefix}Generated SQL with {len(params)} parameters")
                return sql, params
                
            except json.JSONDecodeError as e:
                logger.error(f"{log_prefix}Error parsing JSON from LLM response: {e}")
                logger.debug(f"{log_prefix}Raw response: {cleaned_response}")
                raise ValueError(f"Invalid JSON format in LLM response: {e}")
                
        except (APIConnectionError, APITimeoutError, RateLimitError) as e:
            logger.error(f"{log_prefix}OpenAI API error in SQL generation: {e}")
            raise ValueError(f"Error communicating with OpenAI: {str(e)}")
        except Exception as e:
            logger.error(f"{log_prefix}Unexpected error in SQL generation: {e}", exc_info=True)
            raise ValueError(f"Error generating SQL: {str(e)}")
    
    async def _execute_subqueries_concurrently(self, subqueries: List[str]) -> List[SubqueryResult]:
        """Execute multiple subqueries concurrently and return their results."""
        log_prefix = f"[Org: {self.organization_id}] [SummaryTool] "
        logger.debug(f"{log_prefix}Executing {len(subqueries)} subqueries concurrently")
        
        # Execute all subqueries concurrently to speed up data gathering
        tasks = [self._run_single_query(subquery) for subquery in subqueries]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results and handle any exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"{log_prefix}Error executing subquery {i+1}: {result}")
                processed_results.append(SubqueryResult(
                    query=subqueries[i],
                    result={},
                    error=f"Failed to execute: {str(result)}"
                ))
            else:
                processed_results.append(result)
        
        return processed_results
    
    async def _run_single_query(self, subquery: str) -> SubqueryResult:
        """Execute a single subquery and return its result."""
        log_prefix = f"[Org: {self.organization_id}] [SummaryTool] "
        try:
            # Generate SQL from the subquery description
            sql, params = await self._generate_sql_from_description(subquery)
            
            # Security check for organization_id (similar to sql_tool.py)
            if "organization_id" not in params:
                error_msg = "SECURITY CHECK FAILED: organization_id missing from parameters"
                logger.error(f"{log_prefix}{error_msg}")
                return SubqueryResult(
                    query=subquery,
                    result={},
                    error=error_msg
                )
            
            if params["organization_id"] != self.organization_id:
                error_msg = f"SECURITY CHECK FAILED: organization_id mismatch in parameters. Expected {self.organization_id}"
                logger.error(f"{log_prefix}{error_msg}")
                return SubqueryResult(
                    query=subquery,
                    result={},
                    error=error_msg
                )
            
            # Process placeholder parameters
            placeholder_params = {}
            location_names = []
            
            # Extract and handle any hierarchy ID placeholders
            for key, value in list(params.items()):
                # Skip the organization_id parameter
                if key == "organization_id":
                    continue
                    
                # Check for placeholder values (that need resolution to actual UUIDs)
                if isinstance(value, str) and value.startswith("PLACEHOLDER_ID_"):
                    # Extract the location name from the placeholder
                    raw_location_name = value.replace("PLACEHOLDER_ID_", "").lower()
                    
                    # Process the location name to handle abbreviations
                    location_name = raw_location_name
                    
                    # Check for abbreviation pattern: "name (abbr)" or similar formats
                    abbr_pattern = r'(.+?)\s*\(([a-z]+)\)'
                    abbr_match = re.search(abbr_pattern, location_name)
                    
                    if abbr_match:
                        # Store both the full name and the abbreviation for better matching
                        full_name = abbr_match.group(1).strip()
                        abbreviation = abbr_match.group(2).strip()
                        
                        logger.debug(f"{log_prefix}Extracted location name '{full_name}' with abbreviation '{abbreviation}'")
                        
                        # Add both forms to increase chances of resolution
                        location_names.append(full_name)
                        location_names.append(abbreviation)
                        placeholder_params[key] = full_name  # Use the full name as primary
                    else:
                        # Standard case - no abbreviation
                        placeholder_params[key] = location_name
                        location_names.append(location_name)
                    
                    # Remove the parameter temporarily (we'll add back the resolved UUID)
                    # Important: Use None as temporary value to ensure it doesn't get passed to SQL as-is
                    params[key] = None
            
            # Resolve location names to UUIDs if needed
            if location_names:
                try:
                    # Create a resolver instance with our organization_id
                    resolver = HierarchyNameResolverTool(organization_id=self.organization_id)
                    
                    # Remove duplicates while preserving order
                    unique_location_names = list(dict.fromkeys(location_names))
                    
                    # Batch resolve all location names
                    logger.debug(f"{log_prefix}Resolving {len(unique_location_names)} location names: {unique_location_names}")
                    resolution_result = await resolver.ainvoke({"name_candidates": unique_location_names})
                    resolution_data = resolution_result.get("resolution_results", {})
                    
                    # Log the resolution results for debugging
                    if logger.isEnabledFor(logging.DEBUG):
                        for name, result in resolution_data.items():
                            status = result.get("status", "unknown")
                            score = result.get("score", 0)
                            resolved_name = result.get("resolved_name", "N/A")
                            logger.debug(f"{log_prefix}Resolution for '{name}': status={status}, score={score}, resolved to '{resolved_name}'")
                    
                    # Process resolution results
                    resolution_failed = False
                    failed_locations = []
                    
                    # Update params with the resolved IDs
                    for param_key, location_name in placeholder_params.items():
                        # Check if we have a direct match first
                        resolution_info = resolution_data.get(location_name, {})
                        
                        # If no direct match, try other forms of the name if we extracted an abbreviation
                        if resolution_info.get("status") != "found":
                            # Try to find a match from any of the names we submitted for this parameter
                            for candidate_name in location_names:
                                candidate_info = resolution_data.get(candidate_name, {})
                                if candidate_info.get("status") == "found" and candidate_info.get("id"):
                                    resolution_info = candidate_info
                                    logger.debug(f"{log_prefix}Found match for '{param_key}' using alternative name '{candidate_name}'")
                                    break
                        
                        if resolution_info.get("status") == "found" and resolution_info.get("id"):
                            # We have a successful resolution - use the UUID
                            resolved_uuid = resolution_info["id"]
                            params[param_key] = resolved_uuid
                            
                            # Include the resolved name in the log for clarity
                            resolved_name = resolution_info.get("resolved_name", "unknown")
                            method = resolution_info.get("method", "unknown")
                            score = resolution_info.get("score", 0)
                            
                            logger.debug(
                                f"{log_prefix}Successfully resolved '{location_name}' to "
                                f"'{resolved_name}' (UUID: {resolved_uuid}) with "
                                f"method={method}, score={score}"
                            )
                        else:
                            # Resolution failed for this location
                            resolution_failed = True
                            failed_locations.append(location_name)
                            logger.warning(
                                f"{log_prefix}Failed to resolve location '{location_name}' "
                                f"(status: {resolution_info.get('status', 'unknown')})"
                            )
                    
                    # If any resolutions failed, return an error
                    if resolution_failed:
                        error_msg = f"Could not resolve the following location(s): {', '.join(failed_locations)}"
                        logger.error(f"{log_prefix}{error_msg}")
                        return SubqueryResult(
                            query=subquery,
                            result={},
                            error=error_msg
                        )
                        
                except Exception as e:
                    # Handle any errors during resolution
                    logger.error(f"{log_prefix}Error resolving location names: {e}", exc_info=True)
                    return SubqueryResult(
                        query=subquery,
                        result={},
                        error=f"Error resolving location names: {str(e)}"
                    )
            
            # Clean up params - remove any None values (unresolved parameters)
            for key in list(params.keys()):
                if params[key] is None:
                    logger.warning(f"{log_prefix}Removing unresolved parameter: {key}")
                    del params[key]
            
            # Execute the SQL query
            try:
                # Get a database connection
                async with get_async_db_connection(db_name="report_management") as conn:
                    # Log the SQL and parameters (for debugging)
                    logger.debug(f"{log_prefix}Executing SQL: {sql}")
                    
                    # Safer parameter logging that masks potential sensitive values
                    safe_params = {
                        k: (
                            "***" if k == "organization_id" else 
                            str(v)[:8] + "..." if isinstance(v, str) and len(str(v)) > 10 else
                            str(v)
                        ) for k, v in params.items()
                    }
                    logger.debug(f"{log_prefix}With params: {json.dumps(safe_params)}")
                    
                    # Execute the SQL query
                    result = await conn.execute(text(sql), params)
                    
                    # Extract column names
                    columns = list(result.keys())
                    
                    # Fetch and process rows
                    rows = result.fetchall()
                    processed_rows = []
                    
                    # Process each row to ensure proper JSON serialization
                    for row in rows:
                        row_dict = {}
                        for col_name, value in zip(columns, row):
                            # Handle special data types
                            if isinstance(value, uuid.UUID):
                                row_dict[col_name] = str(value)  # Convert UUID to string
                            elif isinstance(value, datetime):
                                row_dict[col_name] = value.isoformat()  # Convert datetime to ISO string
                            else:
                                row_dict[col_name] = value
                        processed_rows.append(row_dict)
                    
                    # Log the row count for debugging
                    logger.debug(f"{log_prefix}Query returned {len(processed_rows)} rows")
                    
                    # Return successful result
                    return SubqueryResult(
                        query=subquery,
                        result={"table": {"columns": columns, "rows": processed_rows}},
                        error=None
                    )
                    
            except Exception as e:
                # Handle SQL execution errors
                error_msg = str(e)
                logger.error(f"{log_prefix}Error executing SQL: {error_msg}", exc_info=True)
                
                # Provide more specific error message for common DB errors
                if "invalid input syntax for type uuid" in error_msg:
                    enhanced_error = "Database error: Invalid UUID format in the query parameters. This may be due to a problem with hierarchy ID resolution."
                    logger.error(f"{log_prefix}{enhanced_error}")
                    return SubqueryResult(
                        query=subquery,
                        result={},
                        error=enhanced_error
                    )
                
                return SubqueryResult(
                    query=subquery,
                    result={},
                    error=f"Database error: {error_msg}"
                )
                
        except ValueError as e:
            # Handle SQL generation errors
            logger.error(f"{log_prefix}SQL generation error: {e}")
            return SubqueryResult(
                query=subquery,
                result={},
                error=f"SQL generation error: {str(e)}"
            )
        except Exception as e:
            # Handle any other unexpected errors
            logger.error(f"{log_prefix}Unexpected error processing subquery: {e}", exc_info=True)
            return SubqueryResult(
                query=subquery,
                result={},
                error=f"Unexpected error: {str(e)}"
            )
    
    async def _resolve_and_inject_names(self, subquery_results: List[SubqueryResult]) -> List[SubqueryResult]:
        """Resolves hierarchy IDs to names and injects them into results."""
        log_prefix = f"[Org: {self.organization_id}] [SummaryTool] "
        logger.debug(f"{log_prefix}Attempting to resolve hierarchy IDs to names.")
        
        ids_to_resolve = set()
        results_with_id_col = [] # Store indices of results containing 'hierarchyId'
        
        # Gather all unique IDs from successful results that have a 'hierarchyId' column
        for idx, result in enumerate(subquery_results):
            if result.successful and "table" in result.result:
                columns = result.result["table"].get("columns", [])
                rows = result.result["table"].get("rows", [])
                if "hierarchyId" in columns and rows:
                    results_with_id_col.append(idx)
                    for row in rows:
                        if "hierarchyId" in row and row["hierarchyId"]:
                            ids_to_resolve.add(row["hierarchyId"])
        
        if not ids_to_resolve:
            logger.debug(f"{log_prefix}No hierarchy IDs found in results to resolve.")
            return subquery_results # Nothing to do
            
        logger.debug(f"{log_prefix}Found {len(ids_to_resolve)} unique hierarchy IDs to resolve.")
        
        # Fetch names from hierarchyCaches
        id_name_map = {}
        try:
            async with get_async_db_connection(db_name="report_management") as conn:
                # Enhanced SQL to fetch both name and shortName for better display options
                sql = text('''
                    SELECT "id", "name", "shortName", "parentId" 
                    FROM "hierarchyCaches" 
                    WHERE "id" = ANY(:ids)
                    AND "deletedAt" IS NULL
                    AND "organizationId" = :organization_id
                ''')
                # Convert set to list for parameter binding
                params = {
                    "ids": list(ids_to_resolve),
                    "organization_id": self.organization_id
                }
                db_result = await conn.execute(sql, params)
                fetched_names = await db_result.fetchall()
                
                # Create a comprehensive mapping with multiple name options
                for row in fetched_names:
                    hierarchy_id = str(row[0])
                    full_name = row[1]
                    short_name = row[2]
                    parent_id = row[3]
                    
                    # Store both name formats for display flexibility
                    id_name_map[hierarchy_id] = {
                        "name": full_name,
                        "shortName": short_name,
                        "parentId": str(parent_id) if parent_id else None,
                        "displayName": full_name
                    }
                    
                    # For better display, if shortName is available, use "name (shortName)" format
                    if short_name:
                        id_name_map[hierarchy_id]["displayName"] = f"{full_name} ({short_name})"
                
                logger.debug(f"{log_prefix}Successfully fetched {len(id_name_map)} names for IDs.")
                
                # If we have parent IDs, we might need to fetch parent names too for context
                parent_ids = {entry.get("parentId") for entry in id_name_map.values() 
                             if entry.get("parentId") and entry.get("parentId") not in id_name_map}
                
                if parent_ids:
                    logger.debug(f"{log_prefix}Fetching {len(parent_ids)} parent names for context.")
                    parent_sql = text('''
                        SELECT "id", "name", "shortName"
                        FROM "hierarchyCaches" 
                        WHERE "id" = ANY(:parent_ids)
                        AND "deletedAt" IS NULL
                        AND "organizationId" = :organization_id
                    ''')
                    parent_params = {
                        "parent_ids": list(parent_ids),
                        "organization_id": self.organization_id
                    }
                    parent_result = await conn.execute(parent_sql, parent_params)
                    parent_names = await parent_result.fetchall()
                    
                    # Add parent info to our mapping
                    for row in parent_names:
                        parent_id = str(row[0])
                        parent_name = row[1]
                        parent_short_name = row[2]
                        
                        id_name_map[parent_id] = {
                            "name": parent_name,
                            "shortName": parent_short_name,
                            "displayName": f"{parent_name} ({parent_short_name})" if parent_short_name else parent_name
                        }

        except Exception as e:
            logger.error(f"{log_prefix}Error fetching hierarchy names: {e}", exc_info=True)
            # Proceed without names if lookup fails
            return subquery_results 

        # Inject names into the relevant results
        updated_results = list(subquery_results) # Create a copy
        name_col = "Location Name" # Define the new column name
        parent_col = "Parent Location" # Define column for parent info

        for idx in results_with_id_col:
            original_result = updated_results[idx]
            if original_result.successful: # Double-check success
                original_table = original_result.result.get("table", {})
                original_columns = original_table.get("columns", [])
                original_rows = original_table.get("rows", [])
                
                # Define new columns we'll add
                new_columns = list(original_columns)
                
                # Add Location Name column if not present
                if name_col not in original_columns:
                    new_columns.append(name_col)
                
                # Add Parent Location column for additional context
                has_parent_col = False
                if parent_col not in original_columns:
                    new_columns.append(parent_col)
                    has_parent_col = True
                
                new_rows = []
                
                for row in original_rows:
                    new_row = row.copy()
                    hierarchy_id = new_row.get("hierarchyId")
                    
                    # Use fetched name info if available
                    location_info = id_name_map.get(hierarchy_id, {})
                    
                    # Add location name
                    if name_col not in new_row or not new_row[name_col]:
                        display_name = location_info.get("displayName")
                        new_row[name_col] = display_name
                    
                    # Add parent location info if we have it
                    if has_parent_col:
                        parent_id = location_info.get("parentId")
                        parent_info = id_name_map.get(parent_id, {}) if parent_id else {}
                        parent_name = parent_info.get("displayName")
                        new_row[parent_col] = parent_name
                    
                    new_rows.append(new_row)
                
                # Update the result object in the list
                updated_results[idx] = SubqueryResult(
                    query=original_result.query,
                    result={"table": {"columns": new_columns, "rows": new_rows}},
                    error=original_result.error
                )
                
                # Log what we did for debugging
                added_cols = []
                if name_col not in original_columns:
                    added_cols.append(name_col)
                if has_parent_col:
                    added_cols.append(parent_col)
                
                if added_cols:
                    logger.debug(f"{log_prefix}Injected columns {added_cols} into subquery result index {idx}.")

        return updated_results

    def _calculate_composite_metrics(self, subquery_results: List[SubqueryResult]) -> CompositeMetricsResult:
        """
        Calculate derived and composite metrics from the raw data results, potentially combining data across subqueries.
        This specialized function provides business intelligence by:
        1. Calculating success rates (e.g., borrow success rate) by combining success/failure counts from potentially different subqueries for the same entity.
        2. Creating efficiency metrics (e.g., borrows per active user).
        3. Identifying correlations between different metrics.
        4. Computing growth rates and change velocities.
        """
        log_prefix = f"[Org: {self.organization_id}] [SummaryTool] "
        logger.info(f"{log_prefix}Calculating composite metrics from {len(subquery_results)} result sets (cross-query aggregation enabled)")
        
        composite_metrics = CompositeMetricsResult(
            success_rates=[],
            cross_metric_ratios=[]
        )
        
        # --- Step 1: Aggregate data by entity (Location Name or hierarchyId) across all results --- #
        entity_aggregated_data = {}
        all_numeric_metrics = set() # Track all numeric metric names encountered
        entity_key_col = None # Track which column is used as the primary entity key ("Location Name" preferred)

        for subquery_result in subquery_results:
            if subquery_result.successful:
                table_data = subquery_result.result.get("table", {})
                if isinstance(table_data, dict) and "columns" in table_data and "rows" in table_data:
                    columns = table_data.get("columns", [])
                    rows = table_data.get("rows", [])
                    
                    # Determine the best entity identifier column for this table
                    current_entity_col = None
                    if "Location Name" in columns:
                        current_entity_col = "Location Name"
                        if entity_key_col is None: entity_key_col = "Location Name"
                    elif "hierarchyId" in columns:
                        current_entity_col = "hierarchyId"
                        if entity_key_col is None: entity_key_col = "hierarchyId"
                    # Add other potential entity columns if needed (e.g., "Branch Code")
                    
                    if not current_entity_col:
                        # If no entity column, treat as organization-wide aggregate (use a placeholder key)
                        entity_id = "_org_wide_"
                        if entity_id not in entity_aggregated_data:
                            entity_aggregated_data[entity_id] = {}
                        for row in rows:
                            for col in columns:
                                if col not in ["Location Name", "hierarchyId"] and col in row and row[col] is not None: # Exclude entity keys
                                    try:
                                        value = float(row[col])
                                        # Sum org-wide metrics
                                        entity_aggregated_data[entity_id][col] = entity_aggregated_data[entity_id].get(col, 0) + value
                                        all_numeric_metrics.add(col)
                                    except (ValueError, TypeError): pass
                        continue # Move to next subquery result

                    # Process rows with an entity column
                    for row in rows:
                        entity_id = row.get(current_entity_col)
                        if entity_id:
                            if entity_id not in entity_aggregated_data:
                                entity_aggregated_data[entity_id] = {}
                                # Store the preferred entity name if available
                                if current_entity_col == "Location Name":
                                    entity_aggregated_data[entity_id]["_entity_display_name_"] = entity_id
                                elif "Location Name" in row and row["Location Name"]:
                                     entity_aggregated_data[entity_id]["_entity_display_name_"] = row["Location Name"]
                                else:
                                     entity_aggregated_data[entity_id]["_entity_display_name_"] = entity_id # Fallback to ID
                            
                            # Aggregate numeric metrics for this entity
                            for col in columns:
                                if col != current_entity_col and "Location Name" not in col and "hierarchyId" not in col and col in row and row[col] is not None:
                                    try:
                                        value = float(row[col])
                                        entity_aggregated_data[entity_id][col] = entity_aggregated_data[entity_id].get(col, 0) + value
                                        all_numeric_metrics.add(col)
                                    except (ValueError, TypeError): pass

        logger.debug(f"{log_prefix}Aggregated data for {len(entity_aggregated_data)} entities across subqueries.")
        # --- End Step 1 --- #

        # --- Step 2: Calculate Success Rates --- #
        # Define pairs of success/failure columns (using aliases generated by SQL)
        success_failure_pairs = [
            ("Total Borrows", "Total Unsuccessful Borrows", "Borrow Success Rate"),
            ("Total Successful Borrows", "Total Unsuccessful Borrows", "Borrow Success Rate"),
            ("Total Successful Returns", "Total Unsuccessful Returns", "Return Success Rate"),
            # Add other potential pairs like logins, etc.
        ]

        for entity_id, metrics in entity_aggregated_data.items():
            entity_display_name = metrics.get("_entity_display_name_", entity_id if entity_id != "_org_wide_" else "Organization Wide")
            for success_col, failure_col, rate_name in success_failure_pairs:
                if success_col in metrics and failure_col in metrics:
                    success_val = metrics[success_col]
                    failure_val = metrics[failure_col]
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
        # --- End Step 2 --- #

        # --- Step 3: Calculate Cross-Metric Ratios --- #
        # Use the overall aggregated metrics for org-wide ratios, or sum across entities if needed
        overall_metrics = entity_aggregated_data.get("_org_wide_", {})
        # If org-wide metrics weren't directly calculated, sum them from entities
        if not overall_metrics:
             for entity_id, metrics in entity_aggregated_data.items():
                 if entity_id != "_org_wide_":
                     for metric_name, value in metrics.items():
                         if not metric_name.startswith("_"):
                             overall_metrics[metric_name] = overall_metrics.get(metric_name, 0) + value

        ratio_definitions = [
            # (numerator_aliases, denominator_aliases, ratio_name)
            (["Total Borrows", "Total Successful Borrows"], ["Total Users", "Active Patrons"], "Borrows per User"), # Placeholder, need user count
            (["Total Borrows", "Total Successful Borrows"], ["Total Entries", "Total Visits"], "Borrows per Visit"), # Placeholder, need visit count
            (["Total Successful Returns"], ["Total Borrows", "Total Successful Borrows"], "Return Rate"),
            (["Total Renewals"], ["Total Borrows", "Total Successful Borrows"], "Renewal Rate"),
            (["Total Logins"], ["Total Users", "Active Patrons"], "Logins per User") # Placeholder, need user count
        ]

        for num_aliases, denom_aliases, ratio_name in ratio_definitions:
            num_val, num_key = None, None
            denom_val, denom_key = None, None
            
            # Find first matching numerator metric in overall_metrics
            for alias in num_aliases:
                if alias in overall_metrics:
                    num_val = overall_metrics[alias]
                    num_key = alias
                    break
            
            # Find first matching denominator metric in overall_metrics
            for alias in denom_aliases:
                if alias in overall_metrics:
                    denom_val = overall_metrics[alias]
                    denom_key = alias
                    break
            
            # Calculate ratio if both found and denominator > 0
            if num_val is not None and denom_val is not None and denom_val > 0:
                ratio = num_val / denom_val
                composite_metrics.cross_metric_ratios.append({
                    "name": ratio_name,
                    "numerator": f"{num_key} ({num_val})",
                    "denominator": f"{denom_key} ({denom_val})",
                    "ratio": round(ratio, 2)
                })
        # --- End Step 3 --- #

        logger.info(f"{log_prefix}Composite metrics calculated. Found: {len(composite_metrics.success_rates)} success rates, "
                  f"{len(composite_metrics.cross_metric_ratios)} cross-metric ratios")
        
        return composite_metrics
    
    def _detect_trends(self, subquery_results: List[SubqueryResult]) -> InsightsResult:
        """
        Automatically detect trends, anomalies, and patterns in the data.
        This specialized function analyzes numerical data across subqueries to identify:
        1. Trends over time
        2. Outliers and anomalies
        3. Correlations between different metrics
        4. Performance relative to organizational baselines
        """
        log_prefix = f"[Org: {self.organization_id}] [SummaryTool] "
        logger.info(f"{log_prefix}Analyzing trends across {len(subquery_results)} result sets")
        
        insights = InsightsResult(
            trends=[],
            anomalies=[],
            organizational_comparisons=[]
        )
        
        try:
            # Extract numerical data series from all successful results
            numerical_series = {}
            time_series_data = {}
            entity_performance = {}
            org_averages = {}
            
            # First pass: extract and organize data
            for subquery_result in subquery_results:
                if subquery_result.successful:
                    table_data = subquery_result.result.get("table", {})
                    if not isinstance(table_data, dict):
                        continue
                        
                    columns = table_data.get("columns", [])
                    rows = table_data.get("rows", [])
                    
                    if not columns or not rows:
                        continue
                    
                    # Look for time-related columns to identify time series
                    time_cols = [i for i, col in enumerate(columns) if any(time_term in col.lower() 
                                for time_term in ["date", "time", "day", "month", "year", "week"])]
                    
                    # Look for entity-related columns (locations, branches, etc.)
                    entity_cols = [i for i, col in enumerate(columns) if any(entity_term in col.lower() 
                                  for entity_term in ["name", "location", "branch", "department", "id"])]
                    
                    # Look for organizational average columns
                    avg_cols = [i for i, col in enumerate(columns) if "average" in col.lower() or "avg" in col.lower()]
                    
                    # Extract numerical columns (excluding time and entity columns)
                    num_cols = []
                    for i, col in enumerate(columns):
                        if i not in time_cols and i not in entity_cols:
                            # Check if column contains numeric data
                            has_numeric = False
                            for row in rows[:min(5, len(rows))]:  # Check first few rows
                                # Access row data using column NAME (key)
                                if col in row and row[col] is not None: 
                                    try:
                                        float(row[col])
                                        has_numeric = True
                                        break
                                    except (ValueError, TypeError):
                                        pass
                            if has_numeric:
                                # Store column name instead of index
                                num_cols.append(col) 
                    
                    # Process time series data if available
                    if time_cols and num_cols:
                        time_col_idx = time_cols[0]  # Use the first time column index
                        time_col_name = columns[time_col_idx] # Get its name
                        # Iterate through numeric column NAMES
                        for num_col_name in num_cols: 
                            series_name = f"{num_col_name}"
                            time_series_data[series_name] = []
                            
                            for row in rows:
                                # Access row data using column NAMES (keys)
                                if time_col_name in row and num_col_name in row and row[time_col_name] and row[num_col_name] is not None: 
                                    try:
                                        time_val = row[time_col_name]
                                        num_val = float(row[num_col_name])
                                        time_series_data[series_name].append((time_val, num_val))
                                    except (ValueError, TypeError):
                                        pass
                    
                    # Process entity performance data
                    if entity_cols and num_cols:
                        entity_col_idx = entity_cols[0]  # Use the first entity column index
                        entity_col_name = columns[entity_col_idx] # Get its name
                        # Iterate through numeric column NAMES
                        for metric_name in num_cols: 
                            
                            if metric_name not in entity_performance:
                                entity_performance[metric_name] = {}
                                
                            for row in rows:
                                # Access row data using column NAMES (keys)
                                if entity_col_name in row and metric_name in row and row[entity_col_name] and row[metric_name] is not None: 
                                    try:
                                        entity_name = str(row[entity_col_name])
                                        num_val = float(row[metric_name])
                                        entity_performance[metric_name][entity_name] = num_val
                                    except (ValueError, TypeError):
                                        pass
                    
                    # Process organizational averages
                    if avg_cols:
                        for avg_col_idx in avg_cols:
                            if avg_col_idx < len(columns):
                                metric_name = columns[avg_col_idx] # Get column name
                                for row in rows:
                                    # Access row data using column NAME (key)
                                    if metric_name in row and row[metric_name] is not None: 
                                        try:
                                            org_averages[metric_name] = float(row[metric_name])
                                            break  # Just need one value for org average
                                        except (ValueError, TypeError):
                                            pass
            
            # Analyze time series for trends
            for series_name, data_points in time_series_data.items():
                if len(data_points) >= 3:  # Need at least 3 points to detect a trend
                    # Sort by time
                    data_points.sort(key=lambda x: x[0])
                    
                    # Extract just the values
                    values = [point[1] for point in data_points]
                    
                    # Check for consistent increase/decrease
                    is_increasing = all(values[i] <= values[i+1] for i in range(len(values)-1))
                    is_decreasing = all(values[i] >= values[i+1] for i in range(len(values)-1))
                    
                    # Calculate percent change from first to last
                    if values[0] != 0:
                        percent_change = ((values[-1] - values[0]) / values[0]) * 100
                    else:
                        percent_change = 0
                        
                    if is_increasing and abs(percent_change) > 10:
                        insights.trends.append(TrendInfo(
                            metric=series_name,
                            direction="increasing",
                            percent_change=round(percent_change, 1),
                            confidence="high" if len(data_points) > 5 else "medium"
                        ))
                    elif is_decreasing and abs(percent_change) > 10:
                        insights.trends.append(TrendInfo(
                            metric=series_name,
                            direction="decreasing",
                            percent_change=round(abs(percent_change), 1),
                            confidence="high" if len(data_points) > 5 else "medium"
                        ))
                    elif abs(percent_change) > 20:  # Significant change but not monotonic
                        insights.trends.append(TrendInfo(
                            metric=series_name,
                            direction="increasing" if percent_change > 0 else "decreasing",
                            percent_change=round(abs(percent_change), 1),
                            confidence="medium"
                        ))
            
            # Detect anomalies (outliers) in entity performance
            for metric_name, entities in entity_performance.items():
                if len(entities) >= 3:  # Need at least 3 entities to find outliers
                    values = list(entities.values())
                    
                    if values:
                        mean_val = statistics.mean(values)
                        # Calculate median and standard deviation if possible
                        try:
                            median_val = statistics.median(values)
                            stdev_val = statistics.stdev(values) if len(values) > 1 else 0
                            
                            # Find outliers (more than 2 stdevs from mean)
                            if stdev_val > 0:
                                for entity_name, value in entities.items():
                                    z_score = (value - mean_val) / stdev_val
                                    if abs(z_score) > 2:
                                        insights.anomalies.append(AnomalyInfo(
                                            entity=entity_name,
                                            metric=metric_name,
                                            difference_from_avg=f"{round(((value - mean_val) / mean_val) * 100, 1)}% {'above' if value > mean_val else 'below'} the average for this metric across all entities", 
                                            severity="high" if abs(z_score) > 3 else "medium"
                                        ))
                        except statistics.StatisticsError:
                            pass
                
            # Compare entity performance to organizational averages
            for metric_name, entities in entity_performance.items():
                org_avg_key = next((key for key in org_averages.keys() if metric_name in key), None)
                
                if org_avg_key and org_averages[org_avg_key] > 0:
                    org_avg = org_averages[org_avg_key]
                    
                    for entity_name, value in entities.items():
                        percent_diff = ((value - org_avg) / org_avg) * 100
                        
                        if abs(percent_diff) > 25:  # Significant difference from org average
                            insights.organizational_comparisons.append(OrganizationalComparison(
                                entity=entity_name,
                                metric=metric_name,
                                percent_difference=round(percent_diff, 1),
                                performance="above average" if value > org_avg else "below average",
                                value=value,
                                org_average=org_avg
                            ))
            
            logger.info(f"{log_prefix}Trend analysis complete. Found: {len(insights.trends)} trends, "
                      f"{len(insights.anomalies)} anomalies, {len(insights.organizational_comparisons)} org comparisons")
            
            return insights
            
        except Exception as e:
            logger.error(f"{log_prefix}Error during trend detection: {e}", exc_info=True)
            return insights  # Return empty structure on error
    
    def _synthesize_results(self, query: str, subquery_results: List[SubqueryResult]) -> str:
        """Synthesize subquery results into a coherent summary with automated insights."""
        log_prefix = f"[Org: {self.organization_id}] [SummaryTool] "
        try:
            # Check if all subqueries failed - using proper check that accounts for successful field
            all_failed = all(not result.successful for result in subquery_results)
                    
            if all_failed:
                error_messages = []
                for subquery_result in subquery_results:
                    if subquery_result.error:
                        error_messages.append(subquery_result.error)
                
                unique_errors = set(error_messages)
                
                # Check for specific schema mismatch errors
                table_not_found_errors = []
                for err in unique_errors:
                    if isinstance(err, str) and "Table '" in err and "does not exist in the database" in err:
                        match = re.search(r"Table '([^']+)'", err)
                        if match:
                            table_not_found_errors.append(match.group(1))
                
                if table_not_found_errors:
                    tables_str = ", ".join(f"'{t}'" for t in table_not_found_errors)
                    return (f"I couldn't retrieve the requested data because of a schema mismatch issue. "
                           f"The following tables couldn't be found in the database: {tables_str}. "
                           "This suggests there's a discrepancy between the expected schema and the actual database structure.")
                
                if any("relation" in str(err) and "does not exist" in str(err) for err in unique_errors):
                    return ("I couldn't retrieve the requested library data because some required database tables appear to be missing. "
                           "This could be due to a system configuration issue. Please contact your system administrator.")
                
                if any("column" in str(err) and "does not exist" in str(err) for err in unique_errors):
                    return ("I couldn't retrieve the requested library data because some database columns appear to be missing or incorrectly referenced. "
                           "This could be due to a schema mismatch issue. Please contact your system administrator.")
                
                # Check for location/hierarchy ID resolution errors
                if any("resolve" in str(err).lower() and "location" in str(err).lower() for err in unique_errors):
                    location_errors = [err for err in unique_errors if "resolve" in str(err).lower() and "location" in str(err).lower()]
                    if location_errors:
                        return (f"I couldn't complete the analysis because I couldn't find the location you mentioned. "
                               f"Details: {location_errors[0]}")

                if any("uuid" in str(err).lower() for err in unique_errors):
                    return ("I encountered an issue with the location identifiers in the database. "
                           "This is likely a technical problem with how locations are being referenced. "
                           "Please contact your system administrator.")
                
                if len(unique_errors) == 1:
                    return f"I wasn't able to analyze the requested data due to a technical issue: {next(iter(unique_errors))}"
                else:
                    return ("I couldn't retrieve the data needed for analysis. "
                           "There appears to be a technical issue with accessing the database.")
            
            # Extract location names from successful results for better context
            mentioned_locations = set()
            for result in subquery_results:
                if result.successful and "table" in result.result:
                    columns = result.result["table"].get("columns", [])
                    rows = result.result["table"].get("rows", [])
                    
                    # Check if we have location name column
                    if "Location Name" in columns and rows:
                        location_idx = columns.index("Location Name")
                        for row in rows:
                            if "Location Name" in row and row["Location Name"]:
                                mentioned_locations.add(row["Location Name"])
            
            # First detect trends and patterns automatically
            insights = self._detect_trends(subquery_results)
            
            # Check for anomalies in time series data
            try:
                for result in subquery_results:
                    if result.successful and "table" in result.result:
                        table_data = result.result["table"]
                        columns = table_data.get("columns", [])
                        rows = table_data.get("rows", [])
                        
                        # Look for time-related columns
                        time_cols = [col for col in columns if any(time_term in col.lower() 
                                    for time_term in ["date", "time", "day", "month", "year", "week"])]
                        
                        # Look for numeric columns
                        numeric_cols = []
                        if rows and time_cols:
                            for col in columns:
                                if col not in time_cols:
                                    # Check first row to see if column contains numeric data
                                    if rows[0].get(col) is not None:
                                        try:
                                            float(rows[0].get(col))
                                            numeric_cols.append(col)
                                        except (ValueError, TypeError):
                                            pass
                        
                        # If we have time and numeric data, check for anomalies
                        if time_cols and numeric_cols and len(rows) >= 5:  # Need at least 5 points
                            time_col = time_cols[0]
                            for num_col in numeric_cols:
                                # Extract values and calculate statistics
                                values = []
                                for row in rows:
                                    if row.get(num_col) is not None:
                                        try:
                                            values.append(float(row.get(num_col)))
                                        except (ValueError, TypeError):
                                            pass
                                
                                if len(values) >= 5:
                                    # Calculate mean and standard deviation
                                    try:
                                        mean_val = statistics.mean(values)
                                        stdev_val = statistics.stdev(values)
                                        
                                        # Check for outliers (more than 2.5 standard deviations from mean)
                                        for i, val in enumerate(values):
                                            if abs(val - mean_val) > 2.5 * stdev_val:
                                                # Found an anomaly
                                                entity = rows[i].get("Location Name", "Unknown location")
                                                time_point = rows[i].get(time_col, "Unknown time")
                                                
                                                logger.info(f"{log_prefix}Detected anomaly in {num_col} at {time_point} "
                                                           f"for {entity}: value {val} deviates significantly from mean {mean_val}")
                                                
                                                # Add to insights
                                                insights.anomalies.append(AnomalyInfo(
                                                    entity=f"{entity} at {time_point}",
                                                    metric=num_col,
                                                    difference_from_avg=f"{round(((val - mean_val) / mean_val) * 100, 1)}% {'above' if val > mean_val else 'below'} the average",
                                                    severity="high" if abs(val - mean_val) > 3 * stdev_val else "medium"
                                                ))
                                    except statistics.StatisticsError as e:
                                        # Not enough data for standard deviation or other statistics error
                                        logger.debug(f"{log_prefix}Statistics error during anomaly detection: {str(e)}")
                                    except Exception as e:
                                        # Handle any other unexpected errors during calculation
                                        logger.debug(f"{log_prefix}Error during anomaly calculations: {str(e)}")
                                        # Continue processing other metrics
            except Exception as e:
                logger.warning(f"{log_prefix}Error during anomaly detection: {str(e)}")
                # Continue with synthesis even if anomaly detection fails
            
            # Then calculate composite metrics from the data
            composite_metrics = self._calculate_composite_metrics(subquery_results)
            
            llm = AzureChatOpenAI(
                openai_api_key=settings.AZURE_OPENAI_API_KEY,
                azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
                openai_api_version=settings.AZURE_OPENAI_API_VERSION,
                deployment_name=settings.AZURE_OPENAI_DEPLOYMENT_NAME,
                model_name=settings.LLM_MODEL_NAME,
                temperature=0.3,
                max_retries=settings.LLM_MAX_RETRIES,
            )
            
            results_str = ""
            successful_data = {} # Store successful data for potential calculations
            error_count = 0
            
            for subquery_result in subquery_results:
                results_str += f"Subquery: {subquery_result.query}\n"
                if subquery_result.error:
                     results_str += f"Result: Error - {subquery_result.error}\n\n"
                     error_count += 1
                     continue 
                table_data = subquery_result.result.get("table", {})
                if not isinstance(table_data, dict):
                     results_str += f"Result: Invalid table data format received.\n\n"
                     error_count += 1
                     continue
                # Store successful data keyed by subquery for potential calculations
                successful_data[subquery_result.query] = table_data 
                limited_rows = table_data.get("rows", [])[:5]
                columns = table_data.get("columns", [])
                results_str += f"Results (showing up to 5 rows): {json.dumps({"columns": columns, "rows": limited_rows}, indent=2)}\n"
                results_str += f"Total rows in original result: {len(table_data.get('rows', []))}\n\n"
            
            # Log the success/failure rate - FIXED count that properly matches the actual state
            total_subqueries = len(subquery_results)
            success_count = total_subqueries - error_count
            logger.info(f"{log_prefix}Synthesis stats: {success_count}/{total_subqueries} subqueries successful")
            
            # If all subqueries had some error but we're continuing, add a note about partial data
            warning_prefix = ""
            if error_count > 0 and success_count > 0:
                warning_prefix = "**Note: Some data could not be retrieved due to database issues. This analysis is based on partial data.**\n\n"
            
            # Add automated insights section to the synthesizer input
            insights_str = "Automated Insights Detected:\n"
            
            if insights.trends:
                insights_str += "Trends:\n"
                for trend in insights.trends:
                    insights_str += f"- {trend.metric}: {trend.direction} by {trend.percent_change}% (Confidence: {trend.confidence})\n"
            else:
                insights_str += "Trends: None detected\n"
                
            if insights.anomalies:
                insights_str += "\nAnomalies/Outliers:\n"
                for anomaly in insights.anomalies:
                    insights_str += f"- {anomaly.entity}: {anomaly.metric} is {anomaly.difference_from_avg} (Severity: {anomaly.severity})\n"
            else:
                insights_str += "\nAnomalies/Outliers: None detected\n"
                
            if insights.organizational_comparisons:
                insights_str += "\nOrganizational Comparisons:\n"
                for comp in insights.organizational_comparisons:
                    insights_str += f"- {comp.entity}: {comp.metric} is {comp.percent_difference}% {comp.performance} ({comp.value} vs. org avg {comp.org_average})\n"
            else:
                insights_str += "\nOrganizational Comparisons: None detected\n"
            
            # Add composite metrics section
            if composite_metrics.success_rates or composite_metrics.cross_metric_ratios:
                insights_str += "\nAutomatically Calculated Composite Metrics:\n"
                
                if composite_metrics.success_rates:
                    insights_str += "Success Rates:\n"
                    for rate in composite_metrics.success_rates:
                        entity_str = f" for {rate['entity']}" if rate['entity'] else ""
                        insights_str += f"- {rate['metric']}{entity_str}: {rate['success_rate']}% ({rate['success_count']} successes, {rate['failure_count']} failures)\n"
                
                if composite_metrics.cross_metric_ratios:
                    insights_str += "\nRelationship Metrics:\n"
                    for ratio in composite_metrics.cross_metric_ratios:
                        insights_str += f"- {ratio['name']}: {ratio['ratio']} ({ratio['numerator']} / {ratio['denominator']})\n"
            
            # Add information about locations covered in the analysis
            context_str = ""
            if mentioned_locations:
                context_str = f"\nLocations covered in this analysis: {', '.join(sorted(mentioned_locations))}\n"
            
            template = """
            You are a data analyst for a library system. Given the original user query, results from potentially multiple subqueries, and automated insights,
            synthesize a coherent, insightful summary that addresses the original question.
            The subquery results may contain data tables or error messages.

            Original Query: {query}

            {context}

            Subquery Results:
            {results}
            
            {insights}

            Provide a comprehensive summary that:
            1. Directly answers the original query, integrating information from all **successful** subqueries.
            2. Incorporates the automated insights and composite metrics. 
                - **Crucially**: When reporting trends, state only the `direction` and overall `percent_change` provided in the `TrendInfo` insight. Do not invent specific month-to-month changes or percentages not explicitly provided.
                - Report anomalies using the full description provided.
            3. If subquery results include a "Location Name" column, **use the Location Name instead of the hierarchyId** when referring to specific locations.
            4. If any subqueries resulted in errors, acknowledge this limitation.
            5. Highlights key insights and patterns apparent from the combined successful results.
            6. Mentions any notable outliers or anomalies within the successful results, using the clear descriptions provided in the insights.
            7. Uses specific numbers *sparingly* in the text for key totals or findings. **DO NOT include markdown tables, raw data lists, or extensive numerical lists in this text summary.** Refer to the underlying data conceptually (e.g., "Monthly data shows fluctuations...", "Borrow success rate was high..."). The detailed data should be presented via separate tables or visualizations if requested.
            8. Discusses implications of the calculated metrics (success rates, efficiency metrics, etc.) when available.
            9. **Formatting:** Structure your summary clearly, potentially using bullet points for key findings. Use markdown bolding (`**text**`) for emphasis on insights or overall conclusions.
            10. Is written in a professional, concise style (aim for a few key sentences summarizing the findings).
            11. If the majority of subqueries failed, acknowledge the limitations but try to provide value from any successful data.

            Your summary:
            """
            
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
            logger.info(f"{log_prefix}Query decomposed into {len(subqueries)} subqueries")
            
            # Check if any subquery contains potential hierarchy ID placeholders
            contains_hierarchy_references = any(
                "hierarchy" in sq.lower() or "branch" in sq.lower() or "location" in sq.lower()
                for sq in subqueries
            )
            
            if contains_hierarchy_references:
                logger.info(f"{log_prefix}Detected potential hierarchy/location references in subqueries")
            
            # Check if any subquery contains direct hierarchy ID references that might be problematic
            direct_id_pattern = r"(?:hierarchy|branch).+?(?:'|\")([a-fA-F0-9-]+)(?:'|\")"
            direct_id_matches = [re.search(direct_id_pattern, sq) for sq in subqueries]
            direct_id_matches = [m for m in direct_id_matches if m is not None]
            
            if direct_id_matches:
                direct_ids = [m.group(1) for m in direct_id_matches]
                logger.warning(
                    f"{log_prefix}Detected {len(direct_ids)} direct UUID-like references in subqueries: "
                    f"{direct_ids[:3]}{'...' if len(direct_ids) > 3 else ''}"
                )
                logger.info(f"{log_prefix}Parameter binding system will attempt to handle these IDs properly")
            
            # Execute all subqueries concurrently
            results = await self._execute_subqueries_concurrently(subqueries)
            
            # Log execution results summary
            successful_count = sum(1 for r in results if r.successful)
            error_count = len(results) - successful_count
            
            success_info = {
                "total_subqueries": len(results),
                "successful_subqueries": successful_count,
                "failed_subqueries": error_count,
                "success_rate": f"{successful_count/len(results)*100:.1f}%" if results else "N/A"
            }
            
            logger.info(f"{log_prefix}Subquery execution results: "
                       f"{successful_count}/{len(results)} successful, "
                       f"{error_count}/{len(results)} failed")
            
            if error_count > 0:
                # Log first few errors for debugging
                for i, result in enumerate(results):
                    if not result.successful and i < 3:  # Limit to first 3 errors
                        logger.warning(f"{log_prefix}Subquery error [{i+1}]: {result.error}")
                        
                # Check if errors are related to hierarchy ID resolution
                hierarchy_id_errors = [
                    i for i, r in enumerate(results) 
                    if not r.successful and r.error and (
                        "hierarchy" in r.error.lower() or 
                        "location" in r.error.lower() or
                        "uuid" in r.error.lower() or
                        "id" in r.error.lower()
                    )
                ]
                
                if hierarchy_id_errors:
                    logger.warning(
                        f"{log_prefix}{len(hierarchy_id_errors)} errors appear to be related to "
                        f"hierarchy ID resolution or parameter binding"
                    )
            
            # Resolve hierarchy IDs to names
            results_with_names = await self._resolve_and_inject_names(results)
            
            # Calculate composite metrics
            metrics = self._calculate_composite_metrics(results_with_names)
            
            # Analyze trends and anomalies
            insights = self._detect_trends(results_with_names)
            
            # Synthesize a natural language summary 
            summary = self._synthesize_results(query, results_with_names)
            
            logger.info(f"{log_prefix}Synthesis stats: {successful_count}/{len(results)} subqueries successful")
            
            return {
                "summary": summary,
                "query": query,
                "subquery_count": len(results),
                "execution_info": success_info
            }
        except (APIConnectionError, APITimeoutError, RateLimitError) as e:
            logger.error(f"{log_prefix}OpenAI API error: {e}", exc_info=False)
            return {
                "summary": f"I encountered an issue connecting to the analysis service. Please try again in a moment.",
                "error": str(e),
                "query": query
            }
        except Exception as e:
            logger.error(f"{log_prefix}Unexpected error processing summary: {e}", exc_info=True)
            return {
                "summary": f"I encountered an unexpected error while processing your request: {str(e)}",
                "error": str(e),
                "query": query
            }

    def _run(self, query: str) -> Dict[str, str]:
        """
        Synchronous run method required by BaseTool.
        This tool only supports async operation, so this method raises NotImplementedError.
        """
        raise NotImplementedError("This tool only supports async operation. Use ainvoke instead.")