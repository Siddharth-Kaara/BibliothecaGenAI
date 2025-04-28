import json
import logging
import asyncio
import re
from typing import Any, Dict, List, Optional, Tuple

from langchain.prompts import PromptTemplate
from langchain.tools import BaseTool
from langchain_openai import AzureChatOpenAI
from sqlalchemy import text
from langchain_core.output_parsers import StrOutputParser

from app.core.config import settings
from app.db.connection import get_async_db_connection
from app.langchain.tools.sql_tool import SQLQueryTool

logger = logging.getLogger(__name__)

class SummarySynthesizerTool(BaseTool):
    """Tool for synthesizing high-level summaries from data, optimized for concurrent data retrieval."""
    
    name: str = "summary_synthesizer"
    description: str = """\
    **WARNING:** Use this tool ONLY for open-ended, qualitative summaries (e.g., 'summarize activity'). 
    For specific, quantifiable metric comparisons or retrievals (e.g., 'compare borrows for Branch A and B', 'get total renewals last month'), 
    you MUST use the 'sql_query' tool instead.
    
    Creates high-level summaries and insights from data by intelligently decomposing the request and fetching data concurrently.
    Use this tool for complex queries requiring analysis across multiple dimensions or when a narrative summary is preferred.
    Input should be a description of the summary or analysis needed.
    """
    
    organization_id: str
    
    def _decompose_query(self, query: str) -> List[str]:
        """Decompose a complex query into subqueries using LLM."""
        llm = AzureChatOpenAI(
            openai_api_key=settings.AZURE_OPENAI_API_KEY,
            azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
            openai_api_version=settings.AZURE_OPENAI_API_VERSION,
            deployment_name=settings.AZURE_OPENAI_DEPLOYMENT_NAME,
            model_name=settings.LLM_MODEL_NAME,
            temperature=0.1,
        )
        
        template = """
        You are a data analyst. Given the following high-level query or analysis request,
        break it down into 2-5 specific, atomic subqueries that need to be executed to gather the necessary data.

        Context Provided in Query: The input query may contain already resolved entity names and IDs (e.g., '...for Main Library (ID: uuid-...)'). If IDs are provided, formulate subqueries that use these specific IDs for filtering or joining.

        Database Schema Hints:
        - The main hierarchy table is named "hierarchyCaches".
        - The main event data table is named "5".
        - Joins often happen between "5"."hierarchyId" and "hierarchyCaches"."id".
        - Metric Columns in table "5": Borrows (1=Success, 2=Fail), Returns (3=Success, 4=Fail), Logins (5=Success, 6=Fail), Renewals (7=Success, 8=Fail), Payments (32=Success, 33=Fail).

        High-level Query (potentially with context): {query}

        Format your response as a JSON array of strings, each representing a specific subquery description suitable for the sql_query tool.
        **IMPORTANT**: If a subquery retrieves a metric for a specific hierarchy ID (not org-wide), append **", including the organizational average for comparison"** to the subquery description string. This helps provide context later.

        Example if query contains resolved IDs:
        ["Retrieve borrow count for hierarchy ID 'uuid-for-main' last 30 days, including the organizational average for comparison",
         "Retrieve borrow count for hierarchy ID 'uuid-for-argyle' last 30 days, including the organizational average for comparison"]
        Example if no IDs provided (org-wide):
        ["Count total successful borrows across the organization last month"]

        Return ONLY the JSON array without any explanation or comments.
        """
        
        prompt = PromptTemplate(input_variables=["query"], template=template)
        decompose_chain = prompt | llm | StrOutputParser()
        subqueries_str = decompose_chain.invoke({"query": query})
        
        # Clean and parse the JSON
        subqueries_str = subqueries_str.strip()
        if subqueries_str.startswith("```json"):
            subqueries_str = subqueries_str[7:]
        if subqueries_str.endswith("```"):
            subqueries_str = subqueries_str[:-3]
        
        try:
            subqueries = json.loads(subqueries_str.strip())
            if not isinstance(subqueries, list):
                raise ValueError("Subqueries must be a list")
            # Ensure all items are strings
            if not all(isinstance(sq, str) for sq in subqueries):
                 raise ValueError("All items in subqueries list must be strings")
            return subqueries
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing subqueries JSON: {str(e)}")
            logger.debug(f"Raw subqueries string: {subqueries_str}")
            # Fallback: return the original query as a single subquery
            return [query]
        except ValueError as e:
            logger.error(f"Error validating subqueries: {str(e)}")
            logger.debug(f"Problematic subqueries value: {subqueries_str}")
            return [query]
    
    async def _execute_subqueries_concurrently(self, subqueries: List[str]) -> List[Tuple[str, Dict]]:
        """Execute subqueries concurrently using sql_tool._run and asyncio.gather."""
        results = []
        
        # Create SQL tool instance ONCE
        sql_tool = SQLQueryTool(organization_id=self.organization_id)
        
        async def run_single_query(subquery: str) -> Tuple[str, Dict]:
            """Helper coroutine to run one subquery and handle errors."""
            subquery_result_str = "{}" # Initialize for potential error logging
            try:
                # Execute asynchronously using sql_tool's _run (which is now async)
                # It returns a JSON STRING
                subquery_result_str = await sql_tool._run(
                    query_description=subquery, 
                    db_name="report_management" # Assuming default DB
                )
                # Parse the JSON string result
                subquery_result_dict = json.loads(subquery_result_str)
                
                # Check if the PARSED dict contains an error key from the SQL tool itself
                if isinstance(subquery_result_dict, dict) and "error" in subquery_result_dict:
                    error_msg = subquery_result_dict["error"]
                    logger.error(f"SQL tool returned error for subquery '{subquery}': {error_msg}")
                    # Return the error dict directly as it's already the expected structure
                    return (subquery, subquery_result_dict)
                
                # Basic validation of expected structure after successful parse & no error key
                if not isinstance(subquery_result_dict, dict) or "table" not in subquery_result_dict:
                     logger.error(f"Unexpected result format from SQL tool for subquery '{subquery}' after JSON parse. Got: {type(subquery_result_dict)} Keys: {subquery_result_dict.keys() if isinstance(subquery_result_dict, dict) else 'N/A'}")
                     return (subquery, {"error": "Invalid format from SQL tool", "table": {"columns": ["Error"], "rows": [["Invalid format received"]]}, "text": "Invalid format received"})
                
                # Successfully parsed and validated dictionary
                return (subquery, subquery_result_dict)

            except json.JSONDecodeError as json_e:
                 logger.error(f"Failed to decode JSON from SQL tool for subquery '{subquery}': {json_e}. Raw string: '{subquery_result_str}'")
                 return (subquery, {"error": "Failed to parse SQL tool output", "table": {"columns": ["Error"], "rows": [["Invalid JSON received"]]}, "text": "Invalid JSON received"})
            except Exception as e:
                logger.error(f"Exception executing subquery '{subquery}' via _run for org {self.organization_id}: {str(e)}", exc_info=True)
                # Ensure the fallback structure matches what _synthesize_results expects
                return (subquery, {"error": f"Execution Error: {str(e)}", "table": {"columns": ["Error"], "rows": [[str(e)]]}, "text": f"Execution Error: {str(e)}"})

        # Create tasks for all subqueries
        tasks = [run_single_query(sq) for sq in subqueries]
        
        # Run tasks concurrently and gather results
        results = await asyncio.gather(*tasks)
        
        return results # List of (subquery_str, result_dict)
    
    def _synthesize_results(self, query: str, subquery_results: List[Tuple[str, Dict]]) -> str:
        """Synthesize subquery results into a coherent summary."""
        llm = AzureChatOpenAI(
            openai_api_key=settings.AZURE_OPENAI_API_KEY,
            azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
            openai_api_version=settings.AZURE_OPENAI_API_VERSION,
            deployment_name=settings.AZURE_OPENAI_DEPLOYMENT_NAME,
            model_name=settings.LLM_MODEL_NAME,
            temperature=0.3,
        )
        
        results_str = ""
        successful_data = {} # Store successful data for potential calculations
        for subquery, result_dict in subquery_results:
            results_str += f"Subquery: {subquery}\n"
            if isinstance(result_dict, dict) and "error" in result_dict:
                 results_str += f"Result: Error - {result_dict['error']}\n\n"
                 continue 
            table_data = result_dict.get("table", {})
            if not isinstance(table_data, dict):
                 results_str += f"Result: Invalid table data format received.\n\n"
                 continue
            # Store successful data keyed by subquery for potential calculations
            successful_data[subquery] = table_data 
            limited_rows = table_data.get("rows", [])[:5]
            columns = table_data.get("columns", [])
            results_str += f"Results (showing up to 5 rows): {json.dumps({"columns": columns, "rows": limited_rows}, indent=2)}\n"
            results_str += f"Total rows in original result: {len(table_data.get("rows", []))}\n\n"
        
        template = """
        You are a data analyst. Given the original user query and results from potentially multiple subqueries,
        synthesize a coherent, insightful summary that addresses the original question.
        The subquery results may contain data tables or error messages.

        Original Query: {query}

        Subquery Results:
        {results}

        Provide a comprehensive summary that:
        1. Directly answers the original query, integrating information from all **successful** subqueries.
        2. If any subqueries resulted in errors, acknowledge this limitation.
        3. Highlights key insights, patterns, and trends apparent from the combined successful results. Look for comparisons between entities if multiple were queried, or comparisons against organizational averages if provided (e.g., columns ending with 'Org Average...').
        4. Mentions any notable outliers or anomalies within the successful results (e.g., zero values where activity is expected, significant deviations from averages).
        5. Uses specific numbers from the successful results.
        6. **Derived Metrics (Calculate if possible):** If the results provide both successful and unsuccessful counts for an activity (e.g., 'Successful Borrows' and 'Unsuccessful Borrows'), calculate and mention the success rate (Successful / (Successful + Unsuccessful) * 100%). Do this calculation **only if both counts are available and the sum is not zero**.
        7. **Formatting:** Structure your summary clearly, potentially using bullet points for different metrics or entities. Use markdown bolding (`**text**`) for key findings, comparisons, percentages, or important numbers.
        8. Is written in a professional, concise style.

        Your summary:
        """
        
        prompt = PromptTemplate(input_variables=["query", "results"], template=template)
        synthesis_chain = prompt | llm | StrOutputParser()
        
        # Note: The LLM performs the calculations based on the prompt now.
        # We could add Python logic here to pre-calculate rates.
        summary = synthesis_chain.invoke({"query": query, "results": results_str})
        
        return summary.strip()
    
    async def _run(self, query: str) -> Dict[str, str]:
        log_prefix = f"[Org: {self.organization_id}] [SummaryTool] "
        logger.info(f"{log_prefix}Executing. Query: '{query[:100]}...'")
        
        subquery_results: List[Tuple[str, Dict]] = []
        summary = "An error occurred during summary generation."
        
        try:
            subqueries = self._decompose_query(query)
            logger.info(f"{log_prefix}Decomposed into {len(subqueries)} subqueries.")
        except Exception as e:
             logger.error(f"{log_prefix}Error during query decomposition: {e}", exc_info=True)
             return {"text": f"Failed to understand request structure: {e}"} 

        try:
            subquery_results = await self._execute_subqueries_concurrently(subqueries)
        except Exception as e:
            logger.error(f"{log_prefix}Error during concurrent subquery execution: {e}", exc_info=True)
            return {"text": f"Error fetching data: {e}"}
        
        try:
            summary = self._synthesize_results(query, subquery_results)
            logger.info(f"{log_prefix}Completed successfully.")
        except Exception as e:
             logger.error(f"{log_prefix}Error during summary synthesis: {e}", exc_info=True)
             return {"text": f"Failed to synthesize summary: {e}"}
        
        return {"text": summary}
    
    # Implement ainvoke to ensure compatibility with BaseTool
    async def ainvoke(self, input_data: Dict[str, Any], **kwargs: Any) -> Any:
        """Override ainvoke to ensure it calls our async _run method."""
        if not isinstance(input_data, dict):
            raise ValueError(f"Expected dict input, got {type(input_data)}")
        
        query = input_data.get("query", "")
        if not query:
            return {"text": "Error: No query provided in input data."}
            
        return await self._run(query=query, **kwargs)