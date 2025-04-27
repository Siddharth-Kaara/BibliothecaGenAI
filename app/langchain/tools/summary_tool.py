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
from app.db.connection import get_db_engine
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
        """Execute subqueries concurrently using sql_tool._arun and asyncio.gather."""
        results = []
        
        # Create SQL tool instance ONCE
        sql_tool = SQLQueryTool(organization_id=self.organization_id)
        
        async def run_single_query(subquery: str) -> Tuple[str, Dict]:
            """Helper coroutine to run one subquery and handle errors."""
            subquery_result_str = "{}" # Initialize for potential error logging
            try:
                # Execute asynchronously using sql_tool's _arun
                # It returns a JSON STRING
                subquery_result_str = await sql_tool._arun(
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
                logger.error(f"Exception executing subquery '{subquery}' via _arun for org {self.organization_id}: {str(e)}", exc_info=True)
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
    
    async def _arun(self, query: str) -> Dict[str, str]:
        """Run the tool asynchronously with concurrent sub-query execution.
           Returns a dictionary containing ONLY the summary text.
        """
        logger.info(f"Executing ASYNC summary synthesizer tool for org {self.organization_id}")
        
        subquery_results: List[Tuple[str, Dict]] = []
        summary = "An error occurred during summary generation."
        
        # Decompose the query (sync for now, could be async)
        try:
            subqueries = self._decompose_query(query)
            logger.info(f"Decomposed query into {len(subqueries)} subqueries")
        except Exception as e:
             logger.error(f"Error during query decomposition: {e}", exc_info=True)
             return {"text": f"Failed to understand the request structure: {e}"} # Return only text error

        # Execute subqueries concurrently
        try:
            subquery_results = await self._execute_subqueries_concurrently(subqueries)
        except Exception as e:
            logger.error(f"Error during concurrent subquery execution: {e}", exc_info=True)
            # Attempt synthesis with partial/error results if possible, or return error
            # For simplicity now, return a generic text error
            return {"text": f"An error occurred while fetching data: {e}"}
        
        # Synthesize results (sync for now, could be async)
        try:
            summary = self._synthesize_results(query, subquery_results)
            logger.info("Successfully generated summary from concurrent results.")
        except Exception as e:
             logger.error(f"Error during summary synthesis: {e}", exc_info=True)
             # Return only text error
             return {"text": f"Failed to synthesize the final summary: {e}"}
        
        # Return standardized output format with text ONLY
        return {"text": summary}

    def _run(self, query: str) -> Dict[str, str]:
        """Synchronous wrapper for the asynchronous execution logic."""
        logger.warning("Running summary synthesizer synchronously. Consider using async for better performance.")
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                logger.error("Cannot run async logic synchronously from a running event loop.")
                return {"text": "Error: Synchronous execution from async context not fully supported."}
            else:
                result = loop.run_until_complete(self._arun(query))
                return result # Result is already Dict[str, str]
        except RuntimeError as e:
             logger.error(f"RuntimeError running summary synthesizer synchronously: {e}", exc_info=True)
             return {"text": f"Failed to run summary task: {e}"}
        except Exception as e:
             logger.error(f"Unexpected error in sync wrapper for summary synthesizer: {e}", exc_info=True)
             return {"text": f"An unexpected error occurred: {e}"}