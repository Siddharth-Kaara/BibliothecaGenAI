import logging
import json
import asyncio
from typing import Any, Dict, List, Optional, Tuple, TypedDict, Annotated, Sequence
import operator
from pydantic import BaseModel, Field
import functools
from pydantic_core import ValidationError 
import uuid 
import datetime 
import re 
import openai
from openai import APIConnectionError, APITimeoutError, RateLimitError 
import copy
import sqlparse 
from sqlalchemy import inspect, MetaData, text
from fastapi import HTTPException

from app.db.connection import get_async_db_engine
from app.db.schema_definitions import SCHEMA_DEFINITIONS

# LangChain & LangGraph Imports
from langchain_core.messages import BaseMessage, FunctionMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import AzureChatOpenAI
from langchain_core.output_parsers.openai_tools import PydanticToolsParser, JsonOutputToolsParser
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, END
from langgraph.errors import GraphRecursionError
from langchain.tools import BaseTool

# Local Imports
from app.core.config import settings
from app.langchain.tools.sql_tool import SQLExecutionTool
from app.langchain.tools.summary_tool import SummarySynthesizerTool
from app.langchain.tools.hierarchy_resolver_tool import HierarchyNameResolverTool
from app.schemas.chat import ChatData, ApiChartSpecification, TableData
from app.langchain.charting import ChartSpecFinalInstruction, process_and_validate_chart_specs
from app.prompts import AGENT_SYSTEM_PROMPT # Import from prompts

logger = logging.getLogger(__name__)
usage_logger = logging.getLogger("usage") 


# --- Helper Function to Get Schema String --- #
def _format_column_description(col: Dict[str, Any]) -> str:
    # ... (rest of _format_column_description - unchanged)
    pass

@functools.lru_cache(maxsize=4) # Add LRU cache decorator
def _get_schema_string(db_name: str = "report_management") -> str:
    """Generates a formatted string representation of the schema for a given database.
       Uses LRU cache to avoid repeated computation/fetching.
       Gets schema information from predefined SCHEMA_DEFINITIONS.
    """
    # Check cache first implicitly by decorator
    logger.debug(f"[_get_schema_string] Generating/Fetching schema for database: {db_name}") # Log reflects potential fetch

    if db_name not in SCHEMA_DEFINITIONS:
        error_msg = f"No schema definition found for database {db_name}."
        logger.warning(f"[_get_schema_string] {error_msg}")
        return error_msg

    db_info = SCHEMA_DEFINITIONS[db_name]
    schema_info = [
        f"Database: {db_name}",
        f"Description: {db_info['description']}",
        ""
    ]

    for table_name, table_info in db_info['tables'].items():
        # Use the 'name' field if present, otherwise use the key
        physical_table_name = table_info.get('name', table_name)
        schema_info.append(f"Table: {physical_table_name} (Logical Name: {table_name})")
        schema_info.append(f"Description: {table_info['description']}")

        schema_info.append("Columns:")
        for column in table_info['columns']:
            primary_key = " (PRIMARY KEY)" if column.get('primary_key') else ""
            foreign_key = f" (FOREIGN KEY -> {column.get('foreign_key')})" if column.get('foreign_key') else ""
            timestamp_note = " (Timestamp for filtering)" if 'timestamp' in column['type'].lower() else ""
            schema_info.append(f"  {column['name']} ({column['type']}){primary_key}{foreign_key} - {column['description']}{timestamp_note}")

        if 'example_queries' in table_info and table_info['example_queries']:
            schema_info.append("Example queries:")
            for query in table_info['example_queries']:
                schema_info.append(f"  {query}")

        schema_info.append("")  # Empty line between tables

    logger.debug(f"[_get_schema_string] Successfully generated/retrieved schema for {db_name}")
    return "\n".join(schema_info)


# --- Define Structure for LLM Final Response (Used as a Tool/Schema) ---
class FinalApiResponseStructure(BaseModel):
    """Structure for the final API response. Call this function when you have gathered all necessary information and are ready to formulate the final response to the user.
       Use the 'include_tables' field to specify which data tables should be part of the structured response.
       The 'text' field is for natural language ONLY and MUST NOT contain any markdown tables or other structured data formats.
       This structure also includes the chart specifications directly.
    """
    text: str = Field(description="The final natural language text response for the user. This field is for explanatory and summary text ONLY. It MUST NOT contain any markdown tables or other structured data formats. Use the 'include_tables' field to specify table data. Follow the guidelines in the system prompt for generating this text (e.g., brief introductions if data/charts are present).")
    include_tables: List[bool] = Field(
        description="List of booleans indicating which tables from the agent's structured_results (in their original order) should be included in the final API response's 'tables' field. If you discuss data from a specific query result in your 'text' response and that full table would be useful for the user, set the corresponding boolean in this list to True. This is the ONLY way to include structured table data in the response.",
        default_factory=list
    )

    # chart_specs list containing full specifications
    chart_specs: List[ChartSpecFinalInstruction] = Field( # Uses imported ChartSpecFinalInstruction
        default_factory=list,
        description="List of chart specifications to be included in the final API response. The LLM generates these directly when calling this tool."
    )


# --- Define the Agent State ---
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add] # Conversation history
    # Add a field to hold structured results: List of {table, filters}
    structured_results: Annotated[List[Dict[str, Any]], operator.add]
    # Add a field to hold the final structured response once generated
    final_response_structure: Optional[FinalApiResponseStructure]
    # Add request_id for logging context
    request_id: Optional[str] = None
    prompt_tokens: Annotated[int, operator.add]
    completion_tokens: Annotated[int, operator.add]
    # Add field to track tool failures for intelligent adaptation
    failure_patterns: Dict[str, List[Dict[str, Any]]] = Field(default_factory=dict)
    # Add field for recovery instructions when needed
    recovery_guidance: Optional[str] = None
    # Add field to store the latest successful name resolution results
    resolved_location_map: Optional[Dict[str, str]] = None
    # Add field for context about missing entities (generated by analyze_results_node)
    missing_entities_context: Optional[str] = None
    # Add counter for specific SQL security retries
    sql_security_retry_count: Annotated[int, operator.add] = 0


# --- NAnalysis Node to Check for Missing Entities ---
def analyze_results_node(state: AgentState) -> Dict[str, Any]:
    """Analyzes structured results (table + filters) against resolved entities (using IDs). 
       Determines missing entities by checking if queries filtering by specific IDs returned rows *containing* those IDs.
       Generates context for the LLM about missing data.
    """
    request_id = state.get("request_id")
    logger.debug(f"[AnalyzeResultsNode-Final] Entering node...")
    
    resolved_map = state.get("resolved_location_map") # {name_lower: id}
    structured_results = state.get("structured_results", []) # List of {"table": ..., "filters": ...}
    missing_entities_context = None

    if not resolved_map or not structured_results:
        logger.debug("[AnalyzeResultsNode-Final] Skipping analysis: Missing resolved map or structured results.")
        return {"missing_entities_context": None}

    resolved_id_to_name_map = {v: k for k, v in resolved_map.items()} # Reverse map {id: name_lower}
    all_expected_ids = set() # All unique resolved IDs used in ANY successful filter
    found_ids = set() # Resolved IDs for which corresponding data rows were actually found

    # --- Helper to find ID column index --- # 
    def find_id_column_index(columns: List[str]) -> Optional[int]:
        # Prioritize the explicitly added verification column
        verification_alias = "__verification_id__"
        if verification_alias in columns:
            try:
                return columns.index(verification_alias)
            except ValueError:
                pass # Should not happen if check passes

        # Fallback to standard ID column names - ADDED "Hierarchy ID" and ensure case-insensitive check
        id_col_candidates = ["id", "hierarchyId", "hierarchy_id", "location_id", "branch_id", "Hierarchy ID"]
        # Prepare a list of lowercase candidates for efficient checking
        lower_id_col_candidates = [candidate.lower() for candidate in id_col_candidates]

        for idx, col_name in enumerate(columns):
            if col_name.lower() in lower_id_col_candidates:
                logger.debug(f"[AnalyzeResultsNode-Final] Found ID column by candidate name: '{col_name}' (matched via {col_name.lower()})")
                return idx

        # If no standard ID column, check for entity name columns that might contain branch names
        name_col_candidates = ["Branch Name", "name", "branch", "location", "hierarchy_name"]
        for idx, col_name in enumerate(columns):
            if any(candidate.lower() in col_name.lower() for candidate in name_col_candidates):
                logger.debug(f"[AnalyzeResultsNode-Final] Found name column that might contain entity names: '{col_name}'")
                return idx
        
        logger.warning(f"[AnalyzeResultsNode-Final] Could not find verification alias or standard ID column in {columns}")
        return None

    # --- Analyze each structured result --- #
    for result in structured_results:
        filters = result.get("filters", {})
        table = result.get("table", {})
        columns = table.get("columns", [])
        rows = table.get("rows", [])
        
        # 1. Find all resolved IDs used in *this* result's filters
        ids_filtered_in_this_query = set()
        names_filtered_in_this_query = set()
        
        for key, value in filters.items():
            # Check for direct ID matches in filter values
            if isinstance(value, str) and value in resolved_id_to_name_map:
                ids_filtered_in_this_query.add(value)
                names_filtered_in_this_query.add(resolved_id_to_name_map[value])
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, str) and item in resolved_id_to_name_map:
                        ids_filtered_in_this_query.add(item)
                        names_filtered_in_this_query.add(resolved_id_to_name_map[item])
            elif 'id' in key.lower() or key.startswith('res_id_'): 
                 potential_ids = [value] if isinstance(value, str) else value if isinstance(value, list) else []
                 for item in potential_ids:
                      if isinstance(item, str) and item in resolved_id_to_name_map:
                           ids_filtered_in_this_query.add(item)
                           names_filtered_in_this_query.add(resolved_id_to_name_map[item])
            
            # Check for branch name matches in filter values
            elif isinstance(value, str):
                # Check if this value matches any resolved name (case insensitive)
                for res_name, res_id in resolved_map.items():
                    if res_name.lower() in value.lower() or value.lower() in res_name.lower():
                        ids_filtered_in_this_query.add(res_id)
                        names_filtered_in_this_query.add(res_name)
            elif key.lower() in ["branch", "branch_name", "branch1", "branch2", "branch3"]:
                if isinstance(value, str):
                    # Check for exact or partial match against resolved names
                    for res_name, res_id in resolved_map.items():
                        if res_name.lower() in value.lower() or value.lower() in res_name.lower():
                            ids_filtered_in_this_query.add(res_id)
                            names_filtered_in_this_query.add(res_name)
                           
        if not ids_filtered_in_this_query and not names_filtered_in_this_query:
            # This query didn't filter by any specific resolved entities, skip further checks for it
            logger.debug(f"[AnalyzeResultsNode-Final] Query with filters {filters} did not filter by resolved IDs or names. Skipping row check.")
            continue

        # Add these filtered IDs to the set of all expected IDs across the whole request
        all_expected_ids.update(ids_filtered_in_this_query)
        logger.debug(f"[AnalyzeResultsNode-Final] Query filtered by IDs: {ids_filtered_in_this_query} and names: {names_filtered_in_this_query}")

        # 2. Check if rows were returned for this query AND if they contain actual data
        if not rows:
            logger.debug(f"[AnalyzeResultsNode-Final] Query filtering by {ids_filtered_in_this_query} returned 0 rows. None marked as found.")
            continue # No rows, so none of these IDs were found in this result
        
        # NEW: Check if rows contain any actual data beyond just IDs or all nulls
        has_meaningful_data = False
        # Attempt to identify non-ID columns to check for meaningful data
        id_column_names = {col_name for col_name in ["id", "hierarchyId", "hierarchy_id", "location_id", "branch_id", "organizationId", "eventSrc"] if col_name in columns}
        value_column_indices = [i for i, col_name in enumerate(columns) if col_name not in id_column_names]

        if not value_column_indices: # If all columns are considered ID-like, check all non-None
            logger.debug(f"[AnalyzeResultsNode-Final] No distinct value columns identified for {ids_filtered_in_this_query}. Checking all cells for non-null data.")
            for row_idx, row_content in enumerate(rows):
                if not isinstance(row_content, list):
                    logger.warning(f"[AnalyzeResultsNode-Final] Row {row_idx} is not a list: {row_content}. Skipping for meaningful data check.")
                    continue
                if any(cell is not None for cell in row_content):
                    has_meaningful_data = True
                    break
        else:
            logger.debug(f"[AnalyzeResultsNode-Final] Checking value columns (indices: {value_column_indices}) for {ids_filtered_in_this_query} for non-null data.")
            for row_idx, row_content in enumerate(rows):
                if not isinstance(row_content, list):
                    logger.warning(f"[AnalyzeResultsNode-Final] Row {row_idx} is not a list: {row_content}. Skipping for meaningful data check.")
                    continue
                # Check if any of the identified value columns in this row has a non-None value
                if any(row_content[col_idx] is not None for col_idx in value_column_indices if len(row_content) > col_idx):
                    has_meaningful_data = True
                    break
        
        if not has_meaningful_data:
            logger.info(f"[AnalyzeResultsNode-Final] Query filtering by {ids_filtered_in_this_query} returned rows, but all rows appear to contain only NULL values in metric/value columns or no meaningful data. Marking corresponding IDs as 'data not found'.")
            # Do not add to found_ids if no meaningful data, even if rows were returned.
            # The ids_filtered_in_this_query will then correctly be part of 'missing_ids' later.
            continue

        # 3. Determine which of the filtered IDs were actually present in the returned rows (original logic follows)
        #    This part now assumes that if we reached here, 'has_meaningful_data' is True.
        if len(ids_filtered_in_this_query) <= 1 and len(names_filtered_in_this_query) <= 1:
            # If we filtered by only one ID/name and got meaningful data, that entity's data was found
            # This handles cases where barcode transactional data doesn't include the branch ID in the results
            found_ids.update(ids_filtered_in_this_query)
            logger.debug(f"[AnalyzeResultsNode-Final] Query filtered by single ID {ids_filtered_in_this_query} or name {names_filtered_in_this_query} returned rows with meaningful data. Marking as found.")
        else:
            # Multiple IDs/names were filtered. Try to identify entities in the result rows.
            id_col_index = find_id_column_index(columns)
            
            if id_col_index is not None:
                logger.debug(f"[AnalyzeResultsNode-Final] Checking column (Index: {id_col_index}, Name: {columns[id_col_index]}) for entity identification in rows with meaningful data...")
                
                # Check for both direct ID matches and name matches
                for row in rows:
                    if isinstance(row, list) and len(row) > id_col_index:
                        cell_value = row[id_col_index]
                        cell_value_str = str(cell_value) if isinstance(cell_value, uuid.UUID) else cell_value
                        
                        # Direct ID match
                        if isinstance(cell_value_str, str) and cell_value_str in ids_filtered_in_this_query:
                            found_ids.add(cell_value_str)
                        
                        # Name match - compare with branch names
                        elif isinstance(cell_value_str, str):
                            for res_id, res_name in resolved_id_to_name_map.items():
                                # Get the full resolved name from the map using lowercase key
                                full_name = None
                                for k, v in resolved_map.items():
                                    if k == res_name:
                                        if cell_value_str and res_id in ids_filtered_in_this_query and (
                                           cell_value_str.lower() in k.lower() or 
                                           k.lower() in cell_value_str.lower()):
                                            found_ids.add(res_id)
                                            logger.debug(f"[AnalyzeResultsNode-Final] Found entity match: '{cell_value_str}' matches '{k}' (ID: {res_id})")
                
                logger.debug(f"[AnalyzeResultsNode-Final] Entities found in rows for this query: {found_ids}")
            else:
                # If we can't identify specific entities but have meaningful data, assume all filtered entities were found
                # This is more permissive than the previous approach
                found_ids.update(ids_filtered_in_this_query)
                logger.debug(f"[AnalyzeResultsNode-Final] Query filtered by multiple IDs {ids_filtered_in_this_query} returned meaningful data but no ID/name column found. Conservatively marking all as found.")

    # --- Generate final context --- # 
    if not all_expected_ids:
        logger.debug("[AnalyzeResultsNode-Final] No resolved entity IDs were used in any filters.")
        return {"missing_entities_context": None}

    logger.debug(f"[AnalyzeResultsNode-Final] Total Expected IDs across all filters: {all_expected_ids}")
    logger.debug(f"[AnalyzeResultsNode-Final] IDs confirmed present in returned data rows: {found_ids}")

    # Determine which expected IDs are missing (filtered but no corresponding rows found)
    missing_ids = all_expected_ids - found_ids
    
    if missing_ids:
        # Map missing IDs back to their original names (lowercase from resolved_map keys)
        missing_names = [resolved_map.get(resolved_id_to_name_map.get(mid), f"Unknown Entity [ID: {mid}]") 
                         for mid in missing_ids]
        missing_names_str = ", ".join(sorted(list(set(missing_names))))
        
        missing_entities_context = (
            f"IMPORTANT CONTEXT: The user asked for data related to specific entities (including {missing_names_str}). "
            f"While queries were executed filtering for these, no data rows were returned specifically corresponding to {missing_names_str}. "
            f"You MUST explicitly mention this lack of data for {missing_names_str} in your final response."
        )
        logger.info(f"[AnalyzeResultsNode-Final] Generated missing entities context: {missing_entities_context}")
    else:
        logger.debug("[AnalyzeResultsNode-Final] No missing entity IDs detected.")

    return {"missing_entities_context": missing_entities_context}
# --- END Analysis Node ---


# --- LLM and Tools Initialization ---
def get_llm():
    """Get the Azure OpenAI LLM configured with internal retries."""
    logger.info(f"Initializing Azure OpenAI LLM with deployment {settings.AZURE_OPENAI_DEPLOYMENT_NAME}")
    # Ensure model supports tool calling / structured output
    return AzureChatOpenAI(
        openai_api_key=settings.AZURE_OPENAI_API_KEY,
        azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
        openai_api_version=settings.AZURE_OPENAI_API_VERSION,
        deployment_name=settings.AZURE_OPENAI_DEPLOYMENT_NAME, 
        model_name=settings.LLM_MODEL_NAME, 
        temperature=0.15,
        verbose=settings.VERBOSE_LLM,
        max_retries=settings.LLM_MAX_RETRIES, # Pass retry setting
        # streaming=False # Ensure streaming is False if not handled downstream
    )

def get_tools(organization_id: str) -> List[Any]:
    """Instantiate and return operational tools, injecting dependencies."""
    # Instantiate base tools first
    hierarchy_resolver = HierarchyNameResolverTool(organization_id=organization_id)
    sql_tool = SQLExecutionTool(organization_id=organization_id)
    
    # Instantiate summary tool, injecting the base tools
    summary_tool = SummarySynthesizerTool(
        organization_id=organization_id,
        sql_tool=sql_tool, 
        hierarchy_resolver=hierarchy_resolver
    )
    
    return [
        hierarchy_resolver, # Return all tools for the agent to potentially use
        sql_tool,
        summary_tool,
    ]

# Function to bind tools AND the final response structure to the LLM
def create_llm_with_tools_and_final_response_structure(organization_id: str):
    llm = get_llm()
    # Get operational tools first
    operational_tools = get_tools(organization_id)
    # Define all structures the LLM can output as "tools"
    all_bindable_items = operational_tools + [FinalApiResponseStructure]

    # --- Fetch DB Schema String using the helper function --- #
    try:
        db_schema_string = _get_schema_string() # Call the helper function
        logger.debug("Successfully fetched DB schema string to inject into prompt.")
    except Exception as e:
        logger.error(f"Failed to fetch DB schema for prompt injection: {e}", exc_info=True)
        db_schema_string = "Error: Could not retrieve database schema." # Fallback
    # --- End Fetch DB Schema --- #

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", AGENT_SYSTEM_PROMPT), # Use imported constant
            MessagesPlaceholder(variable_name="messages"),
            # --- ADDED Placeholder for Missing Entities Context --- #
            ("system", "{missing_entities_context}"),
        ]
    )

    # --- Calculate Time Context --- #
    now = datetime.datetime.now()
    current_date_str = now.strftime("%Y-%m-%d")
    current_day_name = now.strftime("%A") # e.g., Monday
    current_year_int = now.year
    # --- End Time Context Calculation --- #

    # Format prompt with tool details AND schema
    # Generate descriptions and names ONLY from bindable items (which exclude DatabaseSchemaTool)
    tool_descriptions_list = [
        f"{getattr(item, 'name', getattr(item, '__name__', 'Unknown Tool'))}: {getattr(item, 'description', getattr(item, '__doc__', 'No description'))}"
        for item in all_bindable_items
    ]
    tool_names_list = [
        f"{getattr(item, 'name', getattr(item, '__name__', 'Unknown Tool'))}"
        for item in all_bindable_items
    ]
    prompt = prompt.partial(
        tool_descriptions="\n".join(tool_descriptions_list),
        tool_names=", ".join(tool_names_list),
        db_schema_string=db_schema_string, # Inject the schema string
        # --- Inject Time Context --- #
        current_date=current_date_str,
        current_day=current_day_name,
        current_year=current_year_int,
        # --- ADDED: Ensure context is available even if None initially ---
        missing_entities_context="", 
    )

    # Bind tools for function calling
    llm_with_tools = llm.bind_tools(
        tools=all_bindable_items, # Pass operational tools + FinalApiResponseStructure
        tool_choice=None # Let the LLM decide which tool/structure to call
    )
    
    logger.debug(f"Created LLM runnable bound with tools/structure for org: {organization_id}")
    # Return the combined prompt and LLM runnable
    return prompt | llm_with_tools 

# --- Agent Node: Decides action - call a tool or invoke FinalApiResponseStructure ---
def _get_sql_call_signature(args: Dict[str, Any]) -> tuple:
    """
    Generates a signature for an execute_sql call based on the
    normalized SQL text and its parameters.
    """
    sql = args.get("sql", "")
    params = args.get("params", {})

    # Normalize SQL: lowercase and consolidate whitespace
    # Keep structural elements like LIMIT, OFFSET, ORDER BY
    normalized_sql = ' '.join(sql.lower().split())

    # Create a canonical representation of parameters (sorted tuple of items)
    # Using tuple makes it hashable directly
    params_signature_items = tuple(sorted(params.items()))

    # Return a tuple including the tool name for absolute clarity
    return ("execute_sql", normalized_sql, params_signature_items)

def agent_node(state: AgentState, llm_with_structured_output):
    """Invokes the LLM ONCE to decide the next action or final response structure.
       If the LLM returns a plain AIMessage, it coerces it into FinalApiResponseStructure.
    """
    request_id = state.get("request_id")
    logger.debug(f"[AgentNode] Entering agent node (single invocation logic)...")

    llm_response: Optional[AIMessage] = None
    final_structure: Optional[FinalApiResponseStructure] = None
    operational_calls = []
    # Initialize return dict, preserving retry count unless incremented
    return_dict: Dict[str, Any] = {
        "messages": [],
        "final_response_structure": None,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "resolved_location_map": state.get("resolved_location_map"),
        "sql_security_retry_count": state.get("sql_security_retry_count", 0) # Preserve count
    }

    # Parser for the final response structure
    final_response_parser = PydanticToolsParser(tools=[FinalApiResponseStructure])

    # --- START: Re-integrated Adaptive Error Analysis for SQL Security Error ---
    failure_patterns = state.get("failure_patterns", {})
    recovery_guidance = None
    retry_increment = 0 # Track if we need to increment count

    # Check for SQL security failures
    last_sql_security_failure = None
    current_failure_patterns = state.get("failure_patterns", {})
    if current_failure_patterns:
        logger.debug(f"[AgentNode] Inspecting failure_patterns for SQL security errors: {current_failure_patterns}")
    else:
        logger.debug("[AgentNode] failure_patterns is empty. No SQL security errors to check from patterns.")

    for key, failures_list in current_failure_patterns.items():
        if isinstance(key, tuple) and len(key) > 0 and key[0] == "execute_sql":
            if failures_list:
                potential_failure = failures_list[-1]
                error_type = potential_failure.get("error_type")
                details = potential_failure.get("details", "")
                
                if error_type == "VALIDATION_ERROR" and \
                   "SECURITY CHECK FAILED: Main SQL query MUST include the :organization_id parameter." in details:
                    logger.info(f"[AgentNode] Matched specific SQL security failure (:organization_id missing). Tool Call ID: {potential_failure.get('tool_call_id')}. Details: {details[:200]}...")
                    last_sql_security_failure = potential_failure
                    break
    
    if last_sql_security_failure:
        current_retry_count = state.get("sql_security_retry_count", 0)
        if current_retry_count < settings.MAX_SQL_SECURITY_RETRIES:
            recovery_guidance = """
            CRITICAL SQL CORRECTION INSTRUCTION:
            
            Your previous SQL query failed the security check because it was missing the required :organization_id filter.
            
            The security system checks EACH SQL component SEPARATELY:
            1. Every CTE (WITH clause) must include: WHERE "organizationId" = :organization_id (or relevant org column)
            2. Every subquery must include: WHERE "organizationId" = :organization_id (or relevant org column)
            3. The main query must include: WHERE "organizationId" = :organization_id (or relevant org column)
            
            You MUST add the organization filter correctly to the query for the requested table. Retry the query generation.
            """
            logger.info(f"[AgentNode] SQL Security Error: Generating recovery guidance for LLM: {recovery_guidance[:150]}...")
            retry_increment = 1
            logger.info(f"[AgentNode] SQL Security Error: Incrementing SQL security retry count from {current_retry_count} to {current_retry_count + 1}.")
        else:
            logger.warning(f"[AgentNode] SQL Security Error: Detected, but retry limit ({settings.MAX_SQL_SECURITY_RETRIES}) reached. Will not generate recovery guidance.")

    # Create a copy for preprocessing
    preprocessed_state = _preprocess_state_for_llm(state)

    # If recovery guidance was generated, inject it into the system message
    if recovery_guidance:
        messages = list(preprocessed_state.get("messages", [])) # Ensure mutable list
        system_message_found = False
        for i, msg in enumerate(messages):
            if isinstance(msg, SystemMessage):
                new_content = f"{msg.content}\\n\\n{recovery_guidance}"
                messages[i] = SystemMessage(content=new_content)
                system_message_found = True
                break
        if not system_message_found and messages:
            messages.insert(0, SystemMessage(content=recovery_guidance))
        
        preprocessed_state["messages"] = messages
        return_dict["recovery_guidance"] = recovery_guidance
        # IMPORTANT: Increment the retry count in the state update for this turn
        return_dict["sql_security_retry_count"] = state.get("sql_security_retry_count", 0) + retry_increment
    # --- END: Re-integrated Adaptive Error Analysis ---

    try:
        # --- Inject the missing_entities_context at invocation time --- #
        missing_context = state.get("missing_entities_context") or ""
        invocation_input = {**preprocessed_state, "missing_entities_context": missing_context}
        # logger.debug(f"[AgentNode] Invoking LLM with state (incl. missing context): {invocation_input}") # Keep log concise
        llm_response = llm_with_structured_output.invoke(invocation_input)

        # --- ADDED: Debugging logs for raw LLM output --- #
        logger.debug(f"[AgentNode] Raw LLM response object type: {type(llm_response)}")
        if isinstance(llm_response, AIMessage):
             logger.debug(f"[AgentNode] Raw LLM response content: {llm_response.content!r}")
             logger.debug(f"[AgentNode] Raw LLM response tool_calls: {llm_response.tool_calls}")
             logger.debug(f"[AgentNode] Raw LLM usage_metadata: {getattr(llm_response, 'usage_metadata', None)}")
        # --- END Debugging logs --- #

        # --- Extract Token Usage --- #
        prompt_tokens_turn = 0 # Initialize outside the if for safety in except blocks?
        completion_tokens_turn = 0 # Initialize outside the if for safety in except blocks?
        if llm_response and hasattr(llm_response, 'usage_metadata') and llm_response.usage_metadata:
            metadata = llm_response.usage_metadata
            # --- Map the correct keys: input_tokens → prompt_tokens, output_tokens → completion_tokens --- #
            return_dict["prompt_tokens"] = metadata.get('input_tokens', 0)
            return_dict["completion_tokens"] = metadata.get('output_tokens', 0)
            # --- Log DIRECTLY from metadata OR the dict --- #
            logger.debug(
                f"[AgentNode] Tokens used this turn: "
                f"Prompt={return_dict['prompt_tokens']}, "
                f"Completion={return_dict['completion_tokens']}"
            )
        else:
            # Ensure return_dict has 0s if metadata is missing
            return_dict["prompt_tokens"] = 0
            return_dict["completion_tokens"] = 0
        # --- End Token Usage --- #

        # --- Process LLM Response --- #
        if isinstance(llm_response, AIMessage):
            return_dict["messages"] = [llm_response] # Add message to history regardless
            final_api_call = None

            # --- REVISED LOGIC: Prioritize processing if tool_calls exist --- #
            if llm_response.tool_calls:
                # Separate final structure call from operational calls
                for tc in llm_response.tool_calls:
                    if tc.get("name") == FinalApiResponseStructure.__name__:
                        if final_api_call is None: final_api_call = tc
                        else: logger.warning(f"[AgentNode] Multiple FinalApiResponseStructure calls found, using first.")
                    else:
                        operational_calls.append(tc)

                # CASE 1: FinalApiResponseStructure tool called by LLM
                if final_api_call:
                    logger.debug("[AgentNode] LLM called FinalApiResponseStructure tool.")
                    try:
                        # --- PRE-PARSING VALIDATION --- #
                        args = final_api_call.get("args", {})
                        valid_args = {
                            "text": args.get("text", ""), # Default to empty string if missing
                            "include_tables": args.get("include_tables"), # Keep None if missing initially
                            "chart_specs": args.get("chart_specs", []) # Default to empty list
                        }
                        # Log and discard invalid keys
                        invalid_keys = set(args.keys()) - set(valid_args.keys())
                        if invalid_keys:
                            logger.warning(f"[AgentNode] Stripping invalid keys from FinalApiResponseStructure args: {invalid_keys}")
                        
                        # Ensure include_tables is a list of booleans of the correct length
                        num_results = len(state.get('structured_results', []))
                        llm_include_tables = valid_args["include_tables"]
                        
                        # --- START VALIDATION ---
                        if not isinstance(llm_include_tables, list):
                            logger.warning(f"[AgentNode] 'include_tables' was not a list ({type(llm_include_tables).__name__}). Defaulting to all False for {num_results} result(s).")
                            valid_args["include_tables"] = [False] * num_results
                        elif len(llm_include_tables) != num_results:
                            logger.warning(f"[AgentNode] 'include_tables' length mismatch ({len(llm_include_tables)}) vs results count ({num_results}). Defaulting to all False.")
                            valid_args["include_tables"] = [False] * num_results
                        else:
                            # Ensure all elements are boolean, default to False if not convertible
                            validated_flags = []
                            for i, flag in enumerate(llm_include_tables):
                                try:
                                    validated_flags.append(bool(flag))
                                except Exception:
                                    logger.warning(f"[AgentNode] Could not convert include_tables element at index {i} ('{flag}') to bool. Defaulting to False.")
                                    validated_flags.append(False)
                            valid_args["include_tables"] = validated_flags
                        # --- END REFINED VALIDATION ---

                        # Ensure chart_specs is a list (already defaulted)
                        if not isinstance(valid_args["chart_specs"], list):
                            logger.warning(f"[AgentNode] 'chart_specs' was not a list ({type(valid_args['chart_specs']).__name__}). Defaulting to empty list.")
                            valid_args["chart_specs"] = []
                            
                        # Reconstruct the tool call with only valid args for parsing
                        validated_final_api_call = {
                            "name": FinalApiResponseStructure.__name__,
                            "args": valid_args,
                            "id": final_api_call.get("id", str(uuid.uuid4())) # Preserve or generate ID
                        }
                        # --- END PRE-PARSING VALIDATION ---

                        # Parse using the validated call
                        parsed_final = final_response_parser.invoke(AIMessage(content="", tool_calls=[validated_final_api_call]))
                        if parsed_final:
                            final_structure = parsed_final[0]
                            logger.debug(f"[AgentNode] Successfully parsed FinalApiResponseStructure after validation.")
                            # Clear tool_calls from the AIMessage when final structure is set ---
                            if llm_response: # Ensure llm_response exists
                                llm_response.tool_calls = [] # Clear the tool calls on the original message
                        else:
                            logger.warning("[AgentNode] FinalAPIStructure parser returned empty list after validation. Creating fallback.")
                            final_structure = FinalApiResponseStructure(text="Error parsing final response structure.", include_tables=[], chart_specs=[]) # Fallback
                    except ValidationError as e:
                        logger.warning(f"[AgentNode] FinalAPIStructure validation failed from tool call: {e}. Creating fallback.")
                        final_structure = FinalApiResponseStructure(text="Error validating final response structure.", include_tables=[], chart_specs=[]) # Fallback
                    except Exception as e:
                        logger.warning(f"[AgentNode] Error parsing FinalAPIStructure from tool call: {e}. Creating fallback.")
                        final_structure = FinalApiResponseStructure(text="Error processing final response structure.", include_tables=[], chart_specs=[]) # Fallback

                # CASE 2: Operational tool(s) called by LLM (NO FinalApiResponseStructure)
                elif operational_calls:
                    logger.debug(f"[AgentNode] LLM called {len(operational_calls)} operational tool(s). Applying deduplication.")
                    unique_operational_calls = []
                    discarded_duplicates = []
                    seen_signatures = set()
                    for tc in operational_calls:
                        tool_name = tc.get("name")
                        tool_args = tc.get("args", {})
                        signature = None
                        if tool_name == "execute_sql":
                            signature = _get_sql_call_signature(tool_args)
                        else:
                            signature = (tool_name, json.dumps(tool_args, sort_keys=True))
                            
                        if signature not in seen_signatures:
                            unique_operational_calls.append(tc)
                            seen_signatures.add(signature)
                        else:
                            discarded_duplicates.append(tc)
                    
                    for discarded_tc in discarded_duplicates:
                        # logger.warning(f"[AgentNode] Discarded functionally duplicate operational tool call (ID: {discarded_tc.get('id')})")
                        logger.info(f"[AgentNode] Discarding duplicate operational tool call: Name: {discarded_tc.get('name')}, Args: {str(discarded_tc.get('args', {}))[:100]}..., ID: {discarded_tc.get('id')}")

                    if unique_operational_calls:
                        # Modify the AIMessage in the return dict to only contain unique calls
                        llm_response.tool_calls = unique_operational_calls
                        return_dict["messages"] = [llm_response]
                        logger.info(f"[AgentNode] Proceeding with {len(unique_operational_calls)} unique operational tool call(s).")
                        # No final_structure set here, graph proceeds to 'tools'
                    else:
                        # If all operational calls were duplicates
                        logger.warning("[AgentNode] No unique operational tool calls remaining after deduplication. Creating fallback final structure.")
                        final_structure = FinalApiResponseStructure(
                            text="No valid actions could be performed after processing tool calls. Please try again.",
                            include_tables=[], chart_specs=[]
                        )
                
                # CASE 3: LLM response had tool_calls list, but it was empty or contained unknown calls
                else: 
                    logger.warning("[AgentNode] LLM response had tool_calls list, but contained no recognized Final or operational calls. Creating fallback final structure.")
                    final_structure = FinalApiResponseStructure(
                        text="Received an unexpected tool call format from the language model.",
                        include_tables=[], chart_specs=[]
                    )

            # --- END Handling responses WITH tool_calls --- #

            # --- NEW CASE 4: Plain AIMessage response (NO tool_calls list) --- #
            else: 
                logger.info(f"[AgentNode] LLM AIMessage has NO tool_calls. Coercing content into FinalApiResponseStructure. Content: {llm_response.content[:100]}...")
                if isinstance(llm_response.content, str) and llm_response.content.strip():
                    # --- RE-ADD Cleanup Logic HERE --- #
                    raw_content = llm_response.content.strip()
                    coerced_text = raw_content # Default to raw content
                    try:
                        split_index = raw_content.rfind("\nAIMessage(") 
                        if split_index != -1:
                            if len(raw_content) - split_index < 500: # Heuristic length check
                                potential_text = raw_content[:split_index].strip()
                                if potential_text: 
                                    coerced_text = potential_text
                                    # logger.debug(f"[AgentNode] Successfully cleaned AIMessage representation from coerced text.")
                                    logger.info(f"[AgentNode] Cleaned AIMessage(...) wrapper from coerced text. Original snippet: {raw_content[:100]}..., New snippet: {coerced_text[:100]}...")
                                else:
                                     logger.warning(f"[AgentNode] Coerced content cleanup resulted in empty string. Using original content. Raw: {raw_content}")
                    except Exception as cleanup_err: 
                        logger.warning(f"[AgentNode] Error during AIMessage cleanup: {cleanup_err}. Using raw content.")
                    # --- END RE-ADD Cleanup Logic --- #

                    try:
                        # Use the potentially cleaned text for the final structure
                        final_structure = FinalApiResponseStructure(
                            text=coerced_text, # USE CLEANED TEXT
                            include_tables=[], 
                            chart_specs=[]      
                        )
                        logger.debug("[AgentNode] Successfully coerced plain text content into FinalApiResponseStructure.")
                    except Exception as coerce_err:
                        logger.error(f"[AgentNode] Error coercing plain text content: {coerce_err}", exc_info=True)
                        final_structure = FinalApiResponseStructure(text="I encountered an issue processing the response.", include_tables=[], chart_specs=[])
                else:
                    # No tool calls and no content
                    logger.warning("[AgentNode] LLM AIMessage had no tool calls and no content. Using safe fallback structure.")
                    final_structure = FinalApiResponseStructure(text="I received an empty response.", include_tables=[], chart_specs=[])

        # CASE 5: LLM response was not an AIMessage
        else:
            logger.error(f"[AgentNode] LLM response was not an AIMessage object (Type: {type(llm_response)}). Using safe fallback structure.")
            final_structure = FinalApiResponseStructure(text="An internal error occurred communicating with the language model.", include_tables=[], chart_specs=[])

    # --- Handle Specific Errors (e.g., Content Filter, Connection Error) --- #
    except openai.BadRequestError as e:
        logger.error(f"[AgentNode] OpenAI BadRequestError during LLM invocation: {e}", exc_info=False)
        if e.body and e.body.get('code') == 'content_filter':
            logger.warning(f"[AgentNode] Azure OpenAI Content Filter triggered. Returning safe response.")
            final_structure = FinalApiResponseStructure(text="I cannot process this request due to content policies.", include_tables=[], chart_specs=[])
        else:
            # For other BadRequestErrors (potentially non-retryable client errors)
            final_structure = FinalApiResponseStructure(text=f"An error occurred processing the request ({e.code or 'Unknown'}). Please check your input.", include_tables=[], chart_specs=[])
        return_dict["prompt_tokens"] = 0
        return_dict["completion_tokens"] = 0

    # --- ADDED: Handle specific connection-related errors AFTER retries --- #
    except APIConnectionError as e:
        logger.error(f"[AgentNode] OpenAI APIConnectionError after retries: {e}", exc_info=False)
        final_structure = FinalApiResponseStructure(
            text="I'm having trouble connecting to the language model service right now. Please try your request again in a moment.",
            include_tables=[], chart_specs=[]
        )
        return_dict["prompt_tokens"] = 0
        return_dict["completion_tokens"] = 0
    except (APITimeoutError, RateLimitError) as e:
        # Handle timeouts and rate limits similarly after retries
        logger.error(f"[AgentNode] OpenAI {type(e).__name__} after retries: {e}", exc_info=False)
        final_structure = FinalApiResponseStructure(
            text="The language model service is currently busy or timed out. Please try again shortly.",
            include_tables=[], chart_specs=[]
        )
        return_dict["prompt_tokens"] = 0
        return_dict["completion_tokens"] = 0
    # --- END Specific Error Handling --- #

    # --- Handle General Exceptions --- #
    except Exception as e:
        # This catches any other unexpected errors during the LLM call or node logic
        logger.error(f"[AgentNode] Unhandled Exception during LLM invocation or node processing: {e}", exc_info=True)
        final_structure = FinalApiResponseStructure(text="An unexpected internal error occurred.", include_tables=[], chart_specs=[])
        return_dict["prompt_tokens"] = 0
        return_dict["completion_tokens"] = 0

    # --- Final Step: Update Return Dictionary --- #
    if final_structure:
        return_dict["final_response_structure"] = final_structure
        if return_dict["messages"] and isinstance(return_dict["messages"][0], AIMessage):
             original_message = return_dict["messages"][0]
             # Clear tool calls from message history IF a final structure was set here
             return_dict["messages"] = [AIMessage(content=original_message.content, id=original_message.id, usage_metadata=original_message.usage_metadata)]
             # Ensure operational_calls list is empty if final structure is set
             operational_calls = []
        # *** Persist resolved_location_map even when final structure is set ***
        # The final processing step might need it to add context about missing data.
        # If the map becomes stale due to complex loops, that's a different issue to address if observed.
        # For now, prioritize providing context in the final response.
        pass # Keep the map as it is in return_dict

    # If operational calls were identified and a final_structure was NOT set, they remain in return_dict["messages"][0].tool_calls
    logger.debug(f"[AgentNode] Exiting agent node. Final Structure Set: {final_structure is not None}. Proceeding Tool Calls in Message History: {len(return_dict['messages'][0].tool_calls) if return_dict['messages'] and isinstance(return_dict['messages'][0], AIMessage) else 0}")
    logger.debug(f"[AgentNode] Exiting agent node. Final Structure Set: {final_structure is not None}. Retry Count: {return_dict['sql_security_retry_count']}")
    return return_dict


def _preprocess_state_for_llm(state: AgentState) -> AgentState:
    """
    Preprocess the state to ensure it's optimized for LLM context window AND
    preserves the integrity of AIMessage/ToolMessage pairs.
    This helps prevent issues with the LLM failing due to context limitations
    or invalid message sequences.
    """
    processed_state = {k: v for k, v in state.items()}
    max_messages = settings.MAX_STATE_MESSAGES
    num_messages_before_pruning = 0 # Initialize

    if 'messages' in processed_state and len(processed_state['messages']) > max_messages:
        original_messages = processed_state['messages']
        num_messages_before_pruning = len(original_messages) # Set actual count
        # logger.debug(f"Starting pruning messages from {len(original_messages)} down to target ~{max_messages}")
        logger.info(f"[_preprocess_state_for_llm] Starting message pruning. Original count: {num_messages_before_pruning}, Target: ~{max_messages}")

        preserved_messages_reversed: List[BaseMessage] = []
        system_message: Optional[SystemMessage] = None
        temp_tool_messages: List[ToolMessage] = []
        num_preserved = 0

        # Find and preserve the system message first
        original_without_system = []
        for msg in original_messages:
            if isinstance(msg, SystemMessage) and system_message is None:
                system_message = msg
            else:
                original_without_system.append(msg)

        # Iterate backwards through messages (excluding system message)
        for msg in reversed(original_without_system):
            # Stop preserving if we've hit the target count
            if num_preserved >= max_messages:
                logger.debug(f"Reached approx max message limit ({max_messages}), stopping preservation.")
                break

            if isinstance(msg, ToolMessage):
                # Collect tool messages potentially belonging to the preceding AI turn
                temp_tool_messages.insert(0, msg) # Insert at beginning to maintain order
                logger.debug(f"Temporarily holding ToolMessage (ID: {msg.tool_call_id})")
                # Don't add to preserved_messages_reversed or increment num_preserved yet

            elif isinstance(msg, AIMessage):
                associated_tool_messages = []
                is_pair = False
                if msg.tool_calls:
                    # This AI message called tools. Check if we have its responses in temp.
                    matching_tool_ids = {tc['id'] for tc in msg.tool_calls}
                    # Check if temp_tool_messages contains *only* responses for this AI msg
                    if temp_tool_messages and all(tm.tool_call_id in matching_tool_ids for tm in temp_tool_messages):
                        # Found a complete pair (or set of pairs)
                        associated_tool_messages = temp_tool_messages # These belong together
                        is_pair = True
                        logger.debug(f"Identified AIMessage (ID: {msg.id}) with {len(associated_tool_messages)} matching ToolMessage(s).")
                        # Clear temp ONLY after successful pairing below
                    else:
                        # This AIMessage expected tool calls, but temp_tool_messages is empty or doesn't match.
                        if temp_tool_messages:
                            # logger.warning(f"Discarding {len(temp_tool_messages)} ToolMessage(s) orphaned before AIMessage(TC) (ID: {msg.id}).")
                            logger.warning(f"[_preprocess_state_for_llm] Discarding {len(temp_tool_messages)} ToolMessage(s) (orphaned before AIMessage with tool_calls ID: {msg.id}). Details: {[ (tm.tool_call_id, str(tm.content)[:50]+'...') for tm in temp_tool_messages ]}")
                            temp_tool_messages = [] # Clear orphans before preserving AIMessage(TC)
                        logger.debug(f"Identified orphaned AIMessage(TC) (ID: {msg.id}) - ToolMessages potentially missing/pruned.")
                else:
                     # Simple AIMessage, no tool calls. Any temp_tool_messages are orphans.
                     if temp_tool_messages:
                        # logger.warning(f"Discarding {len(temp_tool_messages)} ToolMessage(s) orphaned before simple AIMessage (ID: {msg.id}).")
                        logger.warning(f"[_preprocess_state_for_llm] Discarding {len(temp_tool_messages)} ToolMessage(s) (orphaned before simple AIMessage ID: {msg.id}). Details: {[ (tm.tool_call_id, str(tm.content)[:50]+'...') for tm in temp_tool_messages ]}")
                        temp_tool_messages = [] # Clear orphans

                # Now, decide whether to preserve this AIMessage and potentially its tools
                messages_to_add = [msg] + associated_tool_messages # AIMsg + its tools (if any)
                current_block_size = len(messages_to_add)

                # Check if adding this block fits
                if num_preserved + current_block_size <= max_messages + 2: # Soft limit
                    preserved_messages_reversed.extend(reversed(messages_to_add)) # Add block newest-first
                    num_preserved += current_block_size
                    logger.debug(f"Preserved {type(msg).__name__} block (size {current_block_size}). Total preserved: {num_preserved}")
                    if is_pair:
                        temp_tool_messages = [] # Crucial: Clear temp ONLY after successful pairing
                else:
                    logger.debug(f"Stopping preservation: Adding {type(msg).__name__} block (size {current_block_size}) would exceed limits ({num_preserved}/{max_messages}).")
                    break # Stop processing older messages

            elif isinstance(msg, HumanMessage):
                # Human message encountered. Any temp_tool_messages are orphans.
                if temp_tool_messages:
                    # logger.warning(f"Discarding {len(temp_tool_messages)} ToolMessage(s) orphaned before HumanMessage.")
                    logger.warning(f"[_preprocess_state_for_llm] Discarding {len(temp_tool_messages)} ToolMessage(s) (orphaned before HumanMessage). Details: {[ (tm.tool_call_id, str(tm.content)[:50]+'...') for tm in temp_tool_messages ]}")
                    temp_tool_messages = [] # Clear orphans

                # Preserve the HumanMessage if space allows
                if num_preserved < max_messages:
                    preserved_messages_reversed.append(msg)
                    num_preserved += 1
                    logger.debug(f"Preserved HumanMessage. Total preserved: {num_preserved}")
                else:
                     logger.debug(f"Stopping preservation: Adding HumanMessage would exceed limits ({num_preserved}/{max_messages}).")
                     break # Stop processing older messages

        # Final assembly
        final_preserved_messages = []
        if system_message:
            final_preserved_messages.append(system_message)

        # Add the messages preserved during backward iteration (newest-first)
        # Reverse them to get chronological order before adding
        preserved_messages_reversed.reverse()
        final_preserved_messages.extend(preserved_messages_reversed)

        processed_state['messages'] = final_preserved_messages
        # logger.debug(f"Final pruned message count: {len(processed_state['messages'])}. Structure preserved: {[(type(m).__name__ + ('(TC)' if isinstance(m, AIMessage) and m.tool_calls else '')) for m in processed_state['messages']]}")
        logger.info(f"[_preprocess_state_for_llm] Message pruning complete. Messages for LLM: {len(processed_state['messages'])} (Original: {num_messages_before_pruning if num_messages_before_pruning > 0 else len(original_messages) if 'original_messages' in locals() else 'N/A'}).")

    # --- Existing table truncation logic remains unchanged --- #
    if 'structured_results' in processed_state and processed_state['structured_results']:
        original_structured_results = state['structured_results'] 
        processed_structured_results = []
        for i, result in enumerate(processed_state['structured_results']):
            processed_result = result.copy()
            if 'table' in processed_result and isinstance(processed_result['table'], dict):
                processed_table = processed_result['table'].copy()
                max_rows = settings.MAX_TABLE_ROWS_IN_STATE
                if 'rows' in processed_table and len(processed_table['rows']) > max_rows:
                    logger.debug(f"Truncating table index {i} rows from {len(processed_table['rows'])} to {max_rows}")
                    processed_table['rows'] = processed_table['rows'][:max_rows]
                    if 'metadata' not in processed_table: processed_table['metadata'] = {}
                    processed_table['metadata']['truncated'] = True
                    original_table = original_structured_results[i].get('table', {})
                    processed_table['metadata']['original_rows'] = len(original_table.get('rows', []))
                processed_result['table'] = processed_table
            processed_structured_results.append(processed_result)
        processed_state['structured_results'] = processed_structured_results

    return processed_state

# --- Helper Function to Apply Resolved IDs to SQL PARAMS  ---
def _preprocess_sql_params(
    params: Optional[Dict[str, Any]], 
    resolved_map: Optional[Dict[str, str]],
    current_organization_id: str, # Trusted organization_id for the request
    log_prefix: str = "[SQL Param Preprocessing]"
) -> Dict[str, Any]:
    """Processes SQL parameters to ensure correct organization_id and resolve placeholders for known ID types."""
    initial_params_copy = params.copy() if params else {} # For logging changes

    if params is None:
        # If LLM provides no params, but SQL expects :organization_id, ensure it's added.
        logger.debug(f"{log_prefix} Original params dictionary is None. Initializing with trusted organization_id.")
        return {"organization_id": current_organization_id}

    processed_params = params.copy() # Work on a copy

    # Ensure organization_id is present and correct, or inject it if missing/wrong from LLM.
    # This is a security measure to enforce data scoping.
    llm_org_id = processed_params.get("organization_id")
    if llm_org_id != current_organization_id:
        logger.warning(
            f"{log_prefix} LLM provided organization_id '{llm_org_id}' does not match trusted ID '{current_organization_id}' or was missing. Overwriting/injecting trusted ID."
        )
        processed_params["organization_id"] = current_organization_id
    else:
        logger.debug(f"{log_prefix} Confirmed organization_id '{current_organization_id}' from LLM matches trusted ID or was correctly set.")

    if not resolved_map:
        logger.debug(f"{log_prefix} resolved_location_map is None or empty. Skipping further parameter substitution.")
        return processed_params
    
    logger.debug(f"{log_prefix} Current resolved_map for substitution: {resolved_map}")

    # Known ID keys that might need resolution if their value isn't a UUID
    known_id_keys_for_resolution = {"branch_id", "hierarchy_id", "location_id", "hierarchyId"} # Case-sensitive match for keys in params

    for key, value in list(processed_params.items()): # Iterate over a copy for safe modification
        logger.debug(f"{log_prefix} Processing param - Key: '{key}', Value: '{value}', Type: {type(value)}")
        substituted_this_iteration = False

        if key in known_id_keys_for_resolution and isinstance(value, str):
            try:
                uuid.UUID(value) # Check if it's already a valid UUID string
                logger.debug(f"{log_prefix} Param '{key}' ('{value}') is already a valid UUID. No substitution needed.")
            except ValueError:
                logger.debug(f"{log_prefix} Param '{key}' ('{value}') is a string but not a UUID. Attempting lookup in resolved_map using value as key.")
                if value in resolved_map: # e.g., if value is 'argyle' (lowercase) and 'argyle' is a key in resolved_map
                    resolved_uuid = resolved_map[value]
                    processed_params[key] = resolved_uuid
                    substituted_this_iteration = True
                    logger.info(f"{log_prefix} SUCCESS: Substituted non-UUID value for known ID key '{key}' (was '{value}') with resolved UUID '{resolved_uuid}' using map key '{value}'.")
                else:
                    logger.warning(f"{log_prefix} FAILED_LOOKUP (val): Param '{key}' has non-UUID value '{value}', but no match found in resolved_map using '{value}' as key.")
        
        # Check for placeholder values like '<resolved_name_id>' if not already substituted
        if not substituted_this_iteration and isinstance(value, str) and value.startswith("<resolved_") and value.endswith("_id>"):
            logger.debug(f"{log_prefix} Param '{key}' has placeholder value '{value}'. Attempting extraction and lookup.")
            # Try to extract key from placeholder itself, e.g., <resolved_argyle_id> -> argyle
            placeholder_content_match = re.match(r"<resolved_([a-zA-Z0-9_.-]+)_id>", value) # Allow . and - in resolved names
            
            extracted_key_from_placeholder = None
            if placeholder_content_match:
                extracted_key_from_placeholder = placeholder_content_match.group(1).lower()
                logger.debug(f"{log_prefix} Extracted key '{extracted_key_from_placeholder}' from placeholder '{value}'.")
                if extracted_key_from_placeholder in resolved_map:
                    resolved_uuid = resolved_map[extracted_key_from_placeholder]
                    processed_params[key] = resolved_uuid
                    substituted_this_iteration = True
                    logger.info(f"{log_prefix} SUCCESS: Substituted placeholder for '{key}' (was '{value}') with resolved UUID '{resolved_uuid}' using extracted map key '{extracted_key_from_placeholder}'.")
                else:
                    logger.warning(f"{log_prefix} FAILED_LOOKUP (extracted_placeholder): Param '{key}' ('{value}'), extracted key '{extracted_key_from_placeholder}' not in resolved_map: {list(resolved_map.keys())}")    
            else:
                logger.warning(f"{log_prefix} Param '{key}' ('{value}') looks like a placeholder but regex did not match content.")

            # Fallback: If placeholder extraction failed or didn't find a match, 
            # try using the parameter key itself (lowercased) if it wasn't a known_id_key.
            # This handles cases like LLM using param `argyle_id = "<some_placeholder>"` where `argyle_id` itself might be the map key
            if not substituted_this_iteration and key.lower() in resolved_map and key not in known_id_keys_for_resolution:
                resolved_uuid = resolved_map[key.lower()]
                processed_params[key] = resolved_uuid
                substituted_this_iteration = True
                logger.info(f"{log_prefix} SUCCESS: Substituted placeholder/value for '{key}' (was '{value}') with resolved UUID '{resolved_uuid}' using param key '{key.lower()}' as map key.")
            elif not substituted_this_iteration and key.lower() not in resolved_map and key not in known_id_keys_for_resolution:
                 logger.warning(f"{log_prefix} FAILED_LOOKUP (param_key): Param '{key}' ('{value}'), its lowercased key '{key.lower()}' also not in resolved_map (and not a known_id_key). Placeholder remains.")

        if not substituted_this_iteration and isinstance(value, str) and value.startswith("<") and value.endswith(">"):
             logger.warning(f"{log_prefix} Param '{key}' ('{value}') appears to be an unresolved placeholder and was not substituted by any rule.")

    # logger.debug(f"{log_prefix} Final processed params: {processed_params}")
    if processed_params != initial_params_copy:
        logger.info(f"{log_prefix} SQL params were modified. Before: {initial_params_copy}, After: {processed_params}")
    else:
        logger.debug(f"{log_prefix} SQL params unchanged by this function. Final: {processed_params}")
    return processed_params

# --- Helper Function to Apply Resolved IDs to SQL STRING (EXISTING) ---
def _apply_resolved_ids_to_sql_args(sql: str, params: Dict[str, Any], resolved_map: Dict[str, str]) -> Tuple[str, Dict[str, Any]]:
    if not resolved_map:
        return sql, params

    logger.debug(f"[SQL Correction] Applying resolved IDs to SQL string using sqlparse. Map: {resolved_map}")
    updated_params = params.copy()
    new_params_added: Dict[str, str] = {}
    param_counter = 0
    modified = False

    try:
        parsed = sqlparse.parse(sql)
        if not parsed:
            logger.warning("[SQL Correction] Failed to parse SQL with sqlparse. Returning original.")
            return sql, params
        
        stmt = parsed[0]

        def traverse_and_replace(token_list: List[sqlparse.sql.Token]):
            nonlocal param_counter, modified
            i = 0
            while i < len(token_list):
                token = token_list[i]
                replacement_made = False
                
                if isinstance(token, sqlparse.sql.Comparison):
                    left_identifier = None
                    operator_token = None
                    right_operand = None
                    
                    comp_tokens = [t for t in token.flatten() if not t.is_whitespace]
                    if len(comp_tokens) >= 3 and isinstance(comp_tokens[0], sqlparse.sql.Identifier):
                        left_identifier = comp_tokens[0]
                        operator_token = comp_tokens[1]
                        right_operand = comp_tokens[2]
                    
                    if left_identifier and operator_token and right_operand:
                        left_str = str(left_identifier).lower()
                        if left_str.endswith('.\"name\"'): # Using .\"name\" to match literal quotes if present
                            alias = left_str.split('.')[0]
                            names_to_resolve = []
                            new_param_dict = {}

                            if operator_token.value == '=' and right_operand.ttype is sqlparse.tokens.String.Single:
                                names_to_resolve = [right_operand.value.strip("'")]
                            elif operator_token.value.upper() == 'IN' and isinstance(right_operand, sqlparse.sql.Parenthesis):
                                names_to_resolve = [ 
                                    t.value.strip("'") for t in right_operand.flatten() 
                                    if t.ttype is sqlparse.tokens.String.Single
                                ]
                            
                            if names_to_resolve:
                                resolved_ids = []
                                param_names_for_sql = []
                                for name in names_to_resolve:
                                    lower_name = name.lower()
                                    if lower_name in resolved_map:
                                        resolved_id = resolved_map[lower_name]
                                        resolved_ids.append(resolved_id)
                                        param_name = f"res_id_{param_counter}"
                                        param_names_for_sql.append(f":{param_name}")
                                        new_param_dict[param_name] = resolved_id
                                        param_counter += 1
                                    else:
                                        logger.warning(f"[SQL Correction] Name '{name}' in SQL filter not in resolved map. Skipping.")

                                if resolved_ids:
                                    id_column_token = sqlparse.sql.Identifier(f'{alias}.\"id\"')
                                    new_comparison_tokens = []
                                    if len(resolved_ids) == 1:
                                        op_token = sqlparse.sql.Token(sqlparse.tokens.Operator, '=')
                                        param_token = sqlparse.sql.Token(sqlparse.tokens.Name.Placeholder, param_names_for_sql[0])
                                        new_comparison_tokens = [id_column_token, 
                                                                 sqlparse.sql.Token(sqlparse.tokens.Whitespace, ' '), 
                                                                 op_token, 
                                                                 sqlparse.sql.Token(sqlparse.tokens.Whitespace, ' '), 
                                                                 param_token]
                                    else:
                                        op_token = sqlparse.sql.Token(sqlparse.tokens.Keyword, 'IN')
                                        param_tokens_inner = []
                                        for idx, param_name_in_sql in enumerate(param_names_for_sql):
                                             param_tokens_inner.append(sqlparse.sql.Token(sqlparse.tokens.Name.Placeholder, param_name_in_sql))
                                             if idx < len(param_names_for_sql) - 1:
                                                 param_tokens_inner.append(sqlparse.sql.Token(sqlparse.tokens.Punctuation, ','))
                                                 param_tokens_inner.append(sqlparse.sql.Token(sqlparse.tokens.Whitespace, ' '))
                                        parenthesis = sqlparse.sql.Parenthesis(param_tokens_inner)
                                        new_comparison_tokens = [id_column_token, 
                                                                 sqlparse.sql.Token(sqlparse.tokens.Whitespace, ' '), 
                                                                 op_token,
                                                                 sqlparse.sql.Token(sqlparse.tokens.Whitespace, ' '), 
                                                                 parenthesis]
                                    
                                    token_list[i:i+1] = new_comparison_tokens
                                    replacement_made = True
                                    modified = True
                                    new_params_added.update(new_param_dict)
                                    logger.info(f"[SQL Correction] Replaced name filter '{str(token).strip()}' with ID filter using sqlparse.")
                                    i += len(new_comparison_tokens) - 1 
                                else:
                                    logger.warning(f"[SQL Correction] Could not resolve names in filter: '{str(token).strip()}'. Keeping original.")
                
                if not replacement_made and hasattr(token, 'tokens'):
                    traverse_and_replace(token.tokens)
                i += 1
        
        traverse_and_replace(stmt.tokens)

    except Exception as e:
        logger.error(f"[SQL Correction] Error during sqlparse processing: {e}. Returning original SQL.", exc_info=True)
        return sql, params

    if modified:
        updated_params.update(new_params_added)
        modified_sql = str(stmt)
        # logger.info(f"[SQL Correction] Applied ID filters to SQL string. Modified SQL tail: ...{modified_sql[-200:]}. Added params: {new_params_added}")
        logger.info(f"[SQL Correction] SQL string was modified by sqlparse. Original (tail): ...{sql[-200:]}, New (tail): ...{modified_sql[-200:]}")
        if new_params_added: # Log only if params were also changed by this function
            logger.info(f"[SQL Correction] Params were also modified by sqlparse name-to-ID replacement. Original relevant: (see SQL string), Added/Updated: {new_params_added}")
        return modified_sql, updated_params
    else:
        # logger.debug("[SQL Correction] No name filters found in SQL string matching resolved map keys using sqlparse.")
        logger.debug(f"[SQL Correction] No modifications made to SQL string by sqlparse based on resolved map. SQL (tail): ...{sql[-200:]}")
        return sql, params

# --- Tool Node Handler ---
async def async_tools_node_handler(state: AgentState, tools: List[Any]) -> Dict[str, Any]:
    request_id = state.get("request_id")
    logger.debug(f"[ToolsNode] Entering tool handler.")
    last_message = state["messages"][-1] if state["messages"] else None
    tool_map = {tool.name: tool for tool in tools}
    tool_execution_results = []
    operational_tool_calls = []
    location_map_this_turn = {} 
    successful_calls = 0 # Initialize successful_calls counter

    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        operational_tool_calls = [
            tc for tc in last_message.tool_calls
            if tc.get("name") != FinalApiResponseStructure.__name__
        ]

    if not operational_tool_calls:
        logger.debug(f"[ToolsNode] No operational tool calls found in last message.")
        return {
            "messages": [],
            "structured_results": state.get("structured_results", []),
            "failure_patterns": state.get("failure_patterns", {}),
            "recovery_guidance": state.get("recovery_guidance"),
            "resolved_location_map": state.get("resolved_location_map")
        }

    logger.debug(f"[ToolsNode] Dispatching {len(operational_tool_calls)} operational tool calls: {[tc.get('name') for tc in operational_tool_calls]}")

    prepared_tool_invocations = []

    for tool_call in operational_tool_calls:
        tool_name = tool_call.get("name")
        tool_id = tool_call.get("id", f"tool_call_{uuid.uuid4()}")
        raw_tool_args = tool_call.get("args", {})
        
        current_tool_instance = tool_map.get(tool_name)
        if not current_tool_instance:
            logger.error(f"[ToolsNode] Tool '{tool_name}' (ID: {tool_id}) requested by LLM not found. Skipping.")
            error_content = json.dumps({"error": {"type": "TOOL_NOT_FOUND", "message": f"Tool '{tool_name}' is not available."}})
            tool_execution_results.append(ToolMessage(content=error_content, name=tool_name, tool_call_id=tool_id))
            continue

        processed_tool_args = raw_tool_args.copy()

        if tool_name == "execute_sql":
            trusted_org_id_for_tool = getattr(current_tool_instance, 'organization_id', None)
            if not trusted_org_id_for_tool:
                 logger.error(f"[ToolsNode] CRITICAL: SQLExecutionTool for call {tool_id} missing organization_id. Cannot process securely.")
                 error_content = json.dumps({"error": {"type": "TOOL_CONFIG_ERROR", "message": "SQL tool is missing organization context."}})
                 tool_execution_results.append(ToolMessage(content=error_content, name=tool_name, tool_call_id=tool_id))
                 continue # Skip this tool call
            
            # 1. Preprocess parameters (e.g., org_id enforcement, non-UUID string to ID resolution)
            current_sql_params = processed_tool_args.get("params")
            logger.debug(f"[ToolsNode] Preprocessing params for 'execute_sql' (ID: {tool_id}) with trusted org_id: {trusted_org_id_for_tool}. Original params: {current_sql_params}")
            preprocessed_params = _preprocess_sql_params(
                params=current_sql_params,
                resolved_map=state.get("resolved_location_map"),
                current_organization_id=trusted_org_id_for_tool 
            )
            processed_tool_args["params"] = preprocessed_params # Update args with param-preprocessed params
            
            # 2. Apply SQL string literal correction (name -> ID in SQL string itself)
            current_sql_string = processed_tool_args.get("sql", "")
            if state.get("resolved_location_map") and current_sql_string: 
                logger.debug(f"[ToolsNode] Applying SQL string literal correction for 'execute_sql' (ID: {tool_id}). SQL before: ...{current_sql_string[-100:]}")
                modified_sql, final_params_after_sql_rewrite = _apply_resolved_ids_to_sql_args(
                    sql=current_sql_string, 
                    params=preprocessed_params, # Pass the already param-preprocessed params
                    resolved_map=state.get("resolved_location_map")
                )
                processed_tool_args["sql"] = modified_sql
                processed_tool_args["params"] = final_params_after_sql_rewrite 
                logger.debug(f"[ToolsNode] Final SQL for call {tool_id} (tail): ...{modified_sql[-200:] if modified_sql else '[EMPTY SQL]'}. Final Params: {final_params_after_sql_rewrite}")
            else:
                logger.debug(f"[ToolsNode] Skipping SQL string literal correction for 'execute_sql' (ID: {tool_id}) (no resolved_map, empty SQL, or already handled). Params remain: {processed_tool_args.get('params')}")

            # --- START: New check for 'hierarchyId' in SELECT clause ---
            final_sql_for_tool = processed_tool_args.get("sql", "")
            resolved_map_for_check = state.get("resolved_location_map")

            # Check only if there are resolved entities and the query likely involves the hierarchyCaches table.
            # Using a simple string check for "hierarchyCaches" as a heuristic.
            if resolved_map_for_check and "hierarchyCaches" in final_sql_for_tool.lower(): # Case-insensitive check for table name
                # Regex to find 'AS "hierarchyId"', 'AS \'hierarchyId\'', 'AS hierarchyId',
                # or ' hierarchyId,' or ' hierarchyId ' or 'SELECT hierarchyId,' or 'SELECT hierarchyId FROM'.
                # This aims to catch common ways the ID column (aliased or direct) might appear.
                # It's case-insensitive for "hierarchyId" and "AS".
                hierarchy_id_pattern = re.compile(
                    r"""
                        (\sAS\s+(['"]?)hierarchyId\2) | # AS "hierarchyId", AS 'hierarchyId', AS hierarchyId
                        (,\s*hierarchyId\s*[,)]) |     # , hierarchyId, or , hierarchyId) -- Matches 'hierarchyId' if it's a column name followed by comma or parenthesis
                        (\s+hierarchyId\s*[,)]) |      # Space before hierarchyId, then comma or parenthesis
                        (SELECT\s+(DISTINCT\s+)?(['"]?)hierarchyId\3\s*[,FROM]) # SELECT hierarchyId, or SELECT hierarchyId FROM (also with DISTINCT)
                    """, 
                    re.IGNORECASE | re.VERBOSE
                )
                
                # To avoid false positives on very long queries, check a reasonable prefix of the SQL
                # where the SELECT clause is expected. If SQL is short, check all of it.
                sql_prefix_to_check = final_sql_for_tool[:500] # Check first 500 chars

                if not hierarchy_id_pattern.search(sql_prefix_to_check):
                    logger.critical(
                        f"[ToolsNode] CRITICAL ALERT: LLM may have failed to include 'hierarchyId' in SELECT "
                        f"for tool_id: {tool_id}. Query involves resolved entities and 'hierarchyCaches'. "
                        f"Downstream analysis by AnalyzeResultsNode may be impaired. "
                        f"SQL (approx start): {final_sql_for_tool[:250]}..."
                    )
            # --- END: New check for 'hierarchyId' in SELECT clause ---
        # End of execute_sql specific processing

        prepared_tool_invocations.append({
            "tool": current_tool_instance,
            "args": processed_tool_args, # Use the (potentially) processed arguments
            "id": tool_id,
            "name": tool_name,
            "retries_left": settings.TOOL_EXECUTION_RETRIES
        })
    # End of loop preparing tool invocations
    
    # Execute all prepared tool invocations concurrently
    execution_tasks = []
    for invocation_detail in prepared_tool_invocations:
        # tool_to_call = invocation_detail["tool"] # Not directly used here, execute_with_retry takes the dict
        # tool_input_args = invocation_detail["args"]
        tool_call_id_for_exec = invocation_detail["id"]
        logger.debug(f"[ToolsNode] Adding tool '{invocation_detail['name']}' (ID: {tool_call_id_for_exec}) to parallel execution queue with args: {invocation_detail['args']}")
        execution_tasks.append(
            execute_with_retry(invocation_detail) # execute_with_retry expects the whole dict
        )
    
    results_from_execution = []
    if execution_tasks: # Only gather if there are tasks
        try:
            results_from_execution = await asyncio.gather(*execution_tasks, return_exceptions=False) # Let execute_with_retry handle exceptions and return structured results
        except Exception as gather_err: # Fallback if gather itself has an issue (rare)
            logger.error(f"[ToolsNode] Unexpected error during asyncio.gather for tool executions: {gather_err}", exc_info=True)
            # Create a generic error message for all calls that were supposed to run
            for tool_call_details_on_error in prepared_tool_invocations:
                tool_execution_results.append(ToolMessage(
                    content=json.dumps({"error": {"type": "TOOL_GATHER_ERROR", "message": f"Async gathering of tool results failed: {str(gather_err)}"}}),
                    name=tool_call_details_on_error["name"],
                    tool_call_id=tool_call_details_on_error["id"]
                ))
    # else: no tasks, results_from_execution remains empty

    # Process results from asyncio.gather (or empty list if no tasks/gather error)
    temp_structured_results = []
    updated_failure_patterns = state.get("failure_patterns", {}).copy()
    current_recovery_guidance: Optional[str] = None

    for result_item in results_from_execution:
        # execute_with_retry is expected to return a dictionary with specific keys
        if not isinstance(result_item, dict): # Defensive check
            logger.error(f"[ToolsNode] Unexpected item type in results_from_execution (expected dict): {type(result_item)} - {str(result_item)[:200]}")
            continue

        tool_call_id_from_res = result_item.get("id")
        tool_name_from_res = result_item.get("name")
        tool_content_str = result_item.get("content_str") # This is the JSON string content of the ToolMessage
        is_error = result_item.get("is_error", False)
        original_args_for_tool = result_item.get("original_args", {}) # Args used for this specific call

        if not (tool_call_id_from_res and tool_name_from_res and tool_content_str is not None):
            logger.error(f"[ToolsNode] Invalid result item from tool execution (missing id, name, or content): {result_item}")
            continue # Skip this malformed result

        # Always append a ToolMessage for graph history
        tool_execution_results.append(ToolMessage(content=tool_content_str, name=tool_name_from_res, tool_call_id=tool_call_id_from_res))

        if not is_error:
            successful_calls += 1
            try:
                tool_content_dict = json.loads(tool_content_str)
                if tool_name_from_res == "hierarchy_name_resolver":
                    # Update map for THIS turn with results from hierarchy_name_resolver
                    current_resolved_map_from_tool = tool_content_dict.get("resolution_results", {})
                    for name, res_data in current_resolved_map_from_tool.items():
                        if isinstance(res_data, dict) and res_data.get("status") == "found" and "id" in res_data:
                            location_map_this_turn[name.lower()] = res_data["id"]
                    if current_resolved_map_from_tool:
                         logger.info(f"[ToolsNode] Updated resolved location map (this turn) with {len(location_map_this_turn)} entries from '{tool_name_from_res}'.")
                
                elif tool_name_from_res == "execute_sql":
                    # Check if the tool itself returned a structured error (e.g., DB connection, permission)
                    # These are errors that SQLExecutionTool reports in its JSON output, not exceptions during invoke.
                    if tool_content_dict.get("error"):
                        logger.warning(f"[ToolsNode] Tool '{tool_name_from_res}' (ID: {tool_call_id_from_res}) executed but returned a structured error: {tool_content_dict.get('error')}")
                        # The error is already in the ToolMessage; no separate structured result here.
                        # It will be handled by the general error processing logic below for failure_patterns.
                        is_error = True # Treat as error for failure pattern logging

                    elif tool_content_dict.get("columns") is not None and tool_content_dict.get("rows") is not None:
                        table_data = {
                            "columns": tool_content_dict["columns"],
                            "rows": tool_content_dict["rows"],
                            "text": tool_content_dict.get("text") # Optional summary text from tool
                        }
                        # Use the original_args_for_tool that were ACTUALLY used for this successful call
                        sql_filters_used = original_args_for_tool.get("params", {})
                        temp_structured_results.append({"table": table_data, "filters": sql_filters_used})
                        logger.info(f"[ToolsNode] Created structured result from successful '{tool_name_from_res}' call (ID: {tool_call_id_from_res}) with {len(table_data['rows'])} rows. Filters used: {sql_filters_used}")
                    else:
                         logger.warning(f"[ToolsNode] Successful '{tool_name_from_res}' call (ID: {tool_call_id_from_res}) returned parsable JSON, but no expected table structure or error field found. Keys: {list(tool_content_dict.keys())}")

            except json.JSONDecodeError:
                logger.warning(f"[ToolsNode] Failed to decode JSON from nominally successful tool call '{tool_name_from_res}' (ID: {tool_call_id_from_res}): {tool_content_str[:150]}...")
                # This is an unexpected state, treat as error for pattern tracking
                is_error = True 
            except Exception as e_struct_proc: # Catch any other errors during processing of successful results
                logger.error(f"[ToolsNode] Error processing successful result from tool '{tool_name_from_res}' (ID: {tool_call_id_from_res}): {e_struct_proc}", exc_info=True)
                is_error = True
        
        # This 'if is_error' block now also catches errors reported *by* the tool in its JSON, or JSON parsing errors of success cases
        if is_error:
            error_info_dict = {}
            try: 
                # Try to parse the content string again if it wasn't already parsed or if it's an error string
                parsed_content_for_error = json.loads(tool_content_str)
                error_info_dict = parsed_content_for_error.get("error", {}) if isinstance(parsed_content_for_error, dict) else {}
            except: # If content_str is not valid JSON or doesn't contain "error"
                pass # error_info_dict remains empty or has its prior state
            
            failure_type = error_info_dict.get("type", "UNKNOWN_TOOL_RUNTIME_ERROR")
            failure_message = error_info_dict.get("message", tool_content_str) # Fallback to full content if no message
            
            # Determine signature for failure pattern tracking
            error_signature_key = tool_name_from_res
            if tool_name_from_res == "execute_sql":
                # Use a simplified signature for SQL to group similar errors
                error_signature_key = _get_sql_call_signature(original_args_for_tool) 

            if error_signature_key not in updated_failure_patterns:
                updated_failure_patterns[error_signature_key] = []
            
            updated_failure_patterns[error_signature_key].append({
                "error_type": failure_type,
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "details": failure_message,
                "tool_call_id": tool_call_id_from_res
            })
            logger.warning(f"[ToolsNode] Failure recorded for tool '{tool_name_from_res}' (ID: {tool_call_id_from_res}), signature key: '{error_signature_key}', type: '{failure_type}'. Details: {failure_message[:150]}...")
    # End of loop processing results_from_execution

    # Combine resolved locations from this turn with existing ones from the state
    final_resolved_map = state.get("resolved_location_map", {}).copy() if state.get("resolved_location_map") else {}
    if location_map_this_turn: # Only update if new resolutions occurred this turn
        final_resolved_map.update(location_map_this_turn)
        logger.info(f"[ToolsNode] Final resolved_location_map contains {len(final_resolved_map)} entries after this turn.")

    # --- Generate Recovery Guidance if there were failures ---
    if len(operational_tool_calls) > 0 and successful_calls < len(operational_tool_calls):
        # Identify tools that were prepared but did not result in a successful (not is_error) execution
        failed_tool_call_ids = set(inv["id"] for inv in prepared_tool_invocations)
        for res_item in results_from_execution:
             if isinstance(res_item, dict) and not res_item.get("is_error") and res_item.get("id") in failed_tool_call_ids:
                 failed_tool_call_ids.remove(res_item.get("id"))
        
        failed_tool_names_this_turn = list(set(inv["name"] for inv in prepared_tool_invocations if inv["id"] in failed_tool_call_ids))

        if failed_tool_names_this_turn:
            current_recovery_guidance = f"Some tools failed in the last step ({', '.join(failed_tool_names_this_turn)}). Please review any errors and try rephrasing your request or correcting the problematic parts."
            logger.info(f"[ToolsNode] Generated recovery guidance: {current_recovery_guidance}")
        else:
             logger.info(f"[ToolsNode] All ({len(operational_tool_calls)}) operational tools completed successfully or had errors handled internally. No specific recovery guidance generated.")

    logger.info(f"[ToolsNode] Updating state with {len(tool_execution_results)} tool messages, {len(temp_structured_results)} new structured results.")
    
    # Prepare the dictionary for updating the state
    update_dict = {
        "messages": tool_execution_results, # These are ToolMessage objects for LangGraph
        "structured_results": state.get("structured_results", []) + temp_structured_results, # Append new structured results
        "failure_patterns": updated_failure_patterns,
        "recovery_guidance": current_recovery_guidance # Will be None if no new guidance
    }
    # Only update resolved_location_map in the state if it actually changed
    if location_map_this_turn or not state.get("resolved_location_map"): # if new items or map was previously None
        update_dict["resolved_location_map"] = final_resolved_map

    logger.info(f"[ToolsNode] Final update_dict keys before returning: {list(update_dict.keys())}")
    return update_dict

def _is_retryable_error(error: Exception) -> bool:
    """Determine if an error should be retried."""
    error_str = str(error).lower()
    if any(term in error_str for term in [
        "timeout", "connection", "network", "temporarily",
        "unavailable", "service", "busy", "rate limit",
        "too many requests", "429", "503", "504"
    ]):
        return True
    if isinstance(error, (TimeoutError, ConnectionError)): # Add specific types if needed
        return True
    # Add specific API error codes from Azure OpenAI if known
    # e.g., if "408" in error_str or isinstance(error, SpecificAzureError)
    return False

# --- Conditional Edge Logic --- #
def should_continue(state: AgentState) -> str:
    """Determines the next step based on the last message and state.
       Routes back to agent for specific, retryable SQL security errors if limit not reached.
       Routes to END for other errors or final structure.
    """
    request_id = state.get("request_id")
    messages = state.get("messages", [])
    last_message = messages[-1] if messages else None
    final_structure_in_state = state.get("final_response_structure")
    current_retry_count = state.get("sql_security_retry_count", 0)

    # Priority 1: If the final structure is already set, we end.
    if final_structure_in_state:
        # logger.debug("[ShouldContinue] Final response structure found in state. Routing to END.")
        logger.info("[ShouldContinue] Routing to END: FinalApiResponseStructure is present in state.")
        return END

    # Priority 2: Check the last message for the specific retryable SQL security error
    if isinstance(last_message, ToolMessage) and last_message.name == "execute_sql":
        is_specific_security_error = False
        try:
            content_data = json.loads(last_message.content)
            if isinstance(content_data, dict) and "error" in content_data:
                error_info = content_data["error"]
                if (isinstance(error_info, dict) and
                    error_info.get("type") == "SECURITY_ERROR" and
                    "SECURITY CHECK FAILED" in error_info.get("message", "") and
                    "organization_id" in error_info.get("message", "")):
                    is_specific_security_error = True
        except (json.JSONDecodeError, TypeError):
            logger.warning(f"[ShouldContinue] Could not parse execute_sql error message content: {last_message.content[:100]}")

        if is_specific_security_error:
            # Check retry limit
            if current_retry_count < settings.MAX_SQL_SECURITY_RETRIES:
                # logger.info(f"[ShouldContinue] Specific SQL Security error detected. Retry count {current_retry_count} < limit {settings.MAX_SQL_SECURITY_RETRIES}. Routing back to agent for retry.")
                logger.info(f"[ShouldContinue] Routing to 'agent': Specific SQL Security error detected. Retry count {current_retry_count} < limit {settings.MAX_SQL_SECURITY_RETRIES}.")
                # The agent_node should have already added guidance and incremented the counter in the state update it returned.
                return "agent"
            else:
                # logger.warning(f"[ShouldContinue] Specific SQL Security error detected, but retry limit ({settings.MAX_SQL_SECURITY_RETRIES}) reached. Routing to END.")
                logger.warning(f"[ShouldContinue] Routing to END: Specific SQL Security error detected, but retry limit ({settings.MAX_SQL_SECURITY_RETRIES}) reached.")
                # Fall through to return END below
        else:
            # Other execute_sql error OR successful execution
            # If error (but not the specific one), treat as non-retryable for this logic.
            # If success, agent needs to process the result.
            is_error = False
            try:
                content_data = json.loads(last_message.content)
                if isinstance(content_data, dict) and "error" in content_data:
                    is_error = True
            except (json.JSONDecodeError, TypeError):
                 is_error = True # Treat unparseable as error

            if is_error:
                # logger.warning(f"[ShouldContinue] Non-retryable execute_sql error detected. Routing to END.")
                logger.warning(f"[ShouldContinue] Routing to END: Non-retryable or unhandled execute_sql error detected in ToolMessage: {last_message.content[:100]}...")
                # Fall through to return END below
            else:
                 logger.debug("[ShouldContinue] Successful execute_sql Tool message found. Routing back to 'agent' to process.")
                 logger.info("[ShouldContinue] Routing to 'agent': Successful execute_sql ToolMessage found.")
                 return "agent"

    # --- [Optional] Generic recursion check ---
    # failure_patterns = state.get("failure_patterns", {})
    # ... (existing recursion check logic can be placed here if needed) ...

    # Priority 3: Analyze the last message from the agent node (AIMessage)
    if isinstance(last_message, AIMessage):
        # ... (existing logic for AIMessage: check tool calls, route to tools or END) ...
         if last_message.tool_calls:
            has_operational_calls = any(
                tc.get("name") != FinalApiResponseStructure.__name__
                for tc in last_message.tool_calls
            )
            if has_operational_calls:
                 logger.debug("[ShouldContinue] Operational tool call(s) found in AIMessage. Routing to 'tools'.")
                 logger.info("[ShouldContinue] Routing to 'tools': AIMessage with operational tool calls detected.")
                 return "tools"
            else: # Only FinalApiResponseStructure or empty tool_calls
                 logger.warning("[ShouldContinue] AIMessage has only FinalApiResponseStructure or empty tool_calls. Routing to END.")
                 logger.info("[ShouldContinue] Routing to END: AIMessage has only FinalApiResponseStructure or empty tool_calls (will be/was processed by agent/tools node).")
                 return END # Should normally be caught by final_structure_in_state check
         else:
             logger.warning("[ShouldContinue] AIMessage with no tool calls. Routing to END.")
             logger.info("[ShouldContinue] Routing to END: AIMessage with no tool calls (implies final text response).")
             return END

    # Priority 4: Handle other ToolMessages (non-SQL or non-error SQL)
    elif isinstance(last_message, ToolMessage): # Already handled sql tool messages above
        # Any other successful tool message (e.g., hierarchy_resolver) should go back to agent
        # logger.debug(f"[ShouldContinue] Successful Tool message ({last_message.name}) found. Routing back to 'agent'.")
        logger.info(f"[ShouldContinue] Routing to 'agent': ToolMessage for '{last_message.name}' found.")
        return "agent"

    # Default/Fallback: If state is unexpected, end.
    # logger.warning(f"[ShouldContinue] Unexpected state or last message type ({type(last_message).__name__}), routing to END.")
    logger.warning(f"[ShouldContinue] Routing to END: Fallback due to unexpected state or last message type ({type(last_message).__name__}).")
    return END

# --- Create LangGraph Agent ---
def create_graph_app(organization_id: str) -> StateGraph:
    """
    Create the updated LangGraph application.
    Agent node generates operational tool calls or FinalApiResponseStructure.
    Tools node executes all operational tools (including hierarchy resolver).
    """
    # LLM binding includes operational tools + FinalApiResponseStructure
    llm_with_bindings = create_llm_with_tools_and_final_response_structure(organization_id)

    # Get *only* operational tools for the handler node
    operational_tools = get_tools(organization_id)

    # Create the agent node wrapper
    agent_node_wrapper = functools.partial(agent_node, llm_with_structured_output=llm_with_bindings)

    # Create the tools node wrapper (passing ONLY operational tools)
    tools_handler_with_tools = functools.partial(async_tools_node_handler, tools=operational_tools)

    # --- Define the graph ---
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("agent", agent_node_wrapper)
    workflow.add_node("tools", tools_handler_with_tools) # Node for ALL operational tools
    workflow.add_node("analyze_results", analyze_results_node) # Add the new analysis node

    # Set the entry point
    workflow.set_entry_point("agent")

    # Define conditional edges from the agent node based on simplified should_continue
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools", # Route all operational tool calls here
            END: END
        }
    )

    # Add edge from the tools node to the analyze_results node
    workflow.add_edge("tools", "analyze_results")

    # Add edge from the analyze_results node back to the agent
    workflow.add_edge("analyze_results", "agent")

    # Compile the graph
    logger.info("Compiling LangGraph workflow...")
    graph_app = workflow.compile()
    logger.info("LangGraph workflow compiled.")
    return graph_app


# --- Refactored process_chat_message ---
async def process_chat_message(
    organization_id: str,
    message: str,
    session_id: Optional[str] = None,
    chat_history: Optional[List[Dict]] = None,
    request_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Processes a user chat message using the refactored LangGraph agent.
    Extracts the final response structure, including chart specifications embedded within it.
    Constructs the final API response, mapping chart specs to the 'visualizations' field using dedicated charting logic.
    """
    req_id = request_id or str(uuid.uuid4())
    logger.info(f"--- Starting request processing ---")
    logger.info(f"Org: {organization_id}, Session: {session_id}, History: {len(chat_history) if chat_history else 0}, User Message: '{message}'") 

    # Standard response templates
    error_response = {
        "status": "error",
        "error": {
            "code": "UNKNOWN_ERROR",
            "message": "An unexpected error occurred",
            "details": None
        },
        "data": None,
        "timestamp": datetime.datetime.now().isoformat()
    }
    
    success_response = {
        "status": "success",
        "data": {
            "text": "",
            "tables": [],
            "visualizations": []
        },
        "error": None,
        "timestamp": datetime.datetime.now().isoformat()
    }

    # Org ID validation
    try: uuid.UUID(organization_id)
    except ValueError: 
        logger.error(f"Invalid organization_id format: {organization_id}")
        error_response["error"] = {"code": "INVALID_INPUT", "message": "Invalid organization identifier.", "details": None}
        return error_response

    # History validation & Initial State construction
    initial_messages: List[BaseMessage] = []
    if chat_history: # Simplified history processing
        for item in chat_history:
            if isinstance(item, dict) and item.get("role") == "user": initial_messages.append(HumanMessage(content=item.get("content", "")))
            elif isinstance(item, dict) and item.get("role") == "assistant": initial_messages.append(AIMessage(content=item.get("content", "")))
    initial_messages.append(HumanMessage(content=message))

    # State Pruning
    if len(initial_messages) > settings.MAX_STATE_MESSAGES:
        logger.warning(f"Initial message count ({len(initial_messages)}) exceeds limit ({settings.MAX_STATE_MESSAGES}). Pruning...")
        initial_messages = initial_messages[-settings.MAX_STATE_MESSAGES:]

    # Initialize token tracking
    total_prompt_tokens = 0
    total_completion_tokens = 0

    # Initialize agent state with failure pattern tracking AND resolved_location_map
    initial_state = AgentState(
        messages=initial_messages,
        structured_results=[], # Initialize as empty list
        final_response_structure=None,
        request_id=req_id,
        prompt_tokens=0,
        completion_tokens=0,
        failure_patterns={},
        recovery_guidance=None,
        resolved_location_map=None, # Initialize as None
        missing_entities_context=None
    )
    logger.debug(f"[ProcessChatMessage] Initial state prepared: {initial_state}") 

    try:
        graph_app = create_graph_app(organization_id)
        if asyncio.iscoroutine(graph_app): graph_app = await graph_app
            
        logger.info(f"Invoking LangGraph workflow...")
        final_state = await graph_app.ainvoke(
            initial_state,
            config=RunnableConfig(recursion_limit=settings.MAX_GRAPH_ITERATIONS, configurable={})
        )
        logger.info(f"LangGraph workflow invocation complete.")
        logger.debug(f"[ProcessChatMessage] Final state received from graph: {final_state}") # ADDED LOG

        # Track token usage
        total_prompt_tokens = final_state.get("prompt_tokens", 0)
        total_completion_tokens = final_state.get("completion_tokens", 0)

        # --- Process tables based on structured_results --- #
        # --- START: Refactored Final Response Preparation --- #
        final_structure = final_state.get("final_response_structure")
        structured_results = final_state.get("structured_results", [])
        request_id = final_state.get("request_id", "N/A") # Get request_id for logging

        if not final_structure:
            logger.error(f"[ReqID: {request_id}] Final state lacks FinalApiResponseStructure. Creating fallback response.")
            final_structure = FinalApiResponseStructure(
                 text="I wasn't able to properly complete this request. Please try again later.",
                 include_tables=[],
                 chart_specs=[]
             )
            final_text = final_structure.text
            llm_chart_specs = final_structure.chart_specs # Use specs from fallback
        else:
            final_text = final_structure.text
            llm_chart_specs = final_structure.chart_specs # Use validated specs from the structure

        # --- START: Local Helper for Number Formatting --- #
        def _format_table_numbers(table_data: Dict[str, Any]) -> Dict[str, Any]:
            """Formats whole number floats (e.g., 234.0) to ints (234) in table rows."""
            # Check if input is a valid dict with 'rows'
            if not isinstance(table_data, dict) or 'rows' not in table_data or not isinstance(table_data['rows'], list):
                logger.warning(f"[ReqID: {request_id}] Invalid input to _format_table_numbers: {type(table_data)}. Returning as is.")
                return table_data
                
            formatted_table = copy.deepcopy(table_data)
            new_rows = []
            for row in formatted_table['rows']:
                if isinstance(row, list):
                    new_row = []
                    for cell in row:
                        if isinstance(cell, float) and not isinstance(cell, int) and cell.is_integer():
                            new_row.append(int(cell)) # Convert float 234.0 -> int 234
                        else:
                            new_row.append(cell) # Keep other types
                    new_rows.append(new_row)
                else:
                    new_rows.append(row) # Keep non-list rows as is
            formatted_table['rows'] = new_rows
            return formatted_table
        # --- END: Local Helper for Number Formatting --- #

        # --- Process Tables for Inclusion --- #
        formatted_tables_to_include = []
        # Use structured_results extracted from final_state
        if structured_results: 
            include_tables_flags = final_structure.include_tables
            # Ensure flags list matches results length
            if isinstance(include_tables_flags, list) and len(include_tables_flags) == len(structured_results):
                logger.debug(f"[ReqID: {request_id}] Processing {sum(include_tables_flags)} table(s) for final response based on flags.")
                for i, include in enumerate(include_tables_flags):
                    if include:
                        # Get table from the structured result item
                        original_table = structured_results[i].get("table")
                        # Basic validation of table structure
                        if not isinstance(original_table, dict) or "columns" not in original_table or "rows" not in original_table:
                            logger.warning(f"[ReqID: {request_id}] Skipping table at index {i} due to unexpected format: {type(original_table)}")
                            continue

                        table_to_process = copy.deepcopy(original_table)
                        
                        # Remove Verification ID Column before formatting
                        VERIFICATION_ID_ALIAS = "__verification_id__"
                        cleaned_table = table_to_process # Default to original if no cleaning needed
                        if VERIFICATION_ID_ALIAS in table_to_process.get("columns", []):
                            try:
                                logger.debug(f"[ReqID: {request_id}] Removing internal verification column '{VERIFICATION_ID_ALIAS}' from final output table index {i}.")
                                alias_index = table_to_process["columns"].index(VERIFICATION_ID_ALIAS)
                                
                                columns_filtered = [col for idx, col in enumerate(table_to_process["columns"]) if idx != alias_index]
                                rows_filtered = [[cell for idx, cell in enumerate(row) if idx != alias_index] 
                                                 for row in table_to_process.get("rows", []) if isinstance(row, list)] # Ensure row is a list

                                cleaned_table = {"columns": columns_filtered, "rows": rows_filtered}
                                # Preserve other metadata if it exists
                                if "metadata" in table_to_process:
                                     cleaned_table["metadata"] = table_to_process["metadata"]
                                     
                                logger.debug(f"[ReqID: {request_id}] Verification column removed successfully from table index {i}.")
                            except (ValueError, IndexError, TypeError) as e: # Added TypeError
                                logger.error(f"[ReqID: {request_id}] Error removing verification column from table index {i}: {e}. Proceeding with original table.", exc_info=True)
                                cleaned_table = table_to_process # Fallback on error

                        # Format numbers on the (potentially) cleaned table
                        formatted_table = _format_table_numbers(cleaned_table)
                        formatted_tables_to_include.append(formatted_table)
            else: # Handle flag mismatch or invalid flags
                logger.warning(f"[ReqID: {request_id}] Skipping table inclusion/formatting. Mismatch or invalid include_tables flags ({type(include_tables_flags).__name__}, length {len(include_tables_flags) if isinstance(include_tables_flags, list) else 'N/A'}) vs structured_results ({len(structured_results)}).")
        else: # No structured results in state
             logger.debug(f"[ReqID: {request_id}] No structured results found in final state, skipping table processing.")
        # --- End Table Processing ---

        # --- Process Chart Specifications using the dedicated function --- #
        validated_visualizations = [] # Initialize empty list for validated specs
        filtered_chart_info = []   # Initialize empty list for info on filtered charts
        
        if llm_chart_specs and isinstance(llm_chart_specs, list):
            try:
                # Get the list of actual tables from structured_results
                tables_for_charting = [res.get("table") for res in structured_results if isinstance(res.get("table"), dict)]
                
                # Call the validation/processing function from charting.py
                validated_visualizations, filtered_chart_info = process_and_validate_chart_specs(
                    chart_specs=llm_chart_specs, # Pass the specs from LLM
                    tables_from_state=tables_for_charting # Pass the actual tables
                )
                
                logger.debug(f"[ReqID: {request_id}] process_and_validate_chart_specs generated {len(validated_visualizations)} valid visualizations and filtered {len(filtered_chart_info)} specs.")
                if filtered_chart_info:
                     logger.warning(f"[ReqID: {request_id}] Filtered chart specs: {filtered_chart_info}")

            except Exception as chart_processing_err:
                 logger.error(f"[ReqID: {request_id}] Error during overall chart processing: {chart_processing_err}", exc_info=True)
                 validated_visualizations = [] # Ensure empty list on error
                 filtered_chart_info = []
        else:
             logger.debug(f"[ReqID: {request_id}] No chart specs found or invalid format in the final LLM response structure.")
        # --- End Chart Processing ---
        
        # Ensure final_text is a string
        final_text_str = str(final_text) if final_text is not None else "An error occurred generating the response text."


        # --- Programmatic Cleanup of Markdown Tables in Text --- #
        markdown_table_pattern = r"^\s*\|.*\|\s*$\n?|^\s*\|?-+:?-+\|?.*$\n?"
        # Check if the pattern exists before attempting removal
        if re.search(markdown_table_pattern, final_text_str, re.MULTILINE):
            logger.warning(f"[ReqID: {request_id}] Found markdown table pattern in final text. Removing it.")
            cleaned_text = re.sub(markdown_table_pattern, "", final_text_str, flags=re.MULTILINE).strip()
            final_text_str = cleaned_text if cleaned_text else "Summary generated (table data provided separately)." 

        # --- Build Final Success Response --- #
        success_response["data"] = {
            "request_id": req_id, 
            "text": final_text_str, 
            "tables": formatted_tables_to_include, 
            # model_dump() instead of deprecated dict()
            "visualizations": [vis.model_dump() for vis in validated_visualizations] 
        }

        # --- Existing logging and return --- #
        logger.info(f"Token Usage - Prompt: {total_prompt_tokens}, Completion: {total_completion_tokens}, Total: {total_prompt_tokens + total_completion_tokens}")
        usage_logger.info(json.dumps({
            "request_id": req_id,
            "organization_id": organization_id,
            "session_id": session_id,
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": total_completion_tokens,
            "total_tokens": total_prompt_tokens + total_completion_tokens,
            "query": message,
            "response_length": len(final_text_str),
            "table_count": len(formatted_tables_to_include),
            # Use the count of the *validated* visualizations
            "visualization_count": len(validated_visualizations) 
        }))
        
        logger.info("Successfully completed chat request")
        return success_response
        
    except GraphRecursionError as e:
        logger.error(f"Recursion limit exceeded: {e}")
        
        # Try to extract partial results from the state
        partial_tables = []
        failure_info = {}
        error_type = "unknown"
        
        # Access partial state if available
        if hasattr(e, "partial_state"):
            partial_state = e.partial_state
            if partial_state:
                # Extract tables for partial results
                partial_tables = [res.get("table") for res in partial_state.get("structured_results", []) if res.get("table")]
                
                # Extract failure patterns to determine error type
                failure_patterns = partial_state.get("failure_patterns", {})
                if failure_patterns:
                    # Analyze the most common failures
                    for tool_name, failures in failure_patterns.items():
                        if len(failures) >= 2:
                            error_msgs = [f.get("error_message", "") for f in failures]
                            
                            # Check for common error types
                            if any("SECURITY CHECK FAILED" in msg for msg in error_msgs):
                                error_type = "security_filter"
                            elif any("syntax" in msg.lower() for msg in error_msgs):
                                error_type = "sql_syntax"
                            
                            failure_info[tool_name] = {
                                "count": len(failures),
                                "last_error": failures[-1].get("error_message", "") if failures else ""
                            }
        
        # Generate appropriate user message based on error type
        user_message = "I wasn't able to complete this complex request due to system limitations."
        
        if error_type == "security_filter":
            user_message = "I couldn't process this complex query due to security requirements. Please try breaking your request into simpler parts, focusing on one aspect at a time."
        elif error_type == "sql_syntax":
            user_message = "I had difficulty generating a valid query for this complex request. Try asking about one metric or location at a time."
        
        # Return partial results if available
        if partial_tables:
            # Return the first available table with a partial success status
            error_response["status"] = "partial_success"
            error_response["error"] = {
                "code": "RECURSION_LIMIT_EXCEEDED",
                "message": "Request partially completed before complexity limit was reached.",
                "details": {"exception": str(e), "failure_info": failure_info}
            }
            error_response["data"] = {
                "text": f"{user_message} Here are the partial results I was able to retrieve:",
                "tables": [partial_tables[0]],  # First table only
                "visualizations": []
            }
        else:
            # No partial results
            error_response["error"] = {
                "code": "RECURSION_LIMIT_EXCEEDED",
                "message": "Request complexity limit exceeded.",
                "details": {"exception": str(e), "failure_info": failure_info}
            }
            error_response["data"] = {
                "text": user_message,
                "tables": [],
                "visualizations": []
            }
        
        logger.info("--- Finished request processing with exception ---")
        return error_response
        
    except openai.APIError as e:
        # Handle OpenAI API errors
        error_code = "LLM_SERVICE_ERROR"
        error_message = "Language model service error"
        
        if isinstance(e, openai.APIConnectionError):
            error_code = "LLM_CONNECTION_ERROR" 
            error_message = "Could not connect to the language model service."
        elif isinstance(e, openai.APITimeoutError):
            error_code = "LLM_TIMEOUT_ERROR"
            error_message = "The language model service timed out."
        elif isinstance(e, openai.RateLimitError):
            error_code = "LLM_RATE_LIMIT_ERROR"
            error_message = "The language model service rate limit was exceeded."
            
        logger.error(f"OpenAI API Error: {error_code} - {e}", exc_info=True)
        error_response["error"] = {"code": error_code, "message": error_message, "details": {"exception": str(e)}}
        logger.info("--- Finished request processing with exception ---")
        return error_response
    
    except ValueError as e:
        # Handle value errors with special case for content policy violations
        if "content management policy violation" in str(e).lower():
            logger.warning(f"Content policy violation: {e}")
            error_response["error"] = {
                "code": "CONTENT_POLICY_VIOLATION",
                "message": "I cannot process this request due to content policies.",
                "details": {"exception": str(e)}
            }
        else:
            logger.error(f"ValueError during processing: {e}", exc_info=True)
            error_response["error"] = {
                "code": "INVALID_INPUT",
                "message": "The request contained invalid input or parameters.",
                "details": {"exception": str(e)}
            }
        logger.info("--- Finished request processing with exception ---")
        return error_response
        
    except Exception as e:
        # Handle all other exceptions
        logger.error(f"Unhandled exception during processing: {e}", exc_info=True)
        error_response["error"] = {
            "code": "UNKNOWN_ERROR", 
            "message": "An unexpected error occurred while processing your request.",
            "details": {"exception": str(e)}
        }
        logger.info("--- Finished request processing with exception ---")
        return error_response


# --- test_azure_openai_connection() ---
async def test_azure_openai_connection() -> bool:
    """Test connection to Azure OpenAI API."""
    try:
        llm = get_llm()
        # Use a simple prompt that doesn't require tool calling for the test
        prompt = ChatPromptTemplate.from_template("Say hello.")
        chain = prompt | llm
        response = await chain.ainvoke({})
        # Check response content directly
        return isinstance(response, AIMessage) and bool(response.content.strip())
    except Exception as e:
        logger.warning(f"Azure OpenAI connection test failed: {str(e)}")
        return False

async def execute_with_retry(invocation_detail: Dict[str, Any]) -> Dict[str, Any]:
    tool_to_call = invocation_detail["tool"]
    tool_input_args = invocation_detail["args"]
    tool_call_id = invocation_detail["id"]
    tool_name = invocation_detail["name"]
    # Default to settings.TOOL_EXECUTION_RETRIES if not present in invocation_detail, ensure it's at least 0
    retries_left = invocation_detail.get("retries_left", getattr(settings, 'TOOL_EXECUTION_RETRIES', 0))
    
    original_args_for_logging = tool_input_args.copy() 

    attempt = 0
    last_exception = None

    # Loop for retries_left + 1 total attempts (e.g., if retries_left is 2, attempts are 0, 1, 2)
    while attempt <= retries_left:
        current_attempt_number = attempt + 1
        max_attempts = retries_left + 1
        logger.info(f"[ToolRetry] Attempt {current_attempt_number}/{max_attempts} for tool '{tool_name}' (ID: {tool_call_id}). Args: {tool_input_args}")
        try:
            tool_output = await tool_to_call.ainvoke(tool_input_args)

            content_str = ""
            if isinstance(tool_output, (dict, list)):
                content_str = json.dumps(tool_output)
            elif isinstance(tool_output, str):
                content_str = tool_output
            elif hasattr(tool_output, 'model_dump_json'): # For Pydantic models
                 content_str = tool_output.model_dump_json()
            else:
                content_str = str(tool_output)

            logger.info(f"[ToolRetry] Success for tool '{tool_name}' (ID: {tool_call_id}) on attempt {current_attempt_number}. Output (first 100 chars): {content_str[:100]}")
            return {
                "id": tool_call_id,
                "name": tool_name,
                "content_str": content_str,
                "is_error": False,
                "original_args": original_args_for_logging 
            }
        except Exception as e:
            log_level_is_debug = getattr(settings, 'LOG_LEVEL', 'INFO').upper() == 'DEBUG'
            logger.warning(f"[ToolRetry] Attempt {current_attempt_number}/{max_attempts} for tool '{tool_name}' (ID: {tool_call_id}) failed. Error: {e}", exc_info=log_level_is_debug)
            last_exception = e
            
            is_retryable = _is_retryable_error(e) # Assumes _is_retryable_error is defined elsewhere

            if attempt < retries_left: # If there are more retries left
                if is_retryable:
                    delay = getattr(settings, 'TOOL_RETRY_DELAY_SECONDS', 1) * (2**attempt) # Exponential backoff
                    logger.info(f"[ToolRetry] Retryable error for '{tool_name}'. Retrying in {delay:.2f} seconds...")
                    await asyncio.sleep(delay)
                else:
                    logger.warning(f"[ToolRetry] Non-retryable error for '{tool_name}' (ID: {tool_call_id}). Stopping retries immediately.")
                    break # Break from while loop, don't retry non-retryable errors
            else: # This was the last attempt
                logger.info(f"[ToolRetry] Last attempt for '{tool_name}' (ID: {tool_call_id}) failed.")
                break # Break loop as no more retries left
            
            attempt += 1
            
    # If all retries failed or a non-retryable error occurred and broke the loop
    # --- Enhanced Error Message Extraction ---
    detailed_error_content = str(last_exception) # Default message
    error_type_str = "TOOL_EXECUTION_FAILED"    # Default type

    current_exception_for_details = last_exception
    # Traverse the cause chain to get the most specific DB-related error message
    # Limit depth to avoid infinite loops in rare cases, though __cause__ chain should be finite
    # Based on logs, the direct __cause__ of ValueError from sql_tool is the SQLAlchemy error.
    potential_cause = getattr(current_exception_for_details, '__cause__', None)
    if potential_cause:
        # Attempt to get a string representation of the cause.
        # SQLAlchemy errors often have good string representations.
        specific_cause_message = str(potential_cause)
        detailed_error_content = specific_cause_message # Prioritize cause message

        # Refine error type based on the cause
        cause_type_name = type(potential_cause).__name__
        if "UndefinedColumnError" in cause_type_name or \
           ("column" in detailed_error_content.lower() and "does not exist" in detailed_error_content.lower()):
            error_type_str = "DATABASE_UNDEFINED_COLUMN_ERROR"
        elif "ProgrammingError" in cause_type_name: # Catches other SQL programming issues
            error_type_str = "DATABASE_PROGRAMMING_ERROR"
        elif "SyntaxError" in cause_type_name or "syntax error" in detailed_error_content.lower():
             error_type_str = "DATABASE_SYNTAX_ERROR"
        # Add more specific SQLAlchemy or DB error types as needed

    # If not overridden by a more specific DB error type, check for other common types
    if error_type_str == "TOOL_EXECUTION_FAILED":
        if isinstance(last_exception, HTTPException): # from FastAPI, not pydantic.errors
            error_type_str = "HTTP_EXCEPTION_IN_TOOL"
        elif isinstance(last_exception, ValidationError): # from Pydantic
            error_type_str = "VALIDATION_ERROR_IN_TOOL"
        elif isinstance(last_exception, TimeoutError) or isinstance(last_exception, APITimeoutError):
            error_type_str = "TIMEOUT_ERROR_IN_TOOL"
        elif isinstance(last_exception, APIConnectionError):
            error_type_str = "CONNECTION_ERROR_IN_TOOL"
        elif isinstance(last_exception, RateLimitError):
            error_type_str = "RATE_LIMIT_ERROR_IN_TOOL"
    # --- End Enhanced Error Message Extraction ---

    # Prepare messages for logging and for the LLM
    final_llm_error_message = f"Tool '{tool_name}' failed. Error details: {detailed_error_content}"
    
    log_message_summary = f"Tool '{tool_name}' (ID: {tool_call_id}) failed. Type: {error_type_str}."
    if attempt >= retries_left : # exhausted retries
        log_message_summary = f"Tool '{tool_name}' (ID: {tool_call_id}) failed after {retries_left + 1} attempts. Type: {error_type_str}."
    
    logger.error(f"{log_message_summary} Full Error: {str(last_exception)}", exc_info=True if getattr(settings, 'LOG_LEVEL', 'INFO').upper() == 'DEBUG' else False)

    return {
        "id": tool_call_id,
        "name": tool_name,
        "content_str": json.dumps({
            "error": {
                "type": error_type_str,
                "message": final_llm_error_message, # More detailed message for LLM
            }
        }),
        "is_error": True,
        "original_args": original_args_for_logging
    }