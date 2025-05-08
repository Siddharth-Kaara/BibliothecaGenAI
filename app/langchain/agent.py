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
       This now includes the chart specifications directly.
    """
    text: str = Field(description="The final natural language text response for the user. Follow the guidelines in the system prompt for generating this text (e.g., brief introductions if data/charts are present).")
    include_tables: List[bool] = Field(
        description="List of booleans indicating which tables from the agent state should be included in the final API response. Match the order of tables in the state.",
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
    # Add field to store the latest successful name resolution results (across turns within request)
    resolved_location_map: Optional[Dict[str, str]] = None
    # Add field for context about missing entities (generated by analyze_results_node)
    missing_entities_context: Optional[str] = None
    # Add counter for specific SQL security retries
    sql_security_retry_count: Annotated[int, operator.add] = 0
    # --- ADDED: Cache for hierarchy resolution results within a single request --- #
    request_hierarchy_cache: Optional[Dict[str, Dict[str, Any]]] = None


# --- NAnalysis Node to Check for Missing Entities ---
def analyze_results_node(state: AgentState) -> Dict[str, Any]:
    """Analyzes structured results (table + filters) against resolved entities (using IDs). 
       Determines missing entities by checking if queries filtering by specific IDs returned rows containing meaningful data.
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

    # --- Analyze each structured result --- #
    for result in structured_results:
        filters = result.get("filters", {})
        table = result.get("table", {})
        columns = table.get("columns", [])
        rows = table.get("rows", [])
        
        # 1. Find all resolved IDs used in *this* result's filters
        ids_filtered_in_this_query = set()
        names_filtered_in_this_query = set() # Keep track for logging/context
        
        # --- Logic to find ids_filtered_in_this_query (unchanged) ---
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
        # --- End logic to find ids_filtered_in_this_query ---
                           
        if not ids_filtered_in_this_query:
            # This query didn't filter by any specific resolved entities, skip further checks for it
            logger.debug(f"[AnalyzeResultsNode-Final] Query with filters {filters} did not filter by resolved IDs. Skipping row check.")
            continue

        # Add these filtered IDs to the set of all expected IDs across the whole request
        all_expected_ids.update(ids_filtered_in_this_query)
        logger.debug(f"[AnalyzeResultsNode-Final] Query filtered by IDs: {ids_filtered_in_this_query} (Names: {names_filtered_in_this_query}) ")

        # 2. Check if rows were returned for this query AND if they contain meaningful data
        if not rows:
            logger.debug(f"[AnalyzeResultsNode-Final] Query filtering by {ids_filtered_in_this_query} returned 0 rows. None marked as found.")
            continue # No rows, so none of these IDs were found in this result
        
        # Check if rows contain any actual data beyond just IDs or all nulls
        has_meaningful_data = False
        # Attempt to identify non-ID columns to check for meaningful data
        id_column_names = {col_name for col_name in ["id", "hierarchyId", "hierarchy_id", "location_id", "branch_id", "organizationId", "eventSrc"] if col_name in columns}
        value_column_indices = [i for i, col_name in enumerate(columns) if col_name not in id_column_names]

        if not value_column_indices: # If all columns are considered ID-like, check all non-None
            logger.debug(f"[AnalyzeResultsNode-Final] No distinct value columns identified for IDs {ids_filtered_in_this_query}. Checking all cells for non-null data.")
            for row_idx, row_content in enumerate(rows):
                if not isinstance(row_content, list):
                    logger.warning(f"[AnalyzeResultsNode-Final] Row {row_idx} is not a list: {row_content}. Skipping for meaningful data check.")
                    continue
                if any(cell is not None for cell in row_content):
                    has_meaningful_data = True
                    break
        else:
            logger.debug(f"[AnalyzeResultsNode-Final] Checking value columns (indices: {value_column_indices}) for IDs {ids_filtered_in_this_query} for non-null data.")
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
            continue # Do not add to found_ids if no meaningful data

        # 3. SIMPLIFIED: If meaningful rows were returned for a query filtering by specific IDs, assume data for ALL those IDs was found.
        logger.debug(f"[AnalyzeResultsNode-Final] Query filtering by IDs {ids_filtered_in_this_query} returned rows with meaningful data. Marking all as found.")
        found_ids.update(ids_filtered_in_this_query)
        # --- REMOVED complex ID/name checking logic within rows --- 

    # --- Generate final context (Unchanged) --- 
    if not all_expected_ids:
        logger.debug("[AnalyzeResultsNode-Final] No resolved entity IDs were used in any filters.")
        return {"missing_entities_context": None}

    logger.debug(f"[AnalyzeResultsNode-Final] Total Expected IDs across all filters: {all_expected_ids}")
    logger.debug(f"[AnalyzeResultsNode-Final] IDs confirmed present in returned data rows: {found_ids}")

    missing_ids = all_expected_ids - found_ids
    
    if missing_ids:
        # Map missing IDs back to their original names using resolved_id_to_name_map
        # Safely get names, use ID as fallback if name somehow missing from reverse map
        missing_names = [resolved_id_to_name_map.get(mid, f"ID:{mid}") for mid in missing_ids]
        missing_names_str = ", ".join(sorted(list(set(missing_names))))
        
        missing_entities_context = (
            f"IMPORTANT CONTEXT: The user asked for data related to specific entities (including {missing_names_str}). "
            f"While queries were executed filtering for these, no meaningful data rows were returned specifically corresponding to {missing_names_str}. "
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

    # Create a copy for preprocessing
    preprocessed_state = _preprocess_state_for_llm(state)

    # Check failure patterns specifically for execute_sql
    sql_failures = failure_patterns.get("execute_sql", [])
    if sql_failures: # Check only if there are any SQL failures
        last_sql_failure = sql_failures[-1] # Get the most recent one
        is_specific_security_error = (
            "SECURITY CHECK FAILED" in last_sql_failure.get("error_message", "") and
            "organization_id" in last_sql_failure.get("error_message", "")
        )

        if is_specific_security_error:
            current_retry_count = state.get("sql_security_retry_count", 0)
            # Check if retry limit allows another attempt
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
                logger.info(f"[AgentNode] Adding SQL security recovery guidance for retry #{current_retry_count + 1}")
                retry_increment = 1 # Set flag to increment the counter in the return dict
            else:
                 logger.warning(f"[AgentNode] SQL security error detected, but retry limit ({settings.MAX_SQL_SECURITY_RETRIES}) reached. Will not generate recovery guidance.")
                 # No guidance, should_continue will see the error and route to END

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
                        logger.warning(f"[AgentNode] Discarded functionally duplicate operational tool call (ID: {discarded_tc.get('id')})")

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
                logger.info("[AgentNode] LLM AIMessage has NO tool_calls. Coercing content into FinalApiResponseStructure.")
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
                                    logger.debug(f"[AgentNode] Successfully cleaned AIMessage representation from coerced text.")
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

    if 'messages' in processed_state and len(processed_state['messages']) > max_messages:
        original_messages = processed_state['messages']
        logger.debug(f"Starting pruning messages from {len(original_messages)} down to target ~{max_messages}")

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
                            logger.warning(f"Discarding {len(temp_tool_messages)} ToolMessage(s) orphaned before AIMessage(TC) (ID: {msg.id}).")
                            temp_tool_messages = [] # Clear orphans before preserving AIMessage(TC)
                        logger.debug(f"Identified orphaned AIMessage(TC) (ID: {msg.id}) - ToolMessages potentially missing/pruned.")
                else:
                     # Simple AIMessage, no tool calls. Any temp_tool_messages are orphans.
                     if temp_tool_messages:
                        logger.warning(f"Discarding {len(temp_tool_messages)} ToolMessage(s) orphaned before simple AIMessage (ID: {msg.id}).")
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
                    logger.warning(f"Discarding {len(temp_tool_messages)} ToolMessage(s) orphaned before HumanMessage.")
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
        logger.debug(f"Final pruned message count: {len(processed_state['messages'])}. Structure preserved: {[(type(m).__name__ + ('(TC)' if isinstance(m, AIMessage) and m.tool_calls else '')) for m in processed_state['messages']]}")

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
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]: # Return type changed to indicate error
    """
    Processes SQL parameters to:
    1. Ensure the correct 'organization_id' is present and matches the trusted ID.
    2. Resolve string parameter values that appear to be names into their corresponding UUIDs
       using the 'resolved_map'.
    3. Identify and flag (or reject) unresolved placeholder values.
    Returns a tuple: (processed_params_dict, None) on success,
                     (None, error_message_string) on critical failure.
    """
    if params is None:
        logger.debug(f"{log_prefix} Original params dictionary is None. Initializing with trusted organization_id.")
        return {"organization_id": current_organization_id}, None

    processed_params = params.copy() # Work on a copy

    # 1. Enforce trusted organization_id
    llm_org_id = processed_params.get("organization_id")
    if llm_org_id != current_organization_id:
        logger.warning(
            f"{log_prefix} LLM provided organization_id '{llm_org_id}' does not match trusted ID '{current_organization_id}' or was missing. Overwriting/injecting trusted ID."
        )
        processed_params["organization_id"] = current_organization_id
    else:
        logger.debug(f"{log_prefix} Confirmed organization_id '{current_organization_id}' from LLM matches trusted ID or was correctly set.")

    if not resolved_map:
        logger.debug(f"{log_prefix} resolved_location_map is None or empty. Skipping name-to-ID substitution for other parameters.")
        # Still check for unresolved placeholders even if map is empty
        for key, value in list(processed_params.items()):
            if isinstance(value, str) and value.startswith("<resolved_") and value.endswith("_id>"):
                 error_msg = f"{log_prefix} CRITICAL_UNRESOLVED_PLACEHOLDER: Param '{key}' has placeholder value '{value}' but resolved_map is unavailable. Query cannot proceed."
                 logger.error(error_msg)
                 return None, error_msg # Critical failure
        return processed_params, None

    logger.debug(f"{log_prefix} Attempting name/placeholder to ID substitution using resolved_map: {resolved_map}")

    for key, value in list(processed_params.items()): # Iterate over a copy for safe modification
        if key == "organization_id": # Already handled, skip.
            continue

        logger.debug(f"{log_prefix} Processing param - Key: '{key}', Value: '{value}', Type: {type(value)}")
        substituted_value = False

        if isinstance(value, str):
            # Attempt to resolve if the value itself is a name (key in resolved_map)
            # Perform case-insensitive lookup for the value in resolved_map keys
            value_lower = value.lower()
            found_match_in_map = False
            resolved_uuid = None

            for map_key_original_case, map_id in resolved_map.items():
                if map_key_original_case.lower() == value_lower:
                    resolved_uuid = map_id
                    found_match_in_map = True
                    logger.info(f"{log_prefix} SUCCESS_NAME_RESOLUTION: Param '{key}' value '{value}' (matched as '{map_key_original_case}') resolved to ID '{resolved_uuid}'.")
                    processed_params[key] = resolved_uuid
                    substituted_value = True
                    break
            
            if found_match_in_map:
                continue # Value substituted, move to next parameter

            # If not directly resolved as a name, check if it's a placeholder like '<resolved_name_id>'
            if value.startswith("<resolved_") and value.endswith("_id>"):
                logger.debug(f"{log_prefix} Param '{key}' has placeholder value '{value}'. Attempting extraction and lookup.")
                placeholder_content_match = re.match(r"<resolved_([a-zA-Z0-9_.-]+)_id>", value)
                
                if placeholder_content_match:
                    extracted_key_from_placeholder = placeholder_content_match.group(1).lower()
                    logger.debug(f"{log_prefix} Extracted key '{extracted_key_from_placeholder}' from placeholder '{value}'.")
                    
                    # Look up the extracted key in resolved_map (case-insensitive)
                    found_placeholder_in_map = False
                    for map_key_original_case, map_id in resolved_map.items():
                        if map_key_original_case.lower() == extracted_key_from_placeholder:
                            resolved_uuid_for_placeholder = map_id
                            processed_params[key] = resolved_uuid_for_placeholder
                            substituted_value = True
                            found_placeholder_in_map = True
                            logger.info(f"{log_prefix} SUCCESS_PLACEHOLDER_RESOLUTION: Placeholder '{value}' for param '{key}' (extracted key '{extracted_key_from_placeholder}' matched as '{map_key_original_case}') resolved to ID '{resolved_uuid_for_placeholder}'.")
                            break
                    
                    if not found_placeholder_in_map:
                        error_msg = f"{log_prefix} CRITICAL_UNRESOLVED_PLACEHOLDER: Param '{key}' ('{value}'), extracted key '{extracted_key_from_placeholder}' NOT FOUND in resolved_map: {list(resolved_map.keys())}. Query cannot proceed."
                        logger.error(error_msg)
                        return None, error_msg # Critical failure
                else:
                    logger.warning(f"{log_prefix} Param '{key}' ('{value}') looks like a placeholder but regex did not match content for key extraction.")
            # Fallback check: if the parameter KEY itself (e.g. "argyle_branch_param") matches a key in resolved_map
            elif key.lower() in resolved_map: # Check original key (lowercase)
                # This case implies the LLM might have set `param_name_expecting_id = "some_actual_id_guid"`
                # but if `value` was not a guid, and key.lower() is in resolved_map, it means the key was a name.
                # This logic might be redundant if the first check (value is a name) is robust.
                # Let's assume for now that if `value` wasn't resolved, but `key.lower()` is in resolved_map,
                # it implies LLM used a name as a key and we should have resolved its value above.
                # So, this block might not be hit if `value` as a name was handled.
                # If we reach here and `substituted_value` is false, it means `value` itself was not a resolvable name or placeholder.
                pass # Covered by initial value check or placeholder check.

        if not substituted_value and isinstance(value, str) and (value.startswith("<") or value.endswith(">")): # General check for other unresolved-looking strings
             logger.warning(f"{log_prefix} UNRESOLVED_SUSPICIOUS_VALUE: Param '{key}' has value '{value}' which was not substituted and looks like a placeholder or unresolved name. Query may fail.")

    logger.debug(f"{log_prefix} Final processed params after name/placeholder resolution: {processed_params}")
    return processed_params, None

# --- Tool Node Handler ---
async def async_tools_node_handler(state: AgentState, tools: List[Any]) -> Dict[str, Any]:
    request_id = state.get("request_id")
    logger.debug(f"[ToolsNode] Entering tool handler.")
    last_message = state["messages"][-1] if state["messages"] else None
    tool_map = {tool.name: tool for tool in tools}
    tool_execution_results = []
    operational_tool_calls = []
    location_map_this_turn = {} 
    # --- MODIFIED: Initialize from state --- #
    request_hierarchy_cache_this_turn = state.get("request_hierarchy_cache", {}).copy() if state.get("request_hierarchy_cache") is not None else {}

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
            "resolved_location_map": state.get("resolved_location_map"),
            # --- ADDED: Return existing cache if no tools run --- #
            "request_hierarchy_cache": state.get("request_hierarchy_cache")
        }

    logger.debug(f"[ToolsNode] Dispatching {len(operational_tool_calls)} operational tool calls: {[tc.get('name') for tc in operational_tool_calls]}")

    prepared_tool_invocations = []
    skipped_tool_calls_due_to_cache = [] # Track skipped calls

    for tool_call in operational_tool_calls:
        tool_name = tool_call.get("name")
        tool_id = tool_call.get("id", f"tool_call_{uuid.uuid4()}")
        raw_tool_args = tool_call.get("args", {})
        
        # --- ADDED: Hierarchy Resolver Cache Check --- #
        if tool_name == "hierarchy_name_resolver":
            name_candidates = raw_tool_args.get("name_candidates", [])
            cached_results = {}
            all_cached = True
            if not isinstance(name_candidates, list):
                logger.warning(f"[ToolsNode] hierarchy_name_resolver (ID: {tool_id}) received non-list candidates: {type(name_candidates)}. Skipping cache check.")
                all_cached = False # Cannot use cache if input is invalid
            else:
                for name in name_candidates:
                    # Normalize cache key lookup (lowercase)
                    cache_key = name.lower()
                    if cache_key in request_hierarchy_cache_this_turn:
                        cached_results[name] = request_hierarchy_cache_this_turn[cache_key]
                    else:
                        all_cached = False
                        break # No need to check further if one is missing
            
            if all_cached and name_candidates:
                logger.info(f"[ToolsNode] CACHE HIT: All {len(name_candidates)} candidates for hierarchy_name_resolver (ID: {tool_id}) found in request cache. Skipping tool execution.")
                # Construct the ToolMessage content from the cache
                # The tool normally returns {'resolution_results': { 'Original Name': { ... details ... }}}
                cache_content = {"resolution_results": cached_results}
                tool_execution_results.append(ToolMessage(
                    content=json.dumps(cache_content),
                    name=tool_name,
                    tool_call_id=tool_id
                ))
                # Update location_map_this_turn directly from cache
                for name, res_data in cached_results.items():
                    if isinstance(res_data, dict) and res_data.get("status") == "found" and "id" in res_data:
                        location_map_this_turn[name.lower()] = res_data["id"]
                # Add to skipped list and continue to next tool call in operational_tool_calls
                skipped_tool_calls_due_to_cache.append(tool_id)
                continue 
            else:
                logger.debug(f"[ToolsNode] CACHE MISS/PARTIAL: hierarchy_name_resolver (ID: {tool_id}) needs execution for some/all candidates: {name_candidates}")
        # --- END Hierarchy Resolver Cache Check --- #

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
            
            # _preprocess_sql_params now returns a tuple (params_or_none, error_message_or_none)
            params_after_preprocessing, preprocess_error_msg = _preprocess_sql_params(
                params=current_sql_params,
                resolved_map=state.get("resolved_location_map"),
                current_organization_id=trusted_org_id_for_tool
            )

            if preprocess_error_msg:
                logger.error(f"[ToolsNode] Parameter preprocessing failed for tool '{tool_name}' (ID: {tool_id}). Error: {preprocess_error_msg}")
                error_content = json.dumps({"error": {"type": "PARAMETER_RESOLUTION_ERROR", "message": preprocess_error_msg}})
                tool_execution_results.append(ToolMessage(content=error_content, name=tool_name, tool_call_id=tool_id))
                continue # Skip this tool call, proceed to the next one in operational_tool_calls
            
            processed_tool_args["params"] = params_after_preprocessing # Update args with successfully preprocessed params
            
            # 2. Apply SQL string literal correction (name -> ID in SQL string itself) 
            # THIS SECTION IS REMOVED
            # current_sql_string = processed_tool_args.get("sql", "")
            # if state.get("resolved_location_map") and current_sql_string: 
            #     logger.debug(f"[ToolsNode] Applying SQL string literal correction for 'execute_sql' (ID: {tool_id}). SQL before: ...{current_sql_string[-100:]}")
            #     modified_sql, final_params_after_sql_rewrite = _apply_resolved_ids_to_sql_args(
            #         sql=current_sql_string, 
            #         params=preprocessed_params, # Pass the already param-preprocessed params
            #         resolved_map=state.get("resolved_location_map")
            #     )
            #     processed_tool_args["sql"] = modified_sql
            #     processed_tool_args["params"] = final_params_after_sql_rewrite 
            #     logger.debug(f"[ToolsNode] Final SQL for call {tool_id} (tail): ...{modified_sql[-200:] if modified_sql else '[EMPTY SQL]'}. Final Params: {final_params_after_sql_rewrite}")
            # else:
            #     logger.debug(f"[ToolsNode] Skipping SQL string literal correction for 'execute_sql' (ID: {tool_id}) (no resolved_map, empty SQL, or already handled). Params remain: {processed_tool_args.get('params')}")
            # End of removal

            logger.debug(f"[ToolsNode] SQL for call {tool_id} (tail): ...{processed_tool_args.get('sql', '')[-200:]}. Final Params after preprocessing: {processed_tool_args.get('params')}")

        # End of execute_sql specific processing

        # Only add if not skipped by cache
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
    # --- ADDED: Keep track of successful tool call IDs for final check ---
    successful_tool_call_ids = set()

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

        # Append ToolMessage (already done above if skipped, do it here only if executed)
        # Check if this ID was skipped by cache
        if tool_call_id_from_res not in skipped_tool_calls_due_to_cache:
             tool_execution_results.append(ToolMessage(content=tool_content_str, name=tool_name_from_res, tool_call_id=tool_call_id_from_res))

        if not is_error:
            # --- Success path ---
            successful_tool_call_ids.add(tool_call_id_from_res) # Mark as successful

            try:
                tool_content_dict = json.loads(tool_content_str)
                if tool_name_from_res == "hierarchy_name_resolver":
                    # --- ADDED: Update request cache --- #
                    current_resolved_map_from_tool = tool_content_dict.get("resolution_results", {})
                    for original_name, res_data in current_resolved_map_from_tool.items():
                        if isinstance(res_data, dict):
                            # Store using original name as key in cache for reconstruction
                            # But use lowercase name for lookup consistency
                            request_hierarchy_cache_this_turn[original_name.lower()] = res_data 
                            # Update location map for current turn if found
                            if res_data.get("status") == "found" and "id" in res_data:
                                location_map_this_turn[original_name.lower()] = res_data["id"]
                    if current_resolved_map_from_tool:
                         logger.info(f"[ToolsNode] Updated resolved location map (this turn) with {len(location_map_this_turn)} entries AND updated request_hierarchy_cache from '{tool_name_from_res}'.")
                    # --- END Update request cache --- #
                
                elif tool_name_from_res == "execute_sql":
                    # Check if the tool itself returned a structured error (e.g., DB connection, permission)
                    if tool_content_dict.get("error"):
                        logger.warning(f"[ToolsNode] Tool '{tool_name_from_res}' (ID: {tool_call_id_from_res}) executed but returned a structured error: {tool_content_dict.get('error')}")
                        is_error = True # Treat as error for failure pattern logging below

                    # --- MODIFIED: Only add structured result if NO error AND rows ARE present ---
                    elif tool_content_dict.get("columns") is not None and tool_content_dict.get("rows") is not None:
                        # Check if rows list is actually populated
                        if tool_content_dict["rows"]: 
                            table_data = {
                                "columns": tool_content_dict["columns"],
                                "rows": tool_content_dict["rows"],
                                "text": tool_content_dict.get("text") 
                            }
                            sql_filters_used = original_args_for_tool.get("params", {})
                            temp_structured_results.append({"table": table_data, "filters": sql_filters_used})
                            logger.info(f"[ToolsNode] Created structured result from successful '{tool_name_from_res}' call (ID: {tool_call_id_from_res}) with {len(table_data['rows'])} rows. Filters used: {sql_filters_used}")
                        else:
                             # Successful execution but returned 0 rows - DO NOT add to structured_results
                             logger.info(f"[ToolsNode] Successful '{tool_name_from_res}' call (ID: {tool_call_id_from_res}) returned 0 rows. Not adding to structured_results.")
                    else:
                         logger.warning(f"[ToolsNode] Successful '{tool_name_from_res}' call (ID: {tool_call_id_from_res}) returned parsable JSON, but no expected table structure or error field found. Keys: {list(tool_content_dict.keys())}")
                         is_error = True # Treat unexpected success format as error

            except json.JSONDecodeError:
                logger.warning(f"[ToolsNode] Failed to decode JSON from nominally successful tool call '{tool_name_from_res}' (ID: {tool_call_id_from_res}): {tool_content_str[:150]}...")
                is_error = True 
            except Exception as e_struct_proc: 
                logger.error(f"[ToolsNode] Error processing successful result from tool '{tool_name_from_res}' (ID: {tool_call_id_from_res}): {e_struct_proc}", exc_info=True)
                is_error = True
        
        # --- Error path (includes errors reported by tool OR processing errors) ---
        if is_error:
            # Log failure pattern (existing logic - unchanged)
            error_info_dict = {}
            try: 
                # Try to parse the content string again if it wasn't already parsed or if it's an error string
                parsed_content_for_error = json.loads(tool_content_str)
                error_info_dict = parsed_content_for_error.get("error", {}) if isinstance(parsed_content_for_error, dict) else {}
            except: # If content_str is not valid JSON or doesn't contain "error"
                pass # error_info_dict remains empty or has its prior state
            
            failure_type = error_info_dict.get("type", "UNKNOWN_TOOL_RUNTIME_ERROR")
            failure_message = error_info_dict.get("message", tool_content_str) # Fallback to full content if no message
            
            # --- ADDED: Specific recovery guidance for unresolved placeholders --- #
            if (tool_name_from_res == "execute_sql" and 
                failure_type == "PARAMETER_RESOLUTION_ERROR" and 
                "CRITICAL_UNRESOLVED_PLACEHOLDER" in failure_message):
                placeholder_guidance = "One or more SQL queries failed because location names were not yet resolved to IDs. Ensure 'hierarchy_name_resolver' is called first for any specific location names before attempting SQL that requires their IDs."
                if current_recovery_guidance:
                    # Check if this specific guidance isn't already present to avoid duplicates in a single turn
                    if placeholder_guidance not in current_recovery_guidance:
                        current_recovery_guidance += f" \n{placeholder_guidance}"
                else:
                    current_recovery_guidance = placeholder_guidance
                logger.info(f"[ToolsNode] Added specific recovery guidance for unresolved SQL placeholders for tool call ID {tool_call_id_from_res}.")
            # --- END ADDED --- #

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
            # --- Ensure successful_calls doesn't increment on error ---
            if tool_call_id_from_res in successful_tool_call_ids:
                 successful_tool_call_ids.remove(tool_call_id_from_res)
                 
    # End of loop processing results_from_execution

    # Combine resolved locations from this turn with existing ones from the state
    final_resolved_map = state.get("resolved_location_map", {}).copy() if state.get("resolved_location_map") else {}
    if location_map_this_turn: # Only update if new resolutions occurred this turn
        final_resolved_map.update(location_map_this_turn)
        logger.info(f"[ToolsNode] Final resolved_location_map contains {len(final_resolved_map)} entries after this turn.")

    # --- Generate Recovery Guidance (logic slightly adjusted) ---
    # Check if any prepared calls are NOT in the set of successful calls
    prepared_call_ids = set(inv["id"] for inv in prepared_tool_invocations)
    failed_tool_call_ids_this_turn = prepared_call_ids - successful_tool_call_ids

    if failed_tool_call_ids_this_turn: # If there were failures this turn
        failed_tool_names_this_turn = list(set(inv["name"] for inv in prepared_tool_invocations if inv["id"] in failed_tool_call_ids_this_turn))
        if failed_tool_names_this_turn:
            current_recovery_guidance = f"Some tools failed in the last step ({', '.join(failed_tool_names_this_turn)}). Please review any errors and try rephrasing your request or correcting the problematic parts."
            logger.info(f"[ToolsNode] Generated recovery guidance: {current_recovery_guidance}")
        else: # Should not happen if failed_tool_call_ids_this_turn is populated
             logger.error("[ToolsNode] Internal logic error: Failed tool IDs found but could not map back to names.")
    else:
        logger.info(f"[ToolsNode] All ({len(prepared_tool_invocations)}) operational tools completed successfully (or returned 0 rows intentionally). No specific recovery guidance generated.")

    logger.info(f"[ToolsNode] Updating state with {len(tool_execution_results)} tool messages, {len(temp_structured_results)} new structured results.")
    
    # Prepare the dictionary for updating the state
    update_dict = {
        "messages": tool_execution_results, 
        "structured_results": state.get("structured_results", []) + temp_structured_results, 
        "failure_patterns": updated_failure_patterns,
        "recovery_guidance": current_recovery_guidance, 
        # --- ADDED: Persist updated cache --- #
        "request_hierarchy_cache": request_hierarchy_cache_this_turn
    }
    # Only update resolved_location_map in the state if it actually changed
    if location_map_this_turn or not state.get("resolved_location_map"): 
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
        logger.debug("[ShouldContinue] Final response structure found in state. Routing to END.")
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
                logger.info(f"[ShouldContinue] Specific SQL Security error detected. Retry count {current_retry_count} < limit {settings.MAX_SQL_SECURITY_RETRIES}. Routing back to agent for retry.")
                # The agent_node should have already added guidance and incremented the counter in the state update it returned.
                return "agent"
            else:
                logger.warning(f"[ShouldContinue] Specific SQL Security error detected, but retry limit ({settings.MAX_SQL_SECURITY_RETRIES}) reached. Routing to END.")
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
                logger.warning(f"[ShouldContinue] Non-retryable execute_sql error detected. Routing to END.")
                # Fall through to return END below
            else:
                 logger.debug("[ShouldContinue] Successful execute_sql Tool message found. Routing back to 'agent' to process.")
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
                 return "tools"
            else: # Only FinalApiResponseStructure or empty tool_calls
                 logger.warning("[ShouldContinue] AIMessage has only FinalApiResponseStructure or empty tool_calls. Routing to END.")
                 return END # Should normally be caught by final_structure_in_state check
         else:
             logger.warning("[ShouldContinue] AIMessage with no tool calls. Routing to END.")
             return END

    # Priority 4: Handle other ToolMessages (non-SQL or non-error SQL)
    elif isinstance(last_message, ToolMessage): # Already handled sql tool messages above
        # Any other successful tool message (e.g., hierarchy_resolver) should go back to agent
        logger.debug(f"[ShouldContinue] Successful Tool message ({last_message.name}) found. Routing back to 'agent'.")
        return "agent"

    # Default/Fallback: If state is unexpected, end.
    logger.warning(f"[ShouldContinue] Unexpected state or last message type ({type(last_message).__name__}), routing to END.")
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

    # Initialize agent state
    initial_state = AgentState(
        messages=initial_messages,
        structured_results=[], 
        final_response_structure=None,
        request_id=req_id,
        prompt_tokens=0,
        completion_tokens=0,
        failure_patterns={},
        recovery_guidance=None,
        resolved_location_map=None, 
        missing_entities_context=None,
        # --- ADDED: Initialize request cache --- #
        request_hierarchy_cache={}
    )
    logger.debug(f"[ProcessChatMessage] Initial state prepared (incl. hierarchy cache): {initial_state}") 

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
        # --- End Chart Processing --- #
        
        # Ensure final_text is a string
        final_text_str = str(final_text) if final_text is not None else "An error occurred generating the response text."

        # --- Programmatic Cleanup of Markdown Tables in Text --- #
        # Regex to find markdown table lines (simplified)
        markdown_table_pattern = r"^\s*\|.*\|\s*$\n?|^\s*\|?-+:?-+\|?.*$\n?"
        # Check if the pattern exists before attempting removal
        if re.search(markdown_table_pattern, final_text_str, re.MULTILINE):
            logger.warning(f"[ReqID: {request_id}] Found markdown table pattern in final text. Removing it.")
            cleaned_text = re.sub(markdown_table_pattern, "", final_text_str, flags=re.MULTILINE).strip()
            # Add a note if text was cleaned (optional, but helpful for debugging)
            # cleaned_text += "\n(Note: Table data is presented in the structured 'tables' field.)" 
            final_text_str = cleaned_text if cleaned_text else "Summary generated (table data provided separately)." # Fallback if cleaning removes everything
        # --- END Cleanup --- #

        # --- Build Final Success Response --- #
        success_response["data"] = {
            "request_id": req_id, 
            "text": final_text_str, # Use the potentially cleaned text
            "tables": formatted_tables_to_include, 
            "visualizations": [vis.model_dump() for vis in validated_visualizations] 
        }
        
        # --- END: Final Response Preparation --- #

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
    error_message_detail = f"Tool '{tool_name}' (ID: {tool_call_id}) failed. Last error: {str(last_exception)}"
    if attempt > retries_left : # exhausted retries
        error_message_detail = f"Tool '{tool_name}' (ID: {tool_call_id}) failed after {retries_left + 1} attempts. Last error: {str(last_exception)}"
    
    logger.error(error_message_detail)
    
    error_type = "TOOL_EXECUTION_FAILED"
    if isinstance(last_exception, HTTPException):
        error_type = "HTTP_EXCEPTION_IN_TOOL"
    elif isinstance(last_exception, ValidationError):
         error_type = "VALIDATION_ERROR_IN_TOOL"
    # Consider adding more specific error types based on common exceptions from your tools

    return {
        "id": tool_call_id,
        "name": tool_name,
        "content_str": json.dumps({"error": {"type": error_type, "message": error_message_detail, "details": str(last_exception)}}),
        "is_error": True,
        "original_args": original_args_for_logging
    }