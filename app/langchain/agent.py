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
def _get_schema_string(db_name: str = "report_management") -> str:
    """Gets schema information from predefined schema definitions as a formatted string."""
    from app.db.schema_definitions import SCHEMA_DEFINITIONS 
    logger.debug(f"[_get_schema_string] Fetching schema for database: {db_name}")
    
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
    
    logger.debug(f"[_get_schema_string] Successfully retrieved schema for {db_name}")
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
    # Add field to store the latest successful name resolution results
    resolved_location_map: Optional[Dict[str, str]] = None
    # Add field for context about missing entities (generated by analyze_results_node)
    missing_entities_context: Optional[str] = None


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

        # Fallback to standard ID column names
        id_col_candidates = ["id", "hierarchyId", "hierarchy_id", "location_id", "branch_id"]
        for idx, col_name in enumerate(columns):
            if col_name in id_col_candidates:
                logger.debug(f"[AnalyzeResultsNode-Final] Found ID column by candidate name: '{col_name}'")
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
        for key, value in filters.items():
            if isinstance(value, str) and value in resolved_id_to_name_map:
                ids_filtered_in_this_query.add(value)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, str) and item in resolved_id_to_name_map:
                        ids_filtered_in_this_query.add(item)
            elif 'id' in key.lower() or key.startswith('res_id_'): 
                 potential_ids = [value] if isinstance(value, str) else value if isinstance(value, list) else []
                 for item in potential_ids:
                      if isinstance(item, str) and item in resolved_id_to_name_map:
                           ids_filtered_in_this_query.add(item)
                           
        if not ids_filtered_in_this_query:
            # This query didn't filter by any specific resolved entities, skip further checks for it
            logger.debug(f"[AnalyzeResultsNode-Final] Query with filters {filters} did not filter by resolved IDs. Skipping row check.")
            continue

        # Add these filtered IDs to the set of all expected IDs across the whole request
        all_expected_ids.update(ids_filtered_in_this_query)
        logger.debug(f"[AnalyzeResultsNode-Final] Query filtered by IDs: {ids_filtered_in_this_query}")

        # 2. Check if rows were returned for this query
        if not rows:
            logger.debug(f"[AnalyzeResultsNode-Final] Query filtering by {ids_filtered_in_this_query} returned 0 rows. None marked as found.")
            continue # No rows, so none of these IDs were found in this result
            
        # 3. Determine which of the filtered IDs were actually present in the returned rows
        ids_present_in_rows = set()
        if len(ids_filtered_in_this_query) == 1:
            # If only one ID was filtered, and rows were returned, that ID is considered found
            ids_present_in_rows = ids_filtered_in_this_query
            logger.debug(f"[AnalyzeResultsNode-Final] Query filtered by single ID {ids_filtered_in_this_query} returned rows. Marking as found.")
        else:
            # Multiple IDs were filtered (likely IN clause). Need to check the result table.
            id_col_index = find_id_column_index(columns)
            if id_col_index is not None:
                logger.debug(f"[AnalyzeResultsNode-Final] Checking ID column (Index: {id_col_index}, Name: {columns[id_col_index]}) for IDs {ids_filtered_in_this_query}...")
                for row in rows:
                    if isinstance(row, list) and len(row) > id_col_index:
                        cell_value = row[id_col_index]
                        cell_value_str = str(cell_value) if isinstance(cell_value, uuid.UUID) else cell_value
                        if isinstance(cell_value_str, str) and cell_value_str in ids_filtered_in_this_query:
                            ids_present_in_rows.add(cell_value_str)
                logger.debug(f"[AnalyzeResultsNode-Final] IDs actually present in rows for this multi-ID query: {ids_present_in_rows}")
            else:
                # Cannot verify which IDs are present if ID column is missing from results
                logger.warning(f"[AnalyzeResultsNode-Final] Query filtered by multiple IDs {ids_filtered_in_this_query}, but no ID column found in results {columns}. Cannot confirm which IDs were found.")
                # Potential Strategy: Assume all filtered IDs were found if rows were returned? Risky.
                # Safer Strategy: Mark none as found if verification isn't possible.
                ids_present_in_rows = set() # Mark none as found if we can't check

        # Add the IDs confirmed present in rows to the overall 'found_ids' set
        found_ids.update(ids_present_in_rows)

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
    """Generates a signature for an execute_sql call, ignoring ORDER BY and LIMIT clauses."""
    sql = args.get("sql", "")
    params = args.get("params", {})
    
    # Attempt to remove ORDER BY and LIMIT clauses from the end of the main query
    # This regex looks for ORDER BY or LIMIT at the end, possibly preceded/followed by whitespace/semicolon
    # It handles potential trailing whitespace and semicolons more robustly.
    # We use re.IGNORECASE as SQL keywords aren't always consistently cased.
    core_sql = re.sub(r'\s*(?:ORDER\s+BY\s+.*?|LIMIT\s+\d+)\s*;?\s*$', '', sql, flags=re.IGNORECASE | re.DOTALL).strip()
    # Fallback if regex fails or removes too much (simple safety check)
    if not core_sql or len(core_sql) < 10: # Arbitrary short length check
        core_sql = sql # Use original if regex result seems invalid
        
    # Create a canonical representation of parameters
    params_signature = json.dumps(params, sort_keys=True)
    
    return (core_sql, params_signature)

def agent_node(state: AgentState, llm_with_structured_output):
    """Invokes the LLM ONCE to decide the next action or final response structure.
       If the LLM returns a plain AIMessage, it coerces it into FinalApiResponseStructure.
    """
    request_id = state.get("request_id")
    logger.debug(f"[AgentNode] Entering agent node (single invocation logic)...")

    llm_response: Optional[AIMessage] = None
    final_structure: Optional[FinalApiResponseStructure] = None
    operational_calls = []
    return_dict: Dict[str, Any] = {
        "messages": [],
        "final_response_structure": None,
        "prompt_tokens": 0, # Initialize, will be updated if call succeeds
        "completion_tokens": 0,
        "resolved_location_map": state.get("resolved_location_map") # Carry over map if exists
    }

    # Parser for the final response structure
    final_response_parser = PydanticToolsParser(tools=[FinalApiResponseStructure])

    # --- Adaptive Error Analysis: Check for recurring failure patterns ---
    failure_patterns = state.get("failure_patterns", {})
    recovery_guidance = None
    
    # Create a copy for preprocessing to avoid modifying the original state
    preprocessed_state = _preprocess_state_for_llm(state)
    
    # Check if we have recurring patterns of the same failure
    if failure_patterns:
        for tool_name, failures in failure_patterns.items():
            # Require at least 2 failures to trigger adaptation
            if len(failures) >= 2:
                # Extract error messages to detect patterns
                error_messages = [f.get("error_message", "") for f in failures[-2:]]
                
                # Check for SQL organization_id security filter failures
                if tool_name == "execute_sql" and any("SECURITY CHECK FAILED" in msg and "organization_id" in msg for msg in error_messages):
                    recovery_guidance = """
                    CRITICAL SQL CORRECTION INSTRUCTION:
                    
                    Your previous SQL queries have failed the security check because they're missing required organization_id filtering.
                    
                    The security system checks EACH SQL component SEPARATELY:
                    1. Every CTE (WITH clause) must include: WHERE "organizationId" = :organization_id
                    2. Every subquery must include: WHERE "organizationId" = :organization_id 
                    3. The main query must include: WHERE "organizationId" = :organization_id
                    
                    Example of CORRECT organization_id filtering in a complex query:
                    ```sql
                    WITH branch_stats AS (
                        SELECT "hierarchyId", SUM("1") AS total_borrows
                        FROM "5"
                        WHERE "organizationId" = :organization_id  -- REQUIRED HERE
                          AND "eventTimestamp" >= NOW() - INTERVAL '30 days'
                        GROUP BY "hierarchyId"
                    )
                    SELECT hc."name" AS "Branch Name", bs.total_borrows AS "Total Borrows"
                    FROM branch_stats bs
                    JOIN "hierarchyCaches" hc ON bs."hierarchyId" = hc."id"
                    WHERE hc."parentId" = :organization_id  -- REQUIRED HERE TOO
                    LIMIT 10;
                    ```
                    
                    If a complex query structure continues to fail:
                    1. Try a simpler query without CTEs
                    2. Break it into multiple separate queries
                    3. Ensure EVERY table reference has organization_id filtering
                    """
                    logger.info(f"[AgentNode] Adding SQL security recovery guidance after {len(failures)} consecutive failures")
                    break
                
                # Check for SQL syntax failures
                elif tool_name == "execute_sql" and any("syntax" in msg.lower() for msg in error_messages):
                    recovery_guidance = """
                    CRITICAL SQL SYNTAX CORRECTION NEEDED:
                    
                    Your previous SQL queries have syntax errors. Please:
                    1. Verify all table and column names are properly double-quoted
                    2. Ensure all CTEs are properly terminated
                    3. Check that all parentheses are balanced
                    4. Simplify the query structure if needed
                    
                    If syntax issues persist, try a completely different query approach.
                    """
                    logger.info(f"[AgentNode] Adding SQL syntax recovery guidance after {len(failures)} syntax failures")
                    break
                
                # Generic recovery for 3+ failures of any type
                elif len(failures) >= 3:
                    recovery_guidance = """
                    CRITICAL STRATEGY CHANGE REQUIRED:
                    
                    Multiple tool calls have failed consecutively. Change your approach:
                    
                    1. Use a significantly different strategy than previous attempts
                    2. Simplify your approach - focus on the most essential part of the request
                    3. If a complex query is failing, try multiple simpler queries instead
                    
                    If this attempt also fails, return a helpful error message explaining the limitations.
                    """
                    logger.info(f"[AgentNode] Adding generic recovery guidance after {len(failures)} consecutive failures")
                    break
    
    # If we've identified recovery guidance, insert it into the system message
    if recovery_guidance:
        messages = preprocessed_state.get("messages", [])
        system_message_found = False
        
        # Find and update system message if it exists
        for i, msg in enumerate(messages):
            if isinstance(msg, SystemMessage):
                # Append recovery guidance to the existing system message
                new_content = f"{msg.content}\n\n{recovery_guidance}"
                messages[i] = SystemMessage(content=new_content)
                system_message_found = True
                break
        
        # If no system message was found, add one at the beginning
        if not system_message_found and messages:
            messages.insert(0, SystemMessage(content=recovery_guidance))
        
        # Update the preprocessed state
        preprocessed_state["messages"] = messages
        
        # Preserve recovery guidance for state tracking
        return_dict["recovery_guidance"] = recovery_guidance
    # --- End Adaptive Error Analysis ---

    try:
        # --- ADDED: Inject the missing_entities_context at invocation time --- #
        missing_context = state.get("missing_entities_context") or "" # Get context or empty string
        invocation_input = {**preprocessed_state, "missing_entities_context": missing_context}
        logger.debug(f"[AgentNode] Invoking LLM with state (incl. missing context): {invocation_input}") # UPDATED LOG
        llm_response = llm_with_structured_output.invoke(invocation_input) # USE invocation_input
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
    return return_dict


def _preprocess_state_for_llm(state: AgentState) -> AgentState:
    """
    Preprocess the state to ensure it's optimized for LLM context window.
    This helps prevent issues with the LLM failing due to context limitations.
    """
    processed_state = {k: v for k, v in state.items()}
    
    # Handle messages pruning 
    if 'messages' in processed_state and len(processed_state['messages']) > settings.MAX_STATE_MESSAGES:
        messages = processed_state['messages']
        logger.debug(f"Pruning messages from {len(messages)} to {settings.MAX_STATE_MESSAGES}")
        # Keep the first system message if present
        system_msg = next((msg for msg in messages if isinstance(msg, SystemMessage)), None)
        first_user_msg = next((msg for msg in messages if isinstance(msg, HumanMessage)), None)
        last_user_msg = next((msg for msg in reversed(messages) if isinstance(msg, HumanMessage)), None)
        tool_messages = [msg for msg in messages if isinstance(msg, ToolMessage)]
        recent_tool_messages = tool_messages[-5:]
        ai_messages = [msg for msg in messages if isinstance(msg, AIMessage)]
        recent_ai_messages = ai_messages[-3:]
        keep_messages = set()
        if system_msg: keep_messages.add(system_msg)
        if first_user_msg: keep_messages.add(first_user_msg)
        if last_user_msg: keep_messages.add(last_user_msg)
        for msg in recent_tool_messages: keep_messages.add(msg)
        for msg in recent_ai_messages: keep_messages.add(msg)
        preserved_messages: List[BaseMessage] = []
        added_ids = set()
        for msg in messages:
             if msg in keep_messages and id(msg) not in added_ids:
                 preserved_messages.append(msg)
                 added_ids.add(id(msg))
        if len(preserved_messages) > settings.MAX_STATE_MESSAGES:
            start_index = 1 if system_msg else 0
            cutoff = len(preserved_messages) - settings.MAX_STATE_MESSAGES
            preserved_messages = ([preserved_messages[0]] if system_msg else []) + preserved_messages[start_index + cutoff:]
        processed_state['messages'] = preserved_messages
        logger.debug(f"After pruning, kept {len(processed_state['messages'])} messages")
    
    # Process tables - limit large ones
    if 'structured_results' in processed_state and processed_state['structured_results']:
        original_structured_results = state['structured_results'] # Get original for row count
        processed_structured_results = []
        for i, result in enumerate(processed_state['structured_results']):
            processed_result = result.copy() # Shallow copy is fine, we only modify table
            if 'table' in processed_result and isinstance(processed_result['table'], dict):
                processed_table = processed_result['table'].copy() # Copy table dict
                max_rows = settings.MAX_TABLE_ROWS_IN_STATE 
                if 'rows' in processed_table and len(processed_table['rows']) > max_rows:
                    logger.debug(f"Truncating table index {i} rows from {len(processed_table['rows'])} to {max_rows}")
                    processed_table['rows'] = processed_table['rows'][:max_rows]
                    if 'metadata' not in processed_table: processed_table['metadata'] = {}
                    processed_table['metadata']['truncated'] = True
                    # Get original row count from corresponding original result
                    original_table = original_structured_results[i].get('table', {})
                    processed_table['metadata']['original_rows'] = len(original_table.get('rows', []))
                # Update the table within the copied result dict
                processed_result['table'] = processed_table
            # Append the processed result (with potentially truncated table) to the new list
            processed_structured_results.append(processed_result)
        # Replace the old list with the processed one
        processed_state['structured_results'] = processed_structured_results

    return processed_state

# --- Helper Function to Apply Resolved IDs --- #
def _apply_resolved_ids_to_sql_args(sql: str, params: Dict[str, Any], resolved_map: Dict[str, str]) -> Tuple[str, Dict[str, Any]]:
    """
    Replaces name-based filters in SQL with ID-based filters using a resolved map.
    Uses sqlparse for robust identification and replacement of name filters.

    Args:
        sql: The original SQL query string.
        params: The original parameters dictionary.
        resolved_map: A dictionary mapping lowercase location names to resolved UUIDs.

    Returns:
        A tuple containing the modified SQL string and the updated parameters dictionary.
    """
    if not resolved_map:
        return sql, params # No map, nothing to do

    logger.debug(f"[SQL Correction] Applying resolved IDs using sqlparse. Map: {resolved_map}")
    updated_params = params.copy()
    new_params_added: Dict[str, str] = {}
    param_counter = 0
    modified = False

    try:
        parsed = sqlparse.parse(sql)
        if not parsed:
            logger.warning("[SQL Correction] Failed to parse SQL with sqlparse. Returning original.")
            return sql, params
        
        stmt = parsed[0] # Assume single statement

        # --- Recursive Token Traversal --- #
        def traverse_and_replace(token_list: List[sqlparse.sql.Token]):
            nonlocal param_counter, modified
            i = 0
            while i < len(token_list):
                token = token_list[i]
                replacement_made = False # Flag to check if token at index i was replaced
                
                # Look for Comparison tokens (e.g., col = 'value', col IN ('v1', 'v2'))
                if isinstance(token, sqlparse.sql.Comparison):
                    left_identifier = None
                    operator_token = None
                    right_operand = None
                    
                    # Simple structure check: Identifier Operator Operand
                    comp_tokens = [t for t in token.flatten() if not t.is_whitespace]
                    if len(comp_tokens) >= 3 and isinstance(comp_tokens[0], sqlparse.sql.Identifier):
                        left_identifier = comp_tokens[0]
                        operator_token = comp_tokens[1]
                        right_operand = comp_tokens[2] # This could be Identifier, Parenthesis, etc.
                    
                    if left_identifier and operator_token and right_operand:
                        # Check if left operand is a name column (e.g., hc."name")
                        left_str = str(left_identifier).lower()
                        if left_str.endswith('."name"'): # Simple check for alias."name"
                            alias = left_str.split('.')[0] # Extract alias (e.g., hc)
                            names_to_resolve = []
                            new_param_dict = {}

                            # Check operator and extract name(s)
                            if operator_token.value == '=' and right_operand.ttype is sqlparse.tokens.String.Single:
                                names_to_resolve = [right_operand.value.strip("'")]
                            elif operator_token.value.upper() == 'IN' and isinstance(right_operand, sqlparse.sql.Parenthesis):
                                # Extract names from IN list ('Name1', 'Name2', ...)
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
                                        logger.warning(f"[SQL Correction] Name '{name}' found in SQL filter but not in resolved map. Skipping.")

                                if resolved_ids:
                                    # Construct replacement SQL segment tokens
                                    id_column_token = sqlparse.sql.Identifier(f'{alias}."id"') # Use captured alias
                                    new_comparison_tokens = []

                                    if len(resolved_ids) == 1:
                                        op_token = sqlparse.sql.Token(sqlparse.tokens.Operator, '=')
                                        param_token = sqlparse.sql.Token(sqlparse.tokens.Name.Placeholder, param_names_for_sql[0])
                                        new_comparison_tokens = [id_column_token, 
                                                                 sqlparse.sql.Token(sqlparse.tokens.Whitespace, ' '), 
                                                                 op_token, 
                                                                 sqlparse.sql.Token(sqlparse.tokens.Whitespace, ' '), 
                                                                 param_token]
                                    else: # IN clause
                                        op_token = sqlparse.sql.Token(sqlparse.tokens.Keyword, 'IN')
                                        # Create parameter tokens list for Parenthesis
                                        param_tokens_inner = []
                                        for idx, param_name in enumerate(param_names_for_sql):
                                             param_tokens_inner.append(sqlparse.sql.Token(sqlparse.tokens.Name.Placeholder, param_name))
                                             if idx < len(param_names_for_sql) - 1:
                                                 param_tokens_inner.append(sqlparse.sql.Token(sqlparse.tokens.Punctuation, ','))
                                                 param_tokens_inner.append(sqlparse.sql.Token(sqlparse.tokens.Whitespace, ' '))
                                        parenthesis = sqlparse.sql.Parenthesis(param_tokens_inner)
                                        
                                        new_comparison_tokens = [id_column_token, 
                                                                 sqlparse.sql.Token(sqlparse.tokens.Whitespace, ' '), 
                                                                 op_token,
                                                                 sqlparse.sql.Token(sqlparse.tokens.Whitespace, ' '), 
                                                                 parenthesis]
                                    
                                    # Replace the original comparison token (token at index i)
                                    token_list[i:i+1] = new_comparison_tokens # Replace slice
                                    replacement_made = True
                                    modified = True
                                    new_params_added.update(new_param_dict)
                                    logger.info(f"[SQL Correction] Replaced name filter '{str(token).strip()}' with ID filter.")
                                    # Adjust index because we potentially replaced 1 token with multiple
                                    i += len(new_comparison_tokens) - 1 
                                else:
                                    logger.warning(f"[SQL Correction] Could not resolve any names in filter: '{str(token).strip()}'. Keeping original.")
                
                # Recursively process nested structures like Parenthesis, Function calls etc.
                if not replacement_made and hasattr(token, 'tokens'):
                    traverse_and_replace(token.tokens)
                
                i += 1 # Move to next token
        # --- End Recursive Traversal --- #
        
        # Start traversal from the top-level statement tokens
        traverse_and_replace(stmt.tokens)

    except Exception as e:
        logger.error(f"[SQL Correction] Error during sqlparse processing: {e}. Returning original SQL.", exc_info=True)
        return sql, params

    if modified:
        updated_params.update(new_params_added)
        modified_sql = str(stmt)
        logger.info(f"[SQL Correction] Applied ID filters using sqlparse. Modified SQL tail: ...{modified_sql[-200:]}. Added params: {new_params_added}")
        return modified_sql, updated_params
    else:
        logger.debug("[SQL Correction] No name filters found matching resolved map keys using sqlparse.")
        return sql, params

# --- Tool Node Handler ---
async def async_tools_node_handler(state: AgentState, tools: List[Any]) -> Dict[str, Any]:
    """
    Enhanced asynchronous tools node handler. Executes OPERATIONAL tools only.
    Corrects execute_sql calls using resolved IDs if available in state.
    Adds resulting tables to the state and stores resolved ID map.
    """
    request_id = state.get("request_id")
    logger.debug(f"[ToolsNode] Entering tool handler.")
    last_message = state["messages"][-1] if state["messages"] else None
    tool_map = {tool.name: tool for tool in tools}
    tool_executions = []
    successful_calls = 0
    operational_tool_calls = []

    # Initialize structure to hold map resolved ONLY in this turn
    location_map_this_turn = {} # Initialize map for THIS turn's results

    # Filter out non-operational "tool calls" (Final API Structure)
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        operational_tool_calls = [
            tc for tc in last_message.tool_calls
            if tc.get("name") != FinalApiResponseStructure.__name__
        ]

    if not operational_tool_calls:
        logger.debug(f"[ToolsNode] No operational tool calls found in last message.")
        return {
            "messages": [],
            "structured_results": state.get("structured_results", {}),
            "failure_patterns": state.get("failure_patterns", {}),
            "recovery_guidance": state.get("recovery_guidance"),
        }

    logger.debug(f"[ToolsNode] Dispatching {len(operational_tool_calls)} operational tool calls: {[tc.get('name') for tc in operational_tool_calls]}")

    # Prepare tool execution details
    for tool_call in operational_tool_calls:
        tool_name = tool_call.get("name")
        tool_id = tool_call.get("id", "")
        tool_args = tool_call.get("args", {})
        if tool_name in tool_map:
            tool_executions.append({
                "tool": tool_map[tool_name],
                "args": tool_args,
                "id": tool_id,
                "name": tool_name,
                "retries_left": settings.TOOL_EXECUTION_RETRIES
            })
        else:
             logger.error(f"[ToolsNode] Operational tool '{tool_name}' requested but not found.")
             tool_executions.append(ToolMessage(
                content=f"Error: Tool '{tool_name}' not found.",
                tool_call_id=tool_id,
                name=tool_name
             ))

    if not tool_executions:
        logger.warning(f"[ToolsNode] No operational tools could be prepared for execution.")
        return {
            "messages": [ToolMessage(content="Error: No operational tools could be prepared for execution.", tool_call_id="", name="")],
            "structured_results": state.get("structured_results", {}),
            "failure_patterns": state.get("failure_patterns", {}),
            "recovery_guidance": state.get("recovery_guidance"),
        }

    sem = asyncio.Semaphore(settings.MAX_CONCURRENT_TOOLS)

    # --- Helper Function to Augment SQL --- #
    def _augment_sql_for_verification(sql: str, id_keys_in_params: List[str]) -> Tuple[str, bool]:
        """Adds a special ID column to SELECT and GROUP BY if needed for verification, using sqlparse more robustly."""
        try:
            parsed = sqlparse.parse(sql)
            if not parsed:
                logger.warning("[SQL Augment] Failed to parse SQL. Skipping augmentation.")
                return sql, False
            
            stmt = parsed[0] # Assume single statement
            id_column_found = False
            verification_alias_found = False
            target_id_column_expr = 'hc."id"' # Default target
            verification_alias = '"__verification_id__"' # Alias to add

            select_list_tokens: Optional[sqlparse.sql.TokenList] = None
            group_by_clause_tokens: Optional[sqlparse.sql.TokenList] = None

            # --- Find SELECT list and GROUP BY clause using sqlparse types --- #
            # Find the IdentifierList directly following the SELECT keyword
            select_keyword_found = False
            for token in stmt.tokens:
                if token.ttype is sqlparse.tokens.DML and token.value.upper() == 'SELECT':
                    select_keyword_found = True
                    continue # Move to the next token, which should be the list
                if select_keyword_found and isinstance(token, sqlparse.sql.IdentifierList):
                    select_list_tokens = token
                    break # Found the select list
                # Handle case where SELECT list isn't explicitly an IdentifierList (e.g., SELECT *)    
                elif select_keyword_found and token.ttype is not sqlparse.tokens.Whitespace:
                     # This might be a single identifier or wildcard, treat it as the list endpoint
                     logger.debug(f"[SQL Augment] Treating token '{token.value}' as SELECT list.")
                     select_list_tokens = token # Use the token itself as marker
                     break

            # Find the GROUP BY clause
            for token in reversed(stmt.tokens):
                if isinstance(token, sqlparse.sql.Where): # Stop searching if we hit WHERE going backwards
                     break
                if token.ttype is sqlparse.tokens.Keyword and token.value.upper() == 'GROUP BY':
                    # Find the actual list following GROUP BY
                    idx = stmt.token_index(token)
                    if idx + 1 < len(stmt.tokens):
                        # Skip whitespace
                        next_token_idx = idx + 1
                        while next_token_idx < len(stmt.tokens) and stmt.tokens[next_token_idx].is_whitespace:
                            next_token_idx += 1
                        if next_token_idx < len(stmt.tokens):
                            group_by_clause_tokens = stmt.tokens[next_token_idx]
                            logger.debug(f"[SQL Augment] Found GROUP BY clause tokens: {group_by_clause_tokens}")
                    break

            if select_list_tokens is None:
                logger.warning("[SQL Augment] Could not reliably identify SELECT list using sqlparse. Skipping augmentation.")
                return sql, False
            # --- End Finding Clauses --- #

            # --- Check if augmentation is needed --- #
            select_list_str = str(select_list_tokens).lower()
            if verification_alias.lower().strip('"') in select_list_str:
                logger.debug(f"[SQL Augment] Verification alias '{verification_alias}' already present. No augmentation needed.")
                return sql, False
            if target_id_column_expr.lower() in select_list_str:
                logger.debug(f"[SQL Augment] Target ID column ({target_id_column_expr}) found in SELECT. No augmentation needed.")
                return sql, False
            # --- End Check --- #

            # --- Augmentation Needed --- #
            logger.info(f"[SQL Augment] Augmenting SQL to include verification ID column ({target_id_column_expr}).")
            
            # 1. Add to SELECT list using token insertion
            # Create the new token sequence: comma, whitespace, identifier AS alias
            select_addition_tokens = [
                sqlparse.sql.Token(sqlparse.tokens.Punctuation, ','),
                sqlparse.sql.Token(sqlparse.tokens.Whitespace, ' '),
                sqlparse.sql.Identifier(sqlparse.parse(f'{target_id_column_expr} AS {verification_alias}')[0].tokens) # Parse to get Identifier structure
            ]
            
            # Insert before the FROM keyword, handled by sqlparse structure
            # Find the last non-whitespace token within the select list or immediately after if it's not an IdentifierList
            if isinstance(select_list_tokens, sqlparse.sql.TokenList):
                select_list_tokens.tokens.extend(select_addition_tokens)
            elif isinstance(select_list_tokens, sqlparse.sql.Token):
                 # If select list was just a single token (like *), insert AFTER it
                 stmt.insert_after(select_list_tokens, select_addition_tokens)
            else:
                 # Fallback/Error: Shouldn't happen if check above passed
                 logger.error("[SQL Augment] Unexpected type for select_list_tokens during insertion. Aborting.")
                 return sql, False

            logger.debug(f"[SQL Augment] Modified SELECT list.")

            # 2. Add to GROUP BY if it exists
            if group_by_clause_tokens is not None:
                # Check if the target ID column is already in the GROUP BY list
                group_by_str = str(group_by_clause_tokens).lower()
                if target_id_column_expr.lower() not in group_by_str:
                    # Create the new token sequence: comma, whitespace, identifier
                    groupby_addition_tokens = [
                        sqlparse.sql.Token(sqlparse.tokens.Punctuation, ','),
                        sqlparse.sql.Token(sqlparse.tokens.Whitespace, ' '),
                        sqlparse.sql.Identifier(sqlparse.parse(target_id_column_expr)[0].tokens) # Just the column
                    ]
                    # Append to the existing GROUP BY list
                    if isinstance(group_by_clause_tokens, sqlparse.sql.IdentifierList):
                         group_by_clause_tokens.tokens.extend(groupby_addition_tokens)
                    elif isinstance(group_by_clause_tokens, sqlparse.sql.Identifier):
                         # If GROUP BY was a single identifier, create a list and replace the original token
                         original_group_by_token = group_by_clause_tokens
                         new_group_by_list = sqlparse.sql.IdentifierList([original_group_by_token] + groupby_addition_tokens)
                         try:
                             original_token_idx = stmt.token_index(original_group_by_token)
                             # Replace the original Identifier token with the new IdentifierList
                             stmt.tokens[original_token_idx] = new_group_by_list
                             logger.debug(f"[SQL Augment] Replaced single GROUP BY token with list.")
                         except ValueError:
                              logger.error("[SQL Augment] Failed to find index of original single GROUP BY token. Aborting GROUP BY modification.")
                        
                    else:
                         logger.warning(f"[SQL Augment] Unexpected token type for GROUP BY clause: {type(group_by_clause_tokens)}. Attempting simple append.")
                         # Less ideal fallback: find token index and insert after
                         group_by_idx = stmt.token_index(group_by_clause_tokens)
                         stmt.insert_at(group_by_idx + 1, groupby_addition_tokens)
                         
                    logger.debug(f"[SQL Augment] Added {target_id_column_expr} to GROUP BY clause.")
                else:
                     logger.debug(f"[SQL Augment] Target ID column {target_id_column_expr} already in GROUP BY.")

            # Return the modified SQL by converting the statement back to string
            augmented_sql = str(stmt)
            logger.info(f"[SQL Augment] SQL Augmented successfully. New SQL tail: ...{augmented_sql[-250:]}")
            return augmented_sql, True

        except Exception as e:
            logger.error(f"[SQL Augment] Error during SQL augmentation: {e}", exc_info=True)
            return sql, False # Return original on error
    # --- END Helper Function --- #

    # Define async execution function with retry logic
    async def execute_with_retry(execution_details):
        tool_name = execution_details["name"]
        args = execution_details["args"]
        current_resolved_map = state.get("resolved_location_map") # Get map from CURRENT state
        # Initialize augmented flag
        sql_was_augmented = False 

        # --- Apply SQL Correction for resolved name literals (if map exists) --- #
        if tool_name == "execute_sql" and current_resolved_map:
            logger.debug(f"[ToolsNode] Checking SQL args for potential ID correction using map: {current_resolved_map}")
            original_sql = args.get("sql")
            original_params = args.get("params")
            if original_sql and original_params:
                try:
                    corrected_sql, corrected_params = _apply_resolved_ids_to_sql_args(
                        original_sql, original_params, current_resolved_map
                    )
                    # Update args ONLY if changes were made
                    if corrected_sql != original_sql or corrected_params != original_params:
                        execution_details["args"] = {"sql": corrected_sql, "params": corrected_params}
                        logger.info(f"[ToolsNode] Corrected SQL arguments for name literals for tool ID {execution_details['id']}")
                        args = execution_details["args"] # Use corrected args going forward
                except Exception as correction_err:
                     logger.error(f"[ToolsNode] Error applying SQL name literal correction for tool {tool_name}: {correction_err}", exc_info=True)
                     # Proceed with original args on correction error
        # --- End SQL Correction for name literals --- #

        # --- Apply SQL Augmentation for Verification ID --- #
        if tool_name == "execute_sql" and current_resolved_map:
            sql_to_check = args.get("sql")
            params_to_check = args.get("params", {})
            resolved_ids_in_params = set()
            id_keys_for_augmentation = [] # Store keys like :argyle_id, :beaches_id

            if sql_to_check:
                resolved_id_values = set(current_resolved_map.values()) # Get all potential IDs
                for key, value in params_to_check.items():
                    if isinstance(value, str) and value in resolved_id_values:
                        resolved_ids_in_params.add(value)
                        id_keys_for_augmentation.append(key) # Store the param key
                    elif isinstance(value, list):
                        for item in value:
                             if isinstance(item, str) and item in resolved_id_values:
                                 resolved_ids_in_params.add(item)
                                 # We don't store list keys currently for augmentation logic trigger
                    # Also check if the key itself implies an ID (e.g., argyle_id)
                    elif isinstance(value, str) and key.endswith('_id') and value in resolved_id_values:
                         resolved_ids_in_params.add(value)
                         id_keys_for_augmentation.append(key)
            
            # Check if MULTIPLE distinct resolved IDs are being filtered
            if len(resolved_ids_in_params) > 1:
                 logger.debug(f"[ToolsNode] Multiple resolved IDs ({resolved_ids_in_params}) found in params for tool {tool_name}. Checking if augmentation needed.")
                 try:
                     augmented_sql, sql_was_augmented = _augment_sql_for_verification(sql_to_check, id_keys_for_augmentation)
                     if sql_was_augmented:
                          execution_details["args"]["sql"] = augmented_sql
                          logger.info(f"[ToolsNode] Augmented SQL for verification for tool ID {execution_details['id']}")
                          args = execution_details["args"] # Use augmented args
                 except Exception as augment_err:
                      logger.error(f"[ToolsNode] Error applying SQL verification augmentation for tool {tool_name}: {augment_err}", exc_info=True)
            else:
                logger.debug(f"[ToolsNode] Augmentation not needed (found {len(resolved_ids_in_params)} resolved IDs in params). Proceeding with original/corrected SQL.")
        # --- End SQL Augmentation for Verification ID --- #


        tool = execution_details["tool"]
        tool_id = execution_details["id"]
        retries = execution_details["retries_left"]
        try:
            async with sem:
                 logger.debug(f"[ToolsNode] Executing tool '{tool_name}' (ID: {tool_id}) with args: {args}")
                 content = await tool.ainvoke(args)
                 if asyncio.iscoroutine(content): content = await content
                 content_str = json.dumps(content, default=str) if isinstance(content, (dict, list)) else str(content)
                 logger.debug(f"[ToolsNode] Tool '{tool_name}' (ID: {tool_id}) completed.")
                 return {
                     "success": True,
                     "message": ToolMessage(content=content_str, tool_call_id=tool_id, name=tool_name),
                     "raw_content": content,
                     "tool_name": tool_name,
                     "tool_id": tool_id,
                     "args": args
                 }
        except Exception as e:
            error_msg = str(e)
            error_msg_for_log = f"Error executing tool '{tool_name}' (ID: {tool_id}): {error_msg}"
            if retries > 0 and _is_retryable_error(e):
                retry_num = settings.TOOL_EXECUTION_RETRIES - retries + 1
                delay = settings.TOOL_RETRY_DELAY * retry_num
                logger.warning(f"[ToolsNode] {error_msg_for_log} - Retrying ({retries} left, delay {delay}s).", exc_info=False)
                await asyncio.sleep(delay)
                execution_details["retries_left"] = retries - 1
                return await execute_with_retry(execution_details)
            else:
                logger.error(f"[ToolsNode] {error_msg_for_log}", exc_info=True)
                return {
                    "success": False,
                    "error": f"{error_msg_for_log}",
                    "error_message": error_msg,
                    "tool_name": tool_name,
                    "tool_id": tool_id,
                    "args": args
                }

    results = await asyncio.gather(*[execute_with_retry(exec_data) for exec_data in tool_executions])

    # --- Result processing: Extract tables and potential new resolved map ---
    new_messages = [] # Initialize lists for THIS turn's results here
    turn_structured_results = [] # NEW: Store structured {table, filters} for this turn
    # Copy failure patterns from input state to potentially update them
    failure_patterns = state.get("failure_patterns", {}).copy() # <<< Copy failure patterns here
    # Initialize filters for THIS turn's successful SQL calls
    successful_sql_filters_this_turn = {}

    for result in results:
        tool_name = result.get("tool_name", "unknown")
        tool_id = result.get("tool_id", "")
        tool_args = result.get("args", {})

        if result["success"]:
            successful_calls += 1
            new_messages.append(result["message"])
            raw_content = result.get("raw_content")
            table_data = None
            parsed_content = None
            sql_params_for_this_call = None # Store params for this specific SQL call

            # JSON parsing logic - Enhanced to handle potential nested structures
            if isinstance(raw_content, str):
                try: 
                    parsed_content = json.loads(raw_content)
                except json.JSONDecodeError:
                    # If direct parsing fails, it might be simple string output
                    parsed_content = raw_content
                    logger.debug(f"Tool {tool_name} returned non-JSON string content: {raw_content[:100]}")
            elif isinstance(raw_content, dict):
                # If it's already a dict, use it directly
                parsed_content = raw_content
            else:
                # Handle other types if necessary (e.g., lists, primitives)
                parsed_content = raw_content
                logger.debug(f"Tool {tool_name} returned non-dict/non-string content of type {type(raw_content)}")

            # Table extraction logic for SQL tool - only add SUCCESSFUL data tables
            if tool_name in ["sql_query", "execute_sql"]:
                # Store the parameters if the SQL execution was successful
                if isinstance(tool_args, dict) and "params" in tool_args:
                    sql_params_for_this_call = tool_args["params"]
                    # Only store if it's not an error table
                    is_error_table = False
                    if isinstance(parsed_content, dict):
                        if isinstance(parsed_content.get("table"), dict) and \
                           isinstance(parsed_content["table"].get("columns"), list) and \
                           len(parsed_content["table"]["columns"]) == 1 and \
                           parsed_content["table"]["columns"][0].lower() == "error":
                            is_error_table = True
                        elif isinstance(parsed_content.get('columns'), list) and \
                             len(parsed_content['columns']) == 1 and \
                             parsed_content['columns'][0].lower() == "error":
                            is_error_table = True
                            
                    if not is_error_table:
                        # Merge params, prioritizing newer calls if keys overlap (though unlikely in single turn)
                        successful_sql_filters_this_turn.update(tool_args["params"])
                        logger.debug(f"[ToolsNode] Stored successful SQL filters for call {tool_id}: {tool_args['params']}")
                    else:
                        logger.debug(f"[ToolsNode] Not storing filters for call {tool_id} as it returned an error table structure.")

                # Check various possible structures for the actual table data
                if isinstance(parsed_content, dict):
                    if isinstance(parsed_content.get("table"), dict) and \
                       isinstance(parsed_content["table"].get("columns"), list) and \
                       isinstance(parsed_content["table"].get("rows"), list):
                        table_data = parsed_content["table"]
                    elif isinstance(parsed_content.get('columns'), list) and \
                         isinstance(parsed_content.get('rows'), list):
                        # Check if it's NOT an error table structure returned by the tool on failure
                        if not (len(parsed_content["columns"]) == 1 and parsed_content["columns"][0].lower() == "error"):
                            table_data = parsed_content
                        else:
                             logger.warning(f"[ToolsNode] Tool '{tool_name}' returned an error table structure despite success=True. Ignoring table.")
                    
                if table_data: # Append ONLY if valid data table found
                    # NEW: Create structured result
                    turn_structured_results.append({
                        "table": table_data,
                        "filters": sql_params_for_this_call or {} # Use params specific to this call
                    })
                    logger.info(f"[ToolsNode] Created structured result from successful '{tool_name}' call (ID: {tool_id}) with {len(table_data.get('rows', []))} rows")
                elif isinstance(parsed_content, dict):
                    # Log if it looked like a dict but didn't contain a valid table structure
                    logger.warning(f"[ToolsNode] Successful '{tool_name}' call (ID: {tool_id}) returned dict, but no valid table structure found. Keys: {list(parsed_content.keys())}")

            # Hierarchy Resolver: Update the map for THIS TURN
            elif tool_name == "hierarchy_name_resolver" and isinstance(parsed_content, dict):
                resolution_results = parsed_content.get("resolution_results", {})
                if resolution_results:
                    temp_map = {}
                    for name, res_info in resolution_results.items():
                        if res_info.get("status") == "found" and res_info.get("id"):
                            temp_map[name.lower()] = str(res_info["id"])
                    if temp_map:
                        # Update the map specific to this execution pass
                        location_map_this_turn.update(temp_map) # <<< Update this turn's map
                        logger.info(f"[ToolsNode] Updated resolved location map this turn with {len(temp_map)} entries.")

        else: # Handle tool execution failures
            error_message = result.get("error_message", "Unknown error")
            logger.warning(f"[ToolsNode] Tool call '{tool_name}' (ID: {tool_id}) failed: {error_message}")
            # ... (Construct ToolMessage with error content) ...
            if not any(getattr(m, 'tool_call_id', None) == tool_id for m in new_messages if isinstance(m, ToolMessage)):
                 new_messages.append(ToolMessage(content=error_message, tool_call_id=tool_id, name=tool_name))

            # Track failure patterns
            if tool_name not in failure_patterns:
                failure_patterns[tool_name] = []
            # ... (Hashing logic for args_hash) ...
            args_hash = None
            try:
                args_str = json.dumps(tool_args, sort_keys=True, default=str) if tool_args else ""
                args_hash = hash(args_str)
            except Exception as hash_err:
                logger.warning(f"[ToolsNode] Could not hash args for {tool_name}: {hash_err}")
                args_hash = hash(str(tool_args)) if tool_args else None
            
            failure_signature = {
                "tool_id": tool_id,
                "error_message": error_message,
                "timestamp": datetime.datetime.now().isoformat(),
                "args_hash": args_hash,
                "args_summary": str(tool_args)[:100] if tool_args else None
            }
            
            failure_patterns[tool_name].append(failure_signature)
            if len(failure_patterns[tool_name]) > 5:
                failure_patterns[tool_name] = failure_patterns[tool_name][-5:]
            
            # CRITICAL: Ensure no error table is added to new_tables

    # --- Update state ---
    logger.info(f"[ToolsNode] Updating state with {len(new_messages)} messages, {len(turn_structured_results)} structured results.")

    # Construct the update dictionary AFTER processing all results
    update_dict: Dict[str, Any] = { # Use Dict here as AgentState type hint isn't perfect for partial updates
        "messages": new_messages, # Messages from THIS turn
        "structured_results": turn_structured_results, # NEW: Use structured results list
        "failure_patterns": failure_patterns, # Potentially updated failure patterns
        # Preserve recovery guidance from the input state
        "recovery_guidance": state.get("recovery_guidance"),
    }

    # Conditionally add the resolved map IF the resolver ran successfully THIS turn
    if location_map_this_turn: # Check if the map for this turn has entries
        update_dict["resolved_location_map"] = location_map_this_turn
    # Otherwise, the key is omitted, preserving the existing state value

    logger.info(f"[ToolsNode] Final update_dict keys before return: {list(update_dict.keys())}")
    logger.debug(f"[ToolsNode] Returning update_dict: {update_dict}")
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

# --- Conditional Edge Logic (Updated for simpler flow) ---
def should_continue(state: AgentState) -> str:
    """Determines the next step based on the last message and state.
       Routes to END if a tool execution error is detected in the last message.
    """
    request_id = state.get("request_id")
    messages = state.get("messages", [])
    last_message = messages[-1] if messages else None
    final_structure_in_state = state.get("final_response_structure")
    failure_patterns = state.get("failure_patterns", {})

    # If the final structure is already set (by agent node), we end.
    if final_structure_in_state:
        logger.debug("[ShouldContinue] Final response structure found in state. Routing to END.")
        return END

    # --- Check for recursive failure patterns ---
    max_identical_failures = 4  # Break after 4 identical failures
    for tool_name, failures in failure_patterns.items():
        if len(failures) >= max_identical_failures:
            # Check for similar error patterns in the last N failures
            recent_failures = failures[-max_identical_failures:]
            
            # Extract error messages for analysis
            error_messages = [failure.get("error_message", "") for failure in recent_failures]
            
            # Count occurrences of each error type
            error_types = {}
            for error_msg in error_messages:
                # Extract error type (first part before colon or first 40 chars)
                if ":" in error_msg:
                    error_type = error_msg.split(":", 1)[0].strip()
                else:
                    error_type = error_msg[:40] if error_msg else "unknown"
                
                error_types[error_type] = error_types.get(error_type, 0) + 1
            
            # If we have a dominant error type (occurs in at least 3 of last 4 failures)
            # or if we have at most 2 different error types total, we're in a recursion pattern
            has_dominant_error = any(count >= 3 for count in error_types.values())
            few_error_types = len(error_types) <= 2
            
            if has_dominant_error or few_error_types:
                logger.warning(f"[ShouldContinue] Detected recursion with {len(failures)} failures for {tool_name}. Error types: {error_types}")
                
                # Determine the most common error type for a tailored message
                common_error_type = max(error_types.items(), key=lambda x: x[1])[0] if error_types else "unknown"
                response_text = "I encountered technical limitations processing this complex request."
                
                # Customize message based on error pattern
                if "security check" in common_error_type.lower():
                    response_text = (
                        "I'm unable to complete this request due to security constraints in how database queries are structured. "
                        "Please try simplifying your request or breaking it into separate questions about specific metrics or locations."
                    )
                elif "syntax" in common_error_type.lower():
                    response_text = (
                        "I'm having difficulty formulating a valid query for this complex request. "
                        "Please try asking about fewer metrics at once or focus on one specific aspect of your question."
                    )
                else:
                    response_text = (
                        "I've reached a technical limitation while processing this complex request. "
                        "Please try simplifying your question or breaking it into separate parts."
                    )
                
                # Create a final response to break the recursion
                final_structure = FinalApiResponseStructure(
                    text=response_text,
                    include_tables=[],
                    chart_specs=[]
                )
                
                # Update state directly
                state["final_response_structure"] = final_structure
                logger.info(f"[ShouldContinue] Breaking recursion with message: {response_text[:100]}...")
                return END
    # --- End recursion check ---

    # --- ADDED: Check for Tool Execution Errors --- #
    if isinstance(last_message, ToolMessage):
        # Check if the tool message content indicates an error (simple check)
        content_str = str(last_message.content).lower()
        if content_str.startswith("error:") or "failed execution:" in content_str or "tool execution failed:" in content_str:
            logger.warning(f"[ShouldContinue] Tool error detected in last message ({last_message.name}). Routing to END.")
            return END
    # --- END Tool Error Check --- #
        
    # Analyze the last message from the agent node (if no tool error detected)
    if isinstance(last_message, AIMessage):
        if last_message.tool_calls:
            # Check if ANY operational tools (excluding FinalApiResponseStructure) were called
            has_operational_calls = any(
                tc.get("name") != FinalApiResponseStructure.__name__ 
                for tc in last_message.tool_calls
            )
            # Check if FinalApiResponseStructure was called (should only happen if final_structure_in_state is already set)
            has_final_call = any(tc.get("name") == FinalApiResponseStructure.__name__ for tc in last_message.tool_calls)

            if has_final_call: 
                logger.warning("[ShouldContinue] FinalApiResponseStructure tool call found, but structure not in state? Routing to END.")
                return END # Should be caught by final_structure_in_state check
            elif has_operational_calls:
                 # This includes HierarchyNameResolverTool, SQLExecutionTool, SummarySynthesizerTool
                 logger.debug("[ShouldContinue] Operational tool call(s) found. Routing to 'tools'.")
                 return "tools" 
            else:
                 logger.warning("[ShouldContinue] AIMessage has tool_calls list but no recognized operational or final calls. Routing to END.")
                 return END 
        else:
             logger.warning("[ShouldContinue] AIMessage with no tool calls reached here (should have been coerced/ended). Routing to END.")
             return END
    elif isinstance(last_message, ToolMessage): 
        # If it was a ToolMessage but NOT an error, loop back to agent to process the result
        logger.debug("[ShouldContinue] Tool message (non-error) found. Routing back to 'agent'.")
        return "agent"
    else:
        # If last message isn't AIMessage or ToolMessage, or state is unexpected, end.
        logger.warning(f"[ShouldContinue] Unexpected last message type ({type(last_message).__name__}) or state, routing to END.")
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
            # Use the previously defined fallback/error structure if final_structure is missing
            logger.error(f"[ReqID: {request_id}] Final state lacks FinalApiResponseStructure. Creating fallback response.")
            # Directly create the fallback and assign to final_structure
            final_structure = FinalApiResponseStructure(
                 text="I wasn't able to properly complete this request. Please try again later.",
                 include_tables=[],
                 chart_specs=[]
             ) 
            # structured_response was already defined above with a fallback
            final_text = final_structure.text
            chart_specs = final_structure.chart_specs
        else:
            final_text = final_structure.text
            chart_specs = final_structure.chart_specs # Use validated specs from the structure

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

        # --- Process Chart Specifications --- #
        visualizations = []
        # Use chart_specs extracted from final_structure
        if chart_specs and isinstance(chart_specs, list): 
             try:
                 # Convert validated Pydantic models to dicts for JSON response
                 visualizations = [spec.dict() for spec in chart_specs] 
                 logger.debug(f"[ReqID: {request_id}] {len(visualizations)} chart specifications included in final response.")
             except Exception as chart_err:
                  logger.error(f"[ReqID: {request_id}] Error processing chart specifications: {chart_err}", exc_info=True)
                  visualizations = [] # Ensure empty list on error
        else:
             logger.debug(f"[ReqID: {request_id}] No chart specs found or invalid format in the final response structure.")
        # --- End Chart Processing --- #
        
        # Ensure final_text is a string
        final_text_str = str(final_text) if final_text is not None else "An error occurred generating the response text."

        # --- Build Final Success Response --- #
        success_response["data"] = {
            "text": final_text_str, 
            "tables": formatted_tables_to_include, # Use the cleaned and formatted tables
            "visualizations": visualizations
        }
        
        # --- END: Refactored Final Response Preparation --- #

        # --- Existing logging and return --- #
        logger.info(f"Token Usage - Prompt: {total_prompt_tokens}, Completion: {total_completion_tokens}, Total: {total_prompt_tokens + total_completion_tokens}")
        # ... (rest of usage logging and return success_response) ...
        usage_logger.info(json.dumps({
            "request_id": req_id,
            "organization_id": organization_id,
            "session_id": session_id,
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": total_completion_tokens,
            "total_tokens": total_prompt_tokens + total_completion_tokens,
            "query": message,
            "response_length": len(final_text_str), # Use final_text_str length
            "table_count": len(formatted_tables_to_include),
            "visualization_count": len(visualizations)
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