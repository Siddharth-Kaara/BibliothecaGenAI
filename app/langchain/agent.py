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
    tables: Annotated[List[Dict[str, Any]], operator.add] # List of tables from sql_query tool calls
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
        "completion_tokens": 0
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
        llm_response = llm_with_structured_output.invoke(preprocessed_state)
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
                        # Add default for include_tables before parsing
                        args = final_api_call.get("args", {})
                        num_tables = len(state.get('tables', []))
                        if "include_tables" not in args: args["include_tables"] = [False] * num_tables
                        if isinstance(args.get("include_tables"), bool): args["include_tables"] = [args["include_tables"]] * max(1, num_tables)
                        
                        parsed_final = final_response_parser.invoke(AIMessage(content="", tool_calls=[final_api_call]))
                        if parsed_final:
                            final_structure = parsed_final[0]
                            logger.debug(f"[AgentNode] Successfully parsed FinalApiResponseStructure from tool call.")
                        else:
                            logger.warning("[AgentNode] FinalAPIStructure parser returned empty list from tool call. Creating fallback.")
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
    if 'tables' in processed_state and processed_state['tables']:
        original_tables = state['tables']
        processed_tables = []
        for i, table in enumerate(processed_state['tables']):
            processed_table = table.copy() 
            # Use config setting for max rows
            max_rows = settings.MAX_TABLE_ROWS_IN_STATE 
            if 'rows' in processed_table and len(processed_table['rows']) > max_rows:
                logger.debug(f"Truncating table index {i} rows from {len(processed_table['rows'])} to {max_rows}")
                processed_table['rows'] = processed_table['rows'][:max_rows]
                if 'metadata' not in processed_table: processed_table['metadata'] = {}
                processed_table['metadata']['truncated'] = True
                processed_table['metadata']['original_rows'] = len(original_tables[i]['rows']) 
            processed_tables.append(processed_table)
        processed_state['tables'] = processed_tables

    return processed_state

# --- Tool Node Handler ---
async def async_tools_node_handler(state: AgentState, tools: List[Any]) -> Dict[str, Any]:
    """
    Enhanced asynchronous tools node handler. Executes OPERATIONAL tools only.
    Ignores FinalApiResponseStructure 'tool calls'.
    Adds resulting tables to the state.
    """
    request_id = state.get("request_id")
    logger.debug(f"[ToolsNode] Entering tool handler.")
    # Get last message, could be AIMessage with tool calls or others
    last_message = state["messages"][-1] if state["messages"] else None
    tool_map = {tool.name: tool for tool in tools}
    tool_executions = []
    new_messages = []
    new_tables = [] # Only tables are extracted here now
    successful_calls = 0
    operational_tool_calls = []
    
    # Initialize update dictionary and failure tracking
    update_dict = {}
    failure_patterns = state.get("failure_patterns", {}).copy()  # Create a copy to avoid modifying the original

    # Filter out non-operational "tool calls" (Final API Structure)
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        operational_tool_calls = [
            tc for tc in last_message.tool_calls 
            if tc.get("name") != FinalApiResponseStructure.__name__ # Only filter out FinalApiResponseStructure
        ]

    if not operational_tool_calls:
        logger.debug(f"[ToolsNode] No operational tool calls found in last message.")
        return {} 

    logger.debug(f"[ToolsNode] Dispatching {len(operational_tool_calls)} operational tool calls: {[tc.get('name') for tc in operational_tool_calls]}")
    # Prepare tool execution details
    for tool_call in operational_tool_calls:
        tool_name = tool_call.get("name")
        tool_id = tool_call.get("id", "")
        tool_args = tool_call.get("args", {})
        if tool_name in tool_map:
            # Use config setting for retries
            tool_executions.append({
                "tool": tool_map[tool_name], 
                "args": tool_args, 
                "id": tool_id, 
                "name": tool_name, 
                "retries_left": settings.TOOL_EXECUTION_RETRIES
            })
        else:
             logger.error(f"[ToolsNode] Operational tool '{tool_name}' requested but not found.")
             new_messages.append(ToolMessage(
                content=f"Error: Tool '{tool_name}' not found.", 
                tool_call_id=tool_id, 
                name=tool_name
             ))

    if not tool_executions:
        logger.warning(f"[ToolsNode] No operational tools could be prepared for execution.")
        return {"messages": new_messages} if new_messages else {}

    sem = asyncio.Semaphore(settings.MAX_CONCURRENT_TOOLS)
    # Define async execution function with retry logic
    async def execute_with_retry(execution_details):
        tool = execution_details["tool"]
        args = execution_details["args"]
        tool_id = execution_details["id"]
        tool_name = execution_details["name"]
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
                     "args": args  # Include args in successful result
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
                    "error_message": error_msg,  # Store original error message separately
                    "tool_name": tool_name, 
                    "tool_id": tool_id,
                    "args": args  # Include args in failed result too
                }

    results = await asyncio.gather(*[execute_with_retry(exec_data) for exec_data in tool_executions])

    # Result processing: Extract only tables
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

            # JSON parsing logic
            if isinstance(raw_content, str):
                try: parsed_content = json.loads(raw_content)
                except json.JSONDecodeError: parsed_content = raw_content
            else: parsed_content = raw_content

            # Table extraction logic
            if isinstance(parsed_content, dict):
                 if tool_name in ["sql_query", "execute_sql"]:
                     if isinstance(parsed_content.get("table"), dict): table_data = parsed_content["table"]
                     elif isinstance(parsed_content.get('columns'), list) and isinstance(parsed_content.get('rows'), list): table_data = parsed_content
                     
                     if table_data and isinstance(table_data.get('columns'), list) and isinstance(table_data.get('rows'), list):
                         new_tables.append(table_data)
                         logger.info(f"[ToolsNode] Extracted table from '{tool_name}' with {len(table_data.get('rows', []))} rows")
                     elif table_data: logger.warning(f"[ToolsNode] '{tool_name}' invalid table structure. Keys: {list(table_data.keys())}")
        else:
            # Handle tool execution failures and track patterns
            error_message = result.get("error_message", "Unknown error")
            logger.warning(f"[ToolsNode] Tool call '{tool_name}' (ID: {tool_id}) failed: {error_message}")
            
            # Append to messages if not already present
            if not any(getattr(m, 'tool_call_id', None) == tool_id for m in new_messages if isinstance(m, ToolMessage)):
                error_content = f"Tool execution failed: {error_message}"
                new_messages.append(ToolMessage(content=error_content, tool_call_id=tool_id, name=tool_name))
            
            # Track failure patterns for later analysis
            if tool_name not in failure_patterns:
                failure_patterns[tool_name] = []
            
            # Create a unique signature for this failure
            # Include more detailed information to help identify patterns
            args_hash = None
            try:
                # Create a deterministic hash of the arguments
                args_str = json.dumps(tool_args, sort_keys=True, default=str) if tool_args else ""
                args_hash = hash(args_str)  # Simple hash for similarity detection
            except Exception as hash_err:
                logger.warning(f"[ToolsNode] Could not hash args for {tool_name}: {hash_err}")
                args_hash = hash(str(tool_args)) if tool_args else None
            
            failure_signature = {
                "tool_id": tool_id,
                "error_message": error_message,
                "timestamp": datetime.datetime.now().isoformat(),
                "args_hash": args_hash,
                # Store a simplified version of the arguments for analysis
                "args_summary": str(tool_args)[:100] if tool_args else None
            }
            
            # Add to failure patterns
            failure_patterns[tool_name].append(failure_signature)
            
            # Keep only last 5 failures per tool to prevent state bloat
            if len(failure_patterns[tool_name]) > 5:
                failure_patterns[tool_name] = failure_patterns[tool_name][-5:]

    # Update state: Only add messages and new tables
    logger.info(f"[ToolsNode] Updating state with {len(new_messages)} messages, {len(new_tables)} tables, {len(failure_patterns)} failure patterns.")
    
    if new_messages:
        update_dict["messages"] = new_messages
    
    if new_tables:
        # Combine with existing tables from state correctly
        existing_tables = state.get('tables', [])
        combined_tables = existing_tables + new_tables
        logger.info(f"[ToolsNode] Found {len(new_tables)} new table(s). State will have {len(combined_tables)} total tables.")
        update_dict["tables"] = combined_tables
    
    # Add updated failure patterns to state
    update_dict["failure_patterns"] = failure_patterns
    
    # Preserve recovery guidance if present
    if "recovery_guidance" in state:
        update_dict["recovery_guidance"] = state["recovery_guidance"]
        
    logger.info(f"[ToolsNode] Final update_dict keys before return: {list(update_dict.keys())}")
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

    # Add edge from the tools node back to the agent
    workflow.add_edge("tools", "agent")

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

    # Initialize agent state with failure pattern tracking
    initial_state = AgentState(
        messages=initial_messages,
        tables=[],
        final_response_structure=None,
        request_id=req_id,
        prompt_tokens=0,
        completion_tokens=0,
        failure_patterns={},
        recovery_guidance=None
    )

    try:
        graph_app = create_graph_app(organization_id)
        if asyncio.iscoroutine(graph_app): graph_app = await graph_app
            
        logger.info(f"Invoking LangGraph workflow...")
        final_state = await graph_app.ainvoke(
            initial_state,
            config=RunnableConfig(recursion_limit=settings.MAX_GRAPH_ITERATIONS, configurable={})
        )
        logger.info(f"LangGraph workflow invocation complete.")

        # Track token usage
        total_prompt_tokens = final_state.get("prompt_tokens", 0)
        total_completion_tokens = final_state.get("completion_tokens", 0)

        # Extract final response structure
        structured_response = final_state.get("final_response_structure")
        if not structured_response:
            logger.warning(f"No final_response_structure found in final state. Generating fallback.")
            structured_response = FinalApiResponseStructure(
                text="I wasn't able to properly complete this request. Please try again with a more specific question.",
                include_tables=[],
                chart_specs=[]
            )
        
        # Process tables
        tables_from_state = final_state.get("tables", [])
        tables_to_include = []
        
        # Handle include_tables flags
        include_tables_flags = structured_response.include_tables
        if include_tables_flags and len(tables_from_state) > 0:
            # Ensure include_tables exists and has the right length
            if not include_tables_flags:
                # Generate default include_tables (all False)
                logger.debug(f"No include_tables specified, defaulting all {len(tables_from_state)} tables to False")
                include_tables_flags = [False] * len(tables_from_state)
            elif len(include_tables_flags) < len(tables_from_state):
                # Pad with False if too short
                missing_count = len(tables_from_state) - len(include_tables_flags)
                logger.debug(f"Padding include_tables with {missing_count} False values to match table count ({len(tables_from_state)})" )
                include_tables_flags.extend([False] * missing_count)
            elif len(include_tables_flags) > len(tables_from_state):
                # Truncate if too long
                logger.debug(f"Truncating include_tables from {len(include_tables_flags)} to match table count ({len(tables_from_state)})" )
                include_tables_flags = include_tables_flags[:len(tables_from_state)]
            
            # Apply flags to filter tables
            tables_to_include = [
                table for idx, table in enumerate(tables_from_state)
                if idx < len(include_tables_flags) and include_tables_flags[idx]
            ]
        
        # Process charts using the dedicated function from charting.py
        visualizations = []
        filtered_charts_info = [] # Initialize list for filtered info
        chart_specs_from_llm = getattr(structured_response, "chart_specs", [])
        if chart_specs_from_llm:
            logger.debug(f"Processing {len(chart_specs_from_llm)} chart specs from LLM...")
            # Capture both return values from the updated function
            visualizations, filtered_charts_info = process_and_validate_chart_specs(chart_specs_from_llm, tables_from_state)
            logger.debug(f"Validated {len(visualizations)} chart specs for API response. Filtered out {len(filtered_charts_info)} specs.")
        else:
             logger.debug("No chart specs found in the final response structure.")

        # --- Modify text response if charts were filtered --- 
        final_text = structured_response.text
        if filtered_charts_info:
            logger.info(f"Modifying text response as {len(filtered_charts_info)} chart(s) were filtered out.")
            # Construct a message about the filtered charts
            filtered_titles = [info.get('title', 'Untitled Chart') for info in filtered_charts_info]
            if len(filtered_titles) == 1:
                warning_suffix = f"\n\n(Note: A chart titled \"{filtered_titles[0]}\" was generated but could not be displayed due to data formatting issues.)"
            else:
                 warning_suffix = f"\n\n(Note: {len(filtered_titles)} charts (including \"{filtered_titles[0]}\") were generated but could not be displayed due to data formatting issues.)"
            
            # Append the warning to the original text
            final_text += warning_suffix
        # --- End text modification ---
        
        # Build successful response
        success_response["data"] = {
            "text": final_text, # Use the potentially modified text
            "tables": tables_to_include,
            "visualizations": visualizations # Use the validated list
        }
        
        # Log token usage
        logger.info(f"Token Usage - Prompt: {total_prompt_tokens}, Completion: {total_completion_tokens}, Total: {total_prompt_tokens + total_completion_tokens}")
        
        # Log detailed usage info
        usage_logger.info(json.dumps({
            "request_id": req_id,
            "organization_id": organization_id,
            "session_id": session_id,
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": total_completion_tokens,
            "total_tokens": total_prompt_tokens + total_completion_tokens,
            "query": message,
            "response_length": len(structured_response.text),
            "table_count": len(tables_to_include),
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
                partial_tables = partial_state.get("tables", [])
                
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