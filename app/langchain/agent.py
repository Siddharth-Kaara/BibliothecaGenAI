import logging
import json
import asyncio
from typing import Any, Dict, List, Optional, Tuple, TypedDict, Annotated, Sequence
import operator
from pydantic import BaseModel, Field
import functools
from pydantic_core import ValidationError # Import for specific error handling
import uuid # Import for UUID validation

# LangChain & LangGraph Imports
from langchain_core.messages import BaseMessage, FunctionMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import AzureChatOpenAI
from langchain_core.output_parsers.openai_tools import PydanticToolsParser, JsonOutputToolsParser
from langchain_core.runnables import RunnableConfig # Import for config
from langgraph.graph import StateGraph, END

# Local Imports
from app.core.config import settings
from app.langchain.tools.sql_tool import SQLQueryTool
from app.langchain.tools.chart_tool import ChartRendererTool
from app.langchain.tools.summary_tool import SummarySynthesizerTool
from app.langchain.tools.hierarchy_resolver_tool import HierarchyNameResolverTool
from app.schemas.chat import ChatData

logger = logging.getLogger(__name__)

# --- Define Structure for LLM Final Response (Used as a Tool/Schema) ---
class FinalApiResponseStructure(BaseModel):
    """Structure for the final API response. Call this function when you have gathered all necessary information and are ready to formulate the final response to the user."""
    text: str = Field(description="The final natural language text response for the user. Follow the guidelines in the system prompt for generating this text (e.g., brief introductions if data/charts are present).")
    include_tables: List[bool] = Field(
        description="List of booleans indicating which tables from the agent state should be included in the final API response. Match the order of tables in the state. Set to false if a table's data is covered by an included visualization (unless explicitly requested).",
        default_factory=list
    )
    include_visualizations: List[bool] = Field(
        description="List of booleans indicating which visualizations from the agent state should be included in the final API response. Match the order of visualizations in the state.",
        default_factory=list
    )


# --- Define the Agent State ---
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add] # Conversation history
    tables: Annotated[List[Dict[str, Any]], operator.add] # List of tables from sql_query tool calls
    visualizations: Annotated[List[Dict[str, Any]], operator.add] # List of visualizations from chart_renderer tool calls
    # Add a field to hold the final structured response once generated
    final_response_structure: Optional[FinalApiResponseStructure]


# --- System Prompt ---
SYSTEM_PROMPT_TEMPLATE = """You are a professional data assistant for the Bibliotheca chatbot API.

Your primary responsibility is to analyze organizational data and provide accurate insights to users based on the request's context.
**Key Context:** The necessary `organization_id` for data scoping is always provided implicitly through the tool context; **NEVER ask the user for it or use placeholders like 'your-organization-id'.**

You have access to a single database: **report_management**.
This database contains:
- Event counts and usage statistics (table '5').
- All necessary organizational hierarchy information (table 'hierarchyCaches').

Based on the user's query and the conversation history (including previous tool results), decide the best course of action. You have the following tools available:

{tool_descriptions}

Available Tools:
{tool_names} # Note: FinalApiResponseStructure will be listed here if bound correctly

Tool Use and Response Guidelines:
1.  **Analyze History:** Always review the conversation history (`messages`) for context, previous tool outputs (`ToolMessage`), and accumulated data (`tables`, `visualizations` in state).
2.  **Adhere to Tool Schemas:** Ensure all arguments provided to tool calls strictly match the tool's defined input schema (`args_schema`).
3.  **Hierarchy Name Resolution (MANDATORY FIRST STEP if names present):**
    *   If the user query mentions specific organizational hierarchy entities by name (e.g., "Main Library", "Argyle Branch"), you MUST call the `hierarchy_name_resolver` tool **ALONE** as your first action. Do *not* call any other tools in the same step.
    *   Pass the list of names as `name_candidates`.
    *   This tool uses the correct `organization_id` from the request context automatically. You do not need to provide it as an argument. **DO NOT ask the user for the organization_id.**
    *   The graph will automatically route you back here after the resolver runs.
    *   Examine the `ToolMessage` from `hierarchy_name_resolver` in the history.
    *   If any status is 'not_found' or 'error': Inform the user via `FinalApiResponseStructure` about the failure. Only proceed for *successfully* resolved names if logical, clearly stating which ones failed.
    *   If all relevant names have status 'found': Proceed to the next step (e.g., `sql_query` or `summary_synthesizer`) using the returned `id` values.
4.  **Database Usage:** ALWAYS use the `report_management` database. Specify `db_name='report_management'` in `sql_query` calls.
    *   Events: table '5'. Hierarchy: `hierarchyCaches`.
    *   Joins: Use **resolved hierarchy IDs** (`JOIN "hierarchyCaches" hc ON "5"."hierarchyId" = hc."id" WHERE hc."id" IN (...)`).
5.  **SQL Query Generation (`sql_query` tool - Use Primarily for Raw Data/Charts):**
    *   Call this tool *after* name resolution (if needed).
    *   **CRITICAL:** When constructing the `query_description` argument for this tool, you **MUST** explicitly include the **resolved hierarchy ID(s)** obtained from the `hierarchy_name_resolver` tool's output (`ToolMessage`). Do NOT just use the resolved name.
    *   Example Description: `"Get borrow counts for hierarchy ID 'ca4b911c-8b54-e811-2a94-0024e880a2b7' last week"` (NOT "Get borrow counts for Maxville Branch last week").
    *   The SQL generating LLM inside `sql_query` is instructed to use these IDs directly for filtering.
    *   Filter appropriately (e.g., by ID, timestamp). Avoid adding non-existent filters like `isActive`.
    *   Follow standard SQL practices.
6.  **Query Consolidation & Aggregation Strategy (`sql_query` tool):**
    *   Simple counts: Request successful count.
    *   Comparisons/Plots/Breakdowns for **Specific Entities**: If calling `sql_query` for multiple resolved IDs (e.g., for a chart), use **ONE** query **grouped by hierarchy identifier** (e.g., `GROUP BY hc."id", hc."name"`).
    *   Avoid multiple `sql_query` calls if one grouped query suffices.
7.  **SQL Output Format:** `sql_query` returns JSON (`{{"table": ..., "text": ...}}`). Added to state.
8.  **Chart Generation (`chart_renderer` tool - CRITICAL STEPS):**
    *   **Step 8a: Identify Need & Exercise Judgment:** Call this tool ONLY when the user explicitly asks for a chart/graph/visualization OR when presenting complex comparisons/trends (e.g., many entities, time-series data) that would **significantly benefit** from visual representation. **For simple comparisons (e.g., 2-3 entities across 2-3 metrics where the absolute numbers are easy to grasp), strongly prefer presenting the data in a text summary (see Guideline #12), as a chart adds little value and incurs overhead. Only override this preference if a chart is explicitly requested by the user.**
    *   **Step 8b: Check Data:** Ensure relevant data exists in the `state['tables']` list (usually from a preceding `sql_query` call).
    *   **Step 8c: Data Reformatting & Transformation (Mandatory if Calling):**
        *   **Input Format:** Recall `sql_query` returns data in a table structure (`{{"table": {{"columns": [...], "rows": [...]}} }}`). Extract this `table` data.
        *   **Required Output Format for Chart Tool:** The `chart_renderer` **requires** the `data` argument in the *exact same* format: `{{"columns": [...], "rows": [...]}}`. Ensure your extracted data adheres to this.
        *   **Transformation for Comparison Bar Charts (If Applicable):** If creating a bar chart comparing multiple metrics side-by-side (e.g., comparing 'Borrows' and 'Renewals' for 'Main Library' and 'Argyle Branch'), you **MUST** transform the data from the typically "wide" format returned by `sql_query` (e.g., columns: `Branch Name`, `Borrows`, `Renewals`) into a "long" format suitable for plotting with a `hue`/`color_column`.
            *   Example Long Format Target:
              ```
              {{ # Outer braces escaped if needed in actual JSON
                'columns': ['Branch Name', 'Metric Type', 'Total Count'],
                'rows': [
                  ['Main Library', 'Successful Borrows', 16452],
                  ['Main Library', 'Successful Renewals', 108],
                  ['Argyle Branch', 'Successful Borrows', 12512],
                  ['Argyle Branch', 'Successful Renewals', 105],
                  # ... include rows for 'Successful Returns' etc. ...
                ]
              }} # Outer braces escaped if needed
              ```
            *   You must perform this transformation logic yourself before calling the tool.
    *   **Step 8d: Explicit Metadata Creation & CONSISTENCY CHECK (Mandatory):**
        *   You **MUST** construct and provide the `metadata` argument explicitly. **DO NOT** rely on the tool's internal defaults.
        *   **Metadata Content (Example for Comparison Bar Chart using Long Format):**
            *   `chart_type`: "bar"
            *   `title`: A descriptive title (e.g., "Comparison of Activity by Branch")
            *   `x_column`: Name of the column with categories (e.g., "Branch Name")
            *   `y_column`: Name of the column with numerical values (e.g., "Total Count")
            *   `color_column`: Name of the column differentiating bars (hue) (e.g., "Metric Type")
            *   `x_label`, `y_label`: Appropriate axis labels.
        *   **Metadata Content (Other Chart Types):** Adapt fields like `x_column`, `y_column` based on the chart type and the columns present in your *final prepared `data`*.
        *   **CONSISTENCY CHECK (MANDATORY):** Before finalizing the `metadata`, **you MUST verify that every column name specified in the `metadata` (e.g., `x_column`, `y_column`, `color_column`) actually exists as a column in the `data` dictionary you are providing (after any reformatting/transformation).** If a column specified in `metadata` is missing in `data`, you MUST either add the required column to `data` or adjust the `metadata` to use a column that *does* exist in `data`. Failure to ensure consistency will cause chart errors.
    *   **Step 8e: Tool Invocation:** Call `chart_renderer` with the prepared (reformatted, potentially transformed) `data` and the explicitly constructed and *verified* `metadata`.
9.  **CRITICAL TOOL CHOICE: `sql_query` vs. `summary_synthesizer`:**

    *   **Use `sql_query` IF AND ONLY IF:** The user asks for a comparison OR retrieval of **specific, quantifiable metrics** (e.g., counts, sums of borrows, returns, renewals, logins) for **specific, resolved entities** (e.g., Main Library [ID: xxx], Argyle Branch [ID: yyy]) over a **defined time period**.
        *   **Goal:** Formulate a **single, efficient `sql_query` call** (per Guideline #11) targeting a single comparison table as output.
        *   **DO NOT use `summary_synthesizer` in this case.**

    *   **Use `summary_synthesizer` ONLY FOR:** More **open-ended, qualitative summary requests** (e.g., "summarize activity," "tell me about the branches") where specific metrics are not the primary focus, or when the exact metrics are unclear.
        *   Call it **directly after** name resolution (if applicable).
        *   Provide context (original query, resolved IDs) in the `query` argument. Its output will be purely text.

10. **Filtering/Case Sensitivity:** Standard SQL rules apply.

11. **Efficiency (for `sql_query`):**
    *   When formulating a direct `sql_query` call (especially for comparisons as per Guideline #9), if comparing simple metrics for multiple entities over the *same* period, make it a *single* call covering the entire comparison.
    *   Example: For comparing borrows for Branch A and Branch B last week, the description should be like "Compare total successful borrows for hierarchy ID 'id-a' and hierarchy ID 'id-b' last week, grouped by branch name/id".

12. **Prioritize Text for Simple Comparisons:** When comparing just a few specific metrics for a small number of entities (e.g., comparing borrows and renewals for two branches), generate a concise text summary highlighting the key figures and differences. **DO NOT** generate a chart for such simple cases unless the user has explicitly asked for one. Use Guideline #13 (previously #12) to decide whether to include the raw data table.

13. **Including Tables/Visualizations & Concise Text (Final Response):** 
    *   **Source Check:** Tables/Visualizations in the state primarily come from direct `sql_query` or `chart_renderer` calls.
    *   **Check State:** Before calling `FinalApiResponseStructure`, examine the `tables` and `visualizations` lists.
    *   **Usefulness Test:** Determine if any table or visualization provides useful, detailed information that directly supports the user's request (especially for comparisons or breakdowns).
    *   **Inclusion Decision:** If a useful table/visualization exists:
        *   Set the corresponding flag in `include_tables` or `include_visualizations` to `[True]` (or e.g., `[True, False]`).
        *   Ensure the `text` field is **CONCISE**, focuses on insights/anomalies, and **REFERENCES** the included item(s). **DO NOT** repeat the detailed data from the table/viz.
    *   **No Inclusion:** If no useful tables/visualizations exist (or came from the tools used), set flags to `[False]` and provide the full answer in the `text` field.
    *   **Summarizer Text:** If the `summary_synthesizer` was used (for open-ended queries), its output is text. Use that text directly in the `FinalApiResponseStructure`'s `text` field (set include flags to `[False]` unless other tools *also* ran and produced useful tables/viz).
    *   **Accuracy:** Ensure the final `text` accurately reflects and correctly references any included items.

14. **Handling Analytical/Comparative Queries (e.g., 'is it busy?'):** 
    *   Your primary goal is providing factual data.
    *   For queries asking for analysis or comparison (like "is branch X busy?", "is Y popular?"), first retrieve the relevant factual data for the specified entity using `sql_query`. The `sql_query` tool has been instructed to attempt to include a simple organization-wide benchmark (like an average) in its results for such queries.
    *   When formulating the final response:
        *   Check the table returned by `sql_query`. If it contains both the specific entity's value AND a benchmark value (e.g., columns like "Total Entries" and "Org Average Entries"), use both to provide context in the `text` field. Example: "Branch X had 1500 entries, which is above the organizational average of 1200."
        *   If the benchmark value is missing (the SQL tool couldn't easily include it), simply state the facts retrieved for the specific entity in the `text` field and explain that assessing relative terms like "busy" or "popular" requires additional context or comparison data not readily available.
        *   Decide whether to include the table based on Guideline #13 .
    *   Do NOT delegate these interpretations to `summary_synthesizer`.

15. **Out-of-Scope Refusal:** Your function is limited to library data. You MUST refuse requests completely unrelated to library data or operations (e.g., general knowledge, coding, football, recipes, opinions on non-library topics).

16. **Invoke `FinalApiResponseStructure` (CRITICAL FINAL STEP):** You **MUST ALWAYS** conclude your response by invoking the `FinalApiResponseStructure` tool. This applies in **ALL** situations, including:
    *   Successfully answering the query using tool results.
    *   Reporting errors encountered during tool execution (e.g., name resolution failure).
    *   Refusing out-of-scope or inappropriate requests (like the example below).
    *   Providing simple greetings or capability descriptions.
    Populate the arguments based on your analysis and the preceding guidelines (especially #13 for table inclusion and accuracy). **Failure to call `FinalApiResponseStructure` as the final step is an error.**

17. **Example (Name Resolution Failure):**
    *   User Query: "Borrows for Main Lib last week"
    *   `hierarchy_name_resolver` runs, returns `status: 'not_found'` for "Main Lib".
    *   Analysis: Name resolution failed.
    *   Your Response: Invoke `FinalApiResponseStructure(text="I couldn't find a hierarchy entity named 'Main Lib'. Please check the name.", include_tables=[], include_visualizations=[])`
18. **Example (Name Resolution Success -> Query -> Summarization):**
    *   User Query: "Compare borrows for Main Library and Argyle Branch last week"
    *   `hierarchy_name_resolver` runs, returns `status: 'found'` with IDs for both.
    *   `sql_query` runs using the resolved IDs and time filter. Adds table data to state.
    *   `summary_synthesizer` runs on the table data. Adds summary text to state (or directly prepares it).
    *   Analysis: Summarization complete based on resolved names.
    *   Your Response: Invoke `FinalApiResponseStructure(text="Over the last week, Main Library (Main) had X borrows, while Argyle Branch (AYL) had Y borrows.", include_tables=[False], include_visualizations=[False])` # Note: Visualization False based on Guideline #12
19. **Example (Greeting/Capability):**
    *   User Query: "Hi" or "What can you do?"
    *   Analysis: No tools needed.
    *   Your Response: Invoke `FinalApiResponseStructure(text="Hello! How can I assist you today?", include_tables=[], include_visualizations=[])`
20. **Example (Out-of-Scope Refusal - Full Query):**
    *   User Query: "Tell me about Rishabh Pant"
    *   Analysis: Out of scope (general knowledge).
    *   Your Response: Invoke `FinalApiResponseStructure(text="I cannot provide information about Rishabh Pant. My function is limited to answering questions about library data.", include_tables=[], include_visualizations=[])` # Note: Still uses the structure!
21. **Example (Combined Factual Answer + Analytical Interpretation With Benchmark - Simple Result):**
    *   User Query: "What was the footfall for Main Library last month, and is it busy?"
    *   `hierarchy_name_resolver` resolves "Main Library".
    *   `sql_query` gets footfall data AND org average. Returns table like: `'[{{"Location Name": "Main Library", "Total Entries": 5000, "Org Average Entries": 4500, "Total Exits": 4950, "Org Average Exits": 4400}}]'`
    *   Analysis: Got factual data + benchmark. Result is simple (one location).
    *   Your Response: Invoke `FinalApiResponseStructure(text="Main Library had 5000 entries and 4950 exits last month. This is slightly above the organizational average of 4500 entries and 4400 exits.", include_tables=[False], include_visualizations=[])` # Note: include_tables is [False]
22. **Example (Combined Factual Answer + Analytical Interpretation Without Benchmark - Simple Result):**
    *   User Query: "What was the footfall for the Annex last month, and is it busy?"
    *   `hierarchy_name_resolver` resolves "Annex".
    *   `sql_query` gets footfall data (e.g., 10 entries, 5 exits). Benchmark calculation was maybe too complex or skipped by LLM.
    *   Analysis: Got factual data. Benchmark missing. Result is simple.
    *   Your Response: Invoke `FinalApiResponseStructure(text="The Annex recorded 10 entries and 5 exits last month. Assessing whether this is considered 'busy' requires comparison with other branches or historical data, which was not readily available.", include_tables=[False], include_visualizations=[])` # Note: include_tables is [False]

**Workflow Summary:** Check names -> Resolve -> Refuse if needed -> **DECIDE: Specific Comparison (-> Direct `sql_query`) OR Open-Ended Summary (-> `summary_synthesizer`)** -> Get data (via chosen tool) -> Process Tool Results (Add table(s)/viz/summary text to state) -> Prepare Chart Data -> Call Renderer if needed -> Formulate Final Response (Check state for tables/viz -> Decide inclusion -> Generate CONCISE text referencing included items OR use full summarizer text -> Ensure accuracy) -> Conclude with `FinalApiResponseStructure`.
"""

# --- LLM and Tools Initialization ---
def get_llm():
    """Get the Azure OpenAI LLM."""
    logger.info(f"Initializing Azure OpenAI LLM with deployment {settings.AZURE_OPENAI_DEPLOYMENT_NAME}")
    # Ensure model supports tool calling / structured output
    return AzureChatOpenAI(
        openai_api_key=settings.AZURE_OPENAI_API_KEY,
        azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
        openai_api_version=settings.AZURE_OPENAI_API_VERSION,
        deployment_name=settings.AZURE_OPENAI_DEPLOYMENT_NAME,
        model_name=settings.LLM_MODEL_NAME, # Ensure this model supports tool calling well
        temperature=0.1,
        verbose=settings.VERBOSE_LLM,
        # streaming=False # Ensure streaming is False if not handled downstream
    )

def get_tools(organization_id: str) -> List[Any]:
    """Get tools for the agent (excluding FinalApiResponseStructure, which is handled via binding)."""
    return [
        HierarchyNameResolverTool(organization_id=organization_id),
        SQLQueryTool(organization_id=organization_id),
        ChartRendererTool(),
        SummarySynthesizerTool(organization_id=organization_id),
    ]

# Function to bind tools AND the final response structure to the LLM
def create_llm_with_tools_and_final_response_structure(organization_id: str):
    llm = get_llm()
    tools = get_tools(organization_id)
    # Bind the operational tools AND the final response structure
    # The LLM will treat FinalApiResponseStructure like another tool it can call
    all_bindable_items = tools + [FinalApiResponseStructure]

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT_TEMPLATE),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    # Format prompt with tool details (including FinalApiResponseStructure)
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
    )

    # Bind tools for function calling
    # llm_with_tools = llm.bind_tools(all_bindable_items) # Use this if Azure OpenAI supports Pydantic directly
    # Or, more explicitly for Azure OpenAI if needed:
    llm_with_tools = llm.bind_tools(
        tools=[*tools, FinalApiResponseStructure],
        tool_choice=None # Let the LLM decide which tool (or final structure) to call
    )
    logger.debug(f"Created LLM bound with tools and FinalApiResponseStructure for org: {organization_id}")
    return prompt | llm_with_tools

# --- Agent Node: Decides action - call a tool or invoke FinalApiResponseStructure ---
def agent_node(state: AgentState, llm_with_structured_output):
    """Invokes the LLM to decide the next action or final response structure.
       Includes a single retry attempt if the initial LLM response is invalid.
    """
    logger.debug(f"Agent node executing. Current state messages: {len(state['messages'])} messages.")
    logger.debug(f"Agent node state: Tables={len(state.get('tables',[]))}, Visualizations={len(state.get('visualizations',[]))}")

    response = None
    final_structure = None
    parser = PydanticToolsParser(tools=[FinalApiResponseStructure])

    for attempt in range(2): # Try up to 2 times (initial + 1 retry)
        logger.debug(f"Invoking LLM (Attempt {attempt + 1}/2)")
        # Invoke the LLM. It will either return tool calls for sql/chart/summary
        # OR a tool call for FinalApiResponseStructure
        response = llm_with_structured_output.invoke(state)
        logger.debug(f"Agent node generated raw response (Attempt {attempt + 1}): {response}")

        is_valid_response = False
        if isinstance(response, AIMessage) and response.tool_calls:
            # Check if it's an operational tool call OR a parsable FinalApiResponseStructure call
            first_tool_call = response.tool_calls[0]
            if first_tool_call.get("name") != FinalApiResponseStructure.__name__:
                is_valid_response = True # It's calling an operational tool
            else:
                # Try parsing FinalApiResponseStructure
                try:
                    # Use a temporary variable to avoid setting state prematurely on first pass
                    parsed_objects = parser.invoke(response)
                    if parsed_objects: # Should be a list with one item
                        potential_final_structure = parsed_objects[0]
                        final_structure = potential_final_structure # Assign if successfully parsed
                        is_valid_response = True
                        logger.info(f"Successfully parsed FinalApiResponseStructure (Attempt {attempt + 1})")
                    else:
                        logger.warning(f"PydanticToolsParser invoked but returned empty list (Attempt {attempt + 1}).")
                except ValidationError as e:
                    # Handle correctable errors ONLY if it's the final attempt, otherwise just log
                    if attempt == 1: # Final attempt
                        logger.warning(f"Initial parsing failed (Attempt {attempt + 1}): {e}. Checking for correctable list errors...")
                        # Check if it's the specific boolean-instead-of-list error
                        is_correctable_error = False
                        correctable_fields = ['include_tables', 'include_visualizations']
                        error_details = e.errors()
                        if all(err.get('type') == 'list_type' and err.get('loc', [None])[0] in correctable_fields for err in error_details):
                            is_correctable_error = True
                        
                        if is_correctable_error:
                            logger.info("Identified correctable boolean-as-list error. Attempting manual correction...")
                            try:
                                raw_args = first_tool_call.get("args", {})
                                corrected_args = raw_args.copy()
                                needs_correction = False
                                for field in correctable_fields:
                                    if field in corrected_args and isinstance(corrected_args[field], bool):
                                        corrected_args[field] = [corrected_args[field]]
                                        needs_correction = True
                                
                                if needs_correction:
                                     final_structure = FinalApiResponseStructure(**corrected_args)
                                     logger.warning(f"Successfully corrected LLM arguments and parsed FinalApiResponseStructure: {final_structure}")
                                     is_valid_response = True # Now it's valid after correction
                                else:
                                     logger.error(f"Caught list_type ValidationError, but no boolean found to correct. Raw args: {raw_args}. Original Error: {e}")
                            except Exception as correction_err:
                                 logger.error(f"Error during manual correction of FinalApiResponseStructure args: {correction_err}", exc_info=True)
                        else:
                             logger.error(f"Uncorrectable ValidationError parsing FinalApiResponseStructure: {e}", exc_info=True)
                    else: # Log validation error on first attempt but don't correct yet
                        logger.warning(f"Parsing FinalApiResponseStructure failed on attempt {attempt + 1}: {e}")
                except Exception as e:
                    logger.error(f"Unexpected error parsing FinalApiResponseStructure (Attempt {attempt + 1}): {e}", exc_info=True)
        
        if is_valid_response:
            break # Exit loop if response is valid (operational tool call or successfully parsed final structure)
        else:
            logger.warning(f"LLM response invalid on attempt {attempt + 1}. Retrying if possible...")
            # Optional: sleep briefly before retry?
            # await asyncio.sleep(0.5)

    # Return the (last) AIMessage and the successfully parsed final structure (if any)
    return {
        "messages": [response] if response else [],
        "final_response_structure": final_structure # Add parsed/corrected structure here
    }

# --- Tool Node Handler ---
async def async_tools_node_handler(state: AgentState, tools: List[Any]) -> Dict[str, Any]:
    """
    Enhanced asynchronous tools node handler. Executes operational tools (Resolver, SQL, Chart, Summary).
    Ignores the FinalApiResponseStructure 'tool call' as it signals the end state.

    Args:
        state: Current agent state.
        tools: List of available operational tools.

    Returns:
        Dict representing updates to the agent state (new messages, potentially tables/visualizations).
    """
    last_message = state["messages"][-1]
    if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
        logger.debug("Tool handler: No tool calls found in last AI message.")
        return {} # No update

    tool_map = {tool.name: tool for tool in tools}
    tool_executions = []

    # Filter out calls to FinalApiResponseStructure, only execute operational tools
    operational_tool_calls = [
        tc for tc in last_message.tool_calls
        if tc.get("name") != FinalApiResponseStructure.__name__
    ]

    if not operational_tool_calls:
        logger.debug("Tool handler: No *operational* tool calls found (might be FinalApiResponseStructure call).")
        # Return empty dict, but the AIMessage with the FinalApiResponseStructure call remains in history.
        # The should_continue logic will handle routing to END based on that AIMessage.
        return {}

    logger.info(f"Tool handler: Found {len(operational_tool_calls)} operational tool calls.")
    for tool_call in operational_tool_calls:
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args", {})
        tool_id = tool_call.get("id", "")

        if tool_name not in tool_map:
            logger.error(f"Tool '{tool_name}' not found in available operational tools list: {list(tool_map.keys())}")
            tool_executions.append({
                "error": f"Tool '{tool_name}' not found.",
                "tool_id": tool_id,
                "tool_name": tool_name
                })
            continue

        # Get the actual tool instance from the map
        current_tool_instance = tool_map[tool_name]

        tool_executions.append({
            # Use the instance we already retrieved
            "tool": current_tool_instance,
            "args": tool_args,
            "id": tool_id,
            "name": tool_name,
            "retries_left": 2
        })

    sem = asyncio.Semaphore(settings.MAX_CONCURRENT_TOOLS) # Limit concurrency based on settings
    
    async def execute_with_retry(execution_details):
        # Check if it's an error placeholder first
        if "error" in execution_details:
             return {
                "success": False,
                "error": execution_details["error"],
                "tool_name": execution_details["tool_name"],
                "tool_id": execution_details["tool_id"]
             }

        async with sem:
            tool = execution_details["tool"]
            args = execution_details["args"]
            tool_id = execution_details["id"]
            tool_name = execution_details["name"]
            retries = execution_details["retries_left"]

            try:
                logger.info(f"Executing tool: {tool_name} with args: {args}")
                # Check if the tool supports async execution
                if hasattr(tool, 'ainvoke'):
                    content = await tool.ainvoke(args)
                else:
                    # Fallback to synchronous execution in a thread pool
                    loop = asyncio.get_event_loop()
                    # Ensure tool.invoke doesn't block the event loop if it's CPU-bound
                    content = await loop.run_in_executor(None, functools.partial(tool.invoke, args))

                logger.info(f"Tool '{tool_name}' execution completed successfully")
                # Ensure content is JSON serializable for ToolMessage
                if isinstance(content, dict):
                    # Attempt to dump dicts to string for ToolMessage consistency
                    try:
                         content_str = json.dumps(content)
                    except TypeError:
                         logger.warning(f"Tool {tool_name} output dict not JSON serializable, using str().")
                         content_str = str(content)
                elif isinstance(content, str):
                    content_str = content # Already a string
                else:
                    content_str = str(content) # Fallback

                return {
                    "success": True,
                    "message": ToolMessage(content=content_str, tool_call_id=tool_id, name=tool_name), # Add name here
                    "raw_content": content, # Keep raw content for parsing table/viz below
                    "tool_name": tool_name
                }
            except Exception as e:
                error_msg = f"Error executing tool '{tool_name}' with args {args}: {str(e)}"
                logger.error(error_msg, exc_info=True) # Log traceback

                if retries > 0 and _is_retryable_error(e):
                    logger.info(f"Retrying tool '{tool_name}', {retries} attempts left")
                    execution_details["retries_left"] = retries - 1
                    await asyncio.sleep(0.5 * (3 - retries))
                    return await execute_with_retry(execution_details)

                return {
                    "success": False,
                    "error": error_msg,
                    "tool_name": tool_name,
                    "tool_id": tool_id
                }

    logger.info(f"Executing {len(tool_executions)} operational tool calls..." + (f" Names: {[t['name'] for t in tool_executions]}" if tool_executions else "")) # Log names
    results = await asyncio.gather(*[execute_with_retry(exec_data) for exec_data in tool_executions])

    # --- Processing results: Initialize lists FIRST --- 
    new_tables = []
    new_visualizations = []
    new_messages = []

    # --- Single Pass: Process results and errors --- 
    for result in results:
        tool_id = result.get("tool_id", "")
        tool_name = result.get("tool_name", "unknown_tool")

        if result["success"]:
            tool_message = result["message"]
            raw_content = result.get("raw_content") # Keep raw content for potential parsing
            new_messages.append(tool_message) # Append the message first

            # Try to parse content IF it's sql_query or chart_renderer for tables/viz
            content_to_parse = None
            if tool_name in ["sql_query", "chart_renderer"]:
                if isinstance(raw_content, dict):
                    content_to_parse = raw_content
                elif isinstance(raw_content, str):
                    content_str = raw_content.strip()
                    if content_str.startswith('{') and content_str.endswith('}'):
                        try:
                            content_to_parse = json.loads(content_str)
                        except json.JSONDecodeError as json_err:
                            logger.warning(f"Failed to parse raw string content as JSON for tool {tool_name}: {json_err}.")

            if content_to_parse: # Only proceed if parsing was relevant and potentially successful
                try:
                    # SQL Query: Extract table
                    if tool_name == "sql_query" and "table" in content_to_parse and content_to_parse["table"]:
                        table_data = content_to_parse["table"]
                        if isinstance(table_data, dict) and 'columns' in table_data and 'rows' in table_data:
                             # Basic dupe check against state only
                            if table_data not in state.get("tables", []):
                                logger.info(f"Extracted table from sql_query: {len(table_data.get('rows', []))} rows.")
                                new_tables.append(table_data)
                            else:
                                 logger.warning("Skipping duplicate table from sql_query.")

                    # Chart Renderer: Extract visualization
                    elif tool_name == "chart_renderer" and "visualization" in content_to_parse and content_to_parse["visualization"]:
                        viz_data = content_to_parse["visualization"]
                        if isinstance(viz_data, dict) and 'type' in viz_data and 'image_url' in viz_data:
                             # Basic dupe check against state only
                            if viz_data not in state.get("visualizations", []):
                                logger.info(f"Extracted visualization: {viz_data.get('type', 'unknown')}")
                                new_visualizations.append(viz_data)
                            else:
                                logger.warning("Skipping duplicate visualization from chart_renderer.")

                except Exception as e:
                    logger.error(f"Error processing extracted data from tool '{tool_name}' result: {str(e)}. Original message already appended.", exc_info=True)
            # else: If not sql_query/chart_renderer or parsing failed, original message is already appended

        else: # Handle tool execution failure
            error_msg = result["error"]
            logger.warning(f"Appending error message for failed tool: {tool_name}")
            new_messages.append(ToolMessage(
                content=f"Tool execution failed: {error_msg}",
                tool_call_id=tool_id,
                name=tool_name 
            ))

    # --- Return updates --- 
    update_dict = {}
    if new_messages:
        update_dict["messages"] = new_messages
    if new_tables:
        update_dict["tables"] = new_tables
    if new_visualizations:
        update_dict["visualizations"] = new_visualizations

    logger.debug(f"Tool handler returning updates: { {k: len(v) if isinstance(v, list) else v for k, v in update_dict.items()} }")
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

# --- NEW Node for Hierarchy Resolution ---
async def resolve_hierarchy_node(state: AgentState, tools: List[Any]) -> Dict[str, Any]:
    """Executes ONLY the hierarchy_name_resolver tool if called.
    Simplified version of async_tools_node_handler.
    """
    logger.debug("Entering resolve_hierarchy_node")
    messages = state.get("messages", [])
    last_message = messages[-1] if messages else None

    if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
        logger.warning("resolve_hierarchy_node: No AIMessage or tool calls found.")
        return {"messages": []}

    # Find the hierarchy resolver tool call
    resolver_tool_call = None
    for tool_call in last_message.tool_calls:
        if tool_call.get("name") == HierarchyNameResolverTool.__name__:
            resolver_tool_call = tool_call
            break

    if not resolver_tool_call:
        logger.warning(f"resolve_hierarchy_node: Expected {HierarchyNameResolverTool.__name__} call, not found.")
        return {"messages": []}

    # Find the corresponding tool implementation
    tool_map = {tool.name: tool for tool in tools}
    resolver_tool = tool_map.get(HierarchyNameResolverTool.__name__)

    if not resolver_tool:
         logger.error(f"resolve_hierarchy_node: {HierarchyNameResolverTool.__name__} tool implementation not found.")
         return {"messages": [ToolMessage(content=f"Error: Tool {HierarchyNameResolverTool.__name__} not available.", tool_call_id=resolver_tool_call['id'], name=HierarchyNameResolverTool.__name__)]}

    try:
        args = resolver_tool_call.get("args", {})
        tool_id = resolver_tool_call.get("id", "")
        tool_name = resolver_tool_call.get("name", "")

        logger.info(f"Executing single tool: {tool_name} with args: {args}")
        tool_output = await resolver_tool.ainvoke(args)

        # Convert output to string if needed (similar logic to async_tools_node_handler)
        if isinstance(tool_output, dict):
            content_str = json.dumps(tool_output)
        else:
            content_str = str(tool_output)

        logger.info(f"Tool '{tool_name}' execution completed successfully")
        return {"messages": [ToolMessage(content=content_str, tool_call_id=tool_id, name=tool_name)]}

    except Exception as e:
        error_msg = f"Error executing tool '{tool_name}' with args {args}: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return {"messages": [ToolMessage(content=f"Tool execution failed: {error_msg}", tool_call_id=tool_id, name=tool_name)]}


# --- Conditional Edge Logic (Updated) ---
def should_continue(state: AgentState) -> str:
    """Determines the next step based on the last message."""
    last_message = state["messages"][-1] if state["messages"] else None

    if not isinstance(last_message, AIMessage):
        # If the last message isn't from the AI (e.g., initial HumanMessage, or a ToolMessage from the handler),
        # we likely need the agent to process it.
        # However, after the tool handler, we *always* loop back to the agent.
        # This condition might occur if the graph somehow ends up here unexpectedly.
        # Let's default to ending, assuming the agent should have produced an AIMessage.
        logger.warning(f"should_continue: Last message is not AIMessage ({type(last_message)}), routing to END.")
        return END

    # Check if the AI decided on the final answer by calling FinalApiResponseStructure
    if last_message.tool_calls:
        for tool_call in last_message.tool_calls:
            if tool_call.get("name") == FinalApiResponseStructure.__name__:
                logger.debug("Conditional edge: FinalApiResponseStructure call detected, routing to END.")
                # The final structure should be in state['final_response_structure'] now
                return END # Signal the graph to end

    # Check if the AI called ONLY the hierarchy resolver tool
    if last_message.tool_calls:
        operational_calls = [tc for tc in last_message.tool_calls if tc.get("name") != FinalApiResponseStructure.__name__]
        is_only_resolver = len(operational_calls) == 1 and operational_calls[0].get("name") == HierarchyNameResolverTool.__name__

        if is_only_resolver:
            logger.debug(f"Conditional edge: Only {HierarchyNameResolverTool.__name__} called, routing to resolve_hierarchy node.")
            return "resolve_hierarchy"

    # Check if the AI called other tools (or a mix)
    if last_message.tool_calls:
         # Filter out any accidental FinalApiResponseStructure calls if logic above missed it
         operational_calls = [tc for tc in last_message.tool_calls if tc.get("name") != FinalApiResponseStructure.__name__]
         if operational_calls:
              logger.debug(f"Conditional edge: Operational tool calls ({len(operational_calls)}) detected, routing to tools node.")
              return "use_tool" # Route to execute tools

    # If the AI message has no tool calls (e.g., a direct text response, which it shouldn't do per the prompt)
    # or only had a FinalApiResponseStructure call (handled above), end the graph.
    # This branch handles cases where the agent might unexpectedly return plain text, or if only the final structure was called.
    logger.debug("Conditional edge: No operational tool calls in AIMessage, routing to END.")
    return END

# --- Create LangGraph Agent (Updated) ---
def create_graph_app(organization_id: str) -> StateGraph:
    """
    Create the updated LangGraph application.
    The agent node now directly generates the final response structure when done.
    """
    # Set up the LLM agent with tools and the final response structure binding
    llm_with_bindings = create_llm_with_tools_and_final_response_structure(organization_id)

    # Get operational tools for the handler node
    operational_tools = get_tools(organization_id)

    # Create the agent node wrapper
    agent_node_wrapper = functools.partial(agent_node, llm_with_structured_output=llm_with_bindings)

    # Create the tools node wrapper (passing all operational tools)
    tools_handler_with_tools = functools.partial(async_tools_node_handler, tools=operational_tools)

    # Create the hierarchy resolver node wrapper (also needs the tools list to find the implementation)
    resolver_handler_with_tools = functools.partial(resolve_hierarchy_node, tools=operational_tools)

    # --- Define the graph ---
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("agent", agent_node_wrapper)
    workflow.add_node("tools", tools_handler_with_tools) # Node for general tools
    workflow.add_node("resolve_hierarchy", resolver_handler_with_tools) # Dedicated node for resolver
    # REMOVED: workflow.add_node("format_for_api", format_for_api_node)

    # Set the entry point
    workflow.set_entry_point("agent")

    # Define conditional edges from the agent node
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "resolve_hierarchy": "resolve_hierarchy", # Route to dedicated resolver node
            "use_tool": "tools", # Route to general tools node
            END: END
        }
    )

    # Add edges from tool nodes back to the agent
    workflow.add_edge("tools", "agent") # General tools loop back
    workflow.add_edge("resolve_hierarchy", "agent") # Resolver node also loops back to agent

    # Compile the graph
    logger.info("Compiling LangGraph workflow...")
    graph_app = workflow.compile()
    logger.info("LangGraph workflow compiled.")
    return graph_app


# --- Refactored process_chat_message (Simplified) ---
async def process_chat_message(
    organization_id: str,
    message: str,
    session_id: Optional[str] = None,
    chat_history: Optional[List[Dict]] = None,
) -> Dict[str, Any]:
    """
    Processes a user chat message using the refactored LangGraph agent.
    Extracts the final response structure directly from the agent's output.
    """
    logger.info(f"Processing chat message for org {organization_id}, session {session_id}. Original message: '{message[:100]}...'")

    # ---> Organization ID Validation <--- 
    try:
        uuid.UUID(organization_id)
        logger.debug(f"Organization ID {organization_id} is valid UUID.")
    except ValueError:
        logger.error(f"Invalid organization_id format: {organization_id}")
        return {"status": "error", "error": "Invalid organization identifier.", "data": None}

    # ---> Chat History Validation & Initial State Construction <--- 
    initial_messages = []
    validated_history = []
    if chat_history:
        logger.debug(f"Processing provided chat history ({len(chat_history)} items)...")
        for i, item in enumerate(chat_history):
            if isinstance(item, dict) and "role" in item and "content" in item:
                role = item["role"].lower()
                content = item["content"]
                if role == "user" or role == "human":
                    validated_history.append(HumanMessage(content=content))
                elif role == "assistant" or role == "ai":
                    # Basic handling - assumes simple text assistant messages for history
                    validated_history.append(AIMessage(content=content))
                else:
                    logger.warning(f"Skipping chat history item {i} with unknown role: {item.get('role')}")
            else:
                logger.warning(f"Skipping invalid chat history item {i}: {item}")
        logger.debug(f"Validated chat history contains {len(validated_history)} messages.")
        initial_messages.extend(validated_history)
    else:
        logger.debug("No chat history provided.")
        
    # Add the current user message
    initial_messages.append(HumanMessage(content=message))

    # ---> State Pruning <--- 
    if len(initial_messages) > settings.MAX_STATE_MESSAGES:
        logger.warning(f"Initial message count ({len(initial_messages)}) exceeds limit ({settings.MAX_STATE_MESSAGES}). Pruning history...")
        # Keep the last N messages
        initial_messages = initial_messages[-settings.MAX_STATE_MESSAGES:]
        logger.info(f"Pruned message history to {len(initial_messages)} messages.")

    initial_state = AgentState(
        messages=initial_messages,
        tables=[],
        visualizations=[],
        final_response_structure=None
    )

    try:
        graph_app = create_graph_app(organization_id)
        # Ensure the graph app is awaited if it's an async compilation
        if asyncio.iscoroutine(graph_app):
            graph_app = await graph_app
            
        logger.info("Invoking LangGraph workflow...")
        # ---> Apply Recursion Limit <--- 
        final_state = await graph_app.ainvoke(
            initial_state,
            config=RunnableConfig(
                 recursion_limit=settings.MAX_GRAPH_ITERATIONS,
                 configurable={})
        )
        logger.info("LangGraph workflow invocation complete.")

        # Extract final response
        if final_state.get("final_response_structure"):
            structured_response = final_state["final_response_structure"]
            logger.info(f"Successfully obtained FinalApiResponseStructure: {structured_response}")
            
            # Filter tables/visualizations based on include flags
            final_tables = [
                table for i, table in enumerate(final_state.get('tables', []))
                if i < len(structured_response.include_tables) and structured_response.include_tables[i]
            ]
            final_visualizations = [
                viz for i, viz in enumerate(final_state.get('visualizations', []))
                if i < len(structured_response.include_visualizations) and structured_response.include_visualizations[i]
            ]

            # --- Mismatch Check --- 
            state_table_count = len(final_state.get('tables', []))
            state_viz_count = len(final_state.get('visualizations', []))
            llm_table_flags = len(structured_response.include_tables)
            llm_viz_flags = len(structured_response.include_visualizations)
            
            if llm_table_flags != state_table_count:
                logger.warning(f"Mismatch: LLM provided {llm_table_flags} include_tables flags ({structured_response.include_tables}), but state has {state_table_count} tables. Recalculating inclusion based on flags.")
                # Recalculate based on available flags
                final_tables = [
                    table for i, table in enumerate(final_state.get('tables', []))
                    if i < llm_table_flags and structured_response.include_tables[i]
                ]
            if llm_viz_flags != state_viz_count:
                logger.warning(f"Mismatch: LLM provided {llm_viz_flags} include_visualizations flags ({structured_response.include_visualizations}), but state has {state_viz_count} visualizations. Recalculating inclusion based on flags.")
                final_visualizations = [
                    viz for i, viz in enumerate(final_state.get('visualizations', []))
                    if i < llm_viz_flags and structured_response.include_visualizations[i]
                ]
            
            logger.info(f"Final response includes {len(final_tables)} tables and {len(final_visualizations)} visualizations.")
            
            response_data = ChatData(
                text=structured_response.text,
                tables=final_tables if final_tables else None, # Return None if empty list
                visualizations=final_visualizations if final_visualizations else None # Return None if empty list
            )
            response_keys = [k for k, v in response_data.model_dump().items() if v is not None]
            logger.info(f"Successfully processed chat message. Response keys: {response_keys}")
            return {"status": "success", "data": response_data.model_dump(exclude_none=True), "error": None}
        else:
            # Fallback if final structure is missing (e.g., graph ended unexpectedly)
            logger.error("Graph execution finished, but FinalApiResponseStructure was not found in the final state.")
            # Check last message for potential error clues
            last_msg_content = str(final_state['messages'][-1].content) if final_state['messages'] else "No messages."
            fallback_text = f"An unexpected error occurred. Could not generate a final response. Last known state: {last_msg_content[:200]}..."
            return {"status": "error", "error": fallback_text, "data": None}

    except Exception as e:
        logger.error(f"Unhandled exception during chat processing for org {organization_id}: {str(e)}", exc_info=True)
        # Check if it's a recursion limit error
        if "recursion limit" in str(e).lower():
             return {"status": "error", "error": f"Processing aborted due to exceeding complexity limit ({settings.MAX_GRAPH_ITERATIONS} iterations). Please simplify your request.", "data": None}
        return {"status": "error", "error": f"An internal error occurred: {str(e)}", "data": None}


# --- test_azure_openai_connection() remains the same ---
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