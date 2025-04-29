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
    # Add request_id for logging context
    request_id: Optional[str] = None


# --- System Prompt ---
SYSTEM_PROMPT_TEMPLATE = """You are a professional data assistant for the Bibliotheca chatbot API.

Your primary responsibility is to analyze organizational data and provide accurate insights to users based on the request's context.
**Key Context:** The necessary `organization_id` for data scoping is always provided implicitly through the tool context; **NEVER ask the user for it or use placeholders like 'your-organization-id'.**

# --- CRITICAL FINAL STEP --- #
**MANDATORY:** You **MUST ALWAYS** conclude your response by invoking the `FinalApiResponseStructure` tool. 
*   This applies in **ALL** situations: successful answers, reporting errors, refusing requests, simple greetings.
*   **DO NOT** provide the final answer as plain text in the message content. Your final output MUST be a call to `FinalApiResponseStructure`.
*   Failure to call `FinalApiResponseStructure` as the absolute final step is an error.
# --- END CRITICAL FINAL STEP --- #

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
    *   **Time Descriptions:** When the user query includes time references (e.g., 'last week', 'last Christmas', 'March'), pass these descriptions **semantically** in the `query_description`. **DO NOT** attempt to calculate the specific date yourself (e.g., do not change 'last Christmas' to 'December 25th, 2024'). Rely on the `sql_query` tool to interpret these descriptions using its internal context.
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

16. **Invoke `FinalApiResponseStructure` (MANDATORY FINAL STEP - REITERATED):** As stated above, you **MUST ALWAYS** conclude your response by invoking the `FinalApiResponseStructure` tool. Populate its arguments based on your analysis and the preceding guidelines (especially #13 for table inclusion and accuracy). **DO NOT output plain text. Your final action MUST be this tool call.**

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
    *   User Query: "Tell me about quantum physics"
    *   Analysis: Out of scope (general knowledge).
    *   Your Response: Invoke `FinalApiResponseStructure(text="I cannot provide information about quantum physics. My function is limited to answering questions about library data.", include_tables=[], include_visualizations=[])` # Note: Still uses the structure!
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
    request_id = state.get("request_id")
    logger.debug(f"[AgentNode] Entering agent node...")

    response = None
    final_structure = None
    parser = PydanticToolsParser(tools=[FinalApiResponseStructure])
    preprocessed_state = _preprocess_state_for_llm(state)
    failures = []

    for attempt in range(3):
        try:
            response = llm_with_structured_output.invoke(preprocessed_state)
            # CRITICAL DEBUG LOG:
            logger.debug(f"[AgentNode] Raw LLM response (Attempt {attempt + 1}): {response.pretty_repr() if isinstance(response, BaseMessage) else response}")

            is_valid_response = False
            if isinstance(response, AIMessage) and response.tool_calls:
                first_tool_call = response.tool_calls[0]
                if first_tool_call.get("name") != FinalApiResponseStructure.__name__:
                    is_valid_response = True # Operational tool call
                else:
                    # Attempt to parse FinalApiResponseStructure
                    try:
                        args = first_tool_call.get("args", {})
                        # Add defaults *before* parsing
                        if "include_tables" not in args: args["include_tables"] = [False] * len(state.get('tables', []))
                        if "include_visualizations" not in args: args["include_visualizations"] = [False] * len(state.get('visualizations', []))
                        for field in ['include_tables', 'include_visualizations']:
                            if field in args and isinstance(args[field], bool): args[field] = [args[field]] * max(1, len(state.get(field.replace('include_', ''), [])))

                        parsed_objects = parser.invoke(response)
                        if parsed_objects:
                            final_structure = parsed_objects[0]
                            is_valid_response = True
                            logger.debug(f"[AgentNode] LLM returned valid FinalApiResponseStructure.")
                        else: failures.append(f"Attempt {attempt + 1}: Parser returned empty list")
                    except ValidationError as e:
                        failures.append(f"Attempt {attempt+1}: FinalAPIStructure validation fail: {e}")
                        logger.warning(f"[AgentNode] FinalAPIStructure validation failed (attempt {attempt+1}). Trying direct init.")
                        try: # Fallback direct init
                            final_structure = FinalApiResponseStructure(**args)
                            is_valid_response = True
                            logger.debug(f"[AgentNode] Created FinalAPIStructure via direct init (attempt {attempt+1}).")
                        except Exception as direct_err: failures.append(f"Attempt {attempt+1}: Direct init failed: {direct_err}")
                    except Exception as e: failures.append(f"Attempt {attempt + 1}: Error parsing FinalAPIStructure: {e}")
            else: failures.append(f"Attempt {attempt + 1}: Response not AIMessage or no tool calls.")

            if is_valid_response: break
        except Exception as e:
            logger.error(f"[AgentNode] Exception during LLM invocation (Attempt {attempt + 1}): {e}", exc_info=True)
            failures.append(f"Attempt {attempt + 1}: LLM invocation exception: {str(e)}")

    # --- Fallback Logic ---
    final_structure_needed = False
    last_response_is_plain_ai = False
    if response and isinstance(response, AIMessage):
        if response.tool_calls: final_structure_needed = (response.tool_calls[0].get("name") == FinalApiResponseStructure.__name__)
        else: last_response_is_plain_ai = True; final_structure_needed = True

    if final_structure_needed and final_structure is None:
        logger.warning(f"[AgentNode] Triggering fallback logic for FinalApiResponseStructure. Failures: {failures}")
        fallback_text = "Error formatting final response."
        if last_response_is_plain_ai and isinstance(response.content, str) and response.content.strip():
             fallback_text = response.content.strip()
             logger.debug(f"[AgentNode] Using LLM's plain text for fallback structure.")
        else: fallback_text = "Could not generate final response structure. Check logs."
        try:
            final_structure = FinalApiResponseStructure(
                text=fallback_text, include_tables=[True] * len(state.get("tables", [])),
                include_visualizations=[True] * len(state.get("visualizations", [])))
            logger.debug(f"[AgentNode] Created fallback FinalApiResponseStructure.")
        except Exception as fallback_err:
            logger.error(f"[AgentNode] Error creating fallback structure: {fallback_err}", exc_info=True)
            final_structure = None

    logger.debug(f"[AgentNode] Exiting agent node.")
    return {"messages": [response] if response else [], "final_response_structure": final_structure}

def _preprocess_state_for_llm(state: AgentState) -> AgentState:
    """
    Preprocess the state to ensure it's optimized for LLM context window.
    This helps prevent issues with the LLM failing due to context limitations.
    """
    # Make a shallow copy of the state to avoid modifying the original
    processed_state = {k: v for k, v in state.items()}
    
    # Handle messages - prioritize keeping tool outputs and recent messages
    if 'messages' in processed_state:
        messages = processed_state['messages']
        
        # If we have too many messages, we need to prune
        if len(messages) > settings.MAX_STATE_MESSAGES:
            logger.debug(f"Pruning messages from {len(messages)} to {settings.MAX_STATE_MESSAGES}")
            
            # Always keep the user's first and last message for context
            first_user_msg = next((msg for msg in messages if isinstance(msg, HumanMessage)), None)
            last_user_msg = next((msg for msg in reversed(messages) if isinstance(msg, HumanMessage)), None)
            
            # Always keep the most recent tool messages as they contain crucial data
            tool_messages = [msg for msg in messages if hasattr(msg, 'tool_call_id')]
            recent_tool_messages = tool_messages[-5:] if len(tool_messages) > 5 else tool_messages
            
            # Create a set of message IDs to keep
            keep_ids = set()
            if first_user_msg:
                keep_ids.add(id(first_user_msg))
            if last_user_msg:
                keep_ids.add(id(last_user_msg))
            for msg in recent_tool_messages:
                keep_ids.add(id(msg))
                
            # Keep a few of the most recent AI messages
            ai_messages = [msg for msg in messages if isinstance(msg, AIMessage)]
            for msg in ai_messages[-2:]:
                keep_ids.add(id(msg))
                
            # Filter messages based on keep_ids set and most recent ones
            priority_messages = [msg for msg in messages if id(msg) in keep_ids]
            remaining_slots = settings.MAX_STATE_MESSAGES - len(priority_messages)
            
            # Fill remaining slots with the most recent messages
            remaining_messages = [msg for msg in reversed(messages) if id(msg) not in keep_ids][:remaining_slots]
            
            # Combine and sort by original order
            preserved_messages = priority_messages + remaining_messages
            original_order = {id(msg): i for i, msg in enumerate(messages)}
            processed_state['messages'] = sorted(preserved_messages, key=lambda msg: original_order.get(id(msg), 999999))
            
            logger.debug(f"After pruning, kept {len(processed_state['messages'])} messages")
    
    # Process tables - limit large ones
    if 'tables' in processed_state and processed_state['tables']:
        for i, table in enumerate(processed_state['tables']):
            # If a table has too many rows, truncate it
            if 'rows' in table and len(table['rows']) > 10:
                table['rows'] = table['rows'][:10]
                if 'metadata' not in table:
                    table['metadata'] = {}
                table['metadata']['truncated'] = True
                table['metadata']['original_rows'] = len(state['tables'][i]['rows'])
    
    return processed_state

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
    request_id = state.get("request_id")
    logger.debug(f"[ToolsNode] Entering tool handler.")
    last_message = state["messages"][-1]
    tool_map = {tool.name: tool for tool in tools}
    tool_executions = []
    new_messages = []; new_tables = []; new_visualizations = []
    successful_calls = 0
    operational_tool_calls = []

    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        operational_tool_calls = [tc for tc in last_message.tool_calls if tc.get("name") != FinalApiResponseStructure.__name__]

    if not operational_tool_calls:
        logger.debug(f"[ToolsNode] No operational tool calls found.")
        return {}

    logger.debug(f"[ToolsNode] Dispatching {len(operational_tool_calls)} tool calls: {[tc.get('name') for tc in operational_tool_calls]}")
    for tool_call in operational_tool_calls:
        tool_name = tool_call.get("name")
        tool_id = tool_call.get("id", "") # Get tool_id here
        if tool_name in tool_map:
            tool_executions.append({"tool": tool_map[tool_name], "args": tool_call.get("args", {}), "id": tool_id, "name": tool_name, "retries_left": 2})
        else:
             logger.error(f"[ToolsNode] Tool '{tool_name}' requested but not found.")
             new_messages.append(ToolMessage(content=f"Error: Tool '{tool_name}' not found.", tool_call_id=tool_id, name=tool_name))

    if not tool_executions: # Handle case where all requested tools were missing
        logger.warning(f"[ToolsNode] No tools could be prepared for execution.")
        return {"messages": new_messages} if new_messages else {}

    sem = asyncio.Semaphore(settings.MAX_CONCURRENT_TOOLS)
    async def execute_with_retry(execution_details):
        # Simplified retry logic focusing on logging
        tool = execution_details["tool"]; args = execution_details["args"]; tool_id = execution_details["id"]; tool_name = execution_details["name"]; retries = execution_details["retries_left"]
        try:
            async with sem:
                 # DEBUG log includes args, might be verbose
                 logger.debug(f"[ToolsNode] Executing tool '{tool_name}' (ID: {tool_id})")
                 content = await tool.ainvoke(args)
                 if asyncio.iscoroutine(content): content = await content
                 content_str = json.dumps(content) if isinstance(content, dict) else str(content)
                 logger.debug(f"[ToolsNode] Tool '{tool_name}' (ID: {tool_id}) completed.")
                 return {"success": True, "message": ToolMessage(content=content_str, tool_call_id=tool_id, name=tool_name), "raw_content": content, "tool_name": tool_name, "tool_id": tool_id}
        except Exception as e:
            error_msg_for_log = f"Error executing tool '{tool_name}' (ID: {tool_id}): {str(e)}"
            if retries > 0 and _is_retryable_error(e):
                logger.warning(f"[ToolsNode] {error_msg_for_log} - Retrying ({retries} left).", exc_info=False)
                await asyncio.sleep(0.5 * (3 - retries)); execution_details["retries_left"] = retries - 1
                return await execute_with_retry(execution_details)
            else:
                logger.error(f"[ToolsNode] {error_msg_for_log}", exc_info=True)
                # Keep original error message structure for return value
                return {"success": False, "error": f"{error_msg_for_log}", "tool_name": tool_name, "tool_id": tool_id}

    results = await asyncio.gather(*[execute_with_retry(exec_data) for exec_data in tool_executions])

    # Simplified result processing
    for result in results:
        tool_name = result.get("tool_name", "unknown")
        tool_id = result.get("tool_id", "") # Ensure tool_id is available for error message
        if result["success"]:
            successful_calls += 1
            new_messages.append(result["message"])
            raw_content = result.get("raw_content")
            # Basic table/viz extraction (no dupe check here for simplicity)
            if isinstance(raw_content, dict):
                 if tool_name == "sql_query" and isinstance(raw_content.get("table"), dict):
                     table_data = raw_content["table"]
                     if isinstance(table_data.get('columns'), list) and isinstance(table_data.get('rows'), list):
                          new_tables.append(table_data) # Add table data
                 elif tool_name == "chart_renderer" and isinstance(raw_content.get("visualization"), dict):
                     viz_data = raw_content["visualization"]
                     if viz_data.get('type') and viz_data.get('image_url'):
                          new_visualizations.append(viz_data) # Add viz data
        else:
            logger.warning(f"[ToolsNode] Tool call '{tool_name}' (ID: {tool_id}) failed. Error appended.")
            # Append error message only if tool mapping didn't already do it
            if not any(getattr(m, 'tool_call_id', None) == tool_id for m in new_messages if isinstance(m, ToolMessage)):
                 # Ensure correct arguments for ToolMessage
                 error_content = f"Tool execution failed: {result.get('error', 'Unknown error')}"
                 new_messages.append(ToolMessage(content=error_content, tool_call_id=tool_id, name=tool_name))


    update_dict = {}
    if new_messages: update_dict["messages"] = new_messages
    if new_tables: update_dict["tables"] = new_tables
    if new_visualizations: update_dict["visualizations"] = new_visualizations

    logger.debug(f"[ToolsNode] Tool handler finished. {successful_calls}/{len(results)} calls successful. State updates: {list(update_dict.keys())}")
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

# --- Node for Hierarchy Resolution ---
async def resolve_hierarchy_node(state: AgentState, tools: List[Any]) -> Dict[str, Any]:
    """Executes ONLY the hierarchy_name_resolver tool if called.
    Simplified version of async_tools_node_handler.
    """
    request_id = state.get("request_id")
    logger.debug(f"[ResolveHierarchyNode] Entering hierarchy resolver node.")
    messages = state.get("messages", [])
    last_message = messages[-1] if messages else None
    resolver_tool_call = None
    tool_name = HierarchyNameResolverTool.__name__ # Tool name is fixed

    if isinstance(last_message, AIMessage) and last_message.tool_calls:
         for tc in last_message.tool_calls:
              if tc.get("name") == tool_name: resolver_tool_call = tc; break

    if not resolver_tool_call:
        logger.warning(f"[ResolveHierarchyNode] Expected {tool_name} call, not found.")
        return {"messages": []}

    tool_map = {tool.name: tool for tool in tools}
    resolver_tool = tool_map.get(tool_name)
    if not resolver_tool:
         logger.error(f"[ResolveHierarchyNode] {tool_name} tool implementation not found.")
         # Ensure error message has correct args
         err_tool_id = resolver_tool_call.get('id', '') if resolver_tool_call else ''
         return {"messages": [ToolMessage(content=f"Error: Tool {tool_name} not available.", tool_call_id=err_tool_id, name=tool_name)]}

    tool_id = resolver_tool_call.get("id", "")
    args = resolver_tool_call.get("args", {})
    try:
        logger.debug(f"[ResolveHierarchyNode] Executing {tool_name} (ID: {tool_id})")
        content = await resolver_tool.ainvoke(args)
        if asyncio.iscoroutine(content): content = await content
        content_str = json.dumps(content) if isinstance(content, dict) else str(content)
        logger.debug(f"[ResolveHierarchyNode] {tool_name} (ID: {tool_id}) completed successfully.")
        return {"messages": [ToolMessage(content=content_str, tool_call_id=tool_id, name=tool_name)]}
    except Exception as e:
        error_msg_for_log = f"Error executing tool '{tool_name}' (ID: {tool_id}): {str(e)}";
        logger.error(f"[ResolveHierarchyNode] {error_msg_for_log}", exc_info=True)
        # Ensure error message has correct args
        return {"messages": [ToolMessage(content=f"Tool execution failed: {str(e)}", tool_call_id=tool_id, name=tool_name)]}


# --- Conditional Edge Logic (Updated) ---
def should_continue(state: AgentState) -> str:
    """Determines the next step based on the last message."""
    request_id = state.get("request_id")
    last_message = state["messages"][-1] if state["messages"] else None
    next_node = END

    if isinstance(last_message, AIMessage):
        if last_message.tool_calls:
            is_final_call = any(tc.get("name") == FinalApiResponseStructure.__name__ for tc in last_message.tool_calls)
            is_resolver_only = len(last_message.tool_calls) == 1 and last_message.tool_calls[0].get("name") == HierarchyNameResolverTool.__name__
            has_operational_calls = any(tc.get("name") != FinalApiResponseStructure.__name__ for tc in last_message.tool_calls)

            if is_final_call: next_node = END
            elif is_resolver_only: next_node = "resolve_hierarchy"
            elif has_operational_calls: next_node = "use_tool"
    else:
        logger.warning(f"[ShouldContinue] Last message not AIMessage ({type(last_message).__name__}), routing to END.")

    logger.debug(f"[ShouldContinue] Routing decision: '{next_node}'.")
    return next_node

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
    request_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Processes a user chat message using the refactored LangGraph agent.
    Extracts the final response structure directly from the agent's output.
    """
    # Generate ReqID if not provided, keep it for potential tracing, but not for logging prefix
    req_id = request_id or str(uuid.uuid4())
     # Log start of processing
    logger.info(f"--- Starting request processing ---")
    # Log full user message at INFO level
    logger.info(f"Org: {organization_id}, Session: {session_id}, History: {len(chat_history) if chat_history else 0}, User Message: '{message}'") 

    # TODO: Load/manage chat history based on session_id (if provided)

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
        logger.debug(f"No chat history provided.")
        
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
        final_response_structure=None,
        request_id=req_id # Pass request_id into state
    )

    try:
        graph_app = create_graph_app(organization_id)
        # Ensure the graph app is awaited if it's an async compilation
        if asyncio.iscoroutine(graph_app):
            graph_app = await graph_app
            
        logger.info(f"Invoking LangGraph workflow...")
        # ---> Apply Recursion Limit <--- 
        final_state = await graph_app.ainvoke(
            initial_state,
            config=RunnableConfig(
                 recursion_limit=settings.MAX_GRAPH_ITERATIONS,
                 configurable={})
        )
        logger.info(f"LangGraph workflow invocation complete.")

        # Extract final response
        if final_state.get("final_response_structure"):
            structured_response = final_state["final_response_structure"]
            logger.info(f"Successfully obtained FinalApiResponseStructure.")
            # Filter tables/visualizations based on include flags
            final_tables = [t for i, t in enumerate(final_state.get('tables', [])) if i < len(structured_response.include_tables) and structured_response.include_tables[i]]
            final_visualizations = [v for i, v in enumerate(final_state.get('visualizations', [])) if i < len(structured_response.include_visualizations) and structured_response.include_visualizations[i]]
            # Log if flags resulted in different counts than available (can indicate LLM confusion)
            if structured_response.include_tables.count(True) != len(final_tables) or structured_response.include_visualizations.count(True) != len(final_visualizations):
                  logger.warning(f"Inclusion mismatch. State had T={len(final_state.get('tables',[]))}/V={len(final_state.get('visualizations',[]))}. Flags T={structured_response.include_tables}/V={structured_response.include_visualizations}. Included T={len(final_tables)}/V={len(final_visualizations)}.")

            response_data = ChatData(text=structured_response.text, tables=final_tables or None, visualizations=final_visualizations or None)
            logger.info(f"Final AI Response Text: {structured_response.text[:250]}{'...' if len(structured_response.text) > 250 else ''}")
            logger.info(f"Success. Returning response.")
            logger.info(f"--- Finished request processing ---")
            return {"status": "success", "data": response_data.model_dump(exclude_none=True), "error": None}
        else:
            logger.error(f"Graph finished, but FinalApiResponseStructure missing. Returning error.")
            last_msg_content = str(final_state['messages'][-1].content) if final_state.get('messages') else "No messages."
            # Simplified error details for log brevity
            error_details = {"last_message_preview": last_msg_content[:100], "cause": "Final structure missing from state"}
            return {"status": "error", "data": None, "error": {"code": "FINAL_STRUCTURE_MISSING", "message": "Unable to generate final response", "details": error_details}}

    except Exception as e:
        logger.error(f"Unhandled exception during processing: {str(e)}", exc_info=True)
        error_code = "INTERNAL_ERROR"; error_message = "Internal error."; error_details = {"exception": str(e)}
        if "recursion limit" in str(e).lower(): error_code = "RECURSION_LIMIT_EXCEEDED"; error_message = "Request complexity limit exceeded."
        return {"status": "error", "data": None, "error": {"code": error_code, "message": error_message, "details": error_details}}
    finally:
        logger.info(f"--- Finished request processing ---")


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