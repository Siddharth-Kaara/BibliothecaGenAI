import logging
import json
import asyncio
from typing import Any, Dict, List, Optional, Tuple, TypedDict, Annotated, Sequence
import operator
from pydantic import BaseModel, Field
import functools
from pydantic_core import ValidationError # Import for specific error handling
import uuid # Import for UUID validation
import datetime # Import datetime

# LangChain & LangGraph Imports
from langchain_core.messages import BaseMessage, FunctionMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import AzureChatOpenAI
from langchain_core.output_parsers.openai_tools import PydanticToolsParser, JsonOutputToolsParser
from langchain_core.runnables import RunnableConfig # Import for config
from langgraph.graph import StateGraph, END
from langchain.tools import BaseTool

# Local Imports
from app.core.config import settings
from app.langchain.tools.sql_tool import SQLExecutionTool
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


# --- Database Schema Tool ---
class DatabaseSchemaTool(BaseTool):
    """Tool for fetching database schema information to help generate SQL queries."""
    
    name: str = "get_schema_info"
    description: str = """Fetches database schema information to help generate SQL queries.
    Use this tool when you need details about database tables, columns, and relationships.
    The output includes table names, column descriptions, primary/foreign keys, and data types."""
    
    class SchemaQueryArgs(BaseModel):
        db_name: str = Field(
            default="report_management", 
            description="The database name to fetch schema for. Default is 'report_management'."
        )
    
    args_schema: type[BaseModel] = SchemaQueryArgs
    
    def _run(self, db_name: str = "report_management") -> str:
        """Get schema information from predefined schema definitions."""
        logger.debug(f"[DatabaseSchemaTool] Fetching schema for database: {db_name}")
        
        from app.db.schema_definitions import SCHEMA_DEFINITIONS
        
        if db_name not in SCHEMA_DEFINITIONS:
            error_msg = f"No schema definition found for database {db_name}."
            logger.warning(f"[DatabaseSchemaTool] {error_msg}")
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
        
        logger.info(f"[DatabaseSchemaTool] Successfully retrieved schema for {db_name}")
        return "\n".join(schema_info)
    
    async def _arun(self, db_name: str = "report_management") -> str:
        """Async implementation of schema retrieval."""
        return self._run(db_name=db_name)


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

# --- Current Time Context --- #
Current Date: {current_date}
Day of the Week: {current_day}
Current Year: {current_year}
# --- End Current Time Context --- #

You have access to a single database: **report_management**.
This database contains:
- Event counts and usage statistics (table '5').
- All necessary organizational hierarchy information (table 'hierarchyCaches').

Based on the user's query and the conversation history (including previous tool results), decide the best course of action. You have the following tools available:

{tool_descriptions}

Available Tools:
{tool_names} # Note: FinalApiResponseStructure will be listed here if bound correctly

# --- ORCHESTRATION GUIDELINES --- #

1. **Analyze Request & History:** Always review the conversation history (`messages`) for context, previous tool outputs (`ToolMessage`), and accumulated data (`tables`, `visualizations` in state).

2. **Hierarchy Name Resolution (MANDATORY for location names):**
   - If the user mentions specific organizational hierarchy entities by name (e.g., "Main Library", "Argyle Branch"), you MUST call the `hierarchy_name_resolver` tool **ALONE** as your first action.
   - Pass the list of names as `name_candidates`.
   - This tool uses the correct `organization_id` from the request context automatically.
   - Examine the `ToolMessage` from `hierarchy_name_resolver` in the history after it runs.
   - If any status is 'not_found' or 'error': Inform the user via `FinalApiResponseStructure`.
   - If all relevant names have status 'found': Proceed to the next step using the returned `id` values.

3. **Database Schema Understanding:**
   - Before generating SQL, use the `get_schema_info` tool to understand the database schema.
   - This will give you detailed information about tables, columns and their descriptions.
   - Default is 'report_management' database.

4. **SQL Generation & Execution:**
   - YOU are responsible for generating the SQL query based on the database schema.
   - After generating SQL, use the `execute_sql` tool to execute it.
   - **CRITICAL:** The arguments for `execute_sql` MUST be a JSON object with the keys "sql" and "params".
     - The "sql" key holds the SQL query string.
     - The "params" key holds a dictionary of parameters. This dictionary **MUST** include "organization_id" and any other parameters used in the query.
     - **IMPORTANT**: The *value* for the `organization_id` key MUST be the actual organization ID string (e.g., "b781b517-8954-e811-2a94-0024e880a2b7"), NOT the literal string 'organization_id'.
     - Example arguments structure: `{{"sql": "SELECT ... WHERE \"organizationId\" = :organization_id", "params": {{"organization_id": "<ACTUAL_ORG_ID_VALUE>", "other_param": "value"}}}}` (Replace `<ACTUAL_ORG_ID_VALUE>`)
   - ALWAYS include the correct organization_id value in the `params` dictionary.
   - Include hierarchy IDs in `params` when applicable.

5. **Visualization Strategy (`chart_renderer` tool):** Follow these steps carefully when using the `chart_renderer`:
   a. **When to Use:** ONLY use this tool if:
      - The user explicitly requests a chart/visualization, OR
      - You are presenting complex data (e.g., comparisons across >3 categories/metrics) where a visual representation significantly aids understanding.
      - **AVOID charts for simple comparisons** (e.g., 2-3 items); prefer a text summary (Guideline #7) unless a chart is explicitly requested.
   b. **Data Prerequisite:** Ensure the data needed for the chart exists in the `state['tables']` list, typically from a preceding `execute_sql` call. If not, call `execute_sql` first.
   c. **Argument Structure:** Call `chart_renderer` with a single JSON object containing two keys: `"data"` and `"metadata"`.
   d. **Data Preparation (`"data"` key):**
      - Extract the `table` data (with `"columns"` and `"rows"`) from the relevant `execute_sql` result.
      - Ensure the `data` argument you pass to `chart_renderer` matches this exact format: `{{"columns": [...], "rows": [...]}}`.
      - **Transformation for Comparison Bars:** If creating a bar chart to compare multiple metrics for the same entities (e.g., borrows vs. renewals per branch), transform the "wide" data from SQL (e.g., columns: `Branch`, `Borrows`, `Renewals`) into a "long" format (e.g., columns: `Branch`, `Metric`, `Value`). You must perform this logic.
   e. **Metadata Creation (`"metadata"` key - CRITICAL):**
      - Construct this dictionary **explicitly**. DO NOT rely on defaults.
      - Include: `"type_hint"` (e.g., "bar", "pie"), `"title"`, `"x_column"`, `"y_column"`.
      - Optionally include: `"color_column"` (for hue/grouping), `"x_label"`, `"y_label"`.
      - **CONSISTENCY CHECK (MANDATORY):** Before calling the tool, **verify that every column name specified in your `metadata` (`x_column`, `y_column`, `color_column`) EXACTLY matches a column name present in your prepared `data` dictionary.** If inconsistent, fix either the `data` or the `metadata`.
   f. **Tool Invocation:** Call `chart_renderer` with the fully prepared `data` and `metadata` arguments.

6. **CRITICAL TOOL CHOICE: `execute_sql` vs. `summary_synthesizer`:**
   - **Use Direct SQL Generation (`execute_sql`) IF AND ONLY IF:** The user asks for a comparison OR retrieval of **specific, quantifiable metrics** (e.g., counts, sums of borrows, returns, renewals, logins) for **specific, resolved entities** (e.g., Main Library [ID: xxx], Argyle Branch [ID: yyy]) over a **defined time period**. Your goal is to generate a single, efficient SQL query.
   - **Use `summary_synthesizer` ONLY FOR:** More **open-ended, qualitative summary requests** (e.g., "summarize activity," "tell me about the branches") where specific metrics are not the primary focus, or when the exact metrics are unclear. Call it directly after name resolution (if applicable), providing context in the `query` argument. Its output will be purely text.

7. **Final Response Formation:**
   - Before calling `FinalApiResponseStructure`, examine the gathered data (`tables`, `visualizations`) in the state.
   - **Inclusion Decision:** Determine if any table or visualization provides useful, detailed information supporting the request (esp. for comparisons/breakdowns). If yes:
     - Set the corresponding flag in `include_tables` or `include_visualizations` to `[True]`.
     - Ensure the `text` field is **CONCISE**, focuses on insights/anomalies, and **REFERENCES** the included item(s). **DO NOT** repeat detailed data.
   - **No Inclusion / Summarizer Text:** If no useful tables/visualizations exist, or if `summary_synthesizer` was used, set inclusion flags to `[False]` and provide the full answer/summary in the `text` field.
   - **Accuracy:** Ensure the final `text` accurately reflects and references any included items.

8. **Out-of-Scope Handling:**
   - Refuse requests unrelated to library data or operations.
   - Still use `FinalApiResponseStructure` to deliver the refusal message.

# --- Workflow Summary --- #
1. Analyze Request & History.
2. **IF** hierarchy names present -> Call `hierarchy_name_resolver` FIRST. Check results; Refuse if needed.
3. **DECIDE** Tool (Guideline #6):
   *   Specific Metrics? -> Plan for `execute_sql` (+ optional `chart_renderer`).
   *   Qualitative Summary? -> Plan for `summary_synthesizer`.
4. **IF** using `execute_sql`:
   *   Call `get_schema_info` if needed.
   *   Generate SQL & Call `execute_sql`.
   *   Call `chart_renderer` if appropriate (follow Guideline #5).
5. **IF** using `summary_synthesizer`:
   *   Call `summary_synthesizer`.
6. Formulate final response text (Guideline #7).
7. **ALWAYS** conclude with `FinalApiResponseStructure`.
# --- End Workflow Summary --- #

# --- SQL GENERATION GUIDELINES --- #

When generating SQL queries to be executed by the `execute_sql` tool, follow these strict guidelines:

1. **Parameter Placeholders:** Use parameter placeholders (e.g., :filter_value, :hierarchy_id) for ALL dynamic values derived from the query description (like names, IDs, specific filter values) EXCEPT for time-related values. **DO NOT** use parameters for date/time calculations.

2. **Parameter Dictionary:** Create a valid parameters dictionary mapping placeholder names (without the colon) to their actual values. This MUST include the correct `organization_id` value (see Orchestration Guideline #4).

3. **Quoting & Naming (CRITICAL):**
   - Quote table and column names with double quotes (e.g., "hierarchyCaches", "createdAt").
   - **YOU MUST use the ACTUAL physical table names (e.g., '5' for events, '8' for footfall) in your SQL queries.**
   - The schema description uses logical names (like 'events') for clarity, but your SQL query MUST use the physical names ('5', '8').
   - **Failure to use the physical table names '5' or '8' will cause an error.**

4. **Mandatory Organization Filtering:** ALWAYS filter results by the organization ID. Use the parameter `:organization_id`.
   - **If querying table '5' (event data):** Add `"5"."organizationId" = :organization_id` to your WHERE clause.
   - **If querying table '8' (footfall data):** Add `"8"."organizationId" = :organization_id` to your WHERE clause.
   - **If querying `hierarchyCaches` for the organization's own details:** Filter using `hc."id" = :organization_id`.
   - **If querying `hierarchyCaches` for LOCATIONS WITHIN the organization (e.g., branches, areas):** You MUST filter by the parent ID being the organization ID. Add `hc."parentId" = :organization_id` to your WHERE clause. DO NOT filter using `hc."organizationId"` as this column doesn't exist.

5. **JOINs for Related Data:** When joining table '5' or '8' and `hierarchyCaches`, use appropriate keys like `"5"."hierarchyId" = hc."id"` or `"8"."hierarchyId" = hc."id"`. Remember to apply the organization filter.

6. **Case Sensitivity:** PostgreSQL is case-sensitive; respect exact table/column capitalization.

7. **Column Selection:** Use specific column selection instead of SELECT *.

8. **Sorting:** Add ORDER BY clauses for meaningful sorting, especially when LIMIT is used.

9. **LIMIT Clause:**
   - For standard SELECT queries retrieving multiple rows, ALWAYS include `LIMIT 50` at the end.
   - **DO NOT** add `LIMIT` for aggregate queries (like COUNT(*), SUM(...)) expected to return a single summary row.

10. **Aggregations (COUNT vs SUM):**
    - Use `COUNT(*)` for "how many records/items".
    - Use `SUM("column_name")` for "total number/sum" based on a specific value column (e.g., total logins from column "5").
    - Ensure `GROUP BY` includes all non-aggregated selected columns.

11. **User-Friendly Aliases:**
    - When selecting columns or using aggregate functions (SUM, COUNT, etc.), ALWAYS use descriptive, user-friendly aliases with title casing using the `AS` keyword.
    - Examples: `SELECT hc."hierarchyId" AS "Hierarchy ID"`, `SELECT COUNT(*) AS "Total Records"`, `SELECT SUM("39") AS "Total Entries"`.
    - Do NOT use code-style aliases like `total_entries` or `hierarchyId`.

12. **Benchmarking for Analytical Queries:**
    - If the user asks for analysis or comparison regarding a specific entity (e.g., "is branch X busy?", "compare borrows for branch Y"), *in addition* to selecting the specific metric(s) for that entity, try to include a simple benchmark for comparison in the same query.
    - **Use CTEs for Benchmarks:** The preferred way to calculate an organization-wide average (or similar benchmark) alongside a specific entity's value is using a Common Table Expression (CTE).
    - **Avoid nested aggregates:** Do NOT use invalid nested aggregate/window functions like `AVG(SUM(...)) OVER ()`.
    - Only include this benchmark if it can be done efficiently. The CTE approach is generally efficient.
    - Ensure both the specific value and the benchmark value have clear, user-friendly aliases.

13. **Time Filtering (Generate SQL Directly):**
    - Use the Current Time Context provided above to resolve relative dates and years.
    - If the user query includes time references (e.g., "last week", "yesterday", "past 3 months", "since June 1st", "before 2024"), you MUST generate the appropriate SQL `WHERE` clause condition directly.
    - Use relevant SQL functions like `NOW()`, `CURRENT_DATE`, `INTERVAL`, `DATE_TRUNC`, `EXTRACT`, `MAKE_DATE`, and comparison operators (`>=`, `<`, `BETWEEN`).
    - **Relative Time Interpretation:** For simple relative terms like "last week", "last month", prioritize straightforward intervals like `NOW() - INTERVAL '7 days'` or `NOW() - INTERVAL '1 month'`. For "yesterday", use `CURRENT_DATE - INTERVAL '1 day'`.
    - **Relative Months/Years:** For month names (e.g., "March") without a year, assume the **current year** ({current_year}). For years alone (e.g., "in 2024"), query the whole year.
    - **Specific Day + Month (No Year):** For dates like "April 1st", construct the date using the **current year** ({current_year}). Use `MAKE_DATE({current_year}, <month_number>, <day_number>)`. (No casting needed as {current_year} is already an integer).
    - **Last Working Day Logic:** Based on the `Current Context` Day of the Week ({current_day}), determine the date of the last working day (assuming Mon-Fri). Filter for events matching *that specific date* using `DATE_TRUNC('day', "timestamp_column") = calculated_date`.
        *   If Today is Monday, Last Working Day = `CURRENT_DATE - INTERVAL '3 days'`..
        *   If Today is Tue-Fri, Last Working Day = `CURRENT_DATE - INTERVAL '1 day'`.
        *   Identify the correct timestamp column (check schema).
    - **Example for "March"**: `WHERE EXTRACT(MONTH FROM "eventTimestamp") = 3 AND EXTRACT(YEAR FROM "eventTimestamp") = {current_year}`
    - **Example for "April 1st"**: `WHERE DATE_TRUNC('day', "eventTimestamp") = MAKE_DATE({current_year}, 4, 1)`
    - **Example for "last working day" (if Today is Monday)**: `WHERE DATE_TRUNC('day', "eventTimestamp") = CURRENT_DATE - INTERVAL '3 days'`
    - **Example for "last working day" (if Today is Wednesday)**: `WHERE DATE_TRUNC('day', "eventTimestamp") = CURRENT_DATE - INTERVAL '1 day'`
    - **DO NOT** use parameters like `:start_date` or `:end_date` for these time calculations.

14. **Footfall Queries (Table \"8\"):**
    - If the query asks generally about "footfall", "visitors", "people entering/leaving", or "how many people visited", calculate **both** the sum of entries (`SUM(\"39\")`) and the sum of exits (`SUM(\"40\")`).
    - Alias them clearly (e.g., `AS "Total Entries"`, `AS "Total Exits"`).
    - If the query specifically asks *only* for entries (e.g., "people came in") or *only* for exits (e.g., "people went out"), then only sum the corresponding column ("39" or "40").

# --- END SQL GENERATION GUIDELINES --- #

**MANDATORY FINAL STEP:** Always conclude by calling `FinalApiResponseStructure` with appropriate arguments for text and table/visualization inclusion.
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
        DatabaseSchemaTool(),
        SQLExecutionTool(organization_id=organization_id),
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

    # --- Calculate Time Context --- #
    now = datetime.datetime.now()
    current_date_str = now.strftime("%Y-%m-%d")
    current_day_name = now.strftime("%A") # e.g., Monday
    current_year_int = now.year
    # --- End Time Context Calculation --- #

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
        # --- Inject Time Context --- #
        current_date=current_date_str,
        current_day=current_day_name,
        current_year=current_year_int,
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
                 if tool_name in ["sql_query", "execute_sql"] and isinstance(raw_content.get("table"), dict):
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