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
from app.langchain.tools.summary_tool import SummarySynthesizerTool
from app.langchain.tools.hierarchy_resolver_tool import HierarchyNameResolverTool
from app.schemas.chat import ChatData, ApiChartSpecification, TableData

logger = logging.getLogger(__name__)

# --- New Instruction Structure included directly in FinalApiResponseStructure ---
class ChartSpecFinalInstruction(BaseModel):
    """Defines the specification for a chart to be rendered by the frontend.
       This structure is generated directly by the LLM within the FinalApiResponseStructure.
    """
    source_table_index: int = Field(description="The 0-based index of the table in the agent's 'tables' state that contains the data for this chart.")
    type_hint: str = Field(description="The suggested chart type for the frontend (e.g., 'bar', 'pie', 'line', 'scatter').")
    title: str = Field(description="The title for the chart.")
    x_column: str = Field(description="The name of the column from the source table to use for the X-axis or labels.")
    y_column: str = Field(description="The name of the column from the source table to use for the Y-axis or values.")
    color_column: Optional[str] = Field(default=None, description="Optional: The name of the column to use for grouping data by color/hue (e.g., for grouped bar charts).")
    x_label: Optional[str] = Field(default=None, description="Optional: A descriptive label for the X-axis. Defaults to x_column if not provided.")
    y_label: Optional[str] = Field(default=None, description="Optional: A descriptive label for the Y-axis. Defaults to y_column if not provided.")

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
    # REMOVED: include_visualizations
    # ADDED: chart_specs list containing full specifications
    chart_specs: List[ChartSpecFinalInstruction] = Field(
        default_factory=list,
        description="List of chart specifications to be included in the final API response. The LLM generates these directly when calling this tool."
    )


# --- Define the Agent State ---
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add] # Conversation history
    tables: Annotated[List[Dict[str, Any]], operator.add] # List of tables from sql_query tool calls
    # REMOVED: visualizations - chart specs are now part of FinalApiResponseStructure
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
*   The `FinalApiResponseStructure` includes fields for `text`, `include_tables`, and importantly, `chart_specs`.
*   Failure to call `FinalApiResponseStructure` as the absolute final step is an error.
# --- END CRITICAL FINAL STEP --- #

# --- Current Time Context --- #
Current Date: {current_date}
Day of the Week: {current_day}
Current Year: {current_year}
# --- End Current Time Context --- #

# --- Database Schema (report_management) --- #
{db_schema_string}
# --- End Database Schema --- #

You have access to a single database: **report_management**. The full schema is provided above.
This database contains:
- Event counts and usage statistics (table '5').
- All necessary organizational hierarchy information (table 'hierarchyCaches').

Based on the user's query and the conversation history (including previous tool results), decide the best course of action. You have the following tools available:

{tool_descriptions}

Available Tools:
{tool_names} # Note: Only operational tools and FinalApiResponseStructure are listed here

# --- ORCHESTRATION GUIDELINES --- #

1. **Analyze Request & History:** Always review the conversation history (`messages`) for context, previous tool outputs (`ToolMessage`), and accumulated data (`tables` in state).

2. **Hierarchy Name Resolution (MANDATORY for location names):**
   - If the user mentions specific organizational hierarchy entities by name (e.g., "Main Library", "Argyle Branch"), you MUST call the `hierarchy_name_resolver` tool **ALONE** as your first action.
   - Pass the list of names as `name_candidates`.
   - This tool uses the correct `organization_id` from the request context automatically.
   - Examine the `ToolMessage` from `hierarchy_name_resolver` in the history after it runs.
   - If any status is 'not_found' or 'error': Inform the user via `FinalApiResponseStructure` (with empty `chart_specs`).
   - If all relevant names have status 'found': Proceed to the next step using the returned `id` values.

3. **Database Schema Understanding:**
   - The full schema for the 'report_management' database is provided above in the prompt.
   - **Refer to this schema directly** when generating SQL queries.

4. **SQL Generation & Execution:**
   - YOU are responsible for generating the SQL query based on the database schema provided above.
   - After generating SQL, use the `execute_sql` tool to execute it. The result will be added to the `tables` list in the state.
   - **CRITICAL TRANSITION:** Once `execute_sql` returns the requested quantitative data (counts, sums, etc.) that directly and sufficientlyanswers the user's query, your **IMMEDIATE NEXT STEP** must be to proceed to Guideline #7 and call `FinalApiResponseStructure`. **Do not call `execute_sql` or any other tool again** unless the results were clearly incomplete or erroneous for the user's request.
   - **CRITICAL:** The arguments for `execute_sql` MUST be a JSON object with the keys "sql" and "params".
     - The "sql" key holds the SQL query string.
     - The "params" key holds a dictionary of parameters. This dictionary **MUST** include "organization_id" and any other parameters used in the query.
     - **IMPORTANT**: The *value* for the `organization_id` key MUST be the actual organization ID string (e.g., "b781b517-8954-e811-2a94-0024e880a2b7"), NOT the literal string 'organization_id'.
     - Example arguments structure: `{{"sql": "SELECT ... WHERE \"organizationId\" = :organization_id", "params": {{"organization_id": "<ACTUAL_ORG_ID_VALUE>", "other_param": "value"}}}}` (Replace `<ACTUAL_ORG_ID_VALUE>`)
   - ALWAYS include the correct organization_id value in the `params` dictionary.
   - Include hierarchy IDs in `params` when applicable.

5. **Chart Specification Strategy :** When formulating the final response using `FinalApiResponseStructure`:
    a. **When to Include Charts:** Populate the `chart_specs` list within `FinalApiResponseStructure` ONLY if:
        - The user explicitly requested a chart/visualization, OR
        - You are presenting complex data (e.g., comparisons across >3 categories/metrics) where a visual representation significantly aids understanding.
        - **AVOID charts for simple comparisons** (e.g., 2-3 items); prefer just the `text` summary unless a chart is explicitly requested.
    b. **Data Prerequisite:** Ensure the data needed for any chart specification exists in the `state['tables']` list, typically from a preceding `execute_sql` call.
    c. **Populating `chart_specs`:** For each chart you decide to include, add a `ChartSpecFinalInstruction` object to the `chart_specs` list within the `FinalApiResponseStructure` arguments.
    d. **`ChartSpecFinalInstruction` Fields (CRITICAL):**
        -   `source_table_index`: Specify the **0-based index** of the relevant table in the current `state['tables']` list. This is crucial for linking the specification to the correct data. (e.g., if the data is in the first table added, use `0`).
        -   `type_hint`: Suggest a chart type (e.g., "bar", "pie", "line").
        -   `title`: Provide a clear, descriptive title.
        -   `x_column`, `y_column`: Specify the exact column names from the source table.
        -   `color_column` (Optional): Specify the column for grouping/hue if relevant (e.g., comparing metrics in a bar chart).
        -   `x_label`, `y_label` (Optional): Provide user-friendly axis labels if needed.
    e. **Data Transformation Consideration (for "bar" `type_hint`):** If creating a bar chart to compare multiple metrics (e.g., borrows vs. renewals per branch) where the source table has columns like `Branch`, `Borrows`, `Renewals` ("wide" format), the backend might transform this into a "long" format (`Branch`, `Metric`, `Value`) before sending it to the frontend. Therefore, when populating the `ChartSpecFinalInstruction` fields:
        *   Set `x_column` to the category column (e.g., "Branch Name").
        *   Set `y_column` to the expected *transformed* value column name (conventionally "Value").
        *   Set `color_column` to the expected *transformed* metric name column (conventionally "Metric").
        *   Adjust `y_label` and `title` accordingly (e.g., `y_label="Count"`, `title="Comparison of Metrics by Branch"`).
    f. **Consistency Check (MANDATORY):** Before finalizing the `FinalApiResponseStructure` call, **verify that `x_column`, `y_column`, and `color_column` (if used) in *each* `ChartSpecFinalInstruction` object within the `chart_specs` list EXACTLY match column names present in the source table specified by `source_table_index`, considering potential backend transformations (Guideline #5e).** If inconsistent, correct the column names.

6. **CRITICAL TOOL CHOICE: `execute_sql` vs. `summary_synthesizer`:**
   - **Use Direct SQL Generation (`execute_sql`) IF AND ONLY IF:** The user asks for a comparison OR retrieval of **specific, quantifiable metrics** (e.g., counts, sums of borrows, returns, renewals, logins) for **specific, resolved entities** (e.g., Main Library [ID: xxx], Argyle Branch [ID: yyy]) over a **defined time period**. Your goal is to generate a single, efficient SQL query. The result table might be used for a chart specification later.
   - **Use `summary_synthesizer` ONLY FOR:** More **open-ended, qualitative summary requests** (e.g., "summarize activity," "tell me about the branches") where specific metrics are not the primary focus, or when the exact metrics are unclear. Call it directly after name resolution (if applicable), providing context in the `query` argument. Its output will be purely text. **Do not include chart specifications when `summary_synthesizer` is used.**

7. **Final Response Formation:**
   - **TRIGGER:** You must reach this step and call `FinalApiResponseStructure` as your final action. This is mandatory **immediately after** receiving the necessary data from `execute_sql` (per Guideline #4) or `summary_synthesizer`.
   - Examine the gathered data (`tables` in state).
   - **Decide which tables to include using the `include_tables` flag in `FinalApiResponseStructure`. Apply the following criteria:**
       *   **Prefer `False` for Redundancy:** If the essential information from a table is fully represented in a chart (listed in `chart_specs`) AND adequately summarized in the `text`, set the corresponding `include_tables` flag to `False` to avoid unnecessary duplication.
       *   **Prefer `False` for Simple Summaries:** If a table contains a simple result (e.g., a single row with a total count) that is clearly stated and explained in the `text`, the table is often redundant; lean towards setting the flag to `False`.
       *   **Prefer `True` for Detail/Explicit Request:** Include a table (set flag to `True`) primarily when it provides detailed data points that are not easily captured in the text or a chart, or if the user explicitly asked for the table or raw data.
       *   Basically Pick `False` unless the user explicitly asks for it, or the table adds some actual and extra value over the text + chart (if there is one) combo
   - Decide which chart specifications to generate and include directly in the `chart_specs` list within `FinalApiResponseStructure` (follow Guideline #5).
   - Ensure the `text` field is **CONCISE**, focuses on insights/anomalies, and **REFERENCES** any included table(s) or chart spec(s). **DO NOT** repeat detailed data. Example: "The table shows X, and the bar chart illustrates the trend for Y."
   - If no useful tables/specs are included, provide the full answer/summary in the `text` field.
   - **Accuracy:** Ensure the final text accurately reflects and references any included items.

8. **Out-of-Scope Handling:**
   - Refuse requests unrelated to library data or operations.
   - Still use `FinalApiResponseStructure` (with empty `chart_specs`) to deliver the refusal message.

# --- Workflow Summary --- #
1. Analyze Request & History.
2. **IF** hierarchy names present -> Call `hierarchy_name_resolver` FIRST. Check results; Refuse if needed.
3. **DECIDE** Tool (Guideline #6):
   *   Specific Metrics? -> Plan for `execute_sql`.
   *   Qualitative Summary? -> Plan for `summary_synthesizer`.
4. **IF** using `execute_sql`:
   *   Refer to the schema provided above in the prompt.
   *   Generate SQL & Call `execute_sql` (via tools node).
5. **IF** using `summary_synthesizer`:
   *   Call `summary_synthesizer` (via tools node).
6. **DECIDE** final response content (text, which tables to include).
7. **DECIDE** if chart(s) needed based on state['tables'].
8. **IF** chart needed -> Prepare `ChartSpecFinalInstruction` object(s) (Guideline #5).
9. **ALWAYS** conclude with `FinalApiResponseStructure` call, populating `text`, `include_tables`, and the `chart_specs` list appropriately.
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

**MANDATORY FINAL STEP:** Always conclude by calling `FinalApiResponseStructure` with appropriate arguments for `text`, `include_tables`, and `chart_specs`.
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
    """Get operational tools for the agent (excluding FinalApiResponseStructure and DatabaseSchemaTool)."""
    return [
        HierarchyNameResolverTool(organization_id=organization_id),
        SQLExecutionTool(organization_id=organization_id),
        SummarySynthesizerTool(organization_id=organization_id),
    ]

# Function to bind tools AND the final response structure to the LLM
def create_llm_with_tools_and_final_response_structure(organization_id: str):
    llm = get_llm()
    # Get operational tools first
    operational_tools = get_tools(organization_id)
    # Define all structures the LLM can output as "tools"
    # Note: ChartSpecFinalInstruction is *part of* FinalApiResponseStructure, not bound separately
    all_bindable_items = operational_tools + [FinalApiResponseStructure]

    # --- Fetch DB Schema String --- #
    try:
        schema_tool = DatabaseSchemaTool() # Instantiate locally to get schema
        db_schema_string = schema_tool._run() # Call synchronous run method
        logger.debug("Successfully fetched DB schema string to inject into prompt.")
    except Exception as e:
        logger.error(f"Failed to fetch DB schema for prompt injection: {e}", exc_info=True)
        db_schema_string = "Error: Could not retrieve database schema." # Fallback
    # --- End Fetch DB Schema --- #

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
    logger.debug(f"Created LLM bound with tools and FinalApiResponseStructure for org: {organization_id}")
    return prompt | llm_with_tools

# --- Agent Node: Decides action - call a tool or invoke FinalApiResponseStructure ---
def agent_node(state: AgentState, llm_with_structured_output):
    """Invokes the LLM to decide the next action or final response structure.
       The final response structure now potentially includes chart specifications.
    """
    request_id = state.get("request_id")
    logger.debug(f"[AgentNode] Entering agent node...")

    llm_response: Optional[AIMessage] = None
    final_structure: Optional[FinalApiResponseStructure] = None
    
    # Parser for the final response structure (which might include chart specs)
    final_response_parser = PydanticToolsParser(tools=[FinalApiResponseStructure])
    
    preprocessed_state = _preprocess_state_for_llm(state)
    failures = []

    for attempt in range(3): # Retry loop
        try:
            llm_response = llm_with_structured_output.invoke(preprocessed_state)
            logger.debug(f"[AgentNode] Raw LLM response (Attempt {attempt + 1}): {llm_response.pretty_repr() if isinstance(llm_response, BaseMessage) else llm_response}")

            if isinstance(llm_response, AIMessage) and llm_response.tool_calls:
                 operational_calls = []
                 final_api_call = None

                 # Check for final structure call first
                 for tc in llm_response.tool_calls:
                     if tc.get("name") == FinalApiResponseStructure.__name__:
                         if final_api_call is None: final_api_call = tc
                         else: logger.warning(f"[AgentNode] Multiple FinalApiResponseStructure calls found, using first.")
                     else:
                         operational_calls.append(tc)

                 # If FinalApiResponseStructure is called, parse it
                 if final_api_call:
                     logger.debug("[AgentNode] Found FinalApiResponseStructure call.")
                     try:
                         # Add default for include_tables before parsing
                         args = final_api_call.get("args", {})
                         num_tables = len(state.get('tables', []))
                         if "include_tables" not in args: args["include_tables"] = [False] * num_tables
                         if isinstance(args.get("include_tables"), bool): args["include_tables"] = [args["include_tables"]] * max(1, num_tables)
                         # chart_specs default is handled by Pydantic model (default_factory=list)
                         
                         parsed_final = final_response_parser.invoke(AIMessage(content="", tool_calls=[final_api_call]))
                         if parsed_final:
                             final_structure = parsed_final[0]
                             logger.debug(f"[AgentNode] LLM returned valid FinalApiResponseStructure (contains {len(final_structure.chart_specs)} chart specs).")
                             break # Success, exit retry loop
                         else:
                             failures.append(f"Attempt {attempt + 1}: FinalAPIStructure parser returned empty list")
                     except ValidationError as e:
                         failures.append(f"Attempt {attempt+1}: FinalAPIStructure validation fail: {e}")
                     except Exception as e:
                         failures.append(f"Attempt {attempt + 1}: Error parsing FinalAPIStructure: {e}")
                 
                 # If no final structure call, but operational calls exist, proceed with tools
                 elif operational_calls:
                      # --- Log Generated SQL (Iterate through all calls) --- #
                      for i, tc in enumerate(operational_calls):
                          if tc.get("name") == "execute_sql":
                              sql_args = tc.get("args", {})
                              generated_sql = sql_args.get("sql", "SQL not found in args")
                              generated_params = sql_args.get("params", "Params not found in args")
                              logger.info(f"[AgentNode] LLM decided to call execute_sql (Call #{i+1} in this turn).")
                              logger.info(f"[AgentNode] Generated SQL #{i+1}: {generated_sql}")
                              logger.info(f"[AgentNode] Generated Params #{i+1}: {generated_params}")
                      # --- End Log Generated SQL --- #
                      logger.debug(f"[AgentNode] Found {len(operational_calls)} operational tool call(s). Proceeding to tools node.")
                      break # Exit retry loop to execute tools
                 
                 # If neither final structure nor operational calls found
                 else:
                      failures.append(f"Attempt {attempt + 1}: No operational tool calls or FinalApiResponseStructure found.")
            else:
                 failures.append(f"Attempt {attempt + 1}: Response not AIMessage or no tool calls.")

        except Exception as e:
            logger.error(f"[AgentNode] Exception during LLM invocation (Attempt {attempt + 1}): {e}", exc_info=True)
            failures.append(f"Attempt {attempt + 1}: LLM invocation exception: {str(e)}")
            llm_response = None # Invalidate response on exception

    # --- Fallback Logic for FinalApiResponseStructure ---
    # Trigger fallback only if the LLM was *supposed* to return the final structure but failed or returned plain text
    needs_fallback_final = False
    if llm_response and isinstance(llm_response, AIMessage):
        if llm_response.tool_calls:
             # If final call was detected in the loop but parsing failed
             if any(tc.get("name") == FinalApiResponseStructure.__name__ for tc in llm_response.tool_calls) and final_structure is None:
                 needs_fallback_final = True
        else: # If LLM returned plain text when structure was expected (end of graph likely)
             # This condition might need refinement depending on graph flow - how do we know structure was *expected*?
             # Assuming for now that plain text from agent often means it *thinks* it's done.
             needs_fallback_final = True 

    if needs_fallback_final:
        logger.warning(f"[AgentNode] Triggering fallback logic for FinalApiResponseStructure. Failures: {failures}")
        fallback_text = "Could not generate final response structure. Check logs."
        if llm_response and isinstance(llm_response.content, str) and llm_response.content.strip():
             fallback_text = llm_response.content.strip()
             logger.debug(f"[AgentNode] Using LLM's plain text for fallback structure.")
        try:
            # Fallback includes empty chart_specs list
            final_structure = FinalApiResponseStructure(
                text=fallback_text, 
                include_tables=[True] * len(state.get("tables", [])), 
                chart_specs=[] 
            )
            logger.debug(f"[AgentNode] Created fallback FinalApiResponseStructure.")
        except Exception as fallback_err:
            logger.error(f"[AgentNode] Error creating fallback structure: {fallback_err}", exc_info=True)
            final_structure = None # Fallback failed

    logger.debug(f"[AgentNode] Exiting agent node.")
    
    # --- Construct Final Return Dictionary --- #
    # Return message for history, and the final structure if parsed/created.
    # Tables are passed through implicitly by LangGraph state mechanism until final processing.
    return_dict: Dict[str, Any] = {
        "messages": [llm_response] if llm_response else [],
        "final_response_structure": final_structure
        # REMOVED: chart_spec_instructions
    }
    # We no longer need to explicitly pass tables here, final processing retrieves them from state.
    # if final_structure:
    #     current_tables = state.get("tables", [])
    #     logger.info(f"[AgentNode] Final step detected. Tables in current state: {len(current_tables)}")
    #     return_dict["tables"] = current_tables
    #     logger.info(f"[AgentNode] Returning dict with keys: {list(return_dict.keys())}")
        
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

    # REMOVED: Pruning logic for chart_spec_instructions as it's no longer in state

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
        if tool_name in tool_map:
            # Use config setting for retries
            tool_executions.append({"tool": tool_map[tool_name], "args": tool_call.get("args", {}), "id": tool_id, "name": tool_name, "retries_left": settings.TOOL_EXECUTION_RETRIES})
        else:
             logger.error(f"[ToolsNode] Operational tool '{tool_name}' requested but not found.")
             new_messages.append(ToolMessage(content=f"Error: Tool '{tool_name}' not found.", tool_call_id=tool_id, name=tool_name))

    if not tool_executions:
        logger.warning(f"[ToolsNode] No operational tools could be prepared for execution.")
        return {"messages": new_messages} if new_messages else {}

    sem = asyncio.Semaphore(settings.MAX_CONCURRENT_TOOLS)
    # Define async execution function with retry logic
    async def execute_with_retry(execution_details):
        tool = execution_details["tool"]; args = execution_details["args"]; tool_id = execution_details["id"]; tool_name = execution_details["name"]; retries = execution_details["retries_left"]
        try:
            async with sem:
                 logger.debug(f"[ToolsNode] Executing tool '{tool_name}' (ID: {tool_id}) with args: {args}")
                 content = await tool.ainvoke(args)
                 if asyncio.iscoroutine(content): content = await content
                 content_str = json.dumps(content, default=str) if isinstance(content, (dict, list)) else str(content)
                 logger.debug(f"[ToolsNode] Tool '{tool_name}' (ID: {tool_id}) completed.")
                 return {"success": True, "message": ToolMessage(content=content_str, tool_call_id=tool_id, name=tool_name), "raw_content": content, "tool_name": tool_name, "tool_id": tool_id}
        except Exception as e:
            error_msg_for_log = f"Error executing tool '{tool_name}' (ID: {tool_id}): {str(e)}"
            if retries > 0 and _is_retryable_error(e):
                retry_num = settings.TOOL_EXECUTION_RETRIES - retries + 1
                delay = settings.TOOL_RETRY_DELAY * retry_num
                logger.warning(f"[ToolsNode] {error_msg_for_log} - Retrying ({retries} left, delay {delay}s).", exc_info=False)
                await asyncio.sleep(delay); execution_details["retries_left"] = retries - 1
                return await execute_with_retry(execution_details)
            else:
                logger.error(f"[ToolsNode] {error_msg_for_log}", exc_info=True)
                return {"success": False, "error": f"{error_msg_for_log}", "tool_name": tool_name, "tool_id": tool_id}

    results = await asyncio.gather(*[execute_with_retry(exec_data) for exec_data in tool_executions])

    # Result processing: Extract only tables
    for result in results:
        tool_name = result.get("tool_name", "unknown")
        tool_id = result.get("tool_id", "")
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
            # Error handling 
            logger.warning(f"[ToolsNode] Tool call '{tool_name}' (ID: {tool_id}) failed. Error appended.")
            if not any(getattr(m, 'tool_call_id', None) == tool_id for m in new_messages if isinstance(m, ToolMessage)):
                 error_content = f"Tool execution failed: {result.get('error', 'Unknown error')}"
                 new_messages.append(ToolMessage(content=error_content, tool_call_id=tool_id, name=tool_name))

    # Update state: Only add messages and new tables
    logger.info(f"[ToolsNode] Updating state with {len(new_messages)} messages, {len(new_tables)} tables.")
    
    update_dict: Dict[str, Any] = {"messages": new_messages}
    if new_tables:
        # Combine with existing tables from state correctly
        existing_tables = state.get('tables', [])
        combined_tables = existing_tables + new_tables
        logger.info(f"[ToolsNode] Found {len(new_tables)} new table(s). State will have {len(combined_tables)} total tables.")
        update_dict["tables"] = combined_tables
    
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


# --- Conditional Edge Logic (Updated for simpler flow) ---
def should_continue(state: AgentState) -> str:
    """Determines the next step based on the last message and state."""
    request_id = state.get("request_id")
    messages = state.get("messages", [])
    last_message = messages[-1] if messages else None
    final_structure_in_state = state.get("final_response_structure")

    # If the final structure is already set (by agent node or fallback), we end.
    if final_structure_in_state:
        logger.debug("[ShouldContinue] Final response structure found in state. Routing to END.")
        return END
        
    # Analyze the last message from the agent node
    if isinstance(last_message, AIMessage):
        if last_message.tool_calls:
            # Check if ONLY the hierarchy resolver was called
            is_resolver_only = len(last_message.tool_calls) == 1 and last_message.tool_calls[0].get("name") == HierarchyNameResolverTool.__name__
            # Check if any operational tools (excluding FinalApiResponseStructure) were called
            has_operational_calls = any(
                tc.get("name") != FinalApiResponseStructure.__name__ 
                for tc in last_message.tool_calls
            )
            # Check if FinalApiResponseStructure was called (shouldn't happen if final_structure_in_state is None, but check anyway)
            has_final_call = any(tc.get("name") == FinalApiResponseStructure.__name__ for tc in last_message.tool_calls)

            if has_final_call: 
                logger.warning("[ShouldContinue] FinalApiResponseStructure tool call found, but structure not in state? Routing to END.")
                return END # Should have been caught by final_structure_in_state check
            elif is_resolver_only:
                 logger.debug("[ShouldContinue] HierarchyResolverTool call found. Routing to 'resolve_hierarchy'.")
                 return "resolve_hierarchy"
            elif has_operational_calls:
                 logger.debug("[ShouldContinue] Operational tool call(s) found. Routing to 'tools'.")
                 return "tools" 
            else:
                 # No operational calls, no final call in AIMessage. Should not happen with current agent logic?
                 logger.warning("[ShouldContinue] AIMessage has tool calls but no recognized operational or final calls. Looping back to agent.")
                 return "agent" # Loop back to agent to reconsider
        else:
             # If AIMessage has no tool calls, loop back to agent to generate final response.
             logger.debug("[ShouldContinue] AIMessage with no tool calls. Looping back to 'agent'.")
             return "agent"
    else:
        # If last message isn't AIMessage or state is unexpected, end.
        logger.warning(f"[ShouldContinue] Last message not AIMessage or unexpected state ({type(last_message).__name__}), routing to END.")
        return END


# --- Create LangGraph Agent (Updated) ---
def create_graph_app(organization_id: str) -> StateGraph:
    """
    Create the updated LangGraph application.
    Agent node generates operational tool calls or FinalApiResponseStructure (containing chart specs).
    Tools node executes only operational tools.
    """
    # LLM binding includes operational tools + FinalApiResponseStructure
    llm_with_bindings = create_llm_with_tools_and_final_response_structure(organization_id)

    # Get *only* operational tools for the handler node
    operational_tools = get_tools(organization_id)

    # Create the agent node wrapper
    agent_node_wrapper = functools.partial(agent_node, llm_with_structured_output=llm_with_bindings)

    # Create the tools node wrapper (passing ONLY operational tools)
    tools_handler_with_tools = functools.partial(async_tools_node_handler, tools=operational_tools)

    # Create the hierarchy resolver node wrapper (passing ONLY operational tools)
    resolver_handler_with_tools = functools.partial(resolve_hierarchy_node, tools=operational_tools)

    # --- Define the graph ---
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("agent", agent_node_wrapper)
    workflow.add_node("tools", tools_handler_with_tools) # Node for general operational tools
    workflow.add_node("resolve_hierarchy", resolver_handler_with_tools) # Dedicated node for resolver

    # Set the entry point
    workflow.set_entry_point("agent")

    # Define conditional edges from the agent node based on simplified should_continue
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "resolve_hierarchy": "resolve_hierarchy",
            "tools": "tools", 
            "agent": "agent", # Loop back if agent needs to generate final response
            END: END
        }
    )

    # Add edges from tool nodes back to the agent
    workflow.add_edge("tools", "agent")
    workflow.add_edge("resolve_hierarchy", "agent")

    # Compile the graph
    logger.info("Compiling LangGraph workflow...")
    graph_app = workflow.compile()
    logger.info("LangGraph workflow compiled.")
    return graph_app


# --- Refactored process_chat_message (Updated for Simpler Chart Specs) ---
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
    Constructs the final API response, mapping chart specs to the 'visualizations' field.
    """
    req_id = request_id or str(uuid.uuid4())
    logger.info(f"--- Starting request processing ---")
    logger.info(f"Org: {organization_id}, Session: {session_id}, History: {len(chat_history) if chat_history else 0}, User Message: '{message}'") 

    # Org ID validation
    try: uuid.UUID(organization_id)
    except ValueError: 
        logger.error(f"Invalid organization_id format: {organization_id}")
        return {"status": "error", "error": {"code": "INVALID_INPUT", "message": "Invalid organization identifier.", "details": None}, "data": None}

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

    # AgentState no longer includes 'visualizations'
    initial_state = AgentState(
        messages=initial_messages,
        tables=[],
        final_response_structure=None,
        request_id=req_id
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

        # Extract final response structure (contains text, include_tables, chart_specs)
        structured_response = final_state.get("final_response_structure")

        if structured_response:
            logger.info(f"Successfully obtained FinalApiResponseStructure.")
            
            # --- Prepare final API data ---
            final_text = structured_response.text
            final_tables_for_api: List[TableData] = []
            final_visualizations_for_api: List[ApiChartSpecification] = [] # Still called visualizations in API

            # Get all tables accumulated in the state
            all_tables_from_state = final_state.get('tables', [])
            
            # Filter tables based on include flags
            include_tables_flags = structured_response.include_tables
            logger.debug(f"Processing {len(all_tables_from_state)} tables with flags: {include_tables_flags}")
            for i, table_dict in enumerate(all_tables_from_state):
                 if isinstance(table_dict, dict) and isinstance(table_dict.get('columns'), list) and isinstance(table_dict.get('rows'), list):
                     if i < len(include_tables_flags) and include_tables_flags[i]:
                         try:
                             final_tables_for_api.append(TableData(**table_dict)) 
                             logger.debug(f"Including table index {i} in final API response.")
                         except Exception as table_parse_err:
                             logger.warning(f"Skipping table index {i} due to parsing error: {table_parse_err}. Data: {str(table_dict)[:200]}...")
                     # else: logger.debug(f"Excluding table index {i} based on include flag.") # Reduced verbosity
                 else:
                      logger.warning(f"Skipping invalid table structure at index {i} in final state.")

            # Construct visualizations (ApiChartSpecification) from chart_specs in the final response structure
            chart_specs_from_llm = structured_response.chart_specs # This is List[ChartSpecFinalInstruction]
            logger.debug(f"Processing {len(chart_specs_from_llm)} chart specifications from FinalApiResponseStructure.")
            
            for i, instruction in enumerate(chart_specs_from_llm):
                 logger.debug(f"Constructing API visualization spec for: {instruction.title}")
                 source_index = instruction.source_table_index
                 
                 # Validate source index against the tables available in the final state
                 if 0 <= source_index < len(all_tables_from_state):
                     source_table_dict = all_tables_from_state[source_index]
                     # Validate source table structure before use
                     if isinstance(source_table_dict, dict) and isinstance(source_table_dict.get('columns'), list) and isinstance(source_table_dict.get('rows'), list):
                          try:
                              source_table_data = TableData(**source_table_dict)
                              api_data_for_chart = source_table_data # Default
                              
                              # Placeholder: Wide-to-long transformation logic
                              if instruction.type_hint == "bar" and instruction.color_column and instruction.y_column == "Value":
                                   logger.warning(f"Bar chart spec '{instruction.title}' might require wide-to-long data transformation; passing original data for now.")
                                   # api_data_for_chart = transform_wide_to_long(source_table_data, instruction)

                              # Construct the ApiChartSpecification (for API response)
                              api_spec = ApiChartSpecification(
                                  type_hint=instruction.type_hint,
                                  title=instruction.title,
                                  x_column=instruction.x_column,
                                  y_column=instruction.y_column,
                                  color_column=instruction.color_column,
                                  x_label=instruction.x_label or instruction.x_column,
                                  y_label=instruction.y_label or instruction.y_column,
                                  data=api_data_for_chart # Use the TableData model here
                              )
                              final_visualizations_for_api.append(api_spec)
                              logger.debug(f"Successfully added API visualization spec: {api_spec.title}")
                          except Exception as spec_build_err:
                               logger.error(f"Error building ApiChartSpecification for instruction index {i}: {spec_build_err}", exc_info=True)
                     else:
                          logger.warning(f"Chart spec '{instruction.title}' references invalid table structure at source index {source_index}. Skipping.")
                 else:
                      logger.warning(f"Chart spec '{instruction.title}' references out-of-bounds table index {source_index} (Num tables: {len(all_tables_from_state)}). Skipping.")

            # Log final counts for verification
            logger.info(f"Final Response - Text: {len(final_text)} chars, Tables: {len(final_tables_for_api)}, Visualizations (Specs): {len(final_visualizations_for_api)}")

            # Construct final ChatData for the API response
            response_data = ChatData(
                text=final_text, 
                tables=final_tables_for_api or None, 
                visualizations=final_visualizations_for_api or None # Assign to the 'visualizations' key
            )
            
            logger.info(f"Success. Returning response.")
            logger.info(f"--- Finished request processing ---")
            return {"status": "success", "data": response_data.model_dump(exclude_none=True), "error": None}
        else:
            # Handle case where graph finished but final structure is missing
            logger.error(f"Graph finished, but FinalApiResponseStructure missing. State keys: {list(final_state.keys())}")
            last_msg_content = str(final_state['messages'][-1].content) if final_state.get('messages') else "No messages."
            error_details = {"last_message_preview": last_msg_content[:100], "cause": "Final structure missing from state"}
            logger.info(f"--- Finished request processing with error ---")
            return {"status": "error", "data": None, "error": {"code": "FINAL_STRUCTURE_MISSING", "message": "Unable to generate final response", "details": error_details}}

    except Exception as e:
        logger.error(f"Unhandled exception during processing: {str(e)}", exc_info=True)
        error_code = "INTERNAL_ERROR"; error_message = "Internal error."; error_details = {"exception": str(e)}
        if "recursion limit" in str(e).lower(): error_code = "RECURSION_LIMIT_EXCEEDED"; error_message = "Request complexity limit exceeded."
        logger.info(f"--- Finished request processing with exception ---")
        return {"status": "error", "data": None, "error": {"code": error_code, "message": error_message, "details": error_details}}


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