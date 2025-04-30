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
import re # Import regex module
import openai # <-- ADD THIS IMPORT

# LangChain & LangGraph Imports
from langchain_core.messages import BaseMessage, FunctionMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import AzureChatOpenAI
from langchain_core.output_parsers.openai_tools import PydanticToolsParser, JsonOutputToolsParser
from langchain_core.runnables import RunnableConfig 
from langgraph.graph import StateGraph, END
from langchain.tools import BaseTool

# Local Imports
from app.core.config import settings
from app.langchain.tools.sql_tool import SQLExecutionTool
from app.langchain.tools.summary_tool import SummarySynthesizerTool
from app.langchain.tools.hierarchy_resolver_tool import HierarchyNameResolverTool
from app.schemas.chat import ChatData, ApiChartSpecification, TableData

logger = logging.getLogger(__name__)
usage_logger = logging.getLogger("usage") # Dedicated logger for usage stats


# --- Helper Function to Get Schema String --- #
def _get_schema_string(db_name: str = "report_management") -> str:
    """Gets schema information from predefined schema definitions as a formatted string."""
    from app.db.schema_definitions import SCHEMA_DEFINITIONS # Keep import local
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


# --- Instruction Structure included directly in FinalApiResponseStructure ---
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
    # Add fields for cumulative token counts
    prompt_tokens: Annotated[int, operator.add]
    completion_tokens: Annotated[int, operator.add]


# --- System Prompt ---
SYSTEM_PROMPT_TEMPLATE = """You are a professional data assistant for the Bibliotheca chatbot API.

Your primary responsibility is to analyze organizational data and provide accurate insights to users based on the request's context.
**Key Context:** The necessary `organization_id` for data scoping is always provided implicitly through the tool context; **NEVER ask the user for it or use placeholders like 'your-organization-id'.**

# --- CRITICAL FINAL STEP --- #
**MANDATORY:** You **MUST ALWAYS** conclude your response by invoking the `FinalApiResponseStructure` tool. 
*   This applies in **ALL** situations: successful answers, reporting errors, refusing requests, simple greetings.
*   **DO NOT** provide the final answer as plain text in the message content. Your final output MUST be a call to `FinalApiResponseStructure`.
*   The `FinalApiResponseStructure` includes fields for `text`, `include_tables`, and importantly, `chart_specs`.
*   **FORMATTING:** Your response *must* be structured as an `AIMessage` object where the `tool_calls` list contains the `FinalApiResponseStructure` call with its arguments. Do not simply write the JSON or the tool name in the text content.
*   Failure to call `FinalApiResponseStructure` as the absolute final step, formatted correctly as a tool call within the `AIMessage.tool_calls` attribute, is an error.

# --- Example: Refusal Tool Call ---
# If you need to refuse a request (e.g., asking for weather), your FINAL output MUST be an AIMessage containing ONLY this tool call:
# AIMessage(
#   content="", # Content should be empty when using tool call
#   tool_calls=[
#     {{
#       "name": "FinalApiResponseStructure",
#       "args": {{
#         "text": "I cannot provide weather information. My capabilities are focused on library data and related insights.",
#         "include_tables": [],
#         "chart_specs": []
#       }},
#       "id": "<some_tool_call_id>" # You generate the ID
#     }}
#   ]
# )
# --- End Example ---
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
- Footfall data for visitor entries and exits (table '8').
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
   - Table names must be used exactly as defined in the schema: '5' for events, '8' for footfall, and 'hierarchyCaches' for hierarchy data.

4. **SQL Generation & Execution:**
   - YOU are responsible for generating the SQL query based on the database schema provided above.
   - After generating SQL, use the `execute_sql` tool to execute it. The result will be added to the `tables` list in the state.
   - **CRITICAL TRANSITION:** Once you have formulated the **single** `execute_sql` tool call that you determine will directly and sufficiently answer the user's quantitative query, your **IMMEDIATE NEXT STEP** must be to prepare and invoke the `FinalApiResponseStructure` tool (Guideline #7). **Do not generate the same `execute_sql` call multiple times in your response; generate it only once and then proceed immediately to the final structure.** Do not call any *other* operational tools after this point unless the initial SQL results (once returned in the next step) are clearly insufficient or erroneous.
   - **CRITICAL:** The arguments for `execute_sql` MUST be a JSON object with the keys "sql" and "params".
     - The "sql" key holds the SQL query string.
     - The "params" key holds a dictionary of parameters. This dictionary **MUST** include "organization_id" and any other parameters used in the query.
     - **IMPORTANT**: The *value* for the `organization_id` key MUST be the actual organization ID string from the context (e.g., "b781b517-8954-e811-2a94-0024e880a2b7"), NOT the literal string 'organization_id'.
     - Example arguments structure:
       ```json
       {{
         "sql": "SELECT ... WHERE \\"organizationId\\" = :organization_id",
         "params": {{
           "organization_id": "<ACTUAL_ORG_ID_VALUE>",
           "other_param": "value"
         }}
       }}
       ```
     - Double-check that the actual organization_id value is inserted correctly before execution.
   - ALWAYS include the correct organization_id value in the `params` dictionary.
   - Include hierarchy IDs in `params` when applicable.
   - **VERIFY SQL SYNTAX:** Before finalizing the tool call arguments, verify that:
     - All table and column names are properly double-quoted (e.g., "5", "hierarchyCaches", "organizationId")
     - All parameters in the SQL are prefixed with a colon (e.g., `:organization_id`)
     - All parameters used in the SQL have a corresponding key in the params dictionary (e.g., if `:branch_id` is in the SQL, `params` must contain a `"branch_id"` key).

5. **Chart Specification Strategy:** When formulating the final response using `FinalApiResponseStructure`:
    a. **When to Include Charts:** Populate the `chart_specs` list within `FinalApiResponseStructure` ONLY if:
        - The user explicitly requested a chart/visualization, OR
        - You are presenting complex data (e.g., comparisons across >3 categories/metrics) where a visual representation significantly aids understanding.
        - **CRITICAL EXCEPTION:** If the user **explicitly asked ONLY for a table** and *not* a chart, **DO NOT** generate any chart specifications, even if the data seems complex. Respect the user's request for a table format. Table(s) and chart(s) should be given simultaneously only in those rare cases where tables have some extra, non-redundant data which would add actual and extra value over the text + chart(s) combo.
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
        *   Set `y_column` to the expected *transformed* value column name (conventionally **"Value"**).
        *   Set `color_column` to the expected *transformed* metric name column (conventionally **"Metric"**).
        *   Adjust `y_label` and `title` accordingly (e.g., `y_label="Count"`, `title="Comparison of Metrics by Branch"`).
        *   **Example:** If the SQL query returns columns `["Location Name", "Total Borrows", "Total Returns"]`, and you want a bar chart comparing borrows and returns, the `ChartSpecFinalInstruction` **MUST** have: `type_hint: "bar"`, `x_column: "Location Name"`, `y_column: "Value"`, `color_column: "Metric"`. Failure to set `y_column` to "Value" and `color_column` to "Metric" in this multi-metric bar chart case will result in an incorrect chart.
    f. **Consistency Check (MANDATORY):** Before finalizing the `FinalApiResponseStructure` call, **verify that:**
        * `source_table_index` is a valid 0-based index for a table that exists in the current `state['tables']` list.
        * `x_column`, `y_column`, and `color_column` (if used) in *each* `ChartSpecFinalInstruction` object within the `chart_specs` list EXACTLY match column names present in the source table specified by `source_table_index`, considering potential backend transformations (Guideline #5e).
        * If any inconsistency is found, correct the column names or chart specifications before the final call.
    g. **LLM Internal Verification Checklist (MANDATORY):** Before calling `FinalApiResponseStructure`, YOU MUST INTERNALLY VERIFY the following for EACH `ChartSpecFinalInstruction` you plan to include:
        1.  **Index Valid?** Is `source_table_index` a valid index (0-based) corresponding to a table currently in `state['tables']`?
        2.  **Columns Exist?** Do the specified `x_column`, `y_column`, and `color_column` (if used) exactly match column names present in that source table?
        3.  **Multi-Metric Correct?** If `type_hint` is 'bar' and it's intended to compare multiple metrics (requiring transformation), are `y_column` correctly set to `"Value"` and `color_column` correctly set to `"Metric"`?
        **ACTION:** If any check fails, FIX the `ChartSpecFinalInstruction` object or OMIT it from the `chart_specs` list before generating the `FinalApiResponseStructure` tool call.
    h. **Examples of Correct `ChartSpecFinalInstruction`:**
        *   **Simple Bar Chart (Top 3 Branches by Borrows):**
            ```json
            {{{{
              "source_table_index": 0, // Assuming table 0 has columns: ["Branch Name", "Total Borrows"]
              "type_hint": "bar",
              "title": "Top 3 Branches by Borrows (Last 30 Days)",
              "x_column": "Branch Name",
              "y_column": "Total Borrows",
              "color_column": null, // No grouping needed
              "x_label": "Branch",
              "y_label": "Borrows"
            }}}}
            ```
        *   **Line Chart (Daily Footfall Entries):**
            ```json
            {{{{
              "source_table_index": 1, // Assuming table 1 has columns: ["Date", "Total Entries"]
              "type_hint": "line",
              "title": "Daily Footfall Entries (Last 7 Days)",
              "x_column": "Date",
              "y_column": "Total Entries",
              "color_column": null,
              "x_label": "Date",
              "y_label": "Entries"
            }}}}
            ```
        *   **Multi-Metric Bar Chart (Borrows vs Returns by Branch - Requires Transformation):**
            ```json
            {{{{
              "source_table_index": 2, // Assuming table 2 has columns: ["Location Name", "Total Borrows", "Total Returns"]
              "type_hint": "bar",
              "title": "Borrows vs Returns by Location (Last Month)",
              "x_column": "Location Name",
              "y_column": "Value", // <-- MUST be "Value"
              "color_column": "Metric", // <-- MUST be "Metric"
              "x_label": "Location",
              "y_label": "Count"
            }}}}
            ```

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
       *   Default to `False` unless the user explicitly asks for it, or the table adds some actual and extra value over the text + chart (if there is one) combo.
   - Decide which chart specifications to generate and include directly in the `chart_specs` list within `FinalApiResponseStructure` (follow Guideline #5).
   - **CRITICAL `text` field Formatting:** Ensure the `text` field is **CONCISE** (1-3 sentences), focuses on insights/anomalies, and **REFERENCES** any included table(s) or chart spec(s). **DO NOT** repeat detailed data. **NEVER include markdown tables or extensive data lists in the `text` field;** use the `include_tables` and `chart_specs` fields for detailed data presentation. Example: "The table below shows X, and the bar chart illustrates the trend for Y."
   - If no useful tables/specs are included, provide the full answer/summary in the `text` field, but still keep it reasonably concise.
   - **Accuracy:** Ensure the final text accurately reflects and references any included items.

8. **Strict Out-of-Scope Handling:**
   - If a request is unrelated to library data or operations (e.g., weather, general knowledge, historical facts outside the library context, calculations), you MUST refuse it directly.
   - **DO NOT** engage with the substance of the out-of-scope request, even if it contains factual errors you could correct. Simply state that the request is outside your capabilities.
   - Use `FinalApiResponseStructure` (with empty `chart_specs` and `include_tables`) to deliver a brief refusal message (e.g., "I cannot answer questions about topics outside of library data and operations."). Ensure it's formatted correctly as a tool call (See example in 'CRITICAL FINAL STEP' section).

# --- Workflow Summary --- #
1. Analyze Request & History.
2. **IF** hierarchy names present -> Call `hierarchy_name_resolver` FIRST. Check results; Refuse if needed.
3. **DECIDE** Tool (Guideline #6):
   *   Specific Metrics? -> Plan for `execute_sql`.
   *   Qualitative Summary? -> Plan for `summary_synthesizer`.
4. **IF** using `execute_sql`:
   *   Refer to the schema provided above in the prompt.
   *   Generate SQL & Call `execute_sql` (via tools node).
   *   Double-check parameter values and SQL syntax (Guideline #4 and #16).
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

2. **Parameter Dictionary:** Create a valid parameters dictionary mapping placeholder names (without the colon) to their actual values. This MUST include the correct `organization_id` value (see Orchestration Guideline #4). Example: For SQL with `:organization_id` and `:branch_id`, the params dictionary must include both keys with their actual values.

3. **Quoting & Naming (CRITICAL):**
   - Quote table and column names with double quotes (e.g., "hierarchyCaches", "createdAt").
   - **YOU MUST use the ACTUAL physical table names (e.g., '5' for events, '8' for footfall) in your SQL queries.**
   - The schema description uses logical names (like 'events') for clarity, but your SQL query MUST use the physical names ('5', '8').
   - **Failure to use the physical table names '5' or '8' will cause an error.**
   - Example correct usage: `SELECT * FROM "5" WHERE ...` (NOT `SELECT * FROM "events" WHERE ...`)

4. **Mandatory Organization Filtering:** ALWAYS filter results by the organization ID. Use the parameter `:organization_id`.
   - **If querying table '5' (event data):** Add `"5"."organizationId" = :organization_id` to your WHERE clause.
   - **If querying table '8' (footfall data):** Add `"8"."organizationId" = :organization_id` to your WHERE clause.
   - **If querying `hierarchyCaches` for the organization's own details:** Filter using `hc."id" = :organization_id`.
   - **If querying `hierarchyCaches` for LOCATIONS WITHIN the organization (e.g., branches, areas):** You MUST filter by the parent ID being the organization ID. Add `hc."parentId" = :organization_id` to your WHERE clause. DO NOT filter using `hc."organizationId"` as this column doesn't exist.

5. **JOINs for Related Data:** When joining table '5' or '8' and `hierarchyCaches`, use appropriate keys:
   - For events: `"5"."hierarchyId" = hc."id"`
   - For footfall: `"8"."hierarchyId" = hc."id"`
   - Always use table aliases for joins (e.g., `hc` for hierarchyCaches).
   - Always apply the organization filter on both tables being joined (e.g., `WHERE "5"."organizationId" = :organization_id AND hc."parentId" = :organization_id`).

6. **Case Sensitivity:** PostgreSQL is case-sensitive; respect exact table/column capitalization as shown in the schema.
   - Example: Use `"eventTimestamp"` not `"EventTimestamp"` or `"eventtimestamp"`.

7. **Column Selection:** Use specific column selection instead of SELECT *. Name all columns explicitly.
   - Example: `SELECT "5"."1" AS "Total Borrows", hc."name" AS "Location Name" FROM ...`

8. **Sorting:** Add ORDER BY clauses for meaningful sorting, especially when LIMIT is used.
   - Example: `ORDER BY "Total Borrows" DESC` or `ORDER BY hc."name" ASC`.

9. **LIMIT Clause:**
   - For standard SELECT queries retrieving multiple rows, ALWAYS include `LIMIT 50` at the end.
   - **DO NOT** add `LIMIT` for aggregate queries (like COUNT(*), SUM(...)) expected to return a single summary row.

10. **Aggregations (COUNT vs SUM):**
    - Use `COUNT(*)` for "how many records/items".
    - Use `SUM("column_name")` for "total number/sum" based on a specific value column.
      - For borrows: `SUM("1")`
      - For returns: `SUM("3")`
      - For logins: `SUM("5")`
      - For renewals: `SUM("7")`
      - For entries (footfall): `SUM("39")`
      - For exits (footfall): `SUM("40")`
    - Ensure `GROUP BY` includes all non-aggregated selected columns.
    - Example: `SELECT hc."name" AS "Location Name", SUM("1") AS "Total Borrows" FROM "5" JOIN "hierarchyCaches" hc ON "5"."hierarchyId" = hc."id" WHERE "5"."organizationId" = :organization_id AND hc."parentId" = :organization_id AND "eventTimestamp" >= NOW() - INTERVAL '7 days' GROUP BY hc."name"`

11. **User-Friendly Aliases:**
    - When selecting columns or using aggregate functions (SUM, COUNT, etc.), ALWAYS use descriptive, user-friendly aliases with title casing using the `AS` keyword.
    - Examples: `SELECT hc."hierarchyId" AS "Hierarchy ID"`, `SELECT COUNT(*) AS "Total Records"`, `SELECT SUM("39") AS "Total Entries"`.
    - Do NOT use code-style aliases like `total_entries` or `hierarchyId`.

12. **Benchmarking for Analytical Queries:**
    - If the user asks for analysis or comparison regarding a specific entity (e.g., "is branch X busy?", "compare borrows for branch Y"), *in addition* to selecting the specific metric(s) for that entity, include a simple benchmark for comparison in the same query.
    - **Use CTEs for Benchmarks:** The preferred way to calculate an organization-wide average (or similar benchmark) alongside a specific entity's value is using a Common Table Expression (CTE).
    - Example:
      ```sql
      WITH org_avg AS (
        SELECT AVG(total_borrows) AS "Org Average Borrows"
        FROM (
          SELECT SUM("1") AS total_borrows
          FROM "5"
          WHERE "organizationId" = :organization_id
            AND "eventTimestamp" >= NOW() - INTERVAL '30 days'
          GROUP BY "hierarchyId"
        ) AS sub
      )
      SELECT
        hc."name" AS "Branch Name",
        SUM("1") AS "Branch Borrows",
        (SELECT "Org Average Borrows" FROM org_avg) AS "Organization Average Borrows"
      FROM "5"
      JOIN "hierarchyCaches" hc ON "5"."hierarchyId" = hc."id"
      WHERE "5"."hierarchyId" = :branch_id
        AND "5"."organizationId" = :organization_id
        AND "eventTimestamp" >= NOW() - INTERVAL '30 days'
      GROUP BY hc."name"
      ```
    - **Avoid nested aggregates:** Do NOT use invalid nested aggregate/window functions like `AVG(SUM(...)) OVER ()`. Use CTEs as shown above.
    - Only include this benchmark if it can be done efficiently. The CTE approach is generally efficient.
    - Ensure both the specific value and the benchmark value have clear, user-friendly aliases.

13. **Time Filtering (Generate SQL Directly):**
    - Use the Current Time Context provided above to resolve relative dates and years.
    - If the user query includes time references (e.g., "last week", "yesterday", "past 3 months", "since June 1st", "before 2024"), you MUST generate the appropriate SQL `WHERE` clause condition directly.
    - Use relevant SQL functions like `NOW()`, `CURRENT_DATE`, `INTERVAL`, `DATE_TRUNC`, `EXTRACT`, `MAKE_DATE`, and comparison operators (`>=`, `<`, `BETWEEN`).
    - **Relative Time Interpretation:**
      - "Today" -> `DATE_TRUNC('day', "eventTimestamp") = CURRENT_DATE`
      - "Yesterday" -> `DATE_TRUNC('day', "eventTimestamp") = CURRENT_DATE - INTERVAL '1 day'`
      - "Last week" -> `"eventTimestamp" >= NOW() - INTERVAL '7 days'`
      - "Last month" -> `"eventTimestamp" >= NOW() - INTERVAL '1 month'`
      - "Last 30 days" -> `"eventTimestamp" >= NOW() - INTERVAL '30 days'`
      - "This year" -> `EXTRACT(YEAR FROM "eventTimestamp") = {current_year}`
    - **Relative Months/Years:** For month names (e.g., "March") without a year, assume the **current year** ({current_year}). For years alone (e.g., "in 2024"), query the whole year.
    - **Specific Day + Month (No Year):** For dates like "April 1st", construct the date using the **current year** ({current_year}). Use `MAKE_DATE({current_year}, <month_number>, <day_number>)`. (No casting needed as {current_year} is already an integer).
    - **Last Working Day Logic:** Based on the `Current Context` Day of the Week ({current_day}), determine the date of the last working day (assuming Mon-Fri). Filter for events matching *that specific date* using `DATE_TRUNC('day', "timestamp_column") = calculated_date`.
        *   If Today is Monday, Last Working Day = `CURRENT_DATE - INTERVAL '3 days'` (Friday).
        *   If Today is Tuesday, Last Working Day = `CURRENT_DATE - INTERVAL '1 day'` (Monday).
        *   If Today is Wednesday through Friday, Last Working Day = `CURRENT_DATE - INTERVAL '1 day'`.
        *   Identify the correct timestamp column (typically "eventTimestamp" for tables '5' and '8').
    - **Examples:**
        *   For "March": `WHERE EXTRACT(MONTH FROM "eventTimestamp") = 3 AND EXTRACT(YEAR FROM "eventTimestamp") = {current_year}`
        *   For "April 1st": `WHERE DATE_TRUNC('day', "eventTimestamp") = MAKE_DATE({current_year}, 4, 1)`
        *   For "last working day" (if Today is Monday): `WHERE DATE_TRUNC('day', "eventTimestamp") = CURRENT_DATE - INTERVAL '3 days'`
        *   For "last working day" (if Today is Wednesday): `WHERE DATE_TRUNC('day', "eventTimestamp") = CURRENT_DATE - INTERVAL '1 day'`
        *   For "Q1 2023": `WHERE "eventTimestamp" >= MAKE_DATE(2023, 1, 1) AND "eventTimestamp" < MAKE_DATE(2023, 4, 1)`
    - **DO NOT** use parameters like `:start_date` or `:end_date` for these time calculations.

14. **Date-Based Aggregation:**
    - If the user asks for metrics 'by date', 'per day', 'daily', or 'along with dates', generate SQL that aggregates the relevant metric(s) using appropriate functions (SUM, COUNT) and explicitly includes `DATE("eventTimestamp") AS "Date"` (or the relevant timestamp column) in the SELECT list.
    - You **MUST** also include `DATE("eventTimestamp")` in the `GROUP BY` clause.
    - Remember to apply necessary time filters (e.g., last 30 days) in the `WHERE` clause.
    - Example: `SELECT DATE("eventTimestamp") AS "Date", SUM("1") AS "Total Borrows" FROM "5" WHERE ... GROUP BY DATE("eventTimestamp") ORDER BY "Date" ASC`

15. **Footfall Queries (Table \"8\"):**
    - If the query asks generally about "footfall", "visitors", "people entering/leaving", or "how many people visited", calculate **both** the sum of entries (`SUM(\"39\")`) and the sum of exits (`SUM(\"40\")`).
    - Alias them clearly (e.g., `AS "Total Entries"`, `AS "Total Exits"`).
    - If the query specifically asks *only* for entries (e.g., "people came in") or *only* for exits (e.g., "people went out"), then only sum the corresponding column ("39" or "40").
    - Example:
      ```sql
      SELECT
        SUM("39") AS "Total Entries",
        SUM("40") AS "Total Exits"
      FROM "8"
      WHERE "organizationId" = :organization_id AND "eventTimestamp" >= NOW() - INTERVAL '7 days'
      ```

16. **Combine Related Metrics in SQL:**
    - When the user asks for multiple related metrics from the same table over the same period (e.g., "borrows and returns", "entries and exits", "successful and unsuccessful borrows"), generate a **single SQL query** that calculates all requested metrics using appropriate aggregate functions (e.g., `SUM("1") AS "Total Borrows", SUM("3") AS "Total Returns"`).
    - **CRITICAL:** **DO NOT** generate separate `execute_sql` tool calls for each metric if they can be efficiently combined into one query. Generate **ONLY ONE** `execute_sql` call containing the single, combined query.
    - **CRITICAL:** **DO NOT** generate multiple `execute_sql` calls if they request the same core data and differ only in presentation (e.g., `ORDER BY` clause). Generate only the single, most relevant query needed to retrieve the necessary data for the final answer.
    - Example:
      ```sql
      SELECT
        hc."name" AS "Location Name",
        SUM("1") AS "Total Borrows",
        SUM("3") AS "Total Returns",
        SUM("7") AS "Total Renewals"
      FROM "5"
      JOIN "hierarchyCaches" hc ON "5"."hierarchyId" = hc."id"
      WHERE "5"."organizationId" = :organization_id
        AND hc."parentId" = :organization_id -- Correct JOIN filter
        AND "eventTimestamp" >= NOW() - INTERVAL '30 days'
      GROUP BY hc."name"
      ORDER BY "Total Borrows" DESC
      LIMIT 50
      ```

17. **SQL Syntax Verification Checklist:**
    Before finalizing the SQL query tool call arguments, verify that:
    - All table and column names used in the SQL query match exactly with the schema provided (including physical names like '5', '8').
    - All table and column names are properly double-quoted.
    - All parameter placeholders in the SQL string are prefixed with colons (:).
    - The `WHERE` clause includes the appropriate organization filter(s) (on table '5', '8', or 'hierarchyCaches' as needed).
    - All parameters used in the SQL string have corresponding key-value entries in the `params` dictionary.
    - For joined tables, appropriate join conditions and table aliases are used.
    - For aggregate queries, all non-aggregated selected columns are included in the `GROUP BY` clause.
    - For multi-row result queries, appropriate `ORDER BY` and `LIMIT 50` clauses are included (unless it's a single-row aggregate result).
    - For time-based queries, the correct SQL time functions and calculations are generated directly in the SQL string, not passed as parameters.

# --- END SQL GENERATION GUIDELINES --- #

**MANDATORY FINAL STEP:** Always conclude by calling `FinalApiResponseStructure`, formatted correctly as a tool call within an AIMessage object (see example above), with appropriate arguments for `text`, `include_tables`, and `chart_specs`.
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
        model_name=settings.LLM_MODEL_NAME, 
        temperature=0.15,
        verbose=settings.VERBOSE_LLM,
        # streaming=False # Ensure streaming is False if not handled downstream
    )

def get_tools(organization_id: str) -> List[Any]:
    """Get operational tools for the agent (excluding FinalApiResponseStructure)."""
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
    """Invokes the LLM to decide the next action or final response structure.
       Includes smarter deduplication for execute_sql calls.
    """
    request_id = state.get("request_id")
    logger.debug(f"[AgentNode] Entering agent node...")

    llm_response: Optional[AIMessage] = None
    final_structure: Optional[FinalApiResponseStructure] = None
    final_api_call = None # Initialize here to ensure variable exists in scope
    
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
                      # --- Log ALL Received Operational Calls --- #
                      logger.info(f"[AgentNode] Received {len(operational_calls)} operational tool call(s) from LLM.")
                      for i, tc in enumerate(operational_calls):
                          tool_name = tc.get("name", "Unknown")
                          tool_args = tc.get("args", {})
                          logger.debug(f"[AgentNode] Raw Call #{i+1}: Name='{tool_name}', Args={tool_args}") # Log basic info
                          if tool_name == "execute_sql":
                              generated_sql = tool_args.get("sql", "SQL not found in args")
                              generated_params = tool_args.get("params", "Params not found in args")
                              # Log the details here, clearly marking it's from the raw response
                              logger.info(f"[AgentNode] Raw SQL Call #{i+1} Received: {generated_sql}")
                              logger.info(f"[AgentNode] Raw Params #{i+1} Received: {generated_params}")

                      # --- Smart Deduplication of Operational Tool Calls --- #
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
                              # For non-SQL tools, use simple name + args signature
                              signature = (tool_name, json.dumps(tool_args, sort_keys=True))
                              
                          if signature not in seen_signatures:
                              unique_operational_calls.append(tc)
                              seen_signatures.add(signature)
                              logger.debug(f"[AgentNode] Keeping unique call (ID: {tc.get('id')}, Signature: {signature})")
                          else:
                              discarded_duplicates.append(tc) # Track discarded functional duplicates
                      # --- End Smart Deduplication --- #

                      # --- Log Discarded Functional Duplicates --- #
                      for discarded_tc in discarded_duplicates:
                           logger.warning(f"[AgentNode] Discarded functionally duplicate operational tool call (ID: {discarded_tc.get('id')}): {discarded_tc.get('name')} with args {discarded_tc.get('args', {})}")

                      # --- Check if any unique calls remain AFTER deduplication ---
                      if not unique_operational_calls:
                           failures.append(f"Attempt {attempt + 1}: No unique operational tool calls remaining after deduplication.")
                           continue # Go to next retry attempt

                      # --- Log Summary of Unique Calls Proceeding --- #
                      logger.info(f"[AgentNode] Proceeding with {len(unique_operational_calls)} unique operational tool call(s): {[c.get('name') for c in unique_operational_calls]}")

                      # Modify the AIMessage to only contain unique calls before adding to state
                      llm_response.tool_calls = unique_operational_calls
                      break # Exit retry loop to execute tools
                 
                 # If neither final structure nor operational calls found
                 else:
                      failures.append(f"Attempt {attempt + 1}: No operational tool calls or FinalApiResponseStructure found.")
            else:
                 failures.append(f"Attempt {attempt + 1}: Response not AIMessage or no tool calls.")

        # --- Catch specific OpenAI errors, especially content filtering --- #
        except openai.BadRequestError as e:
            logger.error(f"[AgentNode] OpenAI BadRequestError during LLM invocation (Attempt {attempt + 1}): {e}", exc_info=False) # Log less verbosely for known errors
            # Check if it's a content filter error
            if e.body and e.body.get('code') == 'content_filter':
                logger.warning(f"[AgentNode] Azure OpenAI Content Filter triggered (Attempt {attempt + 1}). Returning safe response.")
                # Create a safe, generic fallback structure immediately
                safe_fallback_structure = FinalApiResponseStructure(
                    text="I cannot process this request due to content policies.", 
                    include_tables=[],
                    chart_specs=[] 
                )
                # Return immediately, bypassing further retries and standard fallback
                return {
                    "messages": [], 
                    "final_response_structure": safe_fallback_structure,
                    "prompt_tokens": 0, # Assume 0 tokens as the call failed before completion
                    "completion_tokens": 0 
                }
            else:
                # If it's a different BadRequestError, add to failures and potentially retry
                failures.append(f"Attempt {attempt + 1}: OpenAI BadRequestError: {str(e)}")
            llm_response = None # Invalidate response

        except Exception as e:
            logger.error(f"[AgentNode] Unhandled Exception during LLM invocation (Attempt {attempt + 1}): {e}", exc_info=True)
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
        
        # Try to extract text from the raw tool call args if parsing failed but call existed
        if final_api_call and final_api_call.get("args") and isinstance(final_api_call["args"].get("text"), str):
             raw_tool_text = final_api_call["args"]["text"].strip()
             if raw_tool_text:
                 logger.debug("[AgentNode] Using text from raw tool call arguments for fallback structure.")
                 fallback_text = raw_tool_text
             else:
                 logger.debug("[AgentNode] Raw tool call text argument was empty.")
        # Only use llm_response.content if the tool call text wasn't usable
        elif llm_response and isinstance(llm_response.content, str) and llm_response.content.strip():
             raw_content = llm_response.content.strip()
             logger.debug(f"[AgentNode] Using LLM's plain text content for fallback structure as tool call text wasn't available/usable.")
             # --- Attempt to clean up unwanted AIMessage string representation --- #
             try:
                 # Find the start of the unwanted representation
                 split_index = raw_content.index("AIMessage(") 
                 # Take the part before it
                 fallback_text = raw_content[:split_index].strip()
                 # If the part before is empty (e.g., only AIMessage was returned), use a generic message
                 if not fallback_text:
                     logger.warning("[AgentNode] Fallback content cleanup resulted in empty string. Using generic refusal. LLM Content: " + raw_content)
                     fallback_text = "I cannot process this request due to content policies or an internal error."
                 else:
                    logger.debug(f"[AgentNode] Successfully cleaned AIMessage representation from fallback content.")
             except ValueError: 
                 # AIMessage( not found, use raw content as is
                 logger.debug(f"[AgentNode] AIMessage representation not found in fallback content. Using raw content.")
                 fallback_text = raw_content
             # --- End Cleanup --- #

        try:
            # Fallback includes empty chart_specs list
            final_structure = FinalApiResponseStructure(
                text=fallback_text, 
                include_tables=[True] * len(state.get("tables", [])), # Default to True for tables in fallback
                chart_specs=[] 
            )
            logger.debug(f"[AgentNode] Created fallback FinalApiResponseStructure.")
            # --- IMMEDIATE RETURN FOR FALLBACK ---
            # If fallback structure was created, return it immediately with zero tokens for this turn.
            return {
                "messages": [], # No new LLM message to add in fallback
                "final_response_structure": final_structure,
                "prompt_tokens": 0, # No LLM call succeeded for this turn's final structure
                "completion_tokens": 0 
            }
        except Exception as fallback_err:
            logger.error(f"[AgentNode] Error creating fallback structure: {fallback_err}", exc_info=True)
            # If even fallback creation fails, return a minimal error structure
            return {
                "messages": [],
                "final_response_structure": FinalApiResponseStructure(
                    text="An internal error occurred while generating the response.",
                    include_tables=[],
                    chart_specs=[]
                ),
                "prompt_tokens": 0,
                "completion_tokens": 0
            }


    # --- This block is now only reached if fallback was NOT triggered and returned ---
    logger.debug(f"[AgentNode] Exiting agent node (standard path).")
    
    # --- Construct Final Return Dictionary --- #
    # Extract token usage if available (only relevant if LLM call was successful)
    prompt_tokens_turn = 0
    completion_tokens_turn = 0
    if llm_response and hasattr(llm_response, 'usage_metadata') and llm_response.usage_metadata:
        prompt_tokens_turn = llm_response.usage_metadata.get('prompt_tokens', 0)
        completion_tokens_turn = llm_response.usage_metadata.get('completion_tokens', 0)
        logger.debug(f"[AgentNode] Tokens used this turn: Prompt={prompt_tokens_turn}, Completion={completion_tokens_turn}")

    # Return message for history, final structure, and token counts for accumulation
    return_dict: Dict[str, Any] = {
        "messages": [llm_response] if llm_response else [],
        "final_response_structure": final_structure,
        "prompt_tokens": prompt_tokens_turn,          # Return counts for this turn
        "completion_tokens": completion_tokens_turn   # Return counts for this turn
    }

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


# --- Create LangGraph Agent ---
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
        request_id=req_id,
        prompt_tokens=0,
        completion_tokens=0
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
                 
                 # --- Start Stricter Validation --- #
                 # 1. Validate source index range
                 if not (0 <= source_index < len(all_tables_from_state)):
                     logger.warning(f"Discarding chart spec '{instruction.title}' due to invalid source_table_index: {source_index} (Num tables: {len(all_tables_from_state)}). LLM Spec: {instruction}")
                     continue # Skip to the next instruction

                 source_table_dict = all_tables_from_state[source_index]

                 validation_passed = True
                 error_reason = ""
                 source_columns = []
                 
                 # 2. Validate source table structure itself
                 if not (isinstance(source_table_dict, dict) and isinstance(source_table_dict.get('columns'), list) and isinstance(source_table_dict.get('rows'), list)):
                     validation_passed = False
                     error_reason = f"Source table at index {source_index} has invalid structure (not dict with list columns/rows)."
                 else:
                     source_columns = source_table_dict['columns']
                     # 3. Validate X column exists
                     if instruction.x_column not in source_columns:
                         validation_passed = False
                         error_reason = f"Specified x_column '{instruction.x_column}' not found in source table columns: {source_columns}."
                     # 4. Validate Y column exists (check original LLM value, with exception for multi-metric bar)
                     elif instruction.y_column not in source_columns:
                         is_multi_metric_bar = False
                         if instruction.type_hint == "bar":
                             metric_cols = [col for col in source_columns if col != instruction.x_column]
                             if len(metric_cols) >= 2: is_multi_metric_bar = True
                         
                         if not (is_multi_metric_bar and instruction.y_column == "Value"): # Allow Value only if multi-metric bar
                             validation_passed = False
                             error_reason = f"Specified y_column '{instruction.y_column}' not found in source table columns: {source_columns}."
                     # 5. Validate Color column exists (if specified, with exception for multi-metric bar)
                     elif instruction.color_column is not None and instruction.color_column not in source_columns:
                         is_multi_metric_bar = False # Recheck here for clarity
                         if instruction.type_hint == "bar":
                             metric_cols = [col for col in source_columns if col != instruction.x_column]
                             if len(metric_cols) >= 2: is_multi_metric_bar = True
                         
                         if not (is_multi_metric_bar and instruction.color_column == "Metric"): # Allow Metric only if multi-metric bar
                             validation_passed = False
                             error_reason = f"Specified color_column '{instruction.color_column}' not found in source table columns: {source_columns}."

                 if not validation_passed:
                     logger.warning(f"Discarding chart spec '{instruction.title}' due to validation error: {error_reason} LLM Spec: {instruction}")
                     continue # Skip to the next instruction
                 # --- End Stricter Validation --- #

                 # If validation passed, proceed with parsing and potential corrections
                 try:
                     source_table_data = TableData(**source_table_dict)
                     
                     # --- Special Handling for Single-Row Pie Charts (Columns as Categories) --- #
                     is_pie_transformed = False
                     if instruction.type_hint == 'pie' and len(source_table_data.rows) == 1 and len(source_table_data.columns) > 1:
                         logger.info(f"Applying specific transformation for single-row pie chart: '{instruction.title}'")
                         original_columns = source_table_data.columns
                         original_row = source_table_data.rows[0]
                         
                         # Define new structure
                         category_col_name = "Metric" # Or derive from x_column if needed? Let's stick to convention.
                         value_col_name = "Count"   # Or derive from y_column?
                         transformed_columns = [category_col_name, value_col_name]
                         transformed_rows = []
                         
                         # Create new rows from original columns/values
                         for col_index, col_name in enumerate(original_columns):
                             try:
                                 # Attempt to convert value, skip if not possible for a pie chart
                                 value = float(original_row[col_index]) if original_row[col_index] is not None else 0.0
                                 transformed_rows.append([col_name, value])
                             except (ValueError, TypeError):
                                 logger.warning(f"Could not convert value '{original_row[col_index]}' for column '{col_name}' to float for pie chart. Skipping this category.")
                         
                         if not transformed_rows:
                             logger.warning(f"Pie chart '{instruction.title}' transformation resulted in no valid data rows. Skipping chart.")
                             continue # Skip this chart spec
                             
                         # Create the transformed data object
                         api_data_for_chart = TableData(columns=transformed_columns, rows=transformed_rows)
                         
                         # Create and append the corrected ApiChartSpecification directly
                         api_spec = ApiChartSpecification(
                             type_hint="pie",
                             title=instruction.title,
                             x_column=category_col_name, # Use transformed column name
                             y_column=value_col_name,   # Use transformed column name
                             color_column=None,         # Pie charts don't use color_column here
                             x_label=instruction.x_label or category_col_name, # Use LLM label or default
                             y_label=instruction.y_label or value_col_name,   # Use LLM label or default
                             data=api_data_for_chart
                         )
                         final_visualizations_for_api.append(api_spec)
                         logger.debug(f"Successfully added transformed pie chart spec: {api_spec.title}")
                         is_pie_transformed = True
                         continue # Go to next instruction, pie chart handled
                     # --- End Special Handling for Single-Row Pie Charts --- #

                     # --- Standard Processing & Correction Logic (Skip if pie was transformed) --- #
                     if not is_pie_transformed:
                         # --- Correction Logic for Multi-Metric Bar Charts --- #
                         corrected_y_column = instruction.y_column
                         corrected_color_column = instruction.color_column
                         corrected_y_label = instruction.y_label or instruction.y_column
                         apply_wide_to_long = False # Flag to ensure transformation runs
                         
                         if instruction.type_hint == "bar":
                             # Check source_columns (derived during validation) instead of source_table_data.columns
                             metric_columns = [col for col in source_columns if col != instruction.x_column]
                             if len(metric_columns) >= 2:
                                 logger.debug(f"Applying correction for multi-metric bar chart '{instruction.title}'. Forcing y_column='Value', color_column='Metric'. Original LLM spec: y='{instruction.y_column}', color='{instruction.color_column}'")
                                 corrected_y_column = "Value"       # Override
                                 corrected_color_column = "Metric"   # Override
                                 corrected_y_label = "Value"        # Override
                                 apply_wide_to_long = True          # Mark for transformation
                         # --- End Correction Logic --- #
 
                         # --- Apply Wide-to-Long Transformation if needed --- #
                         api_data_for_chart = source_table_data # Start with original
                         # Condition: Apply if flagged by correction logic OR (original LLM spec had color_column for bar chart)
                         if apply_wide_to_long or (instruction.type_hint == "bar" and instruction.color_column is not None and not apply_wide_to_long):
                             logger.info(f"Applying wide-to-long transformation for chart spec: {instruction.title}")
                             # Pass the original x_column name to identify the ID column
                             api_data_for_chart = _transform_wide_to_long(source_table_data, instruction.x_column)
 
                         # Construct the ApiChartSpecification (for API response)
                         api_spec = ApiChartSpecification(
                             type_hint=instruction.type_hint,
                             title=instruction.title,
                             x_column=instruction.x_column,       # Use original from LLM
                             y_column=corrected_y_column,         # Use corrected value
                             color_column=corrected_color_column, # Use corrected value
                             x_label=instruction.x_label or instruction.x_column, # Use original from LLM
                             y_label=corrected_y_label,           # Use corrected value
                             data=api_data_for_chart # Use the potentially transformed TableData here
                         )
                         final_visualizations_for_api.append(api_spec)
                         logger.debug(f"Successfully added API visualization spec: {api_spec.title}")
                 except Exception as spec_build_err:
                     logger.error(f"Error building ApiChartSpecification for instruction index {i} after validation: {spec_build_err}", exc_info=True)

            # Log final counts for verification
            logger.info(f"Final Response - Text: {len(final_text)} chars, Tables: {len(final_tables_for_api)}, Visualizations (Specs): {len(final_visualizations_for_api)}")

            # --- Log Usage Data --- #
            total_prompt_tokens = final_state.get('prompt_tokens', 0)
            total_completion_tokens = final_state.get('completion_tokens', 0)
            # Escape potential quotes/newlines in messages for cleaner single-line log
            escaped_query = json.dumps(message)
            escaped_response = json.dumps(final_text)
            usage_logger.info(
                f"REQUEST_ID={req_id} "
                f"PROMPT_TOKENS={total_prompt_tokens} "
                f"COMPLETION_TOKENS={total_completion_tokens} "
                f"QUERY={escaped_query} " 
                f"RESPONSE={escaped_response}" 
            )
            # --- End Log Usage Data --- #

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
            
            # --- Log Usage Data (Error Case) --- #
            total_prompt_tokens = final_state.get('prompt_tokens', 0)
            total_completion_tokens = final_state.get('completion_tokens', 0)
            error_message = f"Error: Final structure missing. Last message: {last_msg_content[:100]}..."
            escaped_query = json.dumps(message)
            escaped_response = json.dumps(error_message)
            usage_logger.info(
                f"REQUEST_ID={req_id} "
                f"PROMPT_TOKENS={total_prompt_tokens} "
                f"COMPLETION_TOKENS={total_completion_tokens} "
                f"QUERY={escaped_query} "
                f"RESPONSE={escaped_response}" # Log error indicator
            )
            # --- End Log Usage Data (Error Case) --- #
            
            logger.info(f"--- Finished request processing with error ---")
            return {"status": "error", "data": None, "error": {"code": "FINAL_STRUCTURE_MISSING", "message": "Unable to generate final response", "details": error_details}}

    except Exception as e:
        logger.error(f"Unhandled exception during processing: {str(e)}", exc_info=True)
        error_code = "INTERNAL_ERROR"; error_message = "Internal error."; error_details = {"exception": str(e)}
        if "recursion limit" in str(e).lower(): error_code = "RECURSION_LIMIT_EXCEEDED"; error_message = "Request complexity limit exceeded."
        logger.info(f"--- Finished request processing with exception ---")
        return {"status": "error", "data": None, "error": {"code": error_code, "message": error_message, "details": error_details}}


# --- Helper function for Data Transformation --- #
def _transform_wide_to_long(wide_table: TableData, id_column_name: str) -> TableData:
    """Transforms TableData from wide to long format for grouped bar charts."""
    logger.debug(f"[_transform_wide_to_long] Starting transformation for table with ID column: {id_column_name}")
    long_rows = []
    metric_col_name = "Metric" # Convention expected by prompt Guideline #5e
    value_col_name = "Value"   # Convention expected by prompt Guideline #5e

    if not wide_table.rows or not wide_table.columns:
        logger.warning("[_transform_wide_to_long] Input table has no rows or columns. Returning empty.")
        return TableData(columns=[id_column_name, metric_col_name, value_col_name], rows=[], metadata=wide_table.metadata)

    try:
        id_col_index = wide_table.columns.index(id_column_name)
    except ValueError:
        logger.error(f"[_transform_wide_to_long] ID column '{id_column_name}' not found in wide table columns: {wide_table.columns}. Returning original table as fallback.")
        return wide_table # Fallback to original to prevent downstream errors

    value_column_indices = {
        i: col_name for i, col_name in enumerate(wide_table.columns) if i != id_col_index
    }

    if not value_column_indices:
         logger.warning(f"[_transform_wide_to_long] No value columns found besides ID column '{id_column_name}'. Returning original table.")
         return wide_table

    for wide_row in wide_table.rows:
        id_value = wide_row[id_col_index]
        for index, metric_name in value_column_indices.items():
            value = wide_row[index]
            try:
                # Convert numeric types, pass others as is? Or enforce numeric?
                # Let's try converting to float, falling back to None if not possible
                numeric_value = float(value) if value is not None else None
                long_rows.append([id_value, metric_name, numeric_value])
            except (ValueError, TypeError):
                 logger.warning(f"[_transform_wide_to_long] Could not convert value '{value}' for metric '{metric_name}' to float. Appending as None.")
                 long_rows.append([id_value, metric_name, None])

    long_columns = [id_column_name, metric_col_name, value_col_name]
    logger.debug(f"[_transform_wide_to_long] Transformation complete. Produced {len(long_rows)} long format rows.")
    
    new_metadata = wide_table.metadata.copy() if wide_table.metadata else {}
    new_metadata["transformed_from_wide"] = True

    return TableData(columns=long_columns, rows=long_rows, metadata=new_metadata)


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