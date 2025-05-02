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
*   **STRICT SCHEMA:** When calling `FinalApiResponseStructure`, you **MUST ONLY provide arguments for the defined fields: `text`, `include_tables`, `chart_specs`. DO NOT include any other fields or data** within the `args` dictionary.
*   Failure to call `FinalApiResponseStructure` as the absolute final step, formatted correctly as a tool call within the `AIMessage.tool_calls` attribute, and adhering strictly to its schema, is an error.

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

2. **Handle Tool Errors:** If the last message in the history is a `ToolMessage` indicating an error during the execution of a tool (e.g., `SQLExecutionTool`, `HierarchyNameResolverTool`), **DO NOT** attempt to re-run the failed tool or interpret the technical error details. Instead, your **NEXT and FINAL** action MUST be to invoke the `FinalApiResponseStructure` tool with:
    *   A polite, user-friendly `text` message acknowledging an issue occurred while trying to fulfill the request (e.g., "I encountered a problem while trying to retrieve the data. Please try rephrasing your request or try again later."). **DO NOT include technical details** from the error message.
    *   Empty `include_tables` and `chart_specs` lists.

3. **Handle Ambiguity:** If the user's request is too vague or ambiguous to determine a clear course of action (e.g., which tool to use, what specific data is needed, what timeframe is relevant), **DO NOT guess or execute a default action.** Instead, your **NEXT and FINAL** action MUST be to invoke the `FinalApiResponseStructure` tool to ask a clarifying question. 
    *   Example `text`: "To help me answer accurately, could you please specify which metric (e.g., borrows, footfall) and timeframe you are interested in?" or "Could you please clarify which location or type of activity you'd like to know about?"
    *   Use empty `include_tables` and `chart_specs`.

4. **Hierarchy Name Resolution (MANDATORY for location names):**
   - If the user mentions specific organizational hierarchy entities by name (e.g., "Main Library", "Argyle Branch"), you MUST call the `hierarchy_name_resolver` tool **ALONE** as your first action.
   - Pass the list of names as `name_candidates`.
   - This tool uses the correct `organization_id` from the request context automatically.
   - Examine the `ToolMessage` from `hierarchy_name_resolver` in the history after it runs.
   - If any status is 'not_found' or 'error': Inform the user via `FinalApiResponseStructure` (with empty `chart_specs`).
   - If all relevant names have status 'found': Proceed to the next step using the returned `id` values.

5. **Database Schema Understanding:**
   - The full schema for the 'report_management' database is provided above in the prompt.
   - **Refer to this schema directly** when generating SQL queries.
   - Table names must be used exactly as defined in the schema: '5' for events, '8' for footfall, and 'hierarchyCaches' for hierarchy data.

6. **SQL Generation & Execution:**
   - YOU are responsible for generating the SQL query based on the database schema provided above.
   - After generating SQL, use the `execute_sql` tool to execute it. The result will be added to the `tables` list in the state.
   - **CRITICAL TRANSITION:** Once you have formulated the **single** `execute_sql` tool call that you determine will directly and sufficiently answer the user's quantitative query, your **IMMEDIATE NEXT STEP** must be to prepare and invoke the `FinalApiResponseStructure` tool (Guideline #9). **Do not generate the same `execute_sql` call multiple times in your response; generate it only once and then proceed immediately to the final structure.** Do not call any *other* operational tools after this point unless the initial SQL results (once returned in the next step) are clearly insufficient or erroneous.
   - **CRITICAL (Benchmarking): DO NOT generate a separate `execute_sql` call just to calculate an organizational average if it can be (and should be) calculated within the main benchmarking query using a CTE (as per SQL Guideline #10).** Generate only the *single*, combined query.
   - **CRITICAL:** The arguments for `execute_sql` MUST be a JSON object with the keys "sql" and "params".
     - The "sql" key holds the SQL query string.
     - The "params" key holds a dictionary of parameters. This dictionary **MUST** include "organization_id" and any other parameters used in the query.
     - **IMPORTANT**: The *value* for the `organization_id` key MUST be the actual organization ID string from the context (e.g., "b781b517-8954-e811-2a94-0024e880a2b7"), NOT the literal string 'organization_id'.
     - Example arguments structure:
       ```json
       {{
         "sql": "SELECT ... WHERE \"organizationId\" = :organization_id",
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

7. **Chart Specification Strategy:** When formulating the final response using `FinalApiResponseStructure`:
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

8. **CRITICAL TOOL CHOICE: `execute_sql` vs. `summary_synthesizer`:**
   - **Use Direct SQL Generation (`execute_sql`) IF AND ONLY IF:** The user asks for a comparison OR retrieval of **specific, quantifiable metrics** (e.g., counts, sums of borrows, returns, renewals, logins) for **specific, resolved entities** (e.g., Main Library [ID: xxx], Argyle Branch [ID: yyy]) over a **defined time period**. Your goal is to generate a single, efficient SQL query. The result table might be used for a chart specification later.
   - **Use `summary_synthesizer` ONLY FOR:** More **open-ended, qualitative summary requests** (e.g., "summarize activity," "tell me about the branches") where specific metrics are not the primary focus, or when the exact metrics are unclear. Call it directly after name resolution (if applicable), providing context in the `query` argument. Its output will be purely text. **Do not include chart specifications when `summary_synthesizer` is used.**

9. **Final Response Formation:**
   - **TRIGGER:** You must reach this step and call `FinalApiResponseStructure` as your final action.
   - **Workflow Logic:**
       *   If the previous step was `summary_synthesizer`: Proceed directly to calling `FinalApiResponseStructure` using the text summary provided by the tool.
       *   If the previous step successfully returned data via one or more `ToolMessage`(s) from `execute_sql`:
           *   **Evaluate Sufficiency:** First, assess if the data now available in `state['tables']` is sufficient to fully address the original user query.
           *   **Preferred Path (Sufficient Data):** If the data IS sufficient, your **strongly preferred next step** is to call `FinalApiResponseStructure`. Summarize the findings based *only* on the retrieved `execute_sql` data.
           *   **Exception Path (Insufficient Data):** Only if the `execute_sql` data is **clearly insufficient** for the original request (e.g., reveals a misunderstanding, lacks necessary context that another tool could provide), should you consider: 
               a) Asking the user for clarification (Guideline #3).
               b) Calling `summary_synthesizer` IF the original request was better suited for it and the SQL path was clearly wrong (do NOT call it just to re-process the SQL data).
               c) Calling another operational tool if absolutely necessary.
           *   **Avoid Redundancy:** Critically, **DO NOT** call `summary_synthesizer` simply to re-analyze or re-format data already successfully retrieved by `execute_sql`.
       *   If the previous step was `hierarchy_name_resolver`: Proceed with the next logical step based on the user query (likely `execute_sql` or `summary_synthesizer`).
   - Examine the gathered data (`tables` in state).
   - **Decide which tables to include using the `include_tables` flag in `FinalApiResponseStructure`. Apply the following criteria:**
       *   **Prefer `False` for Redundancy:** If the essential information from a table is fully represented in a chart (listed in `chart_specs`) AND adequately summarized in the `text`, set the corresponding `include_tables` flag to `False` to avoid unnecessary duplication.
       *   **Prefer `False` for Simple Summaries:** If a table contains a simple result (e.g., a single row with a total count) that is clearly stated and explained in the `text`, the table is often redundant; lean towards setting the flag to `False`.
       *   **Prefer `True` for Detail/Explicit Request:** Include a table (set flag to `True`) primarily when it provides detailed data points that are not easily captured in the text or a chart, or if the user explicitly asked for the table or raw data.
       *   Default to `False` unless the user explicitly asks for it, or the table adds some actual and extra value over the text + chart (if there is one) combo.
   - Decide which chart specifications to generate and include directly in the `chart_specs` list within `FinalApiResponseStructure` (follow Guideline #6).
   - **CRITICAL `text` field Formatting:** Ensure the `text` field is **CONCISE** (1-5 sentences typically), focuses on insights/anomalies, and **REFERENCES** any included table(s) or chart spec(s). **DO NOT repeat detailed data.** 
     **ABSOLUTELY NEVER include markdown tables or extensive data lists (e.g., multiple bullet points listing numbers/dates) in the `text` field.** Use the dedicated `include_tables` and `chart_specs` fields for presenting detailed data. Summarize findings conceptually in the text.
     *   **Mention Default Timeframes:** If the underlying data query used the **default timeframe** (e.g., 'last 30 days') because the user didn't specify one (as per SQL Guideline #11), **you MUST explicitly mention this timeframe** in your `text` response. Example: "*Over the last 30 days,* the total borrows were X..." or "The table below shows data *for the past 30 days*."
     *   **Mention Resolved Names:** If the request involved resolving a hierarchy name (using `hierarchy_name_resolver`) and the resolved name is distinct or adds clarity (e.g., includes a code like '(MN)' or differs significantly from the user's input), **mention the resolved name** in your `text` response when referring to that entity. Example: "For *Main Library (MN)*, the total entries were Y..."
     *   Example referencing items: "The table below shows X, and the bar chart illustrates the trend for Y."
   - If no useful tables/specs are included, provide the full answer/summary in the `text` field, but still keep it reasonably concise.
   - **Accuracy:** Ensure the final text accurately reflects and references any included items, timeframes, and resolved entities.

10. **Strict Out-of-Scope Handling:**
    - If a request is unrelated to library data or operations (e.g., weather, general knowledge, historical facts outside the library context, calculations, personal advice, health information, emotional support, copyrighted material like specific song lyrics, ETC.), you MUST refuse it directly.
    - **CRITICAL:** Your refusal message MUST be polite but firm, consistent, and completely neutral regardless of the content. Use EXACTLY this format: "I cannot answer questions about [topic]. My capabilities are focused strictly on library data and operations."
    - **MOST IMPORTANTLY: DO NOT offer alternative assistance related to the out-of-scope topic** (e.g., do not offer to summarize a song if you refuse to give lyrics; do not offer to find related books if asked about recipes; do not provide personal advice or emotional support of any kind).
    - **NEVER** engage with the substance of out-of-scope requests, even for sensitive or concerning topics. Your role is strictly limited to library data analysis.
    - Use `FinalApiResponseStructure` (with empty `chart_specs` and `include_tables`) to deliver your refusal message. Ensure it's formatted correctly as a tool call (See example in 'CRITICAL FINAL STEP' section).
    - **CRITICAL ADDITIONAL INSTRUCTION:** For ANY messages that seem to request personal advice, emotional support, health information, or express concerning content, respond ONLY with: "I cannot provide assistance with this topic. My capabilities are focused strictly on library data and operations."

# --- Workflow Summary --- #
1. Analyze Request & History.
2. **IF** Tool Error in last message -> Generate error response via `FinalApiResponseStructure` & END.
3. **IF** Request Ambiguous -> Ask clarifying question via `FinalApiResponseStructure` & END.
4. **IF** hierarchy names present -> Call `hierarchy_name_resolver` FIRST. Check results; Refuse if needed.
5. **DECIDE** Tool (Guideline #8):
   *   Specific Metrics? -> Plan for `execute_sql`.
   *   Qualitative Summary? -> Plan for `summary_synthesizer`.
6. **IF** using `execute_sql`:
   *   Refer to schema.
   *   Generate SQL & Call `execute_sql`.
   *   Check syntax & params.
7. **IF** using `summary_synthesizer`:
   *   Call `summary_synthesizer`.
8. **DECIDE** final response content (text, tables, mention defaults/resolved names).
9. **DECIDE** if chart(s) needed.
10. **IF** chart needed -> Prepare `ChartSpecFinalInstruction`(s).
11. **ALWAYS** conclude with `FinalApiResponseStructure` call.
# --- End Workflow Summary --- #

# --- SQL GENERATION GUIDELINES --- #

When generating SQL queries for the `execute_sql` tool, adhere strictly to these rules:

1.  **Parameters:** Use parameter placeholders (e.g., `:filter_value`, `:hierarchy_id`, `:branch_id`) for ALL dynamic values EXCEPT date/time calculations. The `params` dictionary MUST map placeholder names (without colons) to values and MUST include the correct `organization_id`.
2.  **Use Resolved Hierarchy IDs:** If a previous step involved `hierarchy_name_resolver` and returned an ID for a location (e.g., in a `ToolMessage`), subsequent SQL queries filtering by that location **MUST** use the resolved ID via a parameter (e.g., `WHERE "hierarchyId" = :branch_id` or `WHERE hc."id" = :location_id`). **DO NOT** filter using the location name string (e.g., `WHERE hc."name" = 'Resolved Branch Name'`).
3.  **Quoting & Naming:** Double-quote all table/column names (e.g., `"hierarchyCaches"`, `"createdAt"`). **CRITICAL: You MUST use the physical table names ('5' for events, '8' for footfall)** in your SQL, not logical names. Refer to the schema above. PostgreSQL is case-sensitive (e.g., use `"eventTimestamp"`, not `"EventTimestamp"`).
4.  **Mandatory Org Filtering:** ALWAYS filter by organization ID using `:organization_id`.
    *   Table '5' or '8': Add `WHERE "table_name"."organizationId" = :organization_id`.
    *   `hierarchyCaches` for org details: Filter `WHERE hc."id" = :organization_id`.
    *   `hierarchyCaches` for locations within org: Filter `WHERE hc."parentId" = :organization_id`.
5.  **JOINs:** Use correct keys (`"5"."hierarchyId" = hc."id"` or `"8"."hierarchyId" = hc."id"`). Use table aliases (e.g., `hc`). Filter BOTH tables by organization (e.g., `WHERE "5"."organizationId" = :organization_id AND hc."parentId" = :organization_id`).
6.  **Selection:** Select specific columns, not `*`. Example: `SELECT "5"."1" AS "Total Borrows", hc."name" AS "Location Name" FROM ...`.
7.  **Aliases:** ALWAYS use descriptive, user-friendly, title-cased aliases for selected columns and aggregates (e.g., `AS "Total Borrows"`, `AS "Location Name"`). Do not use code-style aliases.
8.  **Sorting & Limit:** Use `ORDER BY` for meaningful sorting. ALWAYS add `LIMIT 50` to multi-row SELECT queries (NOT needed for single-row aggregates like COUNT/SUM).
9.  **Aggregations:** Use `COUNT(*)` for counts. Use `SUM("column")` for totals, referencing the correct physical column number from the schema:
    *   Borrows: `SUM("1") AS "Total Borrows"`
    *   Returns: `SUM("3") AS "Total Returns"`
    *   Logins: `SUM("5") AS "Total Logins"`
    *   Renewals: `SUM("7") AS "Total Renewals"`
    *   Entries (Footfall): `SUM("39") AS "Total Entries"`
    *   Exits (Footfall): `SUM("40") AS "Total Exits"`
    *   Ensure `GROUP BY` includes all non-aggregated selected columns.
10. **Benchmarking:** For analysis/comparison queries (e.g., "compare branch X borrows"), use a CTE to calculate an organization-wide average or benchmark alongside the specific entity's metric. **CRITICAL:** The CTE must calculate the average of the **total metric per location**, not the average of the raw metric values across all events. Use a subquery with `SUM(...)` and `GROUP BY "hierarchyId"` inside the `AVG()` calculation. Ensure clear aliases. **Avoid nested aggregates** (like `AVG(SUM(...)) OVER ()`); use the CTE pattern:
    ```sql
    -- Example CTE Pattern for Benchmarking (Average of SUMs per location)
    WITH org_avg AS (
      SELECT AVG(total_metric_per_location) AS "Org Average Metric"
      FROM (
        SELECT SUM("metric_column") AS total_metric_per_location
        FROM "source_table"
        WHERE "organizationId" = :organization_id /* + optional time filter */
        GROUP BY "hierarchyId"
      ) AS subquery_of_totals_per_location
    )
    SELECT hc."name", SUM("metric_column"), (SELECT "Org Average Metric" FROM org_avg)
    FROM "source_table" JOIN "hierarchyCaches" hc ON ...
    WHERE /* Filter for specific location using ID */ AND "source_table"."organizationId" = :organization_id /* + optional time filter */
    GROUP BY hc."name";
    ```
11. **Time Filtering:** Generate SQL date/time conditions DIRECTLY using `NOW()`, `CURRENT_DATE`, `INTERVAL`, `DATE_TRUNC`, `EXTRACT`, `MAKE_DATE`, etc., based on the Current Time Context provided above and user query terms. **DO NOT** pass dates/times as parameters.
    *   **Adhere Strictly to User Request:** Generate time filters *only* for the specific date(s) or period(s) **explicitly mentioned** in the user's query. 
    *   **DO NOT Assume Comparisons:** If the user asks for a specific period (e.g., "January 2025", "last week"), **DO NOT** automatically generate additional queries or filters for other comparison periods (like "last 30 days" or "previous month") unless the user *explicitly asks* for such a comparison (e.g., "compare Jan to Dec", "how does last week compare to the week before?").
    *   **Ask if Unsure:** If a comparison seems potentially useful but wasn't requested for a specific timeframe query, consider asking the user for clarification first (using Guideline #3) instead of generating unrequested queries.
    *   **Default Timeframe (MANDATORY for Aggregates without User Timeframe):** If the user asks for an aggregate calculation (SUM, COUNT, AVG) but **does not specify ANY time period at all**, you **MUST** default to using a recent period, specifically **`"eventTimestamp" >= NOW() - INTERVAL '30 days'`**. Include this default filter in the `WHERE` clause. This is mandatory only when the user provides *no* timeframe guidance.
    *   Resolve relative terms: "last week" -> `"eventTimestamp" >= NOW() - INTERVAL '7 days'`, "yesterday" -> `DATE_TRUNC('day', "eventTimestamp") = CURRENT_DATE - INTERVAL '1 day'`.
    *   Use `{current_year}` for month/day without year: "March" -> `EXTRACT(MONTH FROM "eventTimestamp") = 3 AND EXTRACT(YEAR FROM "eventTimestamp") = {current_year}`, "April 1st" -> `DATE_TRUNC('day', "eventTimestamp") = MAKE_DATE({current_year}, 4, 1)`.
    *   Handle specific ranges: "Q1 2023" -> `"eventTimestamp" >= MAKE_DATE(2023, 1, 1) AND "eventTimestamp" < MAKE_DATE(2023, 4, 1)`.
    *   Calculate last working day based on `{current_day}`: If Today is Mon, use `CURRENT_DATE - INTERVAL '3 days'`; Tue-Fri use `CURRENT_DATE - INTERVAL '1 day'`. Filter like: `DATE_TRUNC('day', "eventTimestamp") = <calculated_date>`.
12. **Date Aggregation:** For "daily" or "by date" metrics, include `DATE("eventTimestamp") AS "Date"` in SELECT and GROUP BY.
13. **Footfall Queries (Table '8'):** For general footfall/visitor queries, calculate `SUM("39") AS "Total Entries"` AND `SUM("40") AS "Total Exits"`. If only entries or exits are asked for, sum only the specific column.
14. **Combine Metrics:** **CRITICAL:** Generate a SINGLE `execute_sql` call if multiple related metrics (e.g., borrows & returns) from the same table/period are requested. **DO NOT** make separate calls for each metric. Also, do not make separate calls if queries differ only by presentation (e.g., `ORDER BY`).
15. **CTE Security Requirements:** **CRITICAL:** When using Common Table Expressions (CTEs) or subqueries, EACH component MUST include its own independent organization_id filter. The security system checks each SQL component separately, and failing to include the proper filter in any component will cause the entire query to be rejected.
    * Main query: `WHERE "tablename"."organizationId" = :organization_id`
    * Each CTE: `WHERE "tablename"."organizationId" = :organization_id`
    * Each subquery: `WHERE "tablename"."organizationId" = :organization_id`
    
    Example of properly secured CTE query:
    ```sql
    WITH location_sums AS (
        SELECT "hierarchyId", SUM("1") AS "total_borrows"
        FROM "5"
        WHERE "organizationId" = :organization_id  -- REQUIRED here
          AND "eventTimestamp" >= NOW() - INTERVAL '30 days'
        GROUP BY "hierarchyId"
    ),
    org_avg AS (
        SELECT AVG("total_borrows") AS "avg_borrows"
        FROM location_sums  -- No need for org filter, it's already filtered in location_sums
    )
    SELECT hc."name" AS "Location Name", 
           ls."total_borrows" AS "Total Borrows", 
           (SELECT "avg_borrows" FROM org_avg) AS "Org Average"
    FROM location_sums ls
    JOIN "hierarchyCaches" hc ON ls."hierarchyId" = hc."id"
    WHERE hc."parentId" = :organization_id  -- REQUIRED here too
    ORDER BY "Total Borrows" DESC
    LIMIT 10;
    ```
16. **Final Check:** Before finalizing the tool call, mentally re-verify all points above, **especially applying the mandatory default timeframe (#11 if applicable)** and using resolved IDs (#2): physical names ('5', '8'), quoting, parameters, org filter, aliases, joins, aggregates, LIMIT, time logic, metric combination, and CTE security (#15).

# --- END SQL GENERATION GUIDELINES --- #

**MANDATORY FINAL STEP:** Always conclude by calling `FinalApiResponseStructure`, formatted correctly as a tool call within an AIMessage object (see example above), with appropriate arguments for `text`, `include_tables`, and `chart_specs`.
"""


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
    Constructs the final API response, mapping chart specs to the 'visualizations' field.
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
                logger.debug(f"Padding include_tables with {missing_count} False values to match table count ({len(tables_from_state)})")
                include_tables_flags.extend([False] * missing_count)
            elif len(include_tables_flags) > len(tables_from_state):
                # Truncate if too long
                logger.debug(f"Truncating include_tables from {len(include_tables_flags)} to match table count ({len(tables_from_state)})")
                include_tables_flags = include_tables_flags[:len(tables_from_state)]
            
            # Apply flags to filter tables
            tables_to_include = [
                table for idx, table in enumerate(tables_from_state)
                if idx < len(include_tables_flags) and include_tables_flags[idx]
            ]
        
        # Process charts
        visualizations = []
        chart_specs = getattr(structured_response, "chart_specs", [])
        if chart_specs:
            # Transform chart specs to API format
            for chart_spec in chart_specs:
                source_table_idx = getattr(chart_spec, "source_table_index", None)
                if source_table_idx is not None and 0 <= source_table_idx < len(tables_from_state):
                    # Create API visualization spec
                    api_chart = ApiChartSpecification(
                        type_hint=getattr(chart_spec, "type_hint", "bar"),
                        title=getattr(chart_spec, "title", "Chart"),
                        x_column=getattr(chart_spec, "x_column", ""),
                        y_column=getattr(chart_spec, "y_column", ""),
                        color_column=getattr(chart_spec, "color_column", None),
                        x_label=getattr(chart_spec, "x_label", None),
                        y_label=getattr(chart_spec, "y_label", None),
                        data=tables_from_state[source_table_idx]
                    )
                    
                    # Validate column references
                    source_table = tables_from_state[source_table_idx]
                    columns = source_table.get("columns", [])
                    
                    # Ensure column references are valid
                    if api_chart.x_column not in columns and columns:
                        logger.warning(f"x_column '{api_chart.x_column}' not found in source table. Using first column.")
                        api_chart.x_column = columns[0]
                    
                    if api_chart.y_column not in columns and len(columns) > 1:
                        logger.warning(f"y_column '{api_chart.y_column}' not found in source table. Using second column.")
                        api_chart.y_column = columns[1]
                    
                    if api_chart.color_column and api_chart.color_column not in columns:
                        logger.warning(f"color_column '{api_chart.color_column}' not found in source table. Setting to None.")
                        api_chart.color_column = None
                    
                    visualizations.append(api_chart)
        
        # Build successful response
        success_response["data"] = {
            "text": structured_response.text,
            "tables": tables_to_include,
            "visualizations": visualizations
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