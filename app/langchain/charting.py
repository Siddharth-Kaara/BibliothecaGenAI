import logging
from typing import List, Optional, Dict, Any, Tuple
from pydantic import BaseModel, Field
import numbers # Import for checking numeric types
import copy # Import copy for deep copying

# Local Imports needed for type hints if we add functions later
from app.schemas.chat import ApiChartSpecification, TableData

logger = logging.getLogger(__name__)

# --- Instruction Structure included directly in FinalApiResponseStructure ---
class ChartSpecFinalInstruction(BaseModel):
    """Defines the specification for a chart to be rendered by the frontend.
       This structure is generated directly by the LLM within the FinalApiResponseStructure.
    """
    source_table_index: int = Field(description="The 0-based index of the table in the agent's 'tables' state that contains the data for this chart.")
    type_hint: str = Field(description="The suggested chart type for the frontend (e.g., 'bar', 'pie', 'line', 'scatter').")
    title: str = Field(description="The title for the chart.")
    x_column: str = Field(description="The name of the column from the source table to use for the X-axis or labels.")
    y_columns: List[str] = Field(default_factory=list, description="The name(s) of the column(s) from the source table to use for the Y-axis or values. Multiple for multi-series charts.")
    color_column: Optional[str] = Field(default=None, description="Optional: The name of the column to use for grouping data by color/hue. For multi-series from y_columns, this will be set to 'Metric'.")
    x_label: Optional[str] = Field(default=None, description="Optional: A descriptive label for the X-axis. Defaults to x_column if not provided.")
    y_label: Optional[str] = Field(default=None, description="Optional: A descriptive label for the Y-axis. Defaults to y_column (or 'Value' for multi-series) if not provided.")

# --- Helper function to transform wide summary data for Pie charts ---
def _transform_wide_summary_to_pie_data(
    source_table: Dict[str, Any]
    # Removed llm_spec_x_col, llm_spec_y_col parameters
) -> Optional[Dict[str, Any]]:
    """
    Transforms a single-row, multi-column table (like a summary of multiple metrics)
    into the 2-column (Category, Value) format required for pie charts.
    It uses the original column names as the categories.

    Args:
        source_table: The original table data {'columns': [...], 'rows': [[...]]}.

    Returns:
        A new table dictionary in the format {'columns': ['Category', 'Value'], 'rows': [['Metric1', Val1], ['Metric2', Val2], ...]}
        or None if transformation is not applicable or fails.
    """
    columns = source_table.get("columns", [])
    rows = source_table.get("rows", [])

    # Check if transformation is applicable: 1 row, >= 2 columns
    if len(rows) != 1 or len(columns) < 2:
        logger.debug("[_transform_wide_summary_to_pie_data] Skipping transformation: Data does not match 1 row, >=2 columns pattern.")
        return None # Not the pattern we're targeting

    try:
        # Get the single row of data
        row_data = rows[0]
        if len(row_data) != len(columns):
            logger.warning("[_transform_wide_summary_to_pie_data] Skipping transformation: Row length does not match column count.")
            return None

        # Create the new long-format data
        new_columns = ["Category", "Value"]
        new_rows = []
        for i, col_name in enumerate(columns):
             # Use the original column name as the category
             value = row_data[i]
             new_rows.append([col_name, value])

        if not new_rows:
            logger.warning("[_transform_wide_summary_to_pie_data] Transformation resulted in empty data.")
            return None
            
        transformed_table = {
            "columns": new_columns,
            "rows": new_rows,
            "metadata": {"transformed_for_pie": True} # Mark as transformed
        }
        logger.info(f"Successfully transformed wide summary data for pie chart. Original cols: {columns} -> New cols: {new_columns}")
        return transformed_table

    except (IndexError, TypeError) as e:
        logger.warning(f"Failed to transform wide summary data for pie chart: {e}. Original cols: {columns}", exc_info=True)
        return None # Transformation failed

# --- NEW: Helper function to transform wide summary data for Bar charts ---
def _transform_wide_summary_to_bar_data(
    source_table: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """
    Transforms a single-row, multi-column table (like a summary of multiple metrics)
    into the 2-column (Metric, Value) format suitable for a simple bar chart
    showing total counts per metric.

    Args:
        source_table: The original table data {'columns': [...], 'rows': [[...]]}.

    Returns:
        A new table dictionary in the format {'columns': ['Metric', 'Value'], 'rows': [['Metric1', Val1], ['Metric2', Val2], ...]}
        or None if transformation is not applicable or fails.
    """
    columns = source_table.get("columns", [])
    rows = source_table.get("rows", [])

    # Check if transformation is applicable: 1 row, >= 1 column
    if len(rows) != 1 or len(columns) < 1:
        logger.debug("[_transform_wide_summary_to_bar_data] Skipping transformation: Data does not match 1 row, >=1 column pattern.")
        return None # Not the pattern we're targeting

    try:
        row_data = rows[0]
        if len(row_data) != len(columns):
            logger.warning("[_transform_wide_summary_to_bar_data] Skipping transformation: Row length does not match column count.")
            return None

        # Create the new long-format data
        new_columns = ["Metric", "Value"] # Standard names for this transformation
        new_rows = []
        for i, col_name in enumerate(columns):
             value = row_data[i]
             # Attempt to convert value to numeric, skip if not possible
             numeric_value = None
             if isinstance(value, numbers.Number):
                 numeric_value = value
             elif isinstance(value, str):
                 try: numeric_value = float(value)
                 except (ValueError, TypeError): pass
             
             if numeric_value is not None:
                 new_rows.append([col_name, numeric_value]) # Use original column name as the Metric
             else:
                 logger.warning(f"[_transform_wide_summary_to_bar_data] Skipping metric '{col_name}' as its value '{value}' is not numeric.")

        if not new_rows:
            logger.warning("[_transform_wide_summary_to_bar_data] Transformation resulted in empty data (no numeric metrics found).")
            return None
            
        transformed_table = {
            "columns": new_columns,
            "rows": new_rows,
            "metadata": {"transformed_summary_for_bar": True} # Mark as transformed
        }
        logger.info(f"Successfully transformed wide summary data for bar chart. Original cols: {columns} -> New cols: {new_columns}")
        return transformed_table

    except (IndexError, TypeError) as e:
        logger.warning(f"Failed to transform wide summary data for bar chart: {e}. Original cols: {columns}", exc_info=True)
        return None # Transformation failed

# --- Validation Functions for Specific Chart Types ---

def _validate_pie_chart_spec(api_chart: ApiChartSpecification, columns: List[str], rows: List[List[Any]]) -> Tuple[bool, Optional[str]]:
    """Validates specs specifically for a pie chart.
       Assumes data might have been pre-transformed.
    """
    # Rule 1: No color column allowed
    if api_chart.color_column is not None:
        logger.warning(f"Pie chart spec '{api_chart.title}' had color_column '{api_chart.color_column}'. Invalid.")
        # Lenient: Backend should ensure color_column is null if transformation occurred. Frontend ignores it anyway.
        # We will correct this in the main function if needed.

    # Rule 2: Exactly 2 columns required (category, value) - This validation runs AFTER potential transformation
    if len(columns) != 2:
         logger.warning(f"Pie chart spec '{api_chart.title}': Source table (potentially transformed) does not have exactly 2 columns (has {len(columns)}: {columns}). Invalid.")
         return False, "Pie chart requires source data with exactly 2 columns (category, value)."

    # Rule 3: Ensure x_column and y_column match the *actual* columns present
    # The main function will have already enforced standard names ('Category', 'Value') if transformation occurred.
    if api_chart.x_column not in columns or api_chart.y_column not in columns:
           logger.warning(f"Pie chart spec '{api_chart.title}': Specified x/y columns ('{api_chart.x_column}', '{api_chart.y_column}') don't match actual source table columns {columns}. Invalid.")
           # This check might be redundant if main function enforces names post-transformation, but good safety check.
           return False, f"Pie chart x/y columns ('{api_chart.x_column}', '{api_chart.y_column}') not found in data columns {columns}."
    
    # Rule 3b: Check x and y are different columns
    if api_chart.x_column == api_chart.y_column:
           logger.warning(f"Pie chart spec '{api_chart.title}': x_column and y_column are the same ('{api_chart.x_column}'). Invalid.")
           return False, "Pie chart x_column and y_column must be different."

    # Rule 4: Check if y_column data is numeric (check first row if available)
    if rows:
        # Find index based on potentially corrected column name
        try:
             y_col_index = columns.index(api_chart.y_column)
             first_row_y_val = rows[0][y_col_index]
             if not isinstance(first_row_y_val, numbers.Number):
                 logger.warning(f"Pie chart spec '{api_chart.title}': y_column '{api_chart.y_column}' data ('{first_row_y_val}') does not appear numeric. Invalid.")
                 return False, f"Pie chart requires numeric data for the value column ('{api_chart.y_column}')."
        except (ValueError, IndexError):
             # This case should be caught by Rule 3, but defensively check here too
             logger.warning(f"Pie chart spec '{api_chart.title}': Could not find index for y_column '{api_chart.y_column}' in columns {columns}. Invalid.")
             return False, f"Pie chart y_column '{api_chart.y_column}' not found in data columns."

    return True, None # All pie chart rules passed

def _validate_bar_chart_spec(api_chart: ApiChartSpecification, columns: List[str], rows: List[List[Any]]) -> Tuple[bool, Optional[str]]:
    """Validates specs specifically for a bar chart (basic validation for now).

    Returns:
        (is_valid: bool, failure_reason: Optional[str])
    """
    # Basic check: x and y columns must exist (already handled in main function)
    # Advanced check (Example): If color_column is present, ensure y_column is 'Value'
    # This relates to the multi-metric transformation rule in the prompt
    if api_chart.color_column:
         # This check assumes the transformation mentioned in agent prompt guideline #7e
         # might eventually happen upstream or needs to be handled by frontend if data is wide.
         # For now, we just ensure the specified columns exist, which is done in the main func.
         pass # Placeholder for more complex bar chart logic if needed

    # Basic validation: ensure y column seems numeric
    if rows and api_chart.y_column in columns:
         y_col_index = columns.index(api_chart.y_column)
         first_row_y_val = rows[0][y_col_index]
         if not isinstance(first_row_y_val, numbers.Number):
             logger.warning(f"Bar chart spec '{api_chart.title}': y_column '{api_chart.y_column}' data ('{first_row_y_val}') does not appear numeric. Invalid.")
             return False, f"Bar chart requires numeric data for the value column ('{api_chart.y_column}')."

    return True, None

def _validate_line_chart_spec(api_chart: ApiChartSpecification, columns: List[str], rows: List[List[Any]]) -> Tuple[bool, Optional[str]]:
    """Validates specs specifically for a line chart (basic validation for now).

    Returns:
        (is_valid: bool, failure_reason: Optional[str])
    """
    # Basic check: x and y columns must exist (already handled in main function)
    # Advanced check: Ensure x column is date/time or numeric?

    # Basic validation: ensure y column seems numeric
    if rows and api_chart.y_column in columns:
         y_col_index = columns.index(api_chart.y_column)
         first_row_y_val = rows[0][y_col_index]
         if not isinstance(first_row_y_val, numbers.Number):
             logger.warning(f"Line chart spec '{api_chart.title}': y_column '{api_chart.y_column}' data ('{first_row_y_val}') does not appear numeric. Invalid.")
             return False, f"Line chart requires numeric data for the value column ('{api_chart.y_column}')."
             
    return True, None

# --- Main Processing Function ---
def process_and_validate_chart_specs(
    chart_specs: List[ChartSpecFinalInstruction],
    tables_from_state: List[Dict[str, Any]] # Expects list of dicts like {'columns': [], 'rows': []}
) -> Tuple[List[ApiChartSpecification], List[Dict[str, str]]]: # Return valid charts AND info on filtered ones
    """Processes chart specifications from LLM, validates against tables, and creates API specs.
    Handles data transformation for multi-metric charts and single-row summary tables.

    Args:
        chart_specs: List of chart specifications generated by the LLM.
        tables_from_state: List of tables (as dictionaries) retrieved during agent execution.

    Returns:
        A tuple containing:
        - List of validated ApiChartSpecification objects ready for the API response.
        - List of dictionaries detailing charts that were filtered out (e.g., {'title': '...', 'reason': '...'}).
    """
    visualizations = []
    filtered_out_info = [] 
    if not chart_specs or not tables_from_state:
        return [], []

    for llm_chart_spec in chart_specs:
        spec = copy.deepcopy(llm_chart_spec) 
        spec_title = getattr(spec, "title", "Untitled Chart")
        failure_reason = None
        valid_chart = True
        is_multi_metric = False
        is_pie_transformed = False
        is_summary_transformed = False
        
        # --- START: MODIFIED LOGIC FOR SOURCE TABLE DETERMINATION ---
        source_table_for_processing: Optional[Dict[str, Any]] = None
        columns_from_source_table: List[str] = []
        rows_from_source_table: List[List[Any]] = []

        type_hint = getattr(spec, "type_hint", "bar").lower()
        llm_specified_y_cols = getattr(spec, 'y_columns', [])

        # Attempt to combine data for bar chart summary from multiple single-row tables
        if type_hint == 'bar' and llm_specified_y_cols:
            logger.debug(f"Chart '{spec_title}' (bar type with y_columns: {llm_specified_y_cols}): Checking for multi-table single-row summary pattern.")
            combined_row_data: Dict[str, Any] = {}
            found_any_y_col_metric = False

            for y_col_name in llm_specified_y_cols:
                found_this_metric = False
                for table_idx, table_in_state in enumerate(tables_from_state):
                    if isinstance(table_in_state, dict) and \
                       len(table_in_state.get("rows", [])) == 1 and \
                       y_col_name in table_in_state.get("columns", []):
                        
                        cols_of_this_table = table_in_state["columns"]
                        row_of_this_table = table_in_state["rows"][0]
                        
                        try:
                            metric_idx = cols_of_this_table.index(y_col_name)
                            metric_value = row_of_this_table[metric_idx]

                            # Ensure the metric value is numeric for a summary bar chart
                            if isinstance(metric_value, numbers.Number):
                                combined_row_data[y_col_name] = metric_value
                                found_this_metric = True
                                found_any_y_col_metric = True
                                logger.debug(f"Chart '{spec_title}': Found y_column '{y_col_name}' in table {table_idx} with value {metric_value}.")
                                break # Found this y_col_name, move to the next one
                            else:
                                logger.debug(f"Chart '{spec_title}': y_column '{y_col_name}' in table {table_idx} is not numeric ('{metric_value}'). Skipping.")
                        except (ValueError, IndexError):
                            # Should not happen if y_col_name is in columns, but defensive
                            logger.warning(f"Error accessing y_column '{y_col_name}' from table {table_idx} despite checks.")
                            continue 
                if not found_this_metric:
                    logger.warning(f"Chart '{spec_title}': Specified y_column '{y_col_name}' not found or not numeric in any single-row table.")

            if found_any_y_col_metric and combined_row_data:
                # Successfully combined metrics from potentially multiple tables
                source_table_for_processing = {
                    "columns": list(combined_row_data.keys()),
                    "rows": [list(combined_row_data.values())],
                    "metadata": {"source": "combined_single_row_summary"}
                }
                columns_from_source_table = source_table_for_processing["columns"]
                rows_from_source_table = source_table_for_processing["rows"]
                logger.info(f"Chart '{spec_title}': Successfully created a combined single-row source table for summary bar chart with columns {columns_from_source_table}.")
            else:
                logger.debug(f"Chart '{spec_title}': Did not combine metrics for summary bar. Will use specified source_table_index if valid.")

        # Fallback or default: use the specified source_table_index if combination didn't happen or isn't applicable
        if source_table_for_processing is None:
            source_table_idx = getattr(spec, "source_table_index", None)
            if source_table_idx is None:
                failure_reason = "Missing source table index and could not combine summary data."
                valid_chart = False
            elif not (0 <= source_table_idx < len(tables_from_state)):
                failure_reason = f"Invalid source table index ({source_table_idx})."
                valid_chart = False
            else:
                original_source_table = tables_from_state[source_table_idx]
                if not isinstance(original_source_table, dict) or "columns" not in original_source_table or "rows" not in original_source_table:
                    failure_reason = f"Source table {source_table_idx} has invalid format."
                    valid_chart = False
                else:
                    source_table_for_processing = original_source_table
                    columns_from_source_table = source_table_for_processing.get("columns", [])
                    rows_from_source_table = source_table_for_processing.get("rows", [])
                    if not columns_from_source_table:
                        failure_reason = f"Source table {source_table_idx} (from index) has no columns."
                        valid_chart = False
        
        if not valid_chart or source_table_for_processing is None:
            logger.warning(f"Chart spec '{spec_title}' failed basic validation or source table setup: {failure_reason}")
            filtered_out_info.append({"title": spec_title, "reason": failure_reason or "Source table could not be determined."})
            continue
        # --- END: MODIFIED LOGIC FOR SOURCE TABLE DETERMINATION ---

        try:
            # The rest of the processing uses source_table_for_processing, columns_from_source_table, rows_from_source_table

            allowed_types = ['bar', 'pie', 'line'] # type_hint already lowercased
            if type_hint not in allowed_types:
                failure_reason = f"Unsupported chart type '{type_hint}'. Allowed types: {allowed_types}"
                valid_chart = False
                logger.warning(f"Chart spec '{spec_title}' has unsupported type_hint '{type_hint}'. Skipping.")
                filtered_out_info.append({"title": spec_title, "reason": failure_reason})
                continue

            data_for_chart = source_table_for_processing 
            columns_to_validate = columns_from_source_table
            
            llm_specified_x_col = getattr(spec, 'x_column', "")
            # llm_specified_y_cols already fetched
            llm_specified_color_col = getattr(spec, 'color_column', None)

            if type_hint == 'bar' and len(rows_from_source_table) == 1 and len(columns_from_source_table) >= 1:
                # This transformation is attempted even if data was combined above,
                # _transform_wide_summary_to_bar_data handles the already combined single-row data.
                logger.info(f"Bar chart '{spec_title}' (source cols: {columns_from_source_table}) detected single-row summary pattern. Attempting transformation.")
                transformed_summary_data = _transform_wide_summary_to_bar_data(source_table_for_processing)
                if transformed_summary_data:
                    data_for_chart = transformed_summary_data
                    columns_to_validate = transformed_summary_data["columns"]
                    is_summary_transformed = True
                    logger.debug(f"Bar chart summary transformation successful for '{spec_title}'. New columns: {columns_to_validate}")
                else:
                    logger.warning(f"Bar chart '{spec_title}' summary transformation failed. Chart may be invalid.")

            elif type_hint == 'pie' and len(rows_from_source_table) == 1 and len(columns_from_source_table) >= 2:
                 logger.info(f"Pie chart '{spec_title}' detected wide summary data pattern. Attempting deterministic transformation.")
                 transformed_pie_data = _transform_wide_summary_to_pie_data(source_table_for_processing)
                 if transformed_pie_data:
                     data_for_chart = transformed_pie_data
                     columns_to_validate = transformed_pie_data["columns"]
                     is_pie_transformed = True 
                     logger.debug(f"Pie chart transformation successful for '{spec_title}'. New columns: {columns_to_validate}")
                 else:
                     logger.warning(f"Pie chart '{spec_title}' transformation from wide summary failed. Chart may be invalid.")
            
            elif not is_summary_transformed and not is_pie_transformed and type_hint in ['bar', 'line'] and llm_specified_color_col and len(llm_specified_y_cols) > 1:
                log_prefix = f"Multi-series '{spec_title}' (Inferred from {len(llm_specified_y_cols)} y_columns with explicit color_column)"
                logger.info(f"{log_prefix}: Applying wide-to-long transformation. Y-columns: {llm_specified_y_cols}")

                if not llm_specified_x_col or llm_specified_x_col not in columns_from_source_table:
                    failure_reason = f"{log_prefix} failed: x_column '{llm_specified_x_col}' (specified by LLM) not found in original source table columns {columns_from_source_table}."
                    valid_chart = False
                else:
                    for y_col_check in llm_specified_y_cols:
                        if y_col_check not in columns_from_source_table:
                            failure_reason = f"{log_prefix} failed: y_column '{y_col_check}' (specified by LLM) not found in original source table columns {columns_from_source_table}."
                            valid_chart = False
                            break
                    if not valid_chart: # If any y_column was invalid
                        pass # Failure reason already set
                    else:
                        transformed_data = _transform_wide_to_long(source_table_for_processing, llm_specified_x_col, llm_specified_y_cols)
                        if transformed_data.get("metadata", {}).get("transformed_from_wide_multi_y"):
                            data_for_chart = transformed_data
                            columns_to_validate = transformed_data["columns"]
                            is_multi_metric = True # Mark as multi-metric
                            logger.debug(f"{log_prefix}: Transformation successful. New columns for validation: {columns_to_validate}")
                        else:
                            failure_reason = f"{log_prefix}: Data transformation using _transform_wide_to_long failed. Original columns: {columns_from_source_table}, ID col: {llm_specified_x_col}, Y-cols: {llm_specified_y_cols}."
                            valid_chart = False
            
            elif not is_summary_transformed and not is_pie_transformed and not is_multi_metric and type_hint in ['bar', 'line'] and len(llm_specified_y_cols) > 1:
                log_prefix = f"Multi-series '{spec_title}' (Multiple y_columns: {len(llm_specified_y_cols)} without explicit color_column)"
                logger.info(f"{log_prefix}: Applying standard multi-series transformation. Y-columns: {llm_specified_y_cols}")
                
                if not llm_specified_x_col or llm_specified_x_col not in columns_from_source_table:
                    failure_reason = f"{log_prefix} failed: x_column '{llm_specified_x_col}' (specified by LLM) not found in original source table columns {columns_from_source_table}."
                    valid_chart = False
                else:
                    for y_col_check in llm_specified_y_cols:
                        if y_col_check not in columns_from_source_table:
                            failure_reason = f"{log_prefix} failed: y_column '{y_col_check}' (specified by LLM) not found in original source table columns {columns_from_source_table}."
                            valid_chart = False
                            break
                    if not valid_chart: # If any y_column was invalid
                        pass # Failure reason already set
                    else:
                        transformed_data = _transform_wide_to_long(source_table_for_processing, llm_specified_x_col, llm_specified_y_cols)
                        if transformed_data.get("metadata", {}).get("transformed_from_wide_multi_y"):
                            data_for_chart = transformed_data
                            columns_to_validate = transformed_data["columns"]
                            is_multi_metric = True # Mark as multi-metric
                            logger.debug(f"{log_prefix}: Transformation successful. New columns for validation: {columns_to_validate}")
                        else:
                            failure_reason = f"{log_prefix}: Data transformation using _transform_wide_to_long failed. Original columns: {columns_from_source_table}, ID col: {llm_specified_x_col}, Y-cols: {llm_specified_y_cols}."
                            valid_chart = False
            
            elif not is_summary_transformed and not is_pie_transformed and not is_multi_metric and \
                 type_hint in ['bar', 'line'] and llm_specified_color_col and llm_specified_y_cols and len(llm_specified_y_cols) == 1:
                log_prefix = f"Potential Grouped Chart '{spec_title}' (color_column '{llm_specified_color_col}' with single y_column '{llm_specified_y_cols[0]}')"
                logger.info(f"{log_prefix}: This might be a standard grouped chart if data is long, or an attempt at multi-metric if data is wide and color_column is 'Metric'. The multi-y_column path is preferred for multi-metric.")
                # Current _transform_wide_to_long not called here. This path assumes data might be suitable for grouping or requires a different transform.
                # For now, we proceed without specific transformation for this case, relying on column validation.

            if not valid_chart: 
                logger.warning(f"Chart spec '{spec_title}' failed during/after transformation stage: {failure_reason}")
                filtered_out_info.append({"title": spec_title, "reason": failure_reason})
                continue

            api_y_column = "Value" if is_multi_metric else (llm_specified_y_cols[0] if llm_specified_y_cols else "")
            api_color_column = "Metric" if is_multi_metric else llm_specified_color_col

            api_chart = ApiChartSpecification(
                type_hint=type_hint, title=spec_title,
                x_column=llm_specified_x_col,
                y_column=api_y_column, 
                color_column=api_color_column,
                x_label=getattr(spec, "x_label", None), 
                y_label=getattr(spec, "y_label", None),
                data=TableData(**copy.deepcopy(data_for_chart))
            )

            if is_summary_transformed:
                api_chart.x_column = "Metric"
                api_chart.y_column = "Value"
                api_chart.color_column = None
                if not api_chart.y_label: api_chart.y_label = "Value"
                logger.debug(f"Enforced x='Metric', y='Value', color=None for transformed summary bar '{spec_title}'.")
            elif is_pie_transformed:
                 api_chart.x_column = "Category"; api_chart.y_column = "Value"; api_chart.color_column = None
                 if not api_chart.y_label: api_chart.y_label = "Value"
                 logger.debug(f"Enforced x='Category', y='Value', color=null for transformed pie '{spec_title}'.")
            elif is_multi_metric:
                api_chart.x_column = llm_specified_x_col # Keep the original x-column specified by LLM
                api_chart.y_column = "Value"
                api_chart.color_column = "Metric"
                if not api_chart.y_label: api_chart.y_label = "Value" # Set specific y_label
                logger.debug(f"Enforced y='Value', color='Metric', y_label='Value' for multi-metric chart '{spec_title}'. Original x-column '{llm_specified_x_col}' maintained.")
            elif type_hint == 'pie':
                 api_chart.color_column = None
                 logger.debug(f"Enforced color=null for non-transformed pie '{spec_title}'.")

            final_x_col = api_chart.x_column
            final_y_col = api_chart.y_column
            final_color_col = api_chart.color_column

            if not final_x_col or final_x_col not in columns_to_validate: failure_reason = f"Final x_column '{final_x_col}' not found in data columns {columns_to_validate}."; valid_chart = False
            if valid_chart and (not final_y_col or final_y_col not in columns_to_validate): failure_reason = f"Final y_column '{final_y_col}' not found in data columns {columns_to_validate}."; valid_chart = False
            if valid_chart and final_color_col and final_color_col not in columns_to_validate: failure_reason = f"Final color_column '{final_color_col}' not found in data columns {columns_to_validate}."; valid_chart = False

            if valid_chart:
                if not api_chart.x_label: api_chart.x_label = api_chart.x_column
                if not api_chart.y_label: api_chart.y_label = api_chart.y_column # Default if not set by transformations
            
            if not valid_chart: 
                logger.warning(f"Chart spec '{spec_title}' failed generic column validation after transformations/enforcements: {failure_reason}")
                filtered_out_info.append({"title": spec_title, "reason": failure_reason})
                continue

            type_specific_valid = True; type_specific_reason = None
            current_rows_for_validation = api_chart.data.rows
            current_columns_for_validation = api_chart.data.columns 

            if type_hint == 'pie': type_specific_valid, type_specific_reason = _validate_pie_chart_spec(api_chart, current_columns_for_validation, current_rows_for_validation)
            elif type_hint == 'bar': type_specific_valid, type_specific_reason = _validate_bar_chart_spec(api_chart, current_columns_for_validation, current_rows_for_validation)
            elif type_hint == 'line': type_specific_valid, type_specific_reason = _validate_line_chart_spec(api_chart, current_columns_for_validation, current_rows_for_validation)

            if not type_specific_valid: valid_chart = False; failure_reason = type_specific_reason or "Type-specific validation failed."

            if valid_chart: visualizations.append(api_chart)
            else: 
                logger.warning(f"Skipping chart spec '{spec_title}' due to validation errors: {failure_reason}") # Removed source_table_idx as it might be misleading now
                filtered_out_info.append({"title": spec_title, "reason": failure_reason or "Unknown validation error"})

        except Exception as e: 
            logger.error(f"Error processing chart spec: {spec}. Error: {e}", exc_info=True)
            filtered_out_info.append({"title": getattr(spec, 'title', 'Untitled Chart'), "reason": f"Internal processing error: {e}"}) # Use getattr for title
            continue

    return visualizations, filtered_out_info

# --- Helper function for Data Transformation (Wide to Long for Multi-Series) --- 
def _transform_wide_to_long(
    wide_table: Dict[str, Any], 
    id_column_name: str,
    value_columns_to_melt: List[str]
) -> Dict[str, Any]:
    """Transforms TableData from wide to long format for multi-series charts.

    Args:
        wide_table: The source table dictionary (keys: 'columns', 'rows').
        id_column_name: The name of the column to use as the identifier/category (X-axis).
        value_columns_to_melt: List of column names to be melted into 'Metric' and 'Value'.

    Returns:
        A new table dictionary in long format, or the original if transformation fails.
    """
    logger.debug(f"[_transform_wide_to_long] Starting multi-y transformation. ID col: {id_column_name}, Y-cols to melt: {value_columns_to_melt}")
    metric_col_name = "Metric" 
    value_col_name = "Value"   

    original_rows = wide_table.get('rows', [])
    original_columns = wide_table.get('columns', [])
    original_metadata = wide_table.get('metadata', {})
    long_rows = []

    if not original_rows or not original_columns:
        logger.warning("[_transform_wide_to_long] Input table has no rows or columns. Returning empty.")
        return {"columns": [id_column_name, metric_col_name, value_col_name], "rows": [], "metadata": {**original_metadata, "transformed_from_wide_multi_y": False, "transform_error": "No rows/columns"}}

    try:
        id_col_index = original_columns.index(id_column_name)
    except ValueError:
        logger.error(f"[_transform_wide_to_long] ID column '{id_column_name}' not found in wide table columns: {original_columns}.")
        return {"columns": original_columns, "rows": original_rows, "metadata": {**original_metadata, "transformed_from_wide_multi_y": False, "transform_error": f"ID col {id_column_name} not found"}}

    value_col_indices_map = {}
    for y_col_name in value_columns_to_melt:
        try:
            value_col_indices_map[y_col_name] = original_columns.index(y_col_name)
        except ValueError:
            logger.error(f"[_transform_wide_to_long] Value column '{y_col_name}' (from y_columns spec) not found in wide table columns: {original_columns}.")
            return {"columns": original_columns, "rows": original_rows, "metadata": {**original_metadata, "transformed_from_wide_multi_y": False, "transform_error": f"Y-col {y_col_name} not found"}}
    
    if not value_col_indices_map:
         logger.warning(f"[_transform_wide_to_long] No valid value columns found to melt for ID column '{id_column_name}'.")
         return {"columns": original_columns, "rows": original_rows, "metadata": {**original_metadata, "transformed_from_wide_multi_y": False, "transform_error": "No valid Y-cols to melt"}}

    for wide_row in original_rows:
        if len(wide_row) != len(original_columns):
            logger.warning(f"[_transform_wide_to_long] Skipping row with mismatching column count: {wide_row}")
            continue
            
        id_value = wide_row[id_col_index]
        for metric_name, val_col_idx in value_col_indices_map.items():
            value = wide_row[val_col_idx]
            numeric_value = None
            try:
                if value is not None: numeric_value = float(value)
            except (ValueError, TypeError):
                 logger.warning(f"[_transform_wide_to_long] Could not convert value '{value}' for metric '{metric_name}' to float. Appending as None.")
            
            long_rows.append([id_value, metric_name, numeric_value])

    long_columns = [id_column_name, metric_col_name, value_col_name]
    logger.info(f"[_transform_wide_to_long] Multi-y transformation complete. Produced {len(long_rows)} long format rows.")
    
    new_metadata = original_metadata.copy() if original_metadata else {}
    new_metadata["transformed_from_wide_multi_y"] = True # New metadata key
    new_metadata["original_columns"] = original_columns
    # Remove old key if present to avoid confusion
    if "transformed_from_wide" in new_metadata: del new_metadata["transformed_from_wide"]

    return {"columns": long_columns, "rows": long_rows, "metadata": new_metadata}

# Function to process chart specs will be added here later 