import logging
from typing import List, Optional, Dict, Any, Tuple
from pydantic import BaseModel, Field
import numbers # Import for checking numeric types
import copy # Import copy for deep copying

# Local Imports needed for type hints if we add functions later
from app.schemas.chat import ApiChartSpecification, TableData

logger = logging.getLogger(__name__)

# --- Instruction Structure included directly in FinalApiResponseStructure --- (Moved from agent.py)
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
        # Use deepcopy to avoid modifying the original spec from the LLM state
        # This is important if the same state is reused or logged elsewhere
        spec = copy.deepcopy(llm_chart_spec) 
        spec_title = getattr(spec, "title", "Untitled Chart")
        failure_reason = None
        valid_chart = True
        is_multi_metric = False       # Transformed wide-to-long based on spec
        is_pie_transformed = False    # Transformed wide-to-long for pie
        is_summary_transformed = False # Transformed single-row summary to long for bar

        try:
            source_table_idx = getattr(spec, "source_table_index", None)

            # --- Basic Validation: Index and Table Existence ---
            if source_table_idx is None: failure_reason = "Missing source table index."; valid_chart = False
            elif not (0 <= source_table_idx < len(tables_from_state)): failure_reason = f"Invalid source table index ({source_table_idx})."; valid_chart = False
            else:
                source_table = tables_from_state[source_table_idx]
                if not isinstance(source_table, dict) or "columns" not in source_table or "rows" not in source_table: failure_reason = f"Source table {source_table_idx} has invalid format."; valid_chart = False
                else:
                    columns = source_table.get("columns", []); rows = source_table.get("rows", [])
                    if not columns: failure_reason = f"Source table {source_table_idx} has no columns."; valid_chart = False
            
            if not valid_chart: logger.warning(f"Chart spec '{spec_title}' failed basic validation: {failure_reason}"); filtered_out_info.append({"title": spec_title, "reason": failure_reason}); continue
            # --- End Basic Validation ---

            # --- Type Hint Validation --- 
            type_hint = getattr(spec, "type_hint", "bar").lower()
            allowed_types = ['bar', 'pie', 'line']
            if type_hint not in allowed_types: failure_reason = f"Unsupported chart type '{type_hint}'. Allowed types: {allowed_types}"; valid_chart = False; logger.warning(f"Chart spec '{spec_title}' has unsupported type_hint '{type_hint}'. Skipping."); filtered_out_info.append({"title": spec_title, "reason": failure_reason}); continue
            # --- End Type Hint Validation ---            

            # --- Prepare Data & Column List (Potentially Transform) ---
            data_for_chart = source_table 
            columns_to_validate = columns 
            llm_specified_x_col = getattr(spec, 'x_column', "")
            llm_specified_y_col = getattr(spec, 'y_column', "")
            llm_specified_color_col = getattr(spec, 'color_column', None)

            # --- START: Data Transformations (Prioritized) --- 
            # 1. Handle Single-Row Summary for Bar Chart
            if type_hint == 'bar' and len(rows) == 1 and len(columns) >= 1:
                logger.info(f"Bar chart '{spec_title}' detected single-row summary pattern. Attempting transformation.")
                transformed_summary_data = _transform_wide_summary_to_bar_data(source_table)
                if transformed_summary_data:
                    data_for_chart = transformed_summary_data
                    columns_to_validate = transformed_summary_data["columns"] # ['Metric', 'Value']
                    is_summary_transformed = True
                    logger.debug(f"Bar chart summary transformation successful for '{spec_title}'. New columns: {columns_to_validate}")
                else:
                    logger.warning(f"Bar chart '{spec_title}' summary transformation failed. Chart may be invalid.")
                    # Keep original data, validation likely fails later

            # 2. Handle Wide Summary for Pie Chart (can run even if bar transform failed)
            elif type_hint == 'pie' and len(rows) == 1 and len(columns) >= 2:
                 logger.info(f"Pie chart '{spec_title}' detected wide summary data pattern. Attempting deterministic transformation.")
                 transformed_pie_data = _transform_wide_summary_to_pie_data(source_table)
                 if transformed_pie_data:
                     data_for_chart = transformed_pie_data
                     columns_to_validate = transformed_pie_data["columns"] # ['Category', 'Value']
                     is_pie_transformed = True 
                     logger.debug(f"Pie chart transformation successful for '{spec_title}'. New columns: {columns_to_validate}")
                 else:
                     logger.warning(f"Pie chart '{spec_title}' transformation from wide summary failed. Chart may be invalid.")
            
            # 3. Handle Standard Multi-Metric Bar/Line Transformation (Only if NOT already transformed)
            elif not is_summary_transformed and not is_pie_transformed and type_hint in ['bar', 'line'] and llm_specified_color_col:
                # Check if LLM specified correct multi-metric columns OR if we infer based on presence of color_col
                correctly_specified = (llm_specified_y_col == "Value" and llm_specified_color_col == "Metric")
                if correctly_specified or llm_specified_color_col in columns: # Allow inference if color_col exists
                    log_prefix = f"Multi-metric '{spec_title}' ('{'Correct Spec' if correctly_specified else 'Inferred'}')"
                    logger.info(f"{log_prefix}: Applying wide-to-long transformation.")
                    # Use LLM's x_col as the identifier for transformation
                    if llm_specified_x_col not in columns:
                        failure_reason = f"{log_prefix} failed: x_column '{llm_specified_x_col}' not found in original data."; valid_chart = False
                    else:
                        transformed_data = _transform_wide_to_long(source_table, llm_specified_x_col)
                        if transformed_data.get("metadata", {}).get("transformed_from_wide"):
                            data_for_chart = transformed_data
                            columns_to_validate = transformed_data["columns"] # [id_col, 'Metric', 'Value']
                            is_multi_metric = True
                            logger.debug(f"{log_prefix}: Transformation successful. Columns for validation: {columns_to_validate}")
                        else:
                            failure_reason = f"{log_prefix}: Data transformation failed."; valid_chart = False
                else: # LLM specified color_col but it doesn't exist in columns, cannot infer/transform
                    failure_reason = f"Chart spec '{spec_title}' specified color_column '{llm_specified_color_col}' which is not in source columns {columns}. Cannot transform."; valid_chart = False
            # --- END: Data Transformations ---

            if not valid_chart: logger.warning(f"Chart spec '{spec_title}' failed during transformation stage: {failure_reason}"); filtered_out_info.append({"title": spec_title, "reason": failure_reason}); continue

            # --- Create API Spec object (uses original LLM spec initially) ---
            api_chart = ApiChartSpecification(
                type_hint=type_hint, title=spec_title,
                x_column=llm_specified_x_col, y_column=llm_specified_y_col,
                color_column=llm_specified_color_col,
                x_label=getattr(spec, "x_label", None), y_label=getattr(spec, "y_label", None),
                data=TableData(**copy.deepcopy(data_for_chart)) # Convert dict to TableData
            )

            # --- Enforce Standard Columns AFTER Transformation --- #
            if is_summary_transformed: # Single-row summary bar chart
                api_chart.x_column = "Metric"
                api_chart.y_column = "Value"
                api_chart.color_column = None # No grouping needed for this type
                logger.debug(f"Enforced x='Metric', y='Value', color=None for transformed summary bar '{spec_title}'.")
            elif is_pie_transformed:
                 api_chart.x_column = "Category"; api_chart.y_column = "Value"; api_chart.color_column = None
                 logger.debug(f"Enforced x='Category', y='Value', color=null for transformed pie '{spec_title}'.")
            elif is_multi_metric: # Standard multi-metric bar/line
                api_chart.y_column = "Value"; api_chart.color_column = "Metric"
                logger.debug(f"Enforced y='Value', color='Metric' for multi-metric chart '{spec_title}'.")
            elif type_hint == 'pie': # Non-transformed pie still needs null color
                 api_chart.color_column = None
                 logger.debug(f"Enforced color=null for non-transformed pie '{spec_title}'.")
            # --- End Enforcement --- 

            # --- Generic Column Existence Validation (using FINAL api_chart spec and columns_to_validate) --- #
            final_x_col = api_chart.x_column
            final_y_col = api_chart.y_column
            final_color_col = api_chart.color_column

            if not final_x_col or final_x_col not in columns_to_validate: failure_reason = f"Final x_column '{final_x_col}' not found in data columns {columns_to_validate}."; valid_chart = False
            if valid_chart and (not final_y_col or final_y_col not in columns_to_validate): failure_reason = f"Final y_column '{final_y_col}' not found in data columns {columns_to_validate}."; valid_chart = False
            if valid_chart and final_color_col and final_color_col not in columns_to_validate: failure_reason = f"Final color_column '{final_color_col}' not found in data columns {columns_to_validate}."; valid_chart = False
            # --- End Generic Column Validation --- 

            # --- Default Labels if Missing --- 
            if valid_chart:
                if not api_chart.x_label: api_chart.x_label = api_chart.x_column
                if not api_chart.y_label: api_chart.y_label = api_chart.y_column
            # --- End Default Labels --- 

            if not valid_chart: logger.warning(f"Chart spec '{spec_title}' failed generic column validation: {failure_reason}"); filtered_out_info.append({"title": spec_title, "reason": failure_reason}); continue

            # --- Type-Specific Validation (using FINAL api_chart spec and its data) --- 
            type_specific_valid = True; type_specific_reason = None
            current_rows = api_chart.data.rows; current_columns = api_chart.data.columns

            if type_hint == 'pie': type_specific_valid, type_specific_reason = _validate_pie_chart_spec(api_chart, current_columns, current_rows)
            elif type_hint == 'bar': type_specific_valid, type_specific_reason = _validate_bar_chart_spec(api_chart, current_columns, current_rows)
            elif type_hint == 'line': type_specific_valid, type_specific_reason = _validate_line_chart_spec(api_chart, current_columns, current_rows)

            if not type_specific_valid: valid_chart = False; failure_reason = type_specific_reason or "Type-specific validation failed."
            # --- End Type-Specific Validation --- 

            # Add to list if still valid
            if valid_chart: visualizations.append(api_chart)
            else: logger.warning(f"Skipping chart spec '{spec_title}' due to validation errors (Index: {source_table_idx}): {failure_reason}"); filtered_out_info.append({"title": spec_title, "reason": failure_reason or "Unknown validation error"})

        except Exception as e: logger.error(f"Error processing chart spec: {spec}. Error: {e}", exc_info=True); filtered_out_info.append({"title": spec_title, "reason": f"Internal processing error: {e}"}); continue

    return visualizations, filtered_out_info

# --- Helper function for Data Transformation --- 
def _transform_wide_to_long(wide_table: Dict[str, Any], id_column_name: str) -> Dict[str, Any]:
    """Transforms TableData (dict format) from wide to long format for grouped bar charts.

    Args:
        wide_table: The source table dictionary (keys: 'columns', 'rows').
        id_column_name: The name of the column to use as the identifier/category.

    Returns:
        A new table dictionary in long format, or the original if transformation fails.
    """
    logger.debug(f"[_transform_wide_to_long] Starting transformation for table with ID column: {id_column_name}")
    metric_col_name = "Metric" # Convention expected by prompt Guideline #7e
    value_col_name = "Value"   # Convention expected by prompt Guideline #7e

    original_rows = wide_table.get('rows', [])
    original_columns = wide_table.get('columns', [])
    original_metadata = wide_table.get('metadata', {})
    long_rows = []

    if not original_rows or not original_columns:
        logger.warning("[_transform_wide_to_long] Input table has no rows or columns. Returning empty.")
        return {"columns": [id_column_name, metric_col_name, value_col_name], "rows": [], "metadata": original_metadata}

    try:
        id_col_index = original_columns.index(id_column_name)
    except ValueError:
        logger.error(f"[_transform_wide_to_long] ID column '{id_column_name}' not found in wide table columns: {original_columns}. Returning original table as fallback.")
        return wide_table # Fallback to original to prevent downstream errors

    value_column_indices = {
        i: col_name for i, col_name in enumerate(original_columns) if i != id_col_index
    }

    if not value_column_indices:
         logger.warning(f"[_transform_wide_to_long] No value columns found besides ID column '{id_column_name}'. Returning original table.")
         return wide_table

    for wide_row in original_rows:
        if len(wide_row) != len(original_columns):
            logger.warning(f"[_transform_wide_to_long] Skipping row with mismatching column count: {wide_row}")
            continue
            
        id_value = wide_row[id_col_index]
        for index, metric_name in value_column_indices.items():
            value = wide_row[index]
            try:
                # Convert to float, falling back to None if not possible
                numeric_value = float(value) if value is not None else None
                long_rows.append([id_value, metric_name, numeric_value])
            except (ValueError, TypeError):
                 logger.warning(f"[_transform_wide_to_long] Could not convert value '{value}' for metric '{metric_name}' to float. Appending as None.")
                 long_rows.append([id_value, metric_name, None])

    long_columns = [id_column_name, metric_col_name, value_col_name]
    logger.debug(f"[_transform_wide_to_long] Transformation complete. Produced {len(long_rows)} long format rows.")
    
    new_metadata = original_metadata.copy() if original_metadata else {}
    new_metadata["transformed_from_wide"] = True
    new_metadata["original_columns"] = original_columns

    return {"columns": long_columns, "rows": long_rows, "metadata": new_metadata}

# Function to process chart specs will be added here later 