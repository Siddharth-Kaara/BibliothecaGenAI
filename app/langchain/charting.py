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

# --- NEW: Helper function to transform wide summary data for Pie charts ---
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

    Args:
        chart_specs: List of chart specifications generated by the LLM.
        tables_from_state: List of tables (as dictionaries) retrieved during agent execution.

    Returns:
        A tuple containing:
        - List of validated ApiChartSpecification objects ready for the API response.
        - List of dictionaries detailing charts that were filtered out (e.g., {'title': '...', 'reason': '...'}).
    """
    visualizations = []
    filtered_out_info = [] # Track info for filtered charts
    if not chart_specs or not tables_from_state:
        return [], [] # Return empty lists

    for chart_spec in chart_specs:
        spec_title = getattr(chart_spec, "title", "Untitled Chart")
        failure_reason = None
        valid_chart = True # Start assuming valid
        try:
            source_table_idx = getattr(chart_spec, "source_table_index", None)

            # --- Basic Validation: Index and Table Existence ---
            if source_table_idx is None:
                failure_reason = "Missing source table index."
                valid_chart = False
            elif not (0 <= source_table_idx < len(tables_from_state)):
                failure_reason = f"Invalid source table index ({source_table_idx})."
                valid_chart = False
            else:
                source_table = tables_from_state[source_table_idx]
                if not isinstance(source_table, dict) or "columns" not in source_table or "rows" not in source_table:
                    failure_reason = f"Source table {source_table_idx} has invalid format."
                    valid_chart = False
                else:
                    columns = source_table.get("columns", [])
                    rows = source_table.get("rows", []) # Get rows for type-specific validation
                    if not columns:
                        failure_reason = f"Source table {source_table_idx} has no columns."
                        valid_chart = False
            # --- End Basic Validation ---

            if not valid_chart:
                 logger.warning(f"Chart spec '{spec_title}' failed basic validation: {failure_reason}")
                 filtered_out_info.append({"title": spec_title, "reason": failure_reason})
                 continue # Skip to next chart spec

            # --- Type Hint Validation --- 
            type_hint = getattr(chart_spec, "type_hint", "bar").lower()
            allowed_types = ['bar', 'pie', 'line']
            if type_hint not in allowed_types:
                logger.warning(f"Chart spec '{spec_title}' has unsupported type_hint '{type_hint}'. Skipping.")
                failure_reason = f"Unsupported chart type '{type_hint}'. Allowed types: {allowed_types}"
                valid_chart = False
                filtered_out_info.append({"title": spec_title, "reason": failure_reason})
                continue # Skip to next chart spec
            # --- End Type Hint Validation ---
            
            # --- Prepare Data & Column List (Potentially Transform for Multi-Metric OR Pie Chart) ---
            data_for_chart = source_table # Start with original
            columns_to_validate = columns # Start with original columns
            is_multi_metric = False
            is_pie_transformed = False # Flag for pie transformation
            llm_specified_x_col = getattr(chart_spec, 'x_column', "") # Store original spec for checks
            llm_specified_y_col = getattr(chart_spec, 'y_column', "")
            llm_specified_color_col = getattr(chart_spec, 'color_column', None)
            
            # --- START: Pie Chart Wide Summary Transformation ---
            if type_hint == 'pie':
                 # Check if source table looks like a wide summary (1 row, >= 2 cols)
                 if len(rows) == 1 and len(columns) >= 2:
                     logger.info(f"Pie chart '{spec_title}' detected wide summary data pattern (1 row, cols: {columns}). Attempting deterministic transformation.")
                     # Attempt transformation - relies only on the data shape, ignores LLM column spec for this case.
                     transformed_pie_data = _transform_wide_summary_to_pie_data(source_table)
                     if transformed_pie_data:
                         data_for_chart = transformed_pie_data
                         columns_to_validate = transformed_pie_data["columns"] # Should be ['Category', 'Value']
                         is_pie_transformed = True # Set flag
                         logger.debug(f"Pie chart transformation successful. New columns: {columns_to_validate}")
                     else:
                         # Transformation failed, log and potentially invalidate
                         logger.warning(f"Pie chart '{spec_title}' deterministic transformation from wide summary failed. Chart may be invalid.")
                         # Keep original data for now, validation will likely fail it
            # --- END: Pie Chart Wide Summary Transformation ---

            # Check for multi-metric bar/line case REQUIRING transformation
            # Note: This runs AFTER pie check, transformation logic is separate
            if not is_pie_transformed and type_hint in ['bar', 'line'] and llm_specified_color_col:
                 # CASE 1: LLM correctly specified Value/Metric columns AS PER PROMPT INSTRUCTIONS
                 if llm_specified_y_col == "Value" and llm_specified_color_col == "Metric":
                     logger.info(f"Chart spec '{spec_title}' correctly indicates multi-metric {type_hint}. Applying wide-to-long transformation.")
                     # Validate x_column specified by LLM exists in the original columns
                     if llm_specified_x_col not in columns:
                         logger.warning(f"Multi-metric '{spec_title}' (Correct Spec): Specified x_column '{llm_specified_x_col}' not in source columns {columns}. Skipping.")
                         valid_chart = False
                         failure_reason = f"Specified x_column '{llm_specified_x_col}' not found in original data for transformation."
                     else:
                         transformed_data = _transform_wide_to_long(source_table, llm_specified_x_col) # Use specified x-col
                         # Check if transformation was successful
                         if transformed_data.get("metadata", {}).get("transformed_from_wide"):
                             data_for_chart = transformed_data
                             columns_to_validate = transformed_data["columns"] # Validate against transformed columns
                             is_multi_metric = True
                             logger.debug(f"Transformed columns for validation: {columns_to_validate}")
                         else:
                             logger.warning(f"Multi-metric '{spec_title}' (Correct Spec): Data transformation failed. Skipping chart.")
                             valid_chart = False
                             failure_reason = "Data transformation for multi-metric chart failed."

                 # CASE 2: LLM specified a color_column but DID NOT use y_column="Value" and color_column="Metric". Infer intent.
                 else:
                     logger.warning(f"Chart spec '{spec_title}' has color_column ('{llm_specified_color_col}') but incorrect y/color spec (y='{llm_specified_y_col}', color='{llm_specified_color_col}'). Inferring multi-metric intent.")
                     # Check if LLM's specified x, y, and color columns EXIST in the ORIGINAL data
                     if llm_specified_x_col not in columns:
                         logger.warning(f"Multi-metric '{spec_title}' (Inferred): Specified x_column '{llm_specified_x_col}' not in original source columns {columns}. Cannot transform. Skipping.")
                         valid_chart = False
                         failure_reason = f"Inferred multi-metric failed: x_column '{llm_specified_x_col}' not found in original data."
                     elif llm_specified_y_col not in columns:
                         logger.warning(f"Multi-metric '{spec_title}' (Inferred): Specified y_column '{llm_specified_y_col}' not in original source columns {columns}. Cannot transform. Skipping.")
                         valid_chart = False
                         failure_reason = f"Inferred multi-metric failed: y_column '{llm_specified_y_col}' not found in original data."
                     elif llm_specified_color_col not in columns:
                         logger.warning(f"Multi-metric '{spec_title}' (Inferred): Specified color_column '{llm_specified_color_col}' not in original source columns {columns}. Cannot transform. Skipping.")
                         valid_chart = False
                         failure_reason = f"Inferred multi-metric failed: color_column '{llm_specified_color_col}' not found in original data."
                     else:
                         # Attempt transformation using the LLM's specified x_column as the identifier
                         logger.info(f"Attempting inferred multi-metric transformation using '{llm_specified_x_col}' as ID.")
                         transformed_data = _transform_wide_to_long(source_table, llm_specified_x_col)

                         # Check if transformation was successful
                         if transformed_data.get("metadata", {}).get("transformed_from_wide"):
                             data_for_chart = transformed_data
                             columns_to_validate = transformed_data["columns"] # Validate against transformed columns
                             is_multi_metric = True # Mark as multi-metric AFTER successful transformation
                             logger.info(f"Inferred multi-metric transformation successful. Will enforce y='Value', color='Metric' in final spec.")
                             logger.debug(f"Transformed columns for validation: {columns_to_validate}")
                         else:
                             logger.warning(f"Multi-metric '{spec_title}' (Inferred): Data transformation failed. Skipping chart.")
                             valid_chart = False
                             failure_reason = "Inferred multi-metric data transformation failed."

            if not valid_chart:
                 # Handle failure from multi-metric validation/transformation
                 filtered_out_info.append({"title": spec_title, "reason": failure_reason})
                 continue
            # --- End Data Preparation/Transformation ---

            # Create base API spec object AFTER potential transformation
            # Use DEEP COPY to avoid modifying original spec object from LLM
            api_chart = ApiChartSpecification(
                type_hint=type_hint, # Use original field name 'type_hint'
                title=spec_title,
                x_column=llm_specified_x_col, # Initial values from LLM spec
                y_column=llm_specified_y_col,
                color_column=llm_specified_color_col,
                x_label=getattr(chart_spec, "x_label", None),
                y_label=getattr(chart_spec, "y_label", None),
                # Ensure data is in TableData format before assigning
                data=TableData(**copy.deepcopy(data_for_chart)) # Convert dict to TableData, ensure deep copy
            )

            # --- **ENFORCE Standard Columns if Transformation Occurred** ---
            if is_multi_metric:
                # Regardless of initial spec, if transformation happened, enforce Value/Metric
                api_chart.y_column = "Value"
                api_chart.color_column = "Metric"
                # x_column remains as specified by LLM (and used for transformation)
                logger.debug(f"Enforced y_column='Value' and color_column='Metric' for chart '{spec_title}' due to successful multi-metric transformation.")
            elif is_pie_transformed:
                 # If pie transformation happened, enforce Category/Value and null color
                 api_chart.x_column = "Category" # Standard name from transformer
                 api_chart.y_column = "Value"    # Standard name from transformer
                 api_chart.color_column = None   # Pie must have null color
                 logger.debug(f"Enforced x='Category', y='Value', color=null for chart '{spec_title}' due to successful pie transformation.")
            elif type_hint == 'pie' and not is_pie_transformed:
                 # If it's a pie chart that wasn't transformed (e.g., data was already correct, or failed validation earlier),
                 # still enforce null color column as per pie chart rules.
                 api_chart.color_column = None
                 logger.debug(f"Enforced color_column=null for pie chart '{spec_title}' (no transformation occurred).")

            # --- End Enforcement ---

            # --- Generic Column Existence Validation & Correction (on potentially transformed columns) ---
            # Check if the columns specified *in the api_chart object* exist in the *columns_to_validate*
            spec_x_col = api_chart.x_column
            spec_y_col = api_chart.y_column
            spec_color_col = api_chart.color_column
            
            # Ensure x_column from spec exists
            if not spec_x_col or spec_x_col not in columns_to_validate:
                 logger.warning(f"Chart spec '{spec_title}': Specified x_column '{spec_x_col}' not found in source/transformed columns: {columns_to_validate}. Skipping.")
                 valid_chart = False
                 failure_reason = f"Specified x_column '{spec_x_col}' not found in data."

            # Ensure y_column from spec exists
            if valid_chart and (not spec_y_col or spec_y_col not in columns_to_validate):
                 logger.warning(f"Chart spec '{spec_title}': Specified y_column '{spec_y_col}' not found in source/transformed columns: {columns_to_validate}. Skipping.")
                 valid_chart = False
                 failure_reason = f"Specified y_column '{spec_y_col}' not found in data."

            # Ensure color_column from spec exists if specified
            if valid_chart and spec_color_col and spec_color_col not in columns_to_validate:
                 logger.warning(f"Chart spec '{spec_title}': Specified color_column '{spec_color_col}' not found in source/transformed columns: {columns_to_validate}. Skipping.")
                 valid_chart = False # If color col is specified but not found, chart is invalid
                 failure_reason = f"Specified color_column '{spec_color_col}' not found in data."
            # --- End Generic Column Validation ---

            # --- Default Labels if Missing ---
            if not api_chart.x_label:
                api_chart.x_label = api_chart.x_column # Default to x-column name
            if not api_chart.y_label:
                # For multi-metric, y_column is 'Value', maybe use that or a generic term?
                # Using the y_column value as default for now.
                api_chart.y_label = api_chart.y_column # Default to y-column name
            # --- End Default Labels ---

            if not valid_chart:
                 # Handle failure from generic column validation
                 filtered_out_info.append({"title": spec_title, "reason": failure_reason})
                 continue
                 
            # --- Type-Specific Validation (operates on potentially transformed data and corrected spec) ---
            type_specific_valid = True
            type_specific_reason = None
            current_rows = api_chart.data.rows # Use data directly from TableData object
            current_columns = api_chart.data.columns # Use columns directly from TableData object

            if type_hint == 'pie':
                 # Pass the corrected api_chart object directly
                 type_specific_valid, type_specific_reason = _validate_pie_chart_spec(api_chart, current_columns, current_rows)
            elif type_hint == 'bar':
                 type_specific_valid, type_specific_reason = _validate_bar_chart_spec(api_chart, current_columns, current_rows)
            elif type_hint == 'line':
                 type_specific_valid, type_specific_reason = _validate_line_chart_spec(api_chart, current_columns, current_rows)

            if not type_specific_valid:
                valid_chart = False
                failure_reason = type_specific_reason or "Type-specific validation failed."
            # --- End Type-Specific Validation ---

            # Add to list if still valid
            if valid_chart:
                visualizations.append(api_chart)
            else:
                logger.warning(f"Skipping chart spec '{spec_title}' due to validation errors (Index: {source_table_idx}): {failure_reason}")
                filtered_out_info.append({"title": spec_title, "reason": failure_reason or "Unknown validation error"})

        except Exception as e:
            logger.error(f"Error processing chart spec: {chart_spec}. Error: {e}", exc_info=True)
            filtered_out_info.append({"title": spec_title, "reason": f"Internal processing error: {e}"})
            continue

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