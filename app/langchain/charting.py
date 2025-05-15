import logging
from typing import List, Optional, Dict, Any, Tuple
from pydantic import BaseModel, Field
import numbers # Import for checking numeric types
import copy # Import copy for deep copying

# Local Imports needed for type hints if we add functions later
from app.schemas.chat import ApiChartSpecification, TableData

logger = logging.getLogger(__name__)

# --- Helper function to check if a column name is time-related ---
def _is_time_column(column_name: Optional[str]) -> bool:
    """Check if column name is likely a time dimension for joining"""
    if not column_name: return False
    time_indicators = ["time", "date", "hour", "day", "month", "year", "timestamp", "period"]
    column_name_lower = column_name.lower()
    return any(indicator in column_name_lower for indicator in time_indicators)

# --- Helper function to find a suitable time column from a list ---
def _find_time_column(columns: List[str]) -> Optional[str]:
    """Find the most likely time column from a list of columns"""
    for col in columns:
        if _is_time_column(col):
            return col
    return None

# --- Instruction Structure included directly in FinalApiResponseStructure ---
class ChartSpecFinalInstruction(BaseModel):
    """Defines the specification for a chart to be rendered by the frontend.
       This structure is generated directly by the LLM within the FinalApiResponseStructure.
    """
    source_table_index: int = Field(description="The 0-based index of the table in the agent's 'tables' state that contains the data for this chart.")
    type_hint: str = Field(description="The suggested chart type for the frontend. MUST be one of: 'bar', 'pie', 'line'.")
    title: str = Field(description="The title for the chart.")
    x_column: Optional[str] = Field(default=None, description="The name of the column from the source table to use for the X-axis or labels. Can be null (or omitted by the LLM) for specific pie chart OR single-row summary bar chart transformations (e.g., from wide-summary data where categories are derived directly from y_columns names or inferred from all numeric columns), in which case the backend processing will assign an appropriate x_column (like 'Category' or 'Metric') to the final ApiChartSpecification if a transformation occurred.")
    y_columns: List[str] = Field(default_factory=list, description="The name(s) of the column(s) from the source table to use for the Y-axis or values. Multiple for multi-series charts. If empty for single-row pie/bar summary charts with x_column=null, all numeric columns may be inferred as categories/series.")
    color_column: Optional[str] = Field(default=None, description="Optional: The name of the column to use for grouping data by color/hue. For multi-series from y_columns (melt transformation), this will be set to 'Metric' by the backend if the transformation occurs.")
    x_label: Optional[str] = Field(default=None, description="Optional: A descriptive label for the X-axis. Defaults to x_column if not provided.")
    y_label: Optional[str] = Field(default=None, description="Optional: A descriptive label for the Y-axis. Defaults to y_column (or 'Value' for multi-series) if not provided.")


# --- Helper function to split a spec into single-source parts (fallback) ---
def _split_spec_into_single_source_parts(
    original_spec: ChartSpecFinalInstruction,
    all_tables: List[Dict[str, Any]],
    filtered_out_info: List[Dict[str, str]]
) -> List[ChartSpecFinalInstruction]:
    """
    If the primary source table for the original_spec is insufficient, this function
    attempts to find a single *alternative* table from all_tables that contains
    the original_spec.x_column and all original_spec.y_columns.

    If such an alternative table is found, it returns a list containing a single
    ChartSpecFinalInstruction, updated to point to the alternative table.

    If the primary table is sufficient, it returns the original spec in a list.

    If neither the primary table is sufficient, nor a single alternative table can be found,
    it returns an empty list, indicating the spec (as a single chart concept) is unfulfillable.
    The granular splitting logic has been removed.
    """
    single_source_specs: List[ChartSpecFinalInstruction] = []
    
    primary_table_data_wrapper = all_tables[original_spec.source_table_index]
    primary_table_data = primary_table_data_wrapper.get('table') if isinstance(primary_table_data_wrapper, dict) and 'table' in primary_table_data_wrapper else primary_table_data_wrapper
    
    if not primary_table_data or not isinstance(primary_table_data, dict) or 'columns' not in primary_table_data:
        logger.error(f"Chart '{original_spec.title}': Primary table data for index {original_spec.source_table_index} is malformed or missing 'columns'. Spec: {original_spec.model_dump_json(exclude_none=True)}")
        filtered_out_info.append({"title": original_spec.title, "reason": f"Primary table {original_spec.source_table_index} data malformed."})
        return []
        
    primary_table_cols_set = set(primary_table_data.get("columns", []))
    is_single_row_primary = len(primary_table_data.get("rows", [])) == 1

    original_x_col = original_spec.x_column
    original_y_cols_set = set(original_spec.y_columns) # Ensure it's a set for subset checks

    # Check if primary table is sufficient
    primary_x_ok = False
    if original_x_col is None and original_spec.type_hint == "pie" and is_single_row_primary:
        primary_x_ok = True # x_column=None is valid for single-row pie, y_cols will be categories
    elif original_x_col is not None and original_x_col in primary_table_cols_set:
        primary_x_ok = True

    primary_y_ok = False
    if not original_y_cols_set: # No y_columns specified
        if original_spec.type_hint == "pie" and original_x_col is None and is_single_row_primary:
            # y_columns can be inferred later from numeric cols for this pie case
            primary_y_ok = True
        else:
            # For other cases, if y_columns are required but not provided, it's an issue (unless LLM intends to omit, caught by other validation)
            # For now, if empty, assume not explicitly "not ok" for the splitter's purpose.
            primary_y_ok = True # Or handle as error if y_columns are strictly required by spec type and non-empty
    elif original_y_cols_set.issubset(primary_table_cols_set):
        primary_y_ok = True

    if primary_x_ok and primary_y_ok:
        logger.info(f"Chart '{original_spec.title}': Primary table {original_spec.source_table_index} is sufficient for x_col '{original_x_col}' and y_cols {original_y_cols_set}.")
        single_source_specs.append(original_spec.model_copy(deep=True))
        return single_source_specs

    # --- Attempt to Redirect to a Single Alternative Table ---
    logger.debug(f"Chart '{original_spec.title}': Primary table {original_spec.source_table_index} is insufficient. Looking for a single alternative redirect table.")
    for i, other_data_wrapper in enumerate(all_tables):
        if i == original_spec.source_table_index:
            continue

        other_table_data = other_data_wrapper.get('table') if isinstance(other_data_wrapper, dict) and 'table' in other_data_wrapper else other_data_wrapper
        if not other_table_data or not isinstance(other_table_data, dict) or 'columns' not in other_table_data:
            # logger.warning(f"Chart '{original_spec.title}': Skipping potential redirect to table {i} as its data is malformed.")
            continue
            
        other_table_cols_set = set(other_table_data.get("columns", []))
        is_single_row_other = len(other_table_data.get("rows", [])) == 1
        
        # Check if this other table contains the original x_column AND all original y_columns
        alt_x_ok = False
        if original_x_col is None and original_spec.type_hint == "pie" and is_single_row_other:
             alt_x_ok = True
        elif original_x_col is not None and original_x_col in other_table_cols_set:
            alt_x_ok = True
            
        alt_y_ok = False
        if not original_y_cols_set: # No y_columns specified
            if original_spec.type_hint == "pie" and original_x_col is None and is_single_row_other:
                alt_y_ok = True # y_cols can be inferred
            else:
                alt_y_ok = True # Or handle as error if y_columns are strictly required
        elif original_y_cols_set.issubset(other_table_cols_set):
            alt_y_ok = True

        if alt_x_ok and alt_y_ok:
            logger.info(f"Chart '{original_spec.title}': Redirecting spec to use single alternative table {i} as it contains x_col '{original_x_col}' and all y_cols {original_y_cols_set}.")
            redirected_spec = original_spec.model_copy(deep=True)
            redirected_spec.source_table_index = i
            single_source_specs.append(redirected_spec)
            return single_source_specs # Return immediately with just the redirected spec

    # If no redirect was possible and primary was not sufficient
    logger.warning(
        f"Chart '{original_spec.title}': Primary table {original_spec.source_table_index} was insufficient, and no single alternative redirect table found. "
        f"Original X ok: {primary_x_ok}, Original Y ok: {primary_y_ok}. Spec will be filtered out. "
        f"Spec details: x_col='{original_x_col}', y_cols={original_y_cols_set}, type='{original_spec.type_hint}'"
    )
    # Add to filtered_out_info to explicitly track why it was removed
    filtered_out_info.append({
        "title": original_spec.title,
        "reason": f"Primary table {original_spec.source_table_index} insufficient and no single alternative table found for all columns (x: '{original_x_col}', y: {original_y_cols_set})."
    })
    return [] # Return empty list if no suitable single source found

# --- Helper function to attempt intelligent data merge for a chart spec ---
def _attempt_intelligent_data_merge(
    spec_to_merge: ChartSpecFinalInstruction,
    all_tables: List[Dict[str, Any]],
    original_spec_title: str # For logging continuity
) -> Tuple[Optional[Dict[str, Any]], Optional[List[str]], Optional[str]]:
    """
    Attempts to merge data for a single chart spec that might reference multiple tables,
    if they share a common x_column.
    
    Returns:
        - merged_table_data (Optional[Dict[str, Any]]): The new wide table if merge successful.
        - merged_y_columns (Optional[List[str]]): List of all y-columns in the merged table.
        - failure_reason (Optional[str]): Reason if merge failed.
    """
    primary_table_idx = spec_to_merge.source_table_index
    # Ensure primary_table_idx is valid before accessing all_tables
    if not (0 <= primary_table_idx < len(all_tables)):
        return None, None, f"Primary table index {primary_table_idx} is out of bounds."
        
    primary_table = all_tables[primary_table_idx]
    primary_cols = primary_table.get("columns", [])
    x_col_for_merge = spec_to_merge.x_column

    if not x_col_for_merge or x_col_for_merge not in primary_cols:
        return None, None, f"x_column '{x_col_for_merge}' not in primary source table {primary_table_idx} columns {primary_cols}."

    y_cols_in_primary_source = [yc for yc in spec_to_merge.y_columns if yc in primary_cols]
    y_cols_needing_other_sources = [yc for yc in spec_to_merge.y_columns if yc not in primary_cols]

    if not y_cols_needing_other_sources: # All y_columns are in the primary source, no merge action needed
        # This case should ideally be handled before calling this function if the goal is strictly merging *external* columns.
        # However, returning this indicates no merge was performed for external columns.
        return None, None, "All y_columns already in primary source, no merge action taken by _attempt_intelligent_data_merge."

    logger.info(f"Chart '{original_spec_title}': Attempting merge for x_column '{x_col_for_merge}'. Primary y_cols: {y_cols_in_primary_source}. Missing y_cols: {y_cols_needing_other_sources}.")

    # Structure to hold data: {y_column_name: (table_index_found_in, {x_value: y_value})}
    y_col_data_map: Dict[str, Tuple[int, Dict[Any, Any]]] = {}
    
    # Add y_cols from primary source
    for yc in y_cols_in_primary_source:
        x_to_y_map = {}
        try:
            x_idx = primary_cols.index(x_col_for_merge)
            y_idx = primary_cols.index(yc)
            for row in primary_table.get("rows", []):
                if len(row) > max(x_idx, y_idx):
                    x_to_y_map[row[x_idx]] = row[y_idx]
            y_col_data_map[yc] = (primary_table_idx, x_to_y_map)
        except (ValueError, IndexError) as e:
            # This indicates an internal inconsistency if yc was deemed in primary_cols but index fails
            return None, None, f"Error processing primary source for y-col '{yc}': {e}"

    all_y_columns_successfully_mapped = list(y_cols_in_primary_source) # Start with y_cols we know are in primary

    for missing_yc in y_cols_needing_other_sources:
        found_this_yc_elsewhere = False
        for other_table_idx, other_table_data in enumerate(all_tables):
            if other_table_idx == primary_table_idx: # Already processed primary source y_cols
                continue
            
            other_table_cols = other_table_data.get("columns", [])
            # CRITICAL: The other table MUST also contain the SAME x_column_for_merge
            if x_col_for_merge in other_table_cols and missing_yc in other_table_cols:
                x_to_y_map = {}
                try:
                    x_idx = other_table_cols.index(x_col_for_merge)
                    y_idx = other_table_cols.index(missing_yc)
                    for row in other_table_data.get("rows", []):
                         if len(row) > max(x_idx, y_idx):
                            x_to_y_map[row[x_idx]] = row[y_idx]
                    y_col_data_map[missing_yc] = (other_table_idx, x_to_y_map)
                    all_y_columns_successfully_mapped.append(missing_yc)
                    found_this_yc_elsewhere = True
                    logger.info(f"Chart '{original_spec_title}': Found missing y_column '{missing_yc}' in table {other_table_idx} with common x_column '{x_col_for_merge}'.")
                    break # Found this missing_yc, move to the next missing_yc
                except (ValueError, IndexError) as e:
                    # Should be rare if columns were checked, but good for robustness
                    logger.warning(f"Chart '{original_spec_title}': Error indexing data in table {other_table_idx} for y-col '{missing_yc}': {e}")
                    # Continue to check other tables for this missing_yc, as this one might be malformed
        
        if not found_this_yc_elsewhere:
            # If any single missing_yc cannot be found with a compatible x_column, the merge for *this spec* fails
            return None, None, f"Could not find a source for y_column '{missing_yc}' with compatible x_column '{x_col_for_merge}'."

    # Consolidate all unique x_values from all y_columns successfully mapped
    all_x_values = set()
    for yc_mapped in all_y_columns_successfully_mapped:
        if yc_mapped in y_col_data_map:
            _, x_to_y_map_for_col = y_col_data_map[yc_mapped]
            all_x_values.update(x_to_y_map_for_col.keys())
    
    if not all_x_values: # No data points to merge
        return None, None, "No common x_values found across mapped y_columns or no data to merge."

    # Sort x_values: basic sort, can be enhanced if specific type knowledge (e.g. datetime) is available
    try:
        # Attempt to convert to datetime if possible for sorting, otherwise use original type
        # For now, a basic sort that handles mixed types somewhat gracefully (numbers before strings)
        sorted_x_values = sorted(list(all_x_values), key=lambda x: (isinstance(x, (str, type(None))), x))
    except TypeError: # Handle unorderable types if mixed in a way Python's default sort can't handle
        sorted_x_values = list(all_x_values) # Fallback to unsorted (or original order of discovery)
        logger.warning(f"Chart '{original_spec_title}': Could not sort x-values due to heterogeneous/unorderable types. Proceeding with potentially unsorted x-values.")

    merged_rows = []
    for x_val in sorted_x_values:
        row = [x_val] # First element is the x_value
        for yc in all_y_columns_successfully_mapped: # Iterate in the order they were confirmed
            # yc should be in y_col_data_map if it's in all_y_columns_successfully_mapped
            _, x_to_y_map_for_this_col = y_col_data_map[yc]
            row.append(x_to_y_map_for_this_col.get(x_val)) # Appends None if x_val not present for this y_col
        merged_rows.append(row)

    final_merged_columns = [x_col_for_merge] + all_y_columns_successfully_mapped
    merged_table_data = {
        "columns": final_merged_columns,
        "rows": merged_rows,
        "metadata": {"merged_from_multi_source": True, "original_spec_title": original_spec_title}
    }
    # Log the source tables for the merged y-columns
    source_table_indices_for_merge = sorted(list(set(idx for idx, _ in y_col_data_map.values())))
    logger.info(f"Chart '{original_spec_title}': Successfully merged data for {len(all_y_columns_successfully_mapped)} y-columns from {len(source_table_indices_for_merge)} source table(s) (indices: {source_table_indices_for_merge}). New table columns: {final_merged_columns}. Rows: {len(merged_rows)}.")
    return merged_table_data, all_y_columns_successfully_mapped, None


# --- Helper function to transform wide summary data for Pie charts ---
def _transform_wide_summary_to_pie_data(
    source_table: Dict[str, Any],
    metrics_to_pivot: List[str], # These are the LLM's y_columns
    descriptive_x_col_name: Optional[str] = None # If LLM provided a valid descriptive x_column for single row
) -> Optional[Dict[str, Any]]:
    """
    Transforms a single-row wide-format table into a long format suitable for a pie chart.
    The `metrics_to_pivot` (from LLM's y_columns) become the categories.
    If `descriptive_x_col_name` is provided, it's used as a prefix for the category label.
    """
    if not source_table or "columns" not in source_table or "rows" not in source_table:
        logger.error("Wide summary to pie: Source table is malformed.")
        return None
    if len(source_table["rows"]) != 1:
        logger.error("Wide summary to pie: Expected a single row in the source table.")
        return None

    original_row = source_table["rows"][0]
    original_cols = source_table["columns"]
    
    if not metrics_to_pivot: # If LLM sent empty y_columns, try to infer from all numeric columns
        logger.info("Wide summary to pie: metrics_to_pivot (y_columns) is empty. Inferring numeric columns.")
        metrics_to_pivot = []
        for i, col_name in enumerate(original_cols):
            if isinstance(original_row[i], numbers.Number):
                metrics_to_pivot.append(col_name)
        if not metrics_to_pivot:
            logger.error("Wide summary to pie: No numeric columns found to pivot for pie chart slices.")
            return None
        logger.info(f"Wide summary to pie: Inferred metrics: {metrics_to_pivot}")


    output_rows = []
    # Determine the name for the first column (categories)
    # If LLM provided a descriptive x_column for the single row, and it exists, it's not used for categories directly,
    # but its value might be used as a prefix if 'descriptive_x_col_name' is passed.
    # For x_column: null cases, or if no descriptive_x_col_name, the categories are just metric names.
    
    category_col_final_name = "Category" # Default name for the new category column
    value_col_final_name = "Value"       # Default name for the new value column

    prefix_for_category = ""
    if descriptive_x_col_name and descriptive_x_col_name in original_cols:
        try:
            desc_x_idx = original_cols.index(descriptive_x_col_name)
            prefix_for_category = str(original_row[desc_x_idx]) + ": "
        except (ValueError, IndexError):
            logger.warning(f"Wide summary to pie: Descriptive x_column '{descriptive_x_col_name}' not found or index error.")
            prefix_for_category = ""


    for metric_name in metrics_to_pivot:
        if metric_name in original_cols:
            try:
                metric_idx = original_cols.index(metric_name)
                value = original_row[metric_idx]
                if isinstance(value, numbers.Number): # Ensure the value is numeric
                    output_rows.append([prefix_for_category + metric_name, value])
                else:
                    logger.warning(f"Wide summary to pie: Column '{metric_name}' is not numeric, skipping for pie slice.")
            except (ValueError, IndexError): # Should be rare if metric_name in original_cols
                logger.warning(f"Wide summary to pie: Column '{metric_name}' not found or index error during pivoting.")
        else:
            logger.warning(f"Wide summary to pie: Metric '{metric_name}' not found in source table columns. Skipping.")
            
    if not output_rows:
        logger.error("Wide summary to pie: No valid data rows could be generated after pivoting.")
        return None

    return {
        "columns": [category_col_final_name, value_col_final_name],
        "rows": output_rows,
        "metadata": {"transformed_for_pie": True}
    }

# --- Helper function to transform wide summary data for Bar charts ---
def _transform_wide_summary_to_bar_data(
    source_table: Dict[str, Any],
    metrics_to_bar: List[str] # ADDED: LLM-specified y_columns that are the actual metrics to bar
) -> Optional[Dict[str, Any]]:
    """
    Transforms a single-row wide-format table where specified metrics_to_bar (from y_columns)
    become categories for a bar chart.
    The LLM's original x_column for the single row (if any) is not directly used in the transformed structure's x-axis,
    as the new x-axis becomes "Metric".
    """
    if not source_table or "columns" not in source_table or "rows" not in source_table:
        logger.error("Wide summary to bar: Source table is malformed.")
        return None
    if len(source_table["rows"]) != 1:
        logger.error("Wide summary to bar: Expected a single row in the source table.")
        return None

    original_row = source_table["rows"][0]
    original_cols = source_table["columns"]

    if not metrics_to_bar: # If LLM sent empty y_columns, try to infer from all numeric columns
        logger.info("Wide summary to bar: metrics_to_bar (y_columns) is empty. Inferring numeric columns.")
        metrics_to_bar = []
        for i, col_name in enumerate(original_cols):
            # Exclude potential original descriptive x-column if it was numeric by mistake.
            # This inference is best if x_column was null from LLM.
            if isinstance(original_row[i], numbers.Number):
                 metrics_to_bar.append(col_name)
        if not metrics_to_bar:
            logger.error("Wide summary to bar: No numeric columns found to use as bars.")
            return None
        logger.info(f"Wide summary to bar: Inferred metrics: {metrics_to_bar}")

    output_rows = []
    metric_col_final_name = "Metric" # New X-axis column name
    value_col_final_name = "Value"   # New Y-axis column name

    for metric_name in metrics_to_bar:
        if metric_name in original_cols:
            try:
                metric_idx = original_cols.index(metric_name)
                value = original_row[metric_idx]
                if isinstance(value, numbers.Number): # Ensure the value is numeric
                    output_rows.append([metric_name, value])
                else:
                    logger.warning(f"Wide summary to bar: Column '{metric_name}' is not numeric, skipping for bar.")
            except (ValueError, IndexError):
                 logger.warning(f"Wide summary to bar: Column '{metric_name}' not found or index error during processing.")
        else:
            logger.warning(f"Wide summary to bar: Metric '{metric_name}' not found in source table columns. Skipping.")
            
    if not output_rows:
        logger.error("Wide summary to bar: No valid data rows could be generated after processing metrics.")
        return None

    return {
        "columns": [metric_col_final_name, value_col_final_name],
        "rows": output_rows,
        "metadata": {"transformed_summary_for_bar": True}
    }


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
    if rows and api_chart.y_column in columns:
        try:
            y_idx = columns.index(api_chart.y_column)
            found_numeric_value = False
            for i in range(min(len(rows), 5)): # Check a few rows
                if len(rows[i]) > y_idx and isinstance(rows[i][y_idx], numbers.Number):
                    found_numeric_value = True
                    break
            if not found_numeric_value:
                first_val_for_log = rows[0][y_idx] if rows and len(rows[0]) > y_idx else "N/A"
                logger.warning(f"Bar chart spec '{api_chart.title}': y_column '{api_chart.y_column}' data (e.g., '{first_val_for_log}') does not appear to contain sufficient numeric values. Invalid.")
                return False, f"Bar chart requires numeric data for the value column ('{api_chart.y_column}')."
        except (ValueError, IndexError):
            logger.warning(f"Bar chart spec '{api_chart.title}': Error accessing y_column '{api_chart.y_column}' for numeric check.")
            return False, "Error accessing y_column for numeric check."
    elif not rows and api_chart.y_column in columns: # Columns exist but no data rows
        logger.warning(f"Bar chart spec '{api_chart.title}': y_column '{api_chart.y_column}' exists, but there are no data rows to plot.")
        return False, f"No data available to plot for y_column '{api_chart.y_column}'."
    return True, None

def _validate_line_chart_spec(api_chart: ApiChartSpecification, columns: List[str], rows: List[List[Any]]) -> Tuple[bool, Optional[str]]:
    """Validates specs specifically for a line chart (basic validation for now).

    Returns:
        (is_valid: bool, failure_reason: Optional[str])
    """
    if rows and api_chart.y_column in columns:
        try:
            y_idx = columns.index(api_chart.y_column)
            found_numeric_value = False
            for i in range(min(len(rows), 5)): # Check a few rows
                if len(rows[i]) > y_idx and isinstance(rows[i][y_idx], numbers.Number):
                    found_numeric_value = True
                    break
            if not found_numeric_value:
                first_val_for_log = rows[0][y_idx] if rows and len(rows[0]) > y_idx else "N/A"
                logger.warning(f"Line chart spec '{api_chart.title}': y_column '{api_chart.y_column}' data (e.g., '{first_val_for_log}') does not appear to contain sufficient numeric values. Invalid.")
                return False, f"Line chart requires numeric data for the value column ('{api_chart.y_column}')."
        except (ValueError, IndexError):
            logger.warning(f"Line chart spec '{api_chart.title}': Error accessing y_column '{api_chart.y_column}' for numeric check.")
            return False, "Error accessing y_column for numeric check."
    elif not rows and api_chart.y_column in columns: # Columns exist but no data rows
        logger.warning(f"Line chart spec '{api_chart.title}': y_column '{api_chart.y_column}' exists, but there are no data rows to plot.")
        return False, f"No data available to plot for y_column '{api_chart.y_column}'."
    return True, None


# --- Main Processing Function (Refactored) ---
def process_and_validate_chart_specs(
    llm_chart_specs: List[ChartSpecFinalInstruction], # Renamed for clarity
    tables_from_state: List[Dict[str, Any]]
) -> Tuple[List[ApiChartSpecification], List[Dict[str, str]]]:
    visualizations = []
    filtered_out_info = [] 
    
    if not llm_chart_specs or not tables_from_state:
        return [], []

    specs_to_process_queue: List[ChartSpecFinalInstruction] = list(llm_chart_specs)
    current_tables_in_state = copy.deepcopy(tables_from_state) # To store original + merged tables

    while specs_to_process_queue:
        current_llm_spec = specs_to_process_queue.pop(0)
        spec = copy.deepcopy(current_llm_spec) 
        spec_title = getattr(spec, "title", "Untitled Chart")
        failure_reason = None
        
        # --- START: Robustness addition for Pie Chart x_column from single-row data ---
        # This block is now more about PRE-VALIDATION and identifying clear transformation intent
        # before it hits the main validation/splitting logic.

        if not (0 <= spec.source_table_index < len(tables_from_state)):
            logger.error(f"Chart '{spec.title}': source_table_index {spec.source_table_index} is out of bounds for {len(tables_from_state)} tables. Skipping.")
            filtered_out_info.append({"title": spec.title, "reason": f"Source table index {spec.source_table_index} out of bounds."})
            continue

        current_source_table_wrapper = tables_from_state[spec.source_table_index]
        current_source_table_data = current_source_table_wrapper.get('table') if isinstance(current_source_table_wrapper, dict) and 'table' in current_source_table_wrapper else current_source_table_wrapper
        
        if not current_source_table_data or not isinstance(current_source_table_data, dict) or \
           "columns" not in current_source_table_data or "rows" not in current_source_table_data:
            logger.error(f"Chart '{spec.title}': Source table data for index {spec.source_table_index} is malformed. Skipping.")
            filtered_out_info.append({"title": spec.title, "reason": f"Source table {spec.source_table_index} data malformed."})
            continue

        is_current_spec_single_row = len(current_source_table_data["rows"]) == 1
        current_source_columns_set = set(current_source_table_data["columns"])

        # --- Transformation for Single-Row Pie where x_column is None (y_columns become categories) ---
        if spec.type_hint == "pie" and spec.x_column is None and is_current_spec_single_row:
            logger.info(f"Chart '{spec.title}': Detected single-row Pie with x_column=None. Attempting wide summary to pie transformation.")
            metrics_for_pie = spec.y_columns
            if not metrics_for_pie: # LLM expects inference from all numeric columns
                logger.info(f"Chart '{spec.title}': y_columns also empty, inferring metrics from all numeric columns for pie.")
                metrics_for_pie = [
                    col for col_idx, col in enumerate(current_source_table_data["columns"])
                    if isinstance(current_source_table_data["rows"][0][col_idx], numbers.Number)
                ]
                if not metrics_for_pie:
                    logger.error(f"Chart '{spec.title}': Single-row Pie with x_column=None and y_columns=[] but no numeric columns found. Skipping.")
                    filtered_out_info.append({"title": spec.title, "reason": "Single-row pie (x_column=None, y_columns=[]) had no numeric data."})
                    continue
                spec.y_columns = metrics_for_pie # Update spec with inferred metrics

            transformed_table_data = _transform_wide_summary_to_pie_data(current_source_table_data, metrics_for_pie)
            if transformed_table_data:
                api_x_column = transformed_table_data["columns"][0] # Should be "Category" or similar
                api_y_columns_for_spec = [transformed_table_data["columns"][1]] # Should be "Value"
                api_color_column = None # Pie charts from this transform have no color column conceptually
                
                api_spec = ApiChartSpecification(
                    title=spec.title,
                    type_hint=spec.type_hint,
                    x_column=api_x_column,
                    y_column=api_y_columns_for_spec[0] if api_y_columns_for_spec else None,
                    color_column=api_color_column,
                    x_label=spec.x_label or api_x_column,
                    y_label=spec.y_label or (api_y_columns_for_spec[0] if api_y_columns_for_spec else "Value"),
                    data=TableData(columns=transformed_table_data["columns"], rows=transformed_table_data["rows"])
                )
                visualizations.append(api_spec)
                logger.info(f"Chart '{spec.title}': Successfully transformed single-row pie with x_column=None.")
                continue # Move to next original_llm_spec
            else:
                logger.error(f"Chart '{spec.title}': Failed to transform single-row pie with x_column=None. Skipping.")
                filtered_out_info.append({"title": spec.title, "reason": "Single-row pie (x_column=None) transformation failed."})
                continue
        
        # --- Transformation for Single-Row Bar where y_columns become categories ---
        # This can happen if x_column is None OR if x_column is a general descriptor and y_columns are metric names.
        # The key is is_current_spec_single_row and non-empty y_columns for this transform.
        if spec.type_hint == "bar" and is_current_spec_single_row and spec.y_columns:
            logger.info(f"Chart '{spec.title}': Detected single-row Bar with y_columns. Attempting wide summary to bar transformation.")
            metrics_for_bar = spec.y_columns # y_columns from LLM are the metrics
            # No inference for y_columns here; if LLM sends empty y_columns for this, it's not this transform.
            
            transformed_table_data = _transform_wide_summary_to_bar_data(current_source_table_data, metrics_for_bar)
            if transformed_table_data:
                api_x_column = transformed_table_data["columns"][0] # Should be "Metric"
                api_y_columns_for_spec = [transformed_table_data["columns"][1]] # Should be "Value"
                api_color_column = None # Bar charts from this transform usually don't use color_column
                                        # unless it was a pre-existing column from original row, which is not standard for this transform.
                
                api_spec = ApiChartSpecification(
                    title=spec.title,
                    type_hint=spec.type_hint,
                    x_column=api_x_column,
                    y_column=api_y_columns_for_spec[0] if api_y_columns_for_spec else None,
                    color_column=api_color_column,
                    x_label=spec.x_label or api_x_column,
                    y_label=spec.y_label or (api_y_columns_for_spec[0] if api_y_columns_for_spec else "Value"),
                    data=TableData(columns=transformed_table_data["columns"], rows=transformed_table_data["rows"])
                )
                visualizations.append(api_spec)
                logger.info(f"Chart '{spec.title}': Successfully transformed single-row bar with y_columns as categories.")
                continue # Move to next original_llm_spec
            else:
                logger.error(f"Chart '{spec.title}': Failed to transform single-row bar with y_columns as categories. Skipping.")
                filtered_out_info.append({"title": spec.title, "reason": "Single-row bar (y_columns as categories) transformation failed."})
                continue
        
        # --- Fallback: Original x_column correction logic for Pie charts if x_column WAS provided but seems wrong for single-row ---
        if spec.type_hint == "pie" and spec.x_column is not None and is_current_spec_single_row:
            # This is the old logic: if x_column was given for a single-row pie, but it's one of the y_columns, it's likely a mistake.
            # Or if it's not a descriptive column.
            # The goal here is to correct a potentially confused LLM, *not* to handle x_column:None (that's above).
            
            # Check if x_column is one of the y_columns or not a suitable descriptive column
            is_x_one_of_y = spec.x_column in spec.y_columns
            
            # Try to find a non-numeric, non-y-column to be the descriptive x_column
            potential_descriptive_x = None
            for col_idx, col_name in enumerate(current_source_table_data["columns"]):
                if col_name not in spec.y_columns and not isinstance(current_source_table_data["rows"][0][col_idx], numbers.Number):
                    potential_descriptive_x = col_name
                    break
            
            if is_x_one_of_y or not potential_descriptive_x:
                 # If x_column is bad (is a y_col, or no descriptive found), and y_columns are present,
                 # this implies the LLM *intended* y_columns to be categories but *mistakenly* set x_column.
                 # This should now ideally be x_column: null from LLM, handled above.
                 # This block becomes a safety net.
                if spec.y_columns:
                    logger.warning(f"Chart '{spec.title}': Single-row pie, x_column ('{spec.x_column}') seems problematic. "
                                   f"LLM should have used x_column:null. Attempting transformation with y_columns as categories.")
                    # Attempt transformation as if x_column was null
                    metrics_for_pie_correction = spec.y_columns
                    transformed_table_data_corr = _transform_wide_summary_to_pie_data(current_source_table_data, metrics_for_pie_correction)
                    if transformed_table_data_corr:
                        # ... (similar ApiSpec creation as above for x_column:null case)
                        api_x_column_corr = transformed_table_data_corr["columns"][0]
                        api_y_col_corr = [transformed_table_data_corr["columns"][1]]
                        api_spec_corr = ApiChartSpecification(
                            title=spec.title, type_hint="pie", x_column=api_x_column_corr, y_column=api_y_col_corr[0], color_column=None,
                            x_label=spec.x_label or api_x_column_corr, y_label=spec.y_label or api_y_col_corr[0],
                            data=TableData(columns=transformed_table_data_corr["columns"], rows=transformed_table_data_corr["rows"])
                        )
                        visualizations.append(api_spec_corr)
                        logger.info(f"Chart '{spec.title}': Corrected single-row pie by transforming with y_columns as categories.")
                        continue
                    else:
                        # Fall through to splitter if correction fails.
                        logger.warning(f"Chart '{spec.title}': Failed to correct single-row pie. Passing to splitter.")
                else: # No y_columns to use for categories
                    logger.warning(f"Chart '{spec.title}': Single-row pie, x_column ('{spec.x_column}') problematic and no y_columns for alternative. Passing to splitter.")


            elif potential_descriptive_x and spec.x_column != potential_descriptive_x:
                logger.info(f"Chart '{spec.title}': Single-row pie, x_column ('{spec.x_column}') was non-ideal. "
                              f"Using identified descriptive column '{potential_descriptive_x}' instead for wide-summary transformation.")
                spec.x_column = potential_descriptive_x # Correct x_column to the better descriptive one
                # Now, this spec (with a corrected x_column and original y_columns) will be processed by _transform_wide_summary_to_pie_data
                # if it gets selected by general validation path later. Or it might be handled by the explicit call below.
                # This path needs to ensure it still triggers the transformation.
                metrics_for_pie_desc = spec.y_columns
                if not metrics_for_pie_desc: # Should not happen if x_column was not null
                     logger.error(f"Chart '{spec.title}': Single-row pie with corrected descriptive x_column but no y_columns. Skipping.")
                     filtered_out_info.append({"title": spec.title, "reason": "Single-row pie, corrected x_col, but no y_cols."})
                     continue

                transformed_table_data_desc = _transform_wide_summary_to_pie_data(current_source_table_data, metrics_for_pie_desc, descriptive_x_col_name=spec.x_column)
                if transformed_table_data_desc:
                    # ... (similar ApiSpec creation)
                    api_x_col_desc = transformed_table_data_desc["columns"][0]
                    api_y_col_desc = [transformed_table_data_desc["columns"][1]]
                    api_spec_desc = ApiChartSpecification(
                        title=spec.title, type_hint="pie", x_column=api_x_col_desc, y_column=api_y_col_desc[0], color_column=None,
                        x_label=spec.x_label or api_x_col_desc, y_label=spec.y_label or api_y_col_desc[0],
                        data=TableData(columns=transformed_table_data_desc["columns"], rows=transformed_table_data_desc["rows"])
                    )
                    visualizations.append(api_spec_desc)
                    logger.info(f"Chart '{spec.title}': Successfully transformed single-row pie with corrected descriptive x_column.")
                    continue
                else:
                    logger.warning(f"Chart '{spec.title}': Failed to transform single-row pie with corrected descriptive x_column. Passing to splitter.")


        # --- Standard Validation & Potential Splitting ---
        # If spec wasn't fully processed by a direct transformation above, it goes through standard validation.
        current_spec_list_for_processing = _split_spec_into_single_source_parts(
            original_spec=spec,
            all_tables=tables_from_state,
            filtered_out_info=filtered_out_info
        )
        if not current_spec_list_for_processing:
            logger.error(f"Chart '{spec.title}': Splitting returned no processable specs. Original was: {current_llm_spec.model_dump_json(exclude_none=True)}")
        else:
            logger.info(f"Chart '{spec.title}': Spec is valid for single source table {spec.source_table_index} or is candidate for direct transformation.")
            current_spec_list_for_processing = [spec]
            
        for current_spec_part in current_spec_list_for_processing:
            try:
                idx = current_spec_part.source_table_index
                if not (0 <= idx < len(tables_from_state)):
                    logger.error(f"Chart '{current_spec_part.title}': Invalid source_table_index {idx} after split/redirect. Skipping.")
                    filtered_out_info.append({"title": current_spec_part.title, "reason": f"Invalid source table index {idx} after split/redirect."})
                    continue

                current_source_table_wrapper = tables_from_state[idx]
                current_source_table_data = current_source_table_wrapper.get('table') if isinstance(current_source_table_wrapper, dict) and 'table' in current_source_table_wrapper else current_source_table_wrapper

                if not isinstance(current_source_table_data, dict) or \
                   "columns" not in current_source_table_data or \
                   "rows" not in current_source_table_data:
                    logger.error(f"Chart '{current_spec_part.title}': Source table {idx} data is malformed. Skipping. Data: {str(current_source_table_data)[:200]}")
                    filtered_out_info.append({"title": current_spec_part.title, "reason": f"Source table {idx} data malformed."})
                    continue
                
                transformed_table_data = None
                api_x_column = current_spec_part.x_column
                api_color_column = current_spec_part.color_column

                is_current_spec_single_row = len(current_source_table_data["rows"]) == 1

                if current_spec_part.type_hint == "pie":
                    if is_current_spec_single_row and current_spec_part.x_column is None:
                        metrics_for_pie = current_spec_part.y_columns
                        if not metrics_for_pie: 
                            metrics_for_pie = [
                                col for col_idx, col in enumerate(current_source_table_data["columns"])
                                if isinstance(current_source_table_data["rows"][0][col_idx], numbers.Number)
                            ]
                            if not metrics_for_pie:
                                logger.warning(f"Chart '{current_spec_part.title}' (Pie, x_col=None, single-row): No y_columns specified and no numeric columns found. Cannot transform.")
                                filtered_out_info.append({"title": current_spec_part.title, "reason": "Pie chart from single row with no x_column needs numeric y_columns, but none found/specified."})
                                continue
                            logger.info(f"Chart '{current_spec_part.title}' (Pie, x_col=None, single-row): y_columns empty. Using inferred numeric columns as metrics: {metrics_for_pie}")
                        
                        transformed_data_for_pie = _transform_wide_summary_to_pie_data(
                            current_source_table_data,
                            metrics_to_pivot=metrics_for_pie
                        )
                        if transformed_data_for_pie:
                            transformed_table_data = transformed_data_for_pie
                            api_x_column = transformed_table_data["columns"][0] 
                            logger.info(f"Chart '{current_spec_part.title}': Transformed single-row wide data to long format for pie. New x_column: '{api_x_column}'.")
                        else:
                            logger.warning(f"Chart '{current_spec_part.title}': Failed to transform wide summary to pie data. Skipping.")
                            filtered_out_info.append({"title": current_spec_part.title, "reason": "Failed to transform wide summary to pie data."})
                            continue
                
                elif current_spec_part.type_hint == "bar" and not current_spec_part.x_column:
                    if is_current_spec_single_row and current_spec_part.y_columns:
                        logger.info(f"Chart '{current_spec_part.title}': Bar chart, x_column=None, single-row. Attempting y_columns as categories transform: {current_spec_part.y_columns}")
                        transformed_data_for_bar = _transform_wide_summary_to_bar_data(
                            current_source_table_data,
                            metrics_to_bar=current_spec_part.y_columns
                        )
                        if transformed_data_for_bar:
                            transformed_table_data = transformed_data_for_bar
                            api_x_column = transformed_table_data["columns"][0]
                            logger.info(f"Chart '{current_spec_part.title}': Transformed single-row data for bar (y_cols as categories). New x_column: '{api_x_column}'.")
                        else:
                            logger.warning(f"Chart '{current_spec_part.title}': Failed to transform wide summary to bar data (y_cols as categories). Skipping.")
                            filtered_out_info.append({"title": current_spec_part.title, "reason": "Failed to transform single-row data for bar (y_cols as categories)."})
                            continue
                    else: # x_column is None, but not a single_row_with_y_columns case for bar summary transform
                        logger.warning(f"Chart '{current_spec_part.title}' (Bar): x_column is None, and not eligible for single-row summary transform. An x_column is required. Skipping.")
                        filtered_out_info.append({"title": current_spec_part.title, "reason": "Bar chart requires an x_column unless it's a transformable single-row summary."})
                        continue
                
                elif current_spec_part.type_hint in ["bar", "line"] and current_spec_part.x_column and len(current_spec_part.y_columns or []) > 1:
                    # Potential multi-series bar/line chart from wide data. Attempt wide-to-long transform.
                    if current_spec_part.x_column in current_source_table_data.get("columns", []):
                        logger.info(f"Chart '{current_spec_part.title}': Attempting wide-to-long transform for multi-series {current_spec_part.type_hint} chart. X='{current_spec_part.x_column}', Ys={current_spec_part.y_columns}")
                        melted_data = _transform_wide_to_long(
                            wide_table=current_source_table_data,
                            id_column_name=current_spec_part.x_column,
                            value_columns_to_melt=current_spec_part.y_columns
                        )
                        # Check if transformation was successful and returned expected new columns
                        if melted_data and \
                           "transformed_from_wide_multi_y" in melted_data.get("metadata", {}) and \
                           melted_data["metadata"]["transformed_from_wide_multi_y"] and \
                           len(melted_data.get("columns", [])) == 3: # Expecting id_col, metric_col, value_col
                            
                            transformed_table_data = melted_data
                            api_x_column = melted_data["columns"][0] # This should be the original x_column
                            # api_y_columns_for_spec will be set later to [melted_data["columns"][2]] ('Value')
                            api_color_column = melted_data["columns"][1] # This should be 'Metric'
                            logger.info(f"Chart '{current_spec_part.title}': Successfully transformed to long format. X='{api_x_column}', Y='{melted_data['columns'][2]}', Color='{api_color_column}'.")
                        else:
                            transform_error = melted_data.get("metadata", {}).get("transform_error", "Unknown error during wide-to-long transform") if melted_data else "Transform function returned None"
                            logger.warning(f"Chart '{current_spec_part.title}': Failed to transform wide to long for multi-series. Reason: {transform_error}. Proceeding with original data structure if possible, but it might not be ideal for ApiChartSpecification which expects a single y_column.")
                            # No change to transformed_table_data, api_x_column, api_color_column. Validation later will catch issues if y_columns has multiple items for ApiChartSpec.
                    else:
                        logger.warning(f"Chart '{current_spec_part.title}': x_column '{current_spec_part.x_column}' for multi-series transform not found in source columns. Skipping transform.")


                final_table_data_for_api_spec = transformed_table_data if transformed_table_data else current_source_table_data
                final_cols_set = set(final_table_data_for_api_spec.get("columns", []))
                
                # CRITICAL: Ensure api_x_column is set. ApiChartSpecification requires x_column.
                if api_x_column is None:
                    logger.error(f"Chart '{current_spec_part.title}': x_column is None and was not derived through transformation. ApiChartSpecification requires an x_column. Skipping.")
                    filtered_out_info.append({"title": current_spec_part.title, "reason": "x_column is required but was not provided or derived."})
                    continue

                if api_x_column not in final_cols_set: # Check for api_x_column presence AGAIN after potential transform
                    logger.error(f"Chart '{current_spec_part.title}': Effective x_column '{api_x_column}' not found in final data columns. Skipping. Final columns: {final_cols_set}")
                    filtered_out_info.append({"title": current_spec_part.title, "reason": f"Effective x_column '{api_x_column}' not in final data columns."})
                    continue
            
                api_y_columns_for_spec = [] # This list will ultimately feed the single 'y_column' in ApiChartSpecification
                
                if transformed_table_data:
                    # If data was transformed (pie summary, bar summary, or wide-to-long)
                    # the 'value' column is usually the second or third column.
                    # For pie/bar summary: ["Category", "Value"] -> api_y_columns_for_spec = ["Value"]
                    # For wide-to-long: [id_col, "Metric", "Value"] -> api_y_columns_for_spec = ["Value"]
                    if len(final_table_data_for_api_spec["columns"]) > 1:
                        # Pie/Bar summary transform outputs: "Category", "Value"
                        if final_table_data_for_api_spec.get("metadata", {}).get("transformed_for_pie") or \
                           final_table_data_for_api_spec.get("metadata", {}).get("transformed_summary_for_bar"):
                            value_col_name = final_table_data_for_api_spec["columns"][1] # Should be "Value"
                            if value_col_name in final_cols_set:
                                api_y_columns_for_spec = [value_col_name]
                            else: # Should not happen if transform is correct
                                logger.error(f"Chart '{current_spec_part.title}': Transformed pie/bar data missing expected 'Value' column. Cols: {final_table_data_for_api_spec['columns']}. Skipping.")
                                filtered_out_info.append({"title": current_spec_part.title, "reason": "Transformed pie/bar data missing Value column."})
                                continue
                        # Wide-to-long transform outputs: id_col, "Metric", "Value"
                        elif final_table_data_for_api_spec.get("metadata", {}).get("transformed_from_wide_multi_y"):
                             if len(final_table_data_for_api_spec["columns"]) == 3:
                                value_col_name = final_table_data_for_api_spec["columns"][2] # Should be "Value"
                                if value_col_name in final_cols_set:
                                    api_y_columns_for_spec = [value_col_name]
                                    # api_color_column should have been set during the transform block
                                else: # Should not happen
                                    logger.error(f"Chart '{current_spec_part.title}': Transformed wide-to-long data missing expected 'Value' column at index 2. Cols: {final_table_data_for_api_spec['columns']}. Skipping.")
                                    filtered_out_info.append({"title": current_spec_part.title, "reason": "Transformed wide-to-long missing Value column."})
                                    continue
                             else: # Should not happen
                                logger.error(f"Chart '{current_spec_part.title}': Transformed wide-to-long data does not have 3 columns. Cols: {final_table_data_for_api_spec['columns']}. Skipping.")
                                filtered_out_info.append({"title": current_spec_part.title, "reason": "Transformed wide-to-long not 3 columns."})
                                continue
                        else:
                            # Fallback if transformed_table_data is set but metadata flags are missing (shouldn't happen)
                            # Or if it's a transform type not yet explicitly handled here for y-col derivation
                            logger.warning(f"Chart '{current_spec_part.title}': Data was transformed, but metadata for y-column derivation is unclear. Defaulting to original y_columns spec. This might be incorrect for ApiChartSpecification.")
                            valid_spec_y_cols = [yc for yc in current_spec_part.y_columns if yc in final_cols_set]
                            api_y_columns_for_spec = valid_spec_y_cols # This might contain multiple columns

                    else: # Transformed data has < 2 columns
                        logger.error(f"Chart '{current_spec_part.title}': Transformed data has less than 2 columns. Skipping. Cols: {final_table_data_for_api_spec['columns']}" )
                        filtered_out_info.append({"title": current_spec_part.title, "reason": "Transformed data too few columns."})
                        continue
                else: # No transformation occurred, use y_columns from LLM spec
                    valid_spec_y_cols = [yc for yc in current_spec_part.y_columns if yc in final_cols_set]
                    if len(valid_spec_y_cols) != len(current_spec_part.y_columns):
                        missing_y_cols_in_final = set(current_spec_part.y_columns) - set(valid_spec_y_cols)
                        logger.warning(f"Chart '{current_spec_part.title}': Some LLM y_columns {list(missing_y_cols_in_final)} not in final data. Using only valid: {valid_spec_y_cols}. Final columns: {final_cols_set}")
                        # Do not filter out yet, let validation handle if no y-cols remain or if multiple y-cols are problematic for ApiChartSpec
                    api_y_columns_for_spec = valid_spec_y_cols

                # At this point, api_y_columns_for_spec contains the candidate(s) for the Y-axis.
                # ApiChartSpecification expects a SINGLE y_column.
                if not api_y_columns_for_spec:
                    logger.error(f"Chart '{current_spec_part.title}': No valid y_column could be determined for ApiChartSpecification. Skipping. Original y_cols: {current_spec_part.y_columns}, Final cols: {final_cols_set}")
                    filtered_out_info.append({"title": current_spec_part.title, "reason": "No valid y_column for API spec."})
                    continue
                
                final_y_column_for_api_spec: Optional[str] = None
                # Case 1: Successfully transformed to have a 'Value' y-column and 'Metric' color_column (e.g., by _transform_wide_to_long)
                # or successfully transformed for single-row pie/bar summary (which also results in a 'Value' y-column).
                # In these cases, api_y_columns_for_spec should correctly be ['Value'].
                if len(api_y_columns_for_spec) == 1 and \
                   (api_color_column == "Metric" or \
                    final_table_data_for_api_spec.get("metadata", {}).get("transformed_for_pie") or \
                    final_table_data_for_api_spec.get("metadata", {}).get("transformed_summary_for_bar")):
                    final_y_column_for_api_spec = api_y_columns_for_spec[0]
                    # api_color_column would already be 'Metric' if from _transform_wide_to_long
                    # or should be None/original if from pie/bar summary transforms (frontend handles colors)

                # Case 2: A single, non-transformed y_column was provided by LLM and is valid.
                elif len(api_y_columns_for_spec) == 1 and not transformed_table_data:
                    final_y_column_for_api_spec = api_y_columns_for_spec[0]
                    # api_color_column here would be whatever the LLM provided (if anything) for a single series chart.
                
                # Case 3: Multiple y_columns remain, and it was NOT a successful multi-series transformation. This is an invalid state.
                elif len(api_y_columns_for_spec) > 1 and not api_color_column == "Metric": # and not (transformed_for_pie or transformed_summary_for_bar)
                    logger.error(
                        f"Chart '{current_spec_part.title}': Multiple y_columns ({api_y_columns_for_spec}) remain for ApiChartSpecification "
                        f"which expects a single y_column, and it was not transformed into a standard multi-series format "
                        f"(api_color_column is '{api_color_column}', not 'Metric'). This indicates an issue with the "
                        f"LLM spec or an incomplete transformation for the chart type '{current_spec_part.type_hint}'. Skipping."
                    )
                    filtered_out_info.append({
                        "title": current_spec_part.title,
                        "reason": f"Invalid multi-y-column state for chart type '{current_spec_part.type_hint}' (columns: {api_y_columns_for_spec}). Expected single y-column or melt to multi-series."
                    })
                    continue
                
                # Case 4: Fallback/Error - Should ideally be covered by above.
                else: # Covers len(api_y_columns_for_spec) == 1 but state is inconsistent with transformations
                    logger.error(
                        f"Chart '{current_spec_part.title}': Ambiguous or invalid state for determining final_y_column. "
                        f"api_y_columns_for_spec: {api_y_columns_for_spec}, api_color_column: {api_color_column}, "
                        f"transformed metadata: {final_table_data_for_api_spec.get('metadata', {})}. Skipping."
                    )
                    filtered_out_info.append({
                        "title": current_spec_part.title,
                        "reason": "Ambiguous y-column state for API spec."
                    })
                    continue
                
                if final_y_column_for_api_spec is None: # Should be caught by the continue statements in cases above
                     logger.error(f"Chart '{current_spec_part.title}': final_y_column_for_api_spec could not be determined after conditional checks. This is unexpected. Skipping.")
                     filtered_out_info.append({"title": current_spec_part.title, "reason": "Internal error: final y_column undetermined."})
                     continue

                api_spec = ApiChartSpecification(
                    source_table_index=idx, 
                    type_hint=current_spec_part.type_hint,
                    title=current_spec_part.title,
                    x_column=api_x_column, # Already checked for None
                    y_column=final_y_column_for_api_spec, # Use the determined single y-column
                    color_column=api_color_column, 
                    x_label=current_spec_part.x_label or api_x_column,
                    y_label=current_spec_part.y_label or final_y_column_for_api_spec, # Default y_label to the final y_column
                    data=TableData(**final_table_data_for_api_spec) # Ensure data is cast to TableData model
                )
                
                is_valid_api_spec = False
                validation_msg = "Unknown validation error."
                current_api_spec_cols = api_spec.data.columns if api_spec.data else []
                current_api_spec_rows = api_spec.data.rows if api_spec.data else []

                if not current_api_spec_rows:
                    validation_msg = "No data rows available for the chart after processing."
                    logger.warning(f"Chart '{api_spec.title}': {validation_msg}")
                else:
                    if api_spec.type_hint == "pie":
                        is_valid_api_spec, validation_msg = _validate_pie_chart_spec(api_spec, current_api_spec_cols, current_api_spec_rows)
                    elif api_spec.type_hint == "bar":
                        is_valid_api_spec, validation_msg = _validate_bar_chart_spec(api_spec, current_api_spec_cols, current_api_spec_rows)
                    elif api_spec.type_hint == "line":
                        is_valid_api_spec, validation_msg = _validate_line_chart_spec(api_spec, current_api_spec_cols, current_api_spec_rows)
                    else:
                        validation_msg = f"Unsupported chart type_hint: {api_spec.type_hint}"
                        logger.warning(f"Chart '{api_spec.title}': {validation_msg}")
                
                if is_valid_api_spec:
                    visualizations.append(api_spec)
                    logger.info(f"Chart '{api_spec.title}': Successfully validated. Type: {api_spec.type_hint}, X: '{api_spec.x_column}', Y: {api_spec.y_column}")
                else:
                    logger.error(f"Chart '{api_spec.title}': Failed final validation. Reason: {validation_msg}. Spec: {api_spec.model_dump_json(exclude={'data'})}")
                    filtered_out_info.append({"title": api_spec.title, "reason": f"Failed final validation: {validation_msg}"})

            except Exception as e:
                logger.error(f"Chart '{current_spec_part.title}': Unexpected error processing spec part: {e}", exc_info=True)
                filtered_out_info.append({"title": current_spec_part.title, "reason": f"Unexpected error: {str(e)}"})
                continue

    if not visualizations and llm_chart_specs:
        logger.warning(f"No chart specifications validated from {len(llm_chart_specs)} LLM spec(s). Filtered info: {filtered_out_info}")
    
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
    logger.debug(f"[_transform_wide_to_long] ID col: {id_column_name}, Y-cols to melt: {value_columns_to_melt} from table cols: {wide_table.get('columns')}")
    metric_col_name = "Metric" 
    value_col_name = "Value"   

    original_rows = wide_table.get('rows', [])
    original_columns = wide_table.get('columns', [])
    original_metadata = wide_table.get('metadata', {})
    long_rows = []

    if not original_rows or not original_columns:
        logger.warning("[_transform_wide_to_long] Input table has no rows or columns. Cannot transform.")
        return {"columns": [id_column_name, metric_col_name, value_col_name], "rows": [], "metadata": {**original_metadata, "transformed_from_wide_multi_y": False, "transform_error": "No rows or columns in input table"}}

    try:
        id_col_index = original_columns.index(id_column_name)
    except ValueError:
        logger.error(f"[_transform_wide_to_long] ID column '{id_column_name}' not found in wide table columns: {original_columns}.")
        return {"columns": original_columns, "rows": original_rows, "metadata": {**original_metadata, "transformed_from_wide_multi_y": False, "transform_error": f"ID column '{id_column_name}' not found in source columns {original_columns}"}}

    actual_value_cols_for_melt_map = {} # Stores {y_col_name: index_in_original_columns}
    for y_col_name in value_columns_to_melt:
        try:
            actual_value_cols_for_melt_map[y_col_name] = original_columns.index(y_col_name)
        except ValueError:
            # This y_col_name might not be in original_columns if the table was already a result of a merge 
            # that failed to find this specific y_col, or if LLM spec was wrong.
            logger.warning(f"[_transform_wide_to_long] Y-column '{y_col_name}' (specified for melt) not found in current wide table columns: {original_columns}. It will be skipped for this melt operation.")
            # Continue, as other y_columns in value_columns_to_melt might still be valid and present.
    
    if not actual_value_cols_for_melt_map: # If NO y_columns specified for melt were actually found in the table
         logger.error(f"[_transform_wide_to_long] No valid y-columns from the requested list ({value_columns_to_melt}) were found in the provided table columns ({original_columns}) to perform melt for ID column '{id_column_name}'.")
         # Return a structure indicating failure but with clear metadata
         return {"columns": original_columns, "rows": original_rows, "metadata": {**original_metadata, "transformed_from_wide_multi_y": False, "transform_error": f"No valid Y-columns from {value_columns_to_melt} found in {original_columns} to melt."}}

    for wide_row in original_rows:
        if len(wide_row) != len(original_columns): 
            logger.warning(f"[_transform_wide_to_long] Skipping row with mismatching column count ({len(wide_row)}) vs expected ({len(original_columns)}): {str(wide_row)[:100]}...")
            continue
            
        id_value = wide_row[id_col_index]
        for metric_name, val_col_idx in actual_value_cols_for_melt_map.items(): # Only iterate over y_columns that ARE in this table
            value = wide_row[val_col_idx]
            numeric_value = None # Default to None if conversion fails or value is None
            if value is not None:
                if isinstance(value, numbers.Number):
                    numeric_value = value
                elif isinstance(value, str):
                    try:
                        numeric_value = float(value)
                    except (ValueError, TypeError):
                        logger.debug(f"[_transform_wide_to_long] Could not convert string value '{value}' for metric '{metric_name}' to float. Using None.")
                else: # Other non-numeric, non-string types
                    logger.debug(f"[_transform_wide_to_long] Value '{value}' (type: {type(value)}) for metric '{metric_name}' is not numeric or string representable as float. Using None.")
            
            long_rows.append([id_value, metric_name, numeric_value])


    long_columns = [id_column_name, metric_col_name, value_col_name]
    logger.info(f"[_transform_wide_to_long] Transformation complete. Produced {len(long_rows)} long format rows. Melted y_cols: {list(actual_value_cols_for_melt_map.keys())}")
    
    new_metadata = {**original_metadata, 
                    "transformed_from_wide_multi_y": True, 
                    "original_columns_before_melt": list(original_columns), # Record columns of the table fed into this melt
                    "melted_y_columns": list(actual_value_cols_for_melt_map.keys())}
    if "transformed_from_wide" in new_metadata: del new_metadata["transformed_from_wide"] # Cleanup old key if present for clarity

    return {"columns": long_columns, "rows": long_rows, "metadata": new_metadata}

# Function to process chart specs will be added here later 