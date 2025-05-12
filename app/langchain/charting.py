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
    type_hint: str = Field(description="The suggested chart type for the frontend (e.g., 'bar', 'pie', 'line', 'scatter').")
    title: str = Field(description="The title for the chart.")
    x_column: str = Field(description="The name of the column from the source table to use for the X-axis or labels.")
    y_columns: List[str] = Field(default_factory=list, description="The name(s) of the column(s) from the source table to use for the Y-axis or values. Multiple for multi-series charts.")
    color_column: Optional[str] = Field(default=None, description="Optional: The name of the column to use for grouping data by color/hue. For multi-series from y_columns, this will be set to 'Metric'.")
    x_label: Optional[str] = Field(default=None, description="Optional: A descriptive label for the X-axis. Defaults to x_column if not provided.")
    y_label: Optional[str] = Field(default=None, description="Optional: A descriptive label for the Y-axis. Defaults to y_column (or 'Value' for multi-series) if not provided.")


# --- Helper function to split a spec into single-source parts (fallback) ---
def _split_spec_into_single_source_parts(
    original_spec: ChartSpecFinalInstruction,
    all_tables: List[Dict[str, Any]],
    filtered_out_info: List[Dict[str, str]]
) -> List[ChartSpecFinalInstruction]:
    """
    Fallback: Takes an original spec and splits it into multiple, simpler specs,
    each valid for a single source table and typically a single y-column.
    Tries to use the original x-column where possible, or a fallback time column.
    """
    split_specs = []
    original_title = original_spec.title
    x_col_from_original = original_spec.x_column
    original_y_cols = original_spec.y_columns

    if not original_y_cols:
        logger.warning(f"Chart '{original_title}': No y_columns in spec to split.")
        filtered_out_info.append({"title": original_title, "reason": "No y_columns to process for splitting."})
        return []

    for y_col_candidate in original_y_cols:
        found_source_for_this_y_col = False
        
        # Try primary table first if x_col is there, and y_col is there
        primary_table_idx = original_spec.source_table_index
        if 0 <= primary_table_idx < len(all_tables):
            primary_table = all_tables[primary_table_idx]
            primary_cols = primary_table.get("columns", [])
            if x_col_from_original in primary_cols and y_col_candidate in primary_cols:
                new_spec = copy.deepcopy(original_spec) # Start with a copy of original
                new_spec.source_table_index = primary_table_idx # Ensure correct index
                new_spec.x_column = x_col_from_original # Ensure correct x_column
                new_spec.y_columns = [y_col_candidate] # Single y_column
                # Modify title only if original had multiple y_columns, to differentiate
                new_spec.title = f"{original_title} - {y_col_candidate}" if len(original_y_cols) > 1 else original_title
                split_specs.append(new_spec)
                logger.info(f"Chart '{original_title}': Split off part for y_column '{y_col_candidate}' from its original primary table {primary_table_idx}.")
                found_source_for_this_y_col = True
                # Continue to the next y_col_candidate in the original_spec.y_columns list
                continue 
        
        # If not found in primary (or x_col was not in primary, or y_col not in primary for that x_col)
        # search other tables for this y_col_candidate
        if not found_source_for_this_y_col:
            for other_table_idx, other_table_data in enumerate(all_tables):
                # No need to explicitly skip primary_table_idx again, as the `continue` above handles successful finds in primary.
                # If we are here, it means it wasn't found in primary with the original x_col.
                
                other_table_cols = other_table_data.get("columns", [])
                
                # Attempt 1: Use original x-column if present in this other table
                if x_col_from_original in other_table_cols and y_col_candidate in other_table_cols:
                    new_spec = copy.deepcopy(original_spec)
                    new_spec.source_table_index = other_table_idx
                    new_spec.x_column = x_col_from_original
                    new_spec.y_columns = [y_col_candidate]
                    new_spec.title = f"{original_title} - {y_col_candidate} (from table {other_table_idx})"
                    split_specs.append(new_spec)
                    logger.info(f"Chart '{original_title}': Split off part for y_column '{y_col_candidate}' from other table {other_table_idx} using original x_column '{x_col_from_original}'.")
                    found_source_for_this_y_col = True
                    break # Found source for this y_col_candidate, move to next y_col_candidate
                else: 
                    # Attempt 2: If original x-col not found, try a fallback time column in this other table
                    fallback_x_col = _find_time_column(other_table_cols)
                    if fallback_x_col and y_col_candidate in other_table_cols:
                        new_spec = copy.deepcopy(original_spec)
                        new_spec.source_table_index = other_table_idx
                        new_spec.x_column = fallback_x_col # Use the identified fallback x_column
                        new_spec.y_columns = [y_col_candidate]
                        new_spec.title = f"{original_title} - {y_col_candidate} (from table {other_table_idx}, x-axis: {fallback_x_col})"
                        split_specs.append(new_spec)
                        logger.info(f"Chart '{original_title}': Split off part for y_column '{y_col_candidate}' from table {other_table_idx} using fallback x_column '{fallback_x_col}'.")
                        found_source_for_this_y_col = True
                        break # Found source for this y_col_candidate, move to next y_col_candidate
        
        if not found_source_for_this_y_col:
            logger.warning(f"Chart '{original_title}': Could not find a valid source or x_column for y_column '{y_col_candidate}' during splitting. This metric will be omitted from split results.")
            # Optionally, add to filtered_out_info here if desired for per-y-column failure reasons
            # For now, the overall spec failure will be logged by the calling function if no parts are salvaged.

    if not split_specs and original_y_cols: # If the original spec had y_columns but none could be salvaged
        logger.warning(f"Chart '{original_title}': Original spec with y_columns {original_y_cols} could not be split into any valid parts.")
        # The calling function (`process_and_validate_chart_specs`) will handle adding to filtered_out_info
        # if the original spec ultimately yields no processable parts.

    return split_specs

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
    logger.info(f"Chart '{original_spec_title}': Successfully merged data for {len(all_y_columns_successfully_mapped)} y-columns. New table columns: {final_merged_columns}. Rows: {len(merged_rows)}.")
    return merged_table_data, all_y_columns_successfully_mapped, None


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

# --- Helper function to transform wide summary data for Bar charts ---
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
        
        # Basic spec structure validation
        if not (0 <= spec.source_table_index < len(current_tables_in_state)):
            failure_reason = f"Invalid source_table_index {spec.source_table_index} for {len(current_tables_in_state)} available tables."
        elif not spec.x_column or not spec.y_columns:
            failure_reason = "Missing x_column or y_columns."
        
        if failure_reason:
            logger.warning(f"Chart '{spec_title}': Pre-check failed: {failure_reason}")
            filtered_out_info.append({"title": spec_title, "reason": failure_reason})
            continue

        primary_table_data_for_spec = current_tables_in_state[spec.source_table_index]
        primary_cols_for_spec = primary_table_data_for_spec.get("columns", [])
        
        if not primary_cols_for_spec:
            failure_reason = f"Primary source table {spec.source_table_index} (for spec '{spec_title}') has no columns."
            logger.warning(f"Chart '{spec_title}': {failure_reason}")
            filtered_out_info.append({"title": spec_title, "reason": failure_reason})
            continue

        y_cols_in_primary = [yc for yc in spec.y_columns if yc in primary_cols_for_spec]
        y_cols_not_in_primary = [yc for yc in spec.y_columns if yc not in primary_cols_for_spec]
        x_col_in_primary = spec.x_column in primary_cols_for_spec

        final_spec_to_process_downstream = spec 
        source_table_for_downstream_processing = primary_table_data_for_spec
        ready_for_standard_processing = False

        if x_col_in_primary and not y_cols_not_in_primary:
            logger.info(f"Chart '{spec_title}': Valid single-source spec or all y_cols in primary. source_idx: {spec.source_table_index}")
            ready_for_standard_processing = True
        elif x_col_in_primary and y_cols_not_in_primary:
            logger.info(f"Chart '{spec_title}': x_col '{spec.x_column}' in primary table {spec.source_table_index}, but missing y_cols: {y_cols_not_in_primary}. Attempting merge.")
            merged_table_data, merged_y_cols, merge_fail_reason = _attempt_intelligent_data_merge(spec, current_tables_in_state, spec_title)
            if merged_table_data and merged_y_cols:
                current_tables_in_state.append(merged_table_data)
                final_spec_to_process_downstream = copy.deepcopy(spec) 
                final_spec_to_process_downstream.source_table_index = len(current_tables_in_state) - 1
                final_spec_to_process_downstream.y_columns = merged_y_cols 
                source_table_for_downstream_processing = merged_table_data
                logger.info(f"Chart '{spec_title}': Merge successful. Updated spec to use new table at index {final_spec_to_process_downstream.source_table_index}.")
                ready_for_standard_processing = True
            else:
                logger.warning(f"Chart '{spec_title}': Merge failed (Reason: {merge_fail_reason}). Splitting spec.")
                split_specs = _split_spec_into_single_source_parts(spec, current_tables_in_state, filtered_out_info)
                if split_specs:
                    specs_to_process_queue.extend(split_specs) 
                    logger.info(f"Chart '{spec_title}': Added {len(split_specs)} split parts to processing queue.")
                else:
                    # If splitting also failed to produce anything, the original spec is truly unprocessable.
                    # _split_spec_into_single_source_parts might add to filtered_out_info if it completely fails for all y_cols.
                    if not any(f_info['title'] == spec_title and "split" in f_info['reason'] for f_info in filtered_out_info):
                         filtered_out_info.append({"title": spec_title, "reason": merge_fail_reason or "Merge failed and spec could not be split."})
                continue 
        elif not x_col_in_primary:
            logger.warning(f"Chart '{spec_title}': x_column '{spec.x_column}' not found in primary table {spec.source_table_index}. Splitting spec.")
            split_specs = _split_spec_into_single_source_parts(spec, current_tables_in_state, filtered_out_info)
            if split_specs:
                specs_to_process_queue.extend(split_specs)
                logger.info(f"Chart '{spec_title}': Added {len(split_specs)} split parts to processing queue due to missing x_col in primary.")
            else:
                 if not any(f_info['title'] == spec_title and "split" in f_info['reason'] for f_info in filtered_out_info):
                    filtered_out_info.append({"title": spec_title, "reason": f"x_column '{spec.x_column}' not in primary table and spec could not be split."})
            continue
        else: 
            logger.error(f"Chart '{spec_title}': Unhandled spec condition during merge/split logic. Filtering out.")
            filtered_out_info.append({"title": spec_title, "reason": "Unhandled internal chart processing condition."})
            continue
            
        if not ready_for_standard_processing:
            if not any(f_info['title'] == spec_title for f_info in filtered_out_info):
                 filtered_out_info.append({"title": spec_title, "reason": "Spec was not suitable for direct processing, merge, or split after initial analysis."})
            continue

        try:
            current_spec_instance = final_spec_to_process_downstream
            data_for_chart_obj = source_table_for_downstream_processing 
            
            spec_title_for_api = current_spec_instance.title
            type_hint = getattr(current_spec_instance, "type_hint", "bar").lower()
            current_x_col = current_spec_instance.x_column
            current_y_cols = current_spec_instance.y_columns 
            
            cols_in_data_for_chart_obj = data_for_chart_obj.get("columns", [])
            rows_in_data_for_chart_obj = data_for_chart_obj.get("rows", [])

            if current_x_col not in cols_in_data_for_chart_obj:
                failure_reason = f"Internal Error: x_column '{current_x_col}' for spec '{spec_title_for_api}' not in its designated source table columns {cols_in_data_for_chart_obj}."
                logger.error(failure_reason)
                filtered_out_info.append({"title": spec_title_for_api, "reason": failure_reason})
                continue
            
            valid_y_cols_for_transform = [yc for yc in current_y_cols if yc in cols_in_data_for_chart_obj]
            if not valid_y_cols_for_transform:
                failure_reason = f"Internal Error: No y_columns from spec '{spec_title_for_api}' ({current_y_cols}) found in its designated source table columns {cols_in_data_for_chart_obj}."
                logger.error(failure_reason)
                filtered_out_info.append({"title": spec_title_for_api, "reason": failure_reason})
                continue
            current_y_cols = valid_y_cols_for_transform # Use only the confirmed valid y_columns

            is_multi_metric_transformed = False 
            is_pie_transformed = False
            is_summary_transformed = False
            post_transform_failure_reason = None

            if type_hint == 'bar' and len(rows_in_data_for_chart_obj) == 1 and len(cols_in_data_for_chart_obj) >= 1:
                transformed_s_bar = _transform_wide_summary_to_bar_data(data_for_chart_obj)
                if transformed_s_bar: 
                    data_for_chart_obj = transformed_s_bar
                    cols_in_data_for_chart_obj = data_for_chart_obj.get("columns", []) 
                    is_summary_transformed = True
            elif type_hint == 'pie' and len(rows_in_data_for_chart_obj) == 1 and len(cols_in_data_for_chart_obj) >= 2:
                transformed_s_pie = _transform_wide_summary_to_pie_data(data_for_chart_obj)
                if transformed_s_pie: 
                    data_for_chart_obj = transformed_s_pie
                    cols_in_data_for_chart_obj = data_for_chart_obj.get("columns", []) 
                    is_pie_transformed = True
            elif type_hint in ['bar', 'line'] and len(current_y_cols) > 1:
                # This must use current_y_cols which are confirmed to be in cols_in_data_for_chart_obj
                transformed_long = _transform_wide_to_long(data_for_chart_obj, current_x_col, current_y_cols)
                if transformed_long.get("metadata", {}).get("transformed_from_wide_multi_y"):
                    data_for_chart_obj = transformed_long
                    cols_in_data_for_chart_obj = data_for_chart_obj.get("columns", []) 
                    is_multi_metric_transformed = True
                else:
                    post_transform_failure_reason = transformed_long.get("metadata",{}).get("transform_error", "Melt transform failed")

            if post_transform_failure_reason:
                logger.warning(f"Chart '{spec_title_for_api}' failed data transformation: {post_transform_failure_reason}")
                filtered_out_info.append({"title": spec_title_for_api, "reason": post_transform_failure_reason})
                continue
            
            final_y_col_name_for_api = "Value" if is_multi_metric_transformed else (current_y_cols[0] if current_y_cols else "")
            final_color_col_name_for_api = "Metric" if is_multi_metric_transformed else getattr(current_spec_instance, 'color_column', None)

            api_chart_obj = ApiChartSpecification(
                type_hint=type_hint, title=spec_title_for_api,
                x_column=current_x_col,
                y_column=final_y_col_name_for_api,
                color_column=final_color_col_name_for_api,
                x_label=getattr(current_spec_instance, "x_label", None),
                y_label=getattr(current_spec_instance, "y_label", None),
                data=TableData(**copy.deepcopy(data_for_chart_obj))
            )

            if is_summary_transformed: 
                api_chart_obj.x_column, api_chart_obj.y_column, api_chart_obj.color_column = "Metric", "Value", None
                if not api_chart_obj.y_label: api_chart_obj.y_label = "Value"
            elif is_pie_transformed: 
                api_chart_obj.x_column, api_chart_obj.y_column, api_chart_obj.color_column = "Category", "Value", None
                if not api_chart_obj.y_label: api_chart_obj.y_label = "Value"
            elif is_multi_metric_transformed:
                if not api_chart_obj.y_label: api_chart_obj.y_label = "Value"
            elif type_hint == 'pie': 
                api_chart_obj.color_column = None

            cols_in_api_chart_data = api_chart_obj.data.columns 
            rows_in_api_chart_data = api_chart_obj.data.rows
            final_validation_passed, final_validation_reason = True, ""

            if not api_chart_obj.x_column or api_chart_obj.x_column not in cols_in_api_chart_data:
                final_validation_passed, final_validation_reason = False, f"Final x_column '{api_chart_obj.x_column}' invalid."
            if final_validation_passed and (not api_chart_obj.y_column or api_chart_obj.y_column not in cols_in_api_chart_data):
                final_validation_passed, final_validation_reason = False, f"Final y_column '{api_chart_obj.y_column}' invalid."
            if final_validation_passed and api_chart_obj.color_column and api_chart_obj.color_column not in cols_in_api_chart_data:
                final_validation_passed, final_validation_reason = False, f"Final color_column '{api_chart_obj.color_column}' invalid."

            if final_validation_passed:
                if not api_chart_obj.x_label: api_chart_obj.x_label = api_chart_obj.x_column
                if not api_chart_obj.y_label: api_chart_obj.y_label = api_chart_obj.y_column
                
                type_specific_valid, type_specific_reason = True, None
                if type_hint == 'pie': type_specific_valid, type_specific_reason = _validate_pie_chart_spec(api_chart_obj, cols_in_api_chart_data, rows_in_api_chart_data)
                elif type_hint == 'bar': type_specific_valid, type_specific_reason = _validate_bar_chart_spec(api_chart_obj, cols_in_api_chart_data, rows_in_api_chart_data)
                elif type_hint == 'line': type_specific_valid, type_specific_reason = _validate_line_chart_spec(api_chart_obj, cols_in_api_chart_data, rows_in_api_chart_data)
                
                if not type_specific_valid:
                    final_validation_passed, final_validation_reason = False, type_specific_reason or "Type-specific validation failed."
            
            if final_validation_passed:
                visualizations.append(api_chart_obj)
                logger.info(f"Successfully processed and validated chart: '{api_chart_obj.title}'")
            else:
                logger.warning(f"Chart '{api_chart_obj.title}' failed final validation: {final_validation_reason}")
                filtered_out_info.append({"title": api_chart_obj.title, "reason": final_validation_reason})
        
        except Exception as e:
            logger.error(f"Unhandled error processing chart spec '{spec_title}' (original LLM spec: {current_llm_spec}): {e}", exc_info=True)
            filtered_out_info.append({"title": spec_title, "reason": f"Internal error during final processing: {str(e)}"})
            
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