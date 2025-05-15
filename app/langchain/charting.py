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
    x_column: Optional[str] = Field(default=None, description="The name of the column from the source table to use for the X-axis or labels. Can be null (or omitted by the LLM) for specific pie chart transformations (e.g., from wide-summary data where categories are derived directly from y_columns names), in which case the backend processing will assign an appropriate x_column (like 'Category') to the final ApiChartSpecification if a transformation occurred.")
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
    Splits a chart spec if its x_column or y_columns are not all in the primary source table.
    It tries to find other tables that can source parts of the spec.
    It also attempts to intelligently redirect the entire spec to a single alternative table if possible.
    """
    single_source_specs: List[ChartSpecFinalInstruction] = []
    primary_table_wrapper = all_tables[original_spec.source_table_index]
    primary_table_data = primary_table_wrapper.get('table') if isinstance(primary_table_wrapper, dict) and 'table' in primary_table_wrapper else primary_table_wrapper
    
    if not primary_table_data or 'columns' not in primary_table_data:
        logger.error(f"Chart '{original_spec.title}': Primary table data for index {original_spec.source_table_index} is malformed or missing 'columns'. Spec: {original_spec.model_dump_json()}")
        filtered_out_info.append({"title": original_spec.title, "reason": f"Primary table {original_spec.source_table_index} data malformed."})
        return []
        
    primary_table_cols = primary_table_data.get("columns", [])

    original_x_col = original_spec.x_column
    original_y_cols = list(original_spec.y_columns) # Ensure it's a list for consistent processing
    original_y_cols_set = set(original_y_cols)

    x_column_in_primary_table = original_x_col in primary_table_cols
    y_columns_in_primary_table = [col for col in original_y_cols if col in primary_table_cols]
    y_columns_not_in_primary_table = [col for col in original_y_cols if col not in primary_table_cols]
    all_original_y_cols_in_primary = original_y_cols_set.issubset(set(primary_table_cols))

    # If primary table already contains x_column and all y_columns, no splitting or redirecting needed for source.
    if x_column_in_primary_table and all_original_y_cols_in_primary:
        logger.info(f"Chart '{original_spec.title}': Valid single-source spec. All x/y_cols in primary table {original_spec.source_table_index}.")
        single_source_specs.append(original_spec.model_copy(deep=True))
        return single_source_specs

    # --- START: New "redirect" logic ---
    # If the primary table is insufficient, try to find a single *other* table that contains all required columns.
    logger.debug(f"Chart '{original_spec.title}': Primary table {original_spec.source_table_index} is insufficient (x_in_primary: {x_column_in_primary_table}, all_y_in_primary: {all_original_y_cols_in_primary}). Looking for redirect.")
    for i, other_data_wrapper in enumerate(all_tables):
        if i == original_spec.source_table_index:  # Skip the primary table itself
            continue

        other_table_data = other_data_wrapper.get('table') if isinstance(other_data_wrapper, dict) and 'table' in other_data_wrapper else other_data_wrapper
        if not other_table_data or 'columns' not in other_table_data:
            logger.warning(f"Chart '{original_spec.title}': Skipping potential redirect to table {i} as its data is malformed.")
            continue
            
        other_table_cols_set = set(other_table_data.get("columns", []))
        
        # Check if this other table contains the original x_column AND all original y_columns
        if original_x_col in other_table_cols_set and original_y_cols_set.issubset(other_table_cols_set):
            logger.info(f"Chart '{original_spec.title}': Redirecting spec to use single alternative table {i} as it contains x_col '{original_x_col}' and all y_cols {original_y_cols}.")
            redirected_spec = original_spec.model_copy(deep=True)
            redirected_spec.source_table_index = i
            single_source_specs.append(redirected_spec)
            return single_source_specs # Return immediately with just the redirected spec
    # --- END: New "redirect" logic ---
    
    # If redirect was not possible, proceed with existing splitting logic
    logger.warning(
        f"Chart '{original_spec.title}': Could not redirect. Primary table {original_spec.source_table_index} lacks x_col '{original_x_col}' (present: {x_column_in_primary_table}) or some y_cols (all present: {all_original_y_cols_in_primary}). Attempting granular split."
    )

    if x_column_in_primary_table:
        # Case 1: x_column is in the primary table, but some y_columns are not.
        if y_columns_in_primary_table: # Create a spec for y_columns that ARE in the primary table.
            spec_part_primary = original_spec.model_copy(deep=True)
            spec_part_primary.y_columns = y_columns_in_primary_table
            spec_part_primary.title = f"{original_spec.title} - Part (Source: {original_spec.source_table_index})"
            single_source_specs.append(spec_part_primary)
            logger.info(f"Chart '{original_spec.title}': Split part for y_columns {y_columns_in_primary_table} from primary table {original_spec.source_table_index} using x_column '{original_x_col}'.")

        # For y_columns NOT in the primary table, try to find other tables that have them WITH the x_column.
        for y_col_not_in_primary in y_columns_not_in_primary_table:
            found_alternative_table = False
            for i, other_data_wrapper in enumerate(all_tables):
                if i == original_spec.source_table_index:
                    continue
                
                other_table_data = other_data_wrapper.get('table') if isinstance(other_data_wrapper, dict) and 'table' in other_data_wrapper else other_data_wrapper
                if not other_table_data or 'columns' not in other_table_data: continue

                other_table_cols = other_table_data.get("columns", [])
                if original_x_col in other_table_cols and y_col_not_in_primary in other_table_cols:
                    spec_part_other = original_spec.model_copy(deep=True)
                    spec_part_other.source_table_index = i
                    spec_part_other.y_columns = [y_col_not_in_primary]
                    spec_part_other.title = f"{original_spec.title} - {y_col_not_in_primary} (Source: {i})"
                    single_source_specs.append(spec_part_other)
                    logger.info(f"Chart '{original_spec.title}': Split part for y_column '{y_col_not_in_primary}' from other table {i} using x_column '{original_x_col}'.")
                    found_alternative_table = True
                    break
            if not found_alternative_table:
                logger.warning(f"Chart '{original_spec.title}': Could not find any table containing x_column '{original_x_col}' and y_column '{y_col_not_in_primary}' together.")
                filtered_out_info.append({"title": f"{original_spec.title} (y-col: {y_col_not_in_primary})", "reason": f"Could not find source for y-col '{y_col_not_in_primary}' with x-col '{original_x_col}'."})

    else: # Case 2: x_column is NOT in the primary table.
        logger.warning(f"Chart '{original_spec.title}': x_column '{original_x_col}' not found in primary table {original_spec.source_table_index}. Will try to find y_cols in other tables that also have this x_col.")
        num_parts_added_for_missing_x = 0
        for y_col_to_check in original_y_cols: # Iterate all original y_cols
            found_alternative_table_for_y_col = False
            for i, other_data_wrapper in enumerate(all_tables):
                other_table_data = other_data_wrapper.get('table') if isinstance(other_data_wrapper, dict) and 'table' in other_data_wrapper else other_data_wrapper
                if not other_table_data or 'columns' not in other_table_data: continue
                
                other_table_cols = other_table_data.get("columns", [])
                if original_x_col in other_table_cols and y_col_to_check in other_table_cols:
                    spec_part_other = original_spec.model_copy(deep=True)
                    spec_part_other.source_table_index = i
                    spec_part_other.x_column = original_x_col 
                    spec_part_other.y_columns = [y_col_to_check]
                    spec_part_other.title = f"{original_spec.title} - {y_col_to_check} (Source: {i}, X: '{original_x_col}')"
                    single_source_specs.append(spec_part_other)
                    logger.info(f"Chart '{original_spec.title}': Split part for y_column '{y_col_to_check}' from other table {i} using original x_column '{original_x_col}'.")
                    num_parts_added_for_missing_x +=1
                    found_alternative_table_for_y_col = True
                    break 
            if not found_alternative_table_for_y_col:
                logger.warning(f"Chart '{original_spec.title}': Could not find any table containing original x_column '{original_x_col}' and y_column '{y_col_to_check}' together.")
                filtered_out_info.append({"title": f"{original_spec.title} (y-col: {y_col_to_check})", "reason": f"Could not find source for y-col '{y_col_to_check}' with x-col '{original_x_col}' (primary x-col missing)."})
        
        if num_parts_added_for_missing_x > 0 :
             logger.info(f"Chart '{original_spec.title}': Added {num_parts_added_for_missing_x} split parts due to missing x_col '{original_x_col}' in primary table {original_spec.source_table_index}.")
        elif not original_y_cols: # No y_cols to begin with, and x_col missing from primary
            logger.warning(f"Chart '{original_spec.title}': x_column '{original_x_col}' not in primary table {original_spec.source_table_index}, and no y_columns were specified. Cannot split.")
            filtered_out_info.append({"title": original_spec.title, "reason": f"x_column '{original_x_col}' not in primary table and no y_columns to process."})
        elif original_y_cols: # x_col missing from primary, and no parts could be formed for any y_col
             logger.error(f"Chart '{original_spec.title}': x_column '{original_x_col}' not in primary table {original_spec.source_table_index}, and no alternative source found for any y_columns {original_y_cols} with this x_column.")
             # This general failure for the original spec will be handled by the calling function if single_source_specs is empty.


    if not single_source_specs:
        logger.error(f"Chart '{original_spec.title}': After attempting split/redirect, no valid single-source spec could be derived. Original spec: {original_spec.model_dump_json()}")
        # Ensure a general filtered_out_info entry if not already added by more specific logic above
        if not any(f_info['title'] == original_spec.title for f_info in filtered_out_info):
            filtered_out_info.append({"title": original_spec.title, "reason": "Failed to derive any processable chart spec after split/redirect attempts."})

    return single_source_specs

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
    metrics_to_pivot: List[str] # Added: LLM-specified y_columns that are the actual metrics
) -> Optional[Dict[str, Any]]:
    """
    Transforms a single-row, multi-column table (like a summary of multiple metrics)
    into the 2-column (Category, Value) format required for pie charts.
    It uses the original column names of the *specified metrics_to_pivot* as the categories.

    Args:
        source_table: The original table data {'columns': [...], 'rows': [[...]]}.
        metrics_to_pivot: A list of column names from the source_table that should be pivoted.
                           These are expected to hold numeric values for the pie slices.

    Returns:
        A new table dictionary in the format {'columns': ['Category', 'Value'], 'rows': [['Metric1', Val1], ['Metric2', Val2], ...]}
        or None if transformation is not applicable or fails.
    """
    columns = source_table.get("columns", [])
    rows = source_table.get("rows", [])

    # Check if transformation is applicable: 1 row, >= 1 metric to pivot
    if len(rows) != 1 or not metrics_to_pivot:
        logger.debug(f"[_transform_wide_summary_to_pie_data] Skipping transformation: Data does not match 1 row pattern or no metrics_to_pivot specified. Rows: {len(rows)}, Metrics: {metrics_to_pivot}")
        return None

    try:
        row_data = rows[0]
        if len(row_data) != len(columns):
            logger.warning(f"[_transform_wide_summary_to_pie_data] Skipping transformation: Row length ({len(row_data)}) does not match column count ({len(columns)}).")
            return None

        new_columns = ["Category", "Value"]
        new_rows = []
        
        processed_metrics = 0
        for metric_col_name in metrics_to_pivot:
            if metric_col_name in columns:
                try:
                    col_index = columns.index(metric_col_name)
                    value = row_data[col_index]
                    
                    # Ensure the value is numeric for a pie chart slice
                    if isinstance(value, numbers.Number):
                        new_rows.append([metric_col_name, value]) # metric_col_name becomes the category
                        processed_metrics +=1
                    else:
                        logger.warning(f"[_transform_wide_summary_to_pie_data] Value for metric column '{metric_col_name}' ('{value}') is not numeric. Skipping for pie chart.")
                except (ValueError, IndexError): # Should not happen if metric_col_name in columns and row_data length matches
                    logger.warning(f"[_transform_wide_summary_to_pie_data] Error accessing data for metric column '{metric_col_name}'.", exc_info=True)
            else:
                logger.warning(f"[_transform_wide_summary_to_pie_data] Metric column '{metric_col_name}' specified for pivoting not found in source table columns: {columns}.")

        if not new_rows: # No valid numeric metrics were pivoted
            logger.warning(f"[_transform_wide_summary_to_pie_data] Transformation resulted in empty data. No valid numeric metrics found in {metrics_to_pivot} from columns {columns}.")
            return None
            
        transformed_table = {
            "columns": new_columns,
            "rows": new_rows,
            "metadata": {"transformed_for_pie": True, "original_metrics_pivoted": metrics_to_pivot}
        }
        logger.info(f"Successfully transformed wide summary data for pie chart. Pivoted {processed_metrics} metrics from {metrics_to_pivot}. Original table cols: {columns} -> New cols: {new_columns}")
        return transformed_table

    except Exception as e: # General catch-all for unexpected issues
        logger.warning(f"Failed to transform wide summary data for pie chart: {e}. Original cols: {columns}, Metrics to pivot: {metrics_to_pivot}", exc_info=True)
        return None

# --- Helper function to transform wide summary data for Bar charts ---
def _transform_wide_summary_to_bar_data(
    source_table: Dict[str, Any],
    metrics_to_bar: List[str] # ADDED: LLM-specified y_columns that are the actual metrics to bar
) -> Optional[Dict[str, Any]]:
    """
    Transforms a single-row, multi-column table (like a summary of multiple metrics)
    into the 2-column (Metric, Value) format suitable for a simple bar chart
    showing total counts per metric, based on explicitly specified metrics.

    Args:
        source_table: The original table data {'columns': [...], 'rows': [[...]]}.
        metrics_to_bar: A list of column names from the source_table that should be barred.
                           These are expected to hold numeric values for the bars.

    Returns:
        A new table dictionary in the format {'columns': ['Metric', 'Value'], 'rows': [['Metric1', Val1], ['Metric2', Val2], ...]}
        or None if transformation is not applicable or fails.
    """
    columns = source_table.get("columns", [])
    rows = source_table.get("rows", [])

    # Check if transformation is applicable: 1 row, >= 1 column in source, and metrics specified
    if len(rows) != 1 or len(columns) < 1:
        logger.debug("[_transform_wide_summary_to_bar_data] Skipping transformation: Source data does not match 1 row, >=1 column pattern.")
        return None 

    if not metrics_to_bar:
        logger.warning("[_transform_wide_summary_to_bar_data] Skipping transformation: No 'metrics_to_bar' were specified by the LLM spec.")
        return None 

    try:
        row_data = rows[0]
        if len(row_data) != len(columns):
            logger.warning("[_transform_wide_summary_to_bar_data] Skipping transformation: Row length does not match column count.")
            return None

        new_columns = ["Metric", "Value"] 
        new_rows = []
        processed_metrics_count = 0

        for metric_col_name in metrics_to_bar: # Iterate over specified metrics
            if metric_col_name in columns:
                try:
                    col_index = columns.index(metric_col_name)
                    value = row_data[col_index]
                    
                    numeric_value = None
                    if isinstance(value, numbers.Number):
                        numeric_value = value
                    elif isinstance(value, str):
                        try: numeric_value = float(value)
                        except (ValueError, TypeError): pass
                    
                    if numeric_value is not None:
                        new_rows.append([metric_col_name, numeric_value]) 
                        processed_metrics_count += 1
                    else:
                        logger.warning(f"[_transform_wide_summary_to_bar_data] Skipping specified metric '{metric_col_name}' as its value '{value}' is not numeric.")
                except (ValueError, IndexError):
                     logger.warning(f"[_transform_wide_summary_to_bar_data] Error accessing data for specified metric column '{metric_col_name}'. Skipping.", exc_info=True)
            else:
                logger.warning(f"[_transform_wide_summary_to_bar_data] Specified metric column '{metric_col_name}' for barring not found in source table columns: {columns}. Skipping.")

        if not new_rows: # No valid numeric metrics were successfully barred
            logger.warning(f"[_transform_wide_summary_to_bar_data] Transformation resulted in empty data. No valid numeric metrics found from the specified list: {metrics_to_bar}.")
            return None
            
        transformed_table = {
            "columns": new_columns,
            "rows": new_rows,
            "metadata": {"transformed_summary_for_bar": True, "original_metrics_barred": metrics_to_bar}
        }
        logger.info(f"Successfully transformed wide summary data for bar chart. Barred {processed_metrics_count} specified metrics from {metrics_to_bar}. Original table cols: {columns} -> New cols: {new_columns}")
        return transformed_table

    except Exception as e: # General catch-all for unexpected issues
        logger.warning(f"Failed to transform wide summary data for bar chart: {e}. Original cols: {columns}, Metrics to bar: {metrics_to_bar}", exc_info=True)
        return None


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
        
        # Basic spec structure validation
        # Allow x_column to be None ONLY if type_hint is 'pie' and conditions for wide-summary transform are met.
        is_potential_wide_summary_pie = (
            spec.type_hint == 'pie' and
            spec.y_columns and # y_columns must exist for this specific transform
            spec.x_column is None # LLM explicitly indicates no x_column for this transform
        )

        if not (0 <= spec.source_table_index < len(current_tables_in_state)):
            failure_reason = f"Invalid source_table_index {spec.source_table_index} for {len(current_tables_in_state)} available tables."
        elif not spec.y_columns: # y_columns must not be empty for any chart type
            failure_reason = "Missing y_columns."
        elif not spec.x_column and not is_potential_wide_summary_pie:
            # x_column is missing, AND it's not the allowed case for wide-summary pie with x_column: None
            failure_reason = f"Missing x_column (and not a valid wide-summary pie with x_column=None). Type: {spec.type_hint}"
        
        if failure_reason:
            logger.warning(f"Chart '{spec_title}': Pre-check failed: {failure_reason}")
            filtered_out_info.append({"title": spec_title, "reason": failure_reason})
            continue

        primary_table_data_for_spec = current_tables_in_state[spec.source_table_index]
        primary_cols_for_spec = primary_table_data_for_spec.get("columns", [])
        primary_rows_for_spec = primary_table_data_for_spec.get("rows", [])
        
        if not primary_cols_for_spec:
            failure_reason = f"Primary source table {spec.source_table_index} (for spec '{spec_title}') has no columns."
            logger.warning(f"Chart '{spec_title}': {failure_reason}")
            filtered_out_info.append({"title": spec_title, "reason": failure_reason})
            continue

        y_cols_in_primary = [yc for yc in spec.y_columns if yc in primary_cols_for_spec]
        y_cols_not_in_primary = [yc for yc in spec.y_columns if yc not in primary_cols_for_spec]
        # Re-evaluate x_col_in_primary: True if x_column is specified AND exists.
        # False if x_column is None OR (specified but doesn't exist)
        x_col_in_primary = spec.x_column is not None and spec.x_column in primary_cols_for_spec

        final_spec_to_process_downstream = spec 
        source_table_for_downstream_processing = primary_table_data_for_spec
        ready_for_standard_processing = False

        if x_col_in_primary and not y_cols_not_in_primary:
            logger.info(f"Chart '{spec_title}': Valid single-source spec. All x/y_cols in primary table {spec.source_table_index}.")
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
            is_pie_transformed = False # Indicates if a pie-specific transformation (wide-summary or long-form to 2-col) occurred
            is_summary_transformed = False
            post_transform_failure_reason = None

            # Prepare variables for ApiChartSpecification arguments, allow them to be overridden by transformations
            data_to_use_for_api_spec = data_for_chart_obj # Default to the current data object for the spec
            x_col_for_api_spec = current_x_col
            # Ensure current_y_cols is not empty before trying to access its first element
            y_col_for_api_spec = current_y_cols[0] if current_y_cols else None 
            color_col_for_api_spec = getattr(current_spec_instance, 'color_column', None)
            # Keep track of original y_cols before potential modification for logging/filtering info
            original_y_cols_for_this_spec = list(current_y_cols)


            # Priority for multi-y-column bar/line charts, including single-row data if x_col is a suitable grouper.
            if (type_hint in ['bar', 'line'] and 
                len(current_y_cols) > 1 and 
                current_x_col in cols_in_data_for_chart_obj and 
                all(yc in cols_in_data_for_chart_obj for yc in current_y_cols) and 
                current_x_col not in current_y_cols): # Heuristic: x_col is a separate grouper

                logger.debug(f"Chart '{spec_title_for_api}': Attempting _transform_wide_to_long due to multi-y ({current_y_cols}) and distinct x_col ('{current_x_col}').")
                # original_y_cols_for_this_spec = list(current_y_cols) # Already captured above
                transformed_long = _transform_wide_to_long(data_for_chart_obj, current_x_col, current_y_cols)
                
                if transformed_long.get("metadata", {}).get("transformed_from_wide_multi_y"):
                    data_to_use_for_api_spec = transformed_long # Update data for API spec
                    cols_in_data_for_chart_obj = data_to_use_for_api_spec.get("columns", []) 
                    
                    if "Value" in cols_in_data_for_chart_obj:
                        y_col_for_api_spec = "Value" # Update y_col for API spec
                    if "Metric" in cols_in_data_for_chart_obj:
                        color_col_for_api_spec = "Metric" # Update color_col for API spec

                    is_multi_metric_transformed = True
                    logger.info(f"Chart '{spec_title_for_api}': Successfully applied _transform_wide_to_long. Original y_cols: {original_y_cols_for_this_spec}.")
                else:
                    post_transform_failure_reason = transformed_long.get("metadata",{}).get("transform_error", "Melt transform for multi-y bar/line failed")
                    logger.warning(f"Chart '{spec_title_for_api}': _transform_wide_to_long failed or did not transform. Reason: {post_transform_failure_reason}. Original y_cols: {original_y_cols_for_this_spec}.")
            
            # Then, specific 1-row summary transformations if the multi-metric transform didn't apply or wasn't suitable.
            # (Bar summary)
            if (type_hint == 'bar' and 
                not is_multi_metric_transformed and 
                len(rows_in_data_for_chart_obj) == 1 and 
                len(cols_in_data_for_chart_obj) >= 1):
                
                # original_y_cols_for_this_spec contains the y_columns from the LLM spec
                if original_y_cols_for_this_spec: # Check if LLM specified metrics for the bar chart
                    logger.debug(f"Chart '{spec_title_for_api}': Attempting _transform_wide_summary_to_bar_data for 1-row bar chart, using y_cols from LLM: {original_y_cols_for_this_spec}.")
                    # Pass the LLM's specified y_columns as metrics_to_bar
                    transformed_s_bar = _transform_wide_summary_to_bar_data(data_for_chart_obj, original_y_cols_for_this_spec) 
                    if transformed_s_bar: 
                        data_to_use_for_api_spec = transformed_s_bar 
                        cols_in_data_for_chart_obj = data_to_use_for_api_spec.get("columns", [])
                        x_col_for_api_spec = "Metric"
                        y_col_for_api_spec = "Value"
                        color_col_for_api_spec = None
                        is_summary_transformed = True # This flag indicates this specific type of bar chart transformation
                        logger.info(f"Chart '{spec_title_for_api}': Successfully applied _transform_wide_summary_to_bar_data using LLM-specified metrics: {original_y_cols_for_this_spec}.")
                    else:
                        # Log if transformation failed even with specified metrics from LLM
                        # post_transform_failure_reason might be set if _transform_wide_summary_to_bar_data returned None
                        # and the reason was internal to it (e.g., no numeric data for specified metrics).
                        logger.warning(f"Chart '{spec_title_for_api}': _transform_wide_summary_to_bar_data for 1-row bar chart failed or did not transform, despite LLM specifying y_cols: {original_y_cols_for_this_spec}.")
                else:
                    # LLM requested a bar chart from a 1-row table but didn't specify which columns to bar.
                    # This path should ideally not be hit if LLM follows prompts, which mandate y_columns for this case.
                    # We will not transform here; it will be handled by later logic (e.g. single-series bar) or fail validation.
                    logger.debug(f"Chart '{spec_title_for_api}': Skipping _transform_wide_summary_to_bar_data for 1-row bar chart as no specific y_columns (metrics_to_bar) were provided in the LLM spec. Original y_cols from spec: {original_y_cols_for_this_spec}.")
            # (Pie summary - can also apply if multi-metric for bar/line didn't fit)
            elif (type_hint == 'pie' and 
                  not is_multi_metric_transformed and 
                  len(rows_in_data_for_chart_obj) == 1 and 
                  len(cols_in_data_for_chart_obj) >= 1): # Check for at least 1 col
                # Ensure there are y_columns specified by LLM to guide the pie transformation
                if current_y_cols: # current_y_cols is from the LLM spec
                    logger.debug(f"Chart '{spec_title_for_api}': Attempting _transform_wide_summary_to_pie_data for 1-row pie chart, using y_cols: {current_y_cols}.")
                    transformed_s_pie = _transform_wide_summary_to_pie_data(data_for_chart_obj, current_y_cols) 
                    if transformed_s_pie: 
                        data_to_use_for_api_spec = transformed_s_pie # Update data for API spec
                        cols_in_data_for_chart_obj = data_to_use_for_api_spec.get("columns", [])
                        x_col_for_api_spec = "Category"
                        y_col_for_api_spec = "Value"
                        color_col_for_api_spec = None
                        is_pie_transformed = True # Key flag for pie structure
                        logger.info(f"Chart '{spec_title_for_api}': Successfully applied _transform_wide_summary_to_pie_data.")
                    else:
                        # More specific reason if transform returned None vs. error in metadata
                        post_transform_failure_reason = transformed_s_pie.get("metadata",{}).get("transform_error") if isinstance(transformed_s_pie, dict) else f"Pie chart summary transformation failed for y_cols: {current_y_cols} (transform returned None or non-dict)."
                        logger.warning(f"Chart '{spec_title_for_api}': _transform_wide_summary_to_pie_data failed. Reason: {post_transform_failure_reason}")
                else:
                    post_transform_failure_reason = "Pie chart summary transformation skipped: No y_columns (metrics) specified by LLM for 1-row data."
                    logger.warning(f"Chart '{spec_title_for_api}': {post_transform_failure_reason}")
            
            # Process long-form pie data if not already transformed as wide-summary pie
            if type_hint == 'pie' and not is_pie_transformed and not is_summary_transformed: # Ensure it's a pie not yet processed
                llm_x_col_for_pie = current_spec_instance.x_column # from LLM
                llm_y_col_for_pie = current_spec_instance.y_columns[0] if current_spec_instance.y_columns else None # from LLM

                if (llm_x_col_for_pie and 
                    llm_y_col_for_pie and 
                    llm_x_col_for_pie in data_for_chart_obj.get("columns", []) and 
                    llm_y_col_for_pie in data_for_chart_obj.get("columns", [])):
                    try:
                        x_idx = data_for_chart_obj["columns"].index(llm_x_col_for_pie)
                        y_idx = data_for_chart_obj["columns"].index(llm_y_col_for_pie)
                        
                        new_pie_rows = []
                        source_rows = data_for_chart_obj.get("rows", [])
                        for row_num, row_content in enumerate(source_rows):
                            if len(row_content) > max(x_idx, y_idx):
                                category_val = row_content[x_idx]
                                value_val = row_content[y_idx]
                                if isinstance(value_val, numbers.Number):
                                    new_pie_rows.append([category_val, value_val])
                                else:
                                    logger.warning(f"Chart '{spec_title_for_api}': Pie chart (long-form) at row {row_num} skipping non-numeric y-value '{value_val}' for y-column '{llm_y_col_for_pie}'.")
                            else:
                                logger.warning(f"Chart '{spec_title_for_api}': Pie chart (long-form) at row {row_num} has insufficient data length for specified x/y columns.")
                        
                        if not new_pie_rows:
                            post_transform_failure_reason = f"Pie chart (long-form) processing for x:'{llm_x_col_for_pie}', y:'{llm_y_col_for_pie}' resulted in no valid numeric data rows."
                        else:
                            data_to_use_for_api_spec = {
                                "columns": ["Category", "Value"],
                                "rows": new_pie_rows,
                                "metadata": {**data_for_chart_obj.get("metadata", {}), 
                                             "transformed_long_form_pie": True, 
                                             "original_x_col": llm_x_col_for_pie, 
                                             "original_y_col": llm_y_col_for_pie}
                            }
                            cols_in_data_for_chart_obj = data_to_use_for_api_spec["columns"] # Update for subsequent checks
                            x_col_for_api_spec = "Category"
                            y_col_for_api_spec = "Value"
                            color_col_for_api_spec = None
                            is_pie_transformed = True # Set flag: structure is now standard 2-col for pie
                            logger.info(f"Chart '{spec_title_for_api}': Successfully processed long-form data for pie chart. Original x:'{llm_x_col_for_pie}', y:'{llm_y_col_for_pie}' -> Standardized to 'Category', 'Value'. {len(new_pie_rows)} rows.")
                    except (ValueError, IndexError) as e:
                        post_transform_failure_reason = f"Error preparing long-form pie data. x:'{llm_x_col_for_pie}', y:'{llm_y_col_for_pie}'. Error: {e}"
                        logger.warning(f"Chart '{spec_title_for_api}': {post_transform_failure_reason}", exc_info=True)
                else:
                    post_transform_failure_reason = f"Pie chart (long-form) requires valid x_column ('{llm_x_col_for_pie}') and y_column ('{llm_y_col_for_pie}') from LLM spec to be present in source table columns: {data_for_chart_obj.get('columns', [])}."
                
                if post_transform_failure_reason: # If long-form pie processing failed
                     logger.warning(f"Chart '{spec_title_for_api}': Failed to process long-form pie data. Reason: {post_transform_failure_reason}")
            
            # Fallback for single y-column bar/line if no other transform applied.
            # Or if multi-y was specified but the transform didn't actually change the structure (e.g., y_cols was already just one).
            # This also handles initial len(current_y_cols) == 1.
            elif (type_hint in ['bar', 'line'] and 
                  not is_multi_metric_transformed and 
                  not is_summary_transformed and 
                  not is_pie_transformed): # Added not is_pie_transformed
                 # current_y_cols here is original_y_cols_for_this_spec
                 if len(original_y_cols_for_this_spec) == 1:
                    y_col_for_api_spec = original_y_cols_for_this_spec[0] # Ensure y_col is set
                    logger.debug(f"Chart '{spec_title_for_api}': Processing as a single y-column bar/line chart ('{y_col_for_api_spec}'). No structural data transformation needed beyond ensuring y-col exists.")
                 elif len(original_y_cols_for_this_spec) > 1: 
                    # Log more clearly that other y-columns are being dropped for this specific chart instance
                    y_col_for_api_spec = original_y_cols_for_this_spec[0] # Default to first y-column
                    dropped_y_cols = original_y_cols_for_this_spec[1:]
                    logger.warning(f"Chart '{spec_title_for_api}': Had multiple y-columns {original_y_cols_for_this_spec} but did not result in a multi-metric transform (e.g., melt). Defaulting to use only the first y-column '{y_col_for_api_spec}' for this chart. Dropped y_columns for this instance: {dropped_y_cols}.")
                    # Add to filtered_out_info if this partial processing is considered a filterable event
                    # For enterprise grade, explicit is better.
                    if dropped_y_cols: # If any columns were actually dropped
                         filtered_out_info.append({
                             "title": spec_title_for_api, 
                             "reason": f"Multi-y-column spec ({original_y_cols_for_this_spec}) did not transform as expected (e.g., melt failed or was not applicable for {type_hint}). Proceeding with first y-column '{y_col_for_api_spec}'. Other y-columns ({dropped_y_cols}) were not included in this specific chart output."
                         })
                 elif not original_y_cols_for_this_spec and type_hint in ['bar', 'line']: # No y-columns for bar/line
                     post_transform_failure_reason = f"No y_columns specified for {type_hint} chart '{spec_title_for_api}'."
                     logger.warning(f"Chart '{spec_title_for_api}': {post_transform_failure_reason}")


            # Consolidate post_transform_failure_reason check
            if post_transform_failure_reason:
                # Check if already added to filtered_out_info to avoid duplicates if a more specific message was added above
                if not any(f_info['title'] == spec_title_for_api and f_info['reason'] == post_transform_failure_reason for f_info in filtered_out_info):
                    logger.warning(f"Chart '{spec_title_for_api}' failed data transformation stage: {post_transform_failure_reason}")
                    filtered_out_info.append({"title": spec_title_for_api, "reason": post_transform_failure_reason})
                continue # Skip to next spec in queue
            
            # Final y_column and color_column for API spec constructor
            # y_col_for_api_spec and color_col_for_api_spec should be set correctly by transformation blocks by now
            # Or use their defaults if no transformation occurred related to them.

            # One last check on y_col_for_api_spec if it's None after all transforms (e.g. no y_cols from LLM for bar/line)
            if y_col_for_api_spec is None and type_hint in ['bar', 'line', 'pie']: # Pie should have it set by transforms
                failure_reason = f"Final y_column for chart '{spec_title_for_api}' is None before creating ApiChartSpecification. Original y_cols: {original_y_cols_for_this_spec}."
                logger.error(failure_reason) # This indicates a logic flaw if not caught by post_transform_failure_reason
                filtered_out_info.append({"title": spec_title_for_api, "reason": failure_reason})
                continue

            api_chart_obj = ApiChartSpecification(
                type_hint=type_hint, title=spec_title_for_api,
                x_column=x_col_for_api_spec,
                y_column=y_col_for_api_spec, # This is now the carefully determined y-column
                color_column=color_col_for_api_spec, # This is now the carefully determined color-column
                x_label=getattr(current_spec_instance, "x_label", None),
                y_label=getattr(current_spec_instance, "y_label", None),
                data=TableData(**copy.deepcopy(data_to_use_for_api_spec)) # Use the potentially transformed data
            )

            # Adjustments after ApiChartSpecification creation based on transformation flags
            # These are now simplified as x_col_for_api_spec, y_col_for_api_spec are set by transform blocks

            if is_summary_transformed: # Bar chart from summary
                # x_column, y_column already set to "Metric", "Value" by transform block
                if not api_chart_obj.y_label: api_chart_obj.y_label = "Value"
            elif is_pie_transformed: # Pie from wide-summary OR long-form
                # x_column, y_column already set to "Category", "Value" by transform blocks
                if not api_chart_obj.y_label: api_chart_obj.y_label = "Value"
            elif is_multi_metric_transformed: # Bar/line from melt
                # y_column already set to "Value", color_column to "Metric" by transform block
                if not api_chart_obj.y_label: api_chart_obj.y_label = "Value"
            # No specific 'elif type_hint == 'pie':' needed here anymore for color_column,
            # as is_pie_transformed block already sets color_column to None.
            # And if it's not transformed, color_col_for_api_spec default will be used.

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