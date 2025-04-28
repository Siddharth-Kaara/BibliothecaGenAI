import base64
import json
import logging
import os
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple
import asyncio

import pandas as pd
import matplotlib
matplotlib.use('Agg') # Use Agg backend for non-interactive environments (important for servers)
import matplotlib.pyplot as plt
import seaborn as sns
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.tools import BaseTool
from langchain_openai import AzureChatOpenAI

from app.core.config import settings

# Use standard logger
logger = logging.getLogger(__name__)

# Set a default Seaborn style
sns.set_theme(style="whitegrid")

# Default metadata for common chart scenarios
# These are simple defaults, can be expanded
DEFAULT_FALLBACK_METADATA = {
    "bar": {
        "title": "Bar Chart",
        "x_column": None, # Determined dynamically based on first non-numeric column
        "y_column": None, # Determined dynamically based on first numeric column
        "color_column": None, 
        "x_label": "Category",
        "y_label": "Value"
    },
    "pie": {
         "title": "Pie Chart",
         "x_column": None, # Category column
         "y_column": None, # Value column
         "color_column": None,
         "description": "Distribution of categories"
    }
    # Add defaults for line, scatter etc. if needed
}

# Define a constant for the maximum recommended categories for a pie chart
MAX_PIE_CATEGORIES = 10

SCALE_RATIO_THRESHOLD = 20 # Threshold for switching bar chart to faceted plot

class ChartRendererTool(BaseTool):
    """Tool for generating chart visualizations using Matplotlib/Seaborn."""
    
    name: str = "chart_renderer"
    description: str = """
    Generates charts and visualizations from data using Matplotlib/Seaborn.
    Use this tool when you need to create a bar chart, pie chart, line chart, scatter plot, etc.
    Input MUST be a dictionary containing:
     - 'data': A dictionary with 'columns': list[str] and 'rows': list[list[any]].
     - 'metadata': A dictionary detailing the chart (e.g., 'chart_type', 'title', 'x_column', 'y_column', 'color_column').
    """
    
    def _generate_chart_metadata(self, query: str, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate chart metadata from query and data using LLM."""
        logger.debug("Generating chart metadata...")
        if not data:
            logger.error("No data provided to _generate_chart_metadata")
            raise ValueError("No data provided for chart generation")
        
        # Convert data to string format
        data_str = json.dumps(data[:10], indent=2)  # Limit to 10 rows for LLM
        
        # Create an Azure OpenAI instance
        llm = AzureChatOpenAI(
            openai_api_key=settings.AZURE_OPENAI_API_KEY,
            azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
            openai_api_version=settings.AZURE_OPENAI_API_VERSION,
            deployment_name=settings.AZURE_OPENAI_DEPLOYMENT_NAME,
            model_name=settings.LLM_MODEL_NAME,
            temperature=0.1,
        )
        
        # Create prompt template
        template = """
        You are a data visualization expert. Given the following data and a query, determine the best chart type
        and provide metadata needed to create the visualization using Matplotlib/Seaborn.
        
        Data (sample):
        {data}
        
        Query: {query}
        
        Respond with a JSON object with the following structure:
        {{
            "chart_type": "bar|pie|line|scatter",
            "title": "Chart title",
            "x_column": "Name of column for x-axis (or categories/labels for pie)",
            "y_column": "Name of column for y-axis (or values/sizes for pie)",
            "color_column": "Optional column for color differentiation (hue)",
            "description": "Brief description of what the chart shows"
        }}
        
        Return ONLY the JSON object without any explanation.
        """
        
        prompt = PromptTemplate(
            input_variables=["data", "query"],
            template=template,
        )
        
        # Create a chain for metadata generation
        metadata_chain = LLMChain(llm=llm, prompt=prompt)
        
        # Generate metadata
        logger.debug("Invoking LLM chain for chart metadata...")
        metadata_str = metadata_chain.run(
            data=data_str,
            query=query,
        )
        logger.debug(f"Raw metadata string from LLM: {metadata_str}")
        
        # Clean and parse the JSON
        metadata_str = metadata_str.strip().removeprefix("```json").removesuffix("```").strip()
        
        try:
            metadata = json.loads(metadata_str)
            logger.debug(f"Successfully parsed chart metadata: {metadata}")
            return metadata
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing chart metadata JSON: {str(e)}", exc_info=True)
            logger.debug(f"Problematic raw metadata string: {metadata_str}")
            raise ValueError(f"Invalid chart metadata format: {str(e)}")
    
    def _create_dataframe(self, data: Dict[str, Any]) -> pd.DataFrame:
        """Create pandas DataFrame from dictionary with 'columns' and 'rows'."""
        if not isinstance(data, dict) or "columns" not in data or "rows" not in data:
            raise ValueError("Invalid 'data' format provided to _create_dataframe. Must contain 'columns' and 'rows'.")
        
        logger.debug("Creating pandas DataFrame...")
        try:
            df = pd.DataFrame(data['rows'], columns=data['columns'])
            logger.debug(f"DataFrame created successfully with shape {df.shape}")
            return df
        except Exception as e:
            logger.error(f"Error creating DataFrame: {e}", exc_info=True)
            raise ValueError(f"Could not create DataFrame from provided data: {e}")

    def _transform_for_comparison_bar(self, df: pd.DataFrame, metadata: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Transforms wide data to long format for comparative bar charts using hue.
           Also adjusts metadata to match the new structure.
        """
        x_col = metadata.get("x_column")
        # Identify potential Y columns (numeric columns other than x_col)
        potential_y_cols = [ 
            col for col in df.columns 
            if col != x_col and pd.api.types.is_numeric_dtype(df[col])
        ]

        if not x_col or x_col not in df.columns:
            logger.warning(f"Invalid or missing x_column ('{x_col}') for transformation. Cannot transform.")
            return df, metadata # Return original
        
        if len(potential_y_cols) <= 1:
            logger.debug(f"Only one or zero numeric Y columns found ({potential_y_cols}). No transformation needed for comparison bar.")
            return df, metadata # No transformation needed
        
        logger.info(f"Attempting wide-to-long transformation for bar chart comparison. X='{x_col}', Ys={potential_y_cols}")
        try:
            df_long = pd.melt(df, 
                              id_vars=[x_col], 
                              value_vars=potential_y_cols, 
                              var_name="Metric Type", # New column for original Y names
                              value_name="Value")      # New column for the values
            
            # Adjust metadata
            new_metadata = metadata.copy()
            new_metadata['y_column'] = "Value" # Y-axis is now the 'Value' column
            new_metadata['color_column'] = "Metric Type" # Hue is now the 'Metric Type' column
            # Ensure labels reflect the change if not explicitly set by LLM
            if "y_label" not in new_metadata or not new_metadata["y_label"]:
                new_metadata["y_label"] = "Value"
                logger.debug("Setting y_label to 'Value' after transformation.")

            logger.info(f"Data transformed successfully to long format. New shape: {df_long.shape}")
            return df_long, new_metadata
        except Exception as e:
            logger.error(f"Error during wide-to-long transformation: {e}", exc_info=True)
            # Fallback to original data if transformation fails
            return df, metadata

    def _render_bar_chart(self, ax: plt.Axes, df: pd.DataFrame, metadata: Dict[str, Any]):
        """Render a bar chart using Seaborn on the provided Axes."""
        x_col_meta = metadata.get("x_column")
        y_col_meta = metadata.get("y_column")
        hue_col_meta = metadata.get("color_column") # Use color_column for hue

        actual_cols = df.columns.tolist()
        x_col = x_col_meta if x_col_meta in actual_cols else actual_cols[0] if len(actual_cols) > 0 else None
        y_col = y_col_meta if y_col_meta in actual_cols else actual_cols[1] if len(actual_cols) > 1 else None
        # Use the color_column from metadata as the definitive hue source
        hue_col = hue_col_meta if hue_col_meta and hue_col_meta in actual_cols else None

        # Get color mapping from metadata if provided
        color_mapping = metadata.get("color_mapping")
        palette = None
        if isinstance(color_mapping, dict) and color_mapping:
            # Ensure mapping keys exist in the x-column data for bar chart
            valid_mapping = {k: v for k, v in color_mapping.items() if k in df[x_col].unique()}
            if valid_mapping:
                palette = valid_mapping
                logger.debug(f"Using provided color mapping (palette): {palette}")
            else:
                logger.warning("Provided color_mapping keys do not match data categories. Using default palette.")

        if x_col != x_col_meta or y_col != y_col_meta:
             # This warning might trigger if transformation adjusted metadata correctly
             logger.warning(f"Metadata columns ('{x_col_meta}', '{y_col_meta}') possibly adjusted or invalid. Using derived ('{x_col}', '{y_col}').")

        if not x_col or not y_col:
            raise ValueError("Could not determine valid x and y columns for bar chart")

        # Log the actual columns being used, especially hue_col
        logger.debug(f"Rendering bar chart with x='{x_col}', y='{y_col}', hue='{hue_col}', palette='{bool(palette)}'")
        try:
            sns.barplot(data=df, x=x_col, y=y_col, hue=hue_col, ax=ax, errorbar=None, palette=palette)
        except Exception as plot_err:
             logger.error(f"Error during seaborn barplot rendering: {plot_err}", exc_info=True)
             # Re-raise or handle gracefully (e.g., render a text message on the plot)
             ax.text(0.5, 0.5, f"Error rendering bar chart:\n{plot_err}", ha='center', va='center', wrap=True)

    def _render_pie_chart(self, ax: plt.Axes, df: pd.DataFrame, metadata: Dict[str, Any]):
        """Render a pie chart using Matplotlib on the provided Axes."""
        label_col_meta = metadata.get("x_column") # Labels from x_column
        value_col_meta = metadata.get("y_column") # Values from y_column
        
        actual_cols = df.columns.tolist()
        label_col = label_col_meta if label_col_meta in actual_cols else actual_cols[0] if len(actual_cols) > 0 else None
        value_col = value_col_meta if value_col_meta in actual_cols else actual_cols[1] if len(actual_cols) > 1 else None

        if label_col != label_col_meta or value_col != value_col_meta:
             logger.warning(f"Metadata columns ('{label_col_meta}', '{value_col_meta}') possibly invalid. Using derived ('{label_col}', '{value_col}').")

        if not label_col or not value_col:
            raise ValueError("Could not determine valid label and value columns for pie chart")
        
        if not pd.api.types.is_numeric_dtype(df[value_col]):
            raise ValueError(f"Value column '{value_col}' for pie chart must be numeric")
            
        logger.debug(f"Rendering pie chart with labels='{label_col}', values='{value_col}'")
        # Prepare data - often better to aggregate first if multiple rows per category
        pie_data = df.set_index(label_col)[value_col]
        
        try:
             ax.pie(pie_data, labels=pie_data.index, autopct='%1.1f%%', startangle=90)
             ax.axis('equal')  # Equal aspect ratio ensures a circular pie chart
        except Exception as plot_err:
             logger.error(f"Error during matplotlib pie chart rendering: {plot_err}", exc_info=True)
             ax.text(0.5, 0.5, f"Error rendering pie chart:\n{plot_err}", ha='center', va='center', wrap=True)

    def _render_line_chart(self, ax: plt.Axes, df: pd.DataFrame, metadata: Dict[str, Any]):
        """Render a line chart using Seaborn on the provided Axes."""
        x_col_meta = metadata.get("x_column")
        y_col_meta = metadata.get("y_column")
        hue_col_meta = metadata.get("color_column")
        
        actual_cols = df.columns.tolist()
        x_col = x_col_meta if x_col_meta in actual_cols else actual_cols[0] if len(actual_cols) > 0 else None
        y_col = y_col_meta if y_col_meta in actual_cols else actual_cols[1] if len(actual_cols) > 1 else None
        hue_col = hue_col_meta if hue_col_meta and hue_col_meta in actual_cols else None

        if x_col != x_col_meta or y_col != y_col_meta:
             logger.warning(f"Metadata columns ('{x_col_meta}', '{y_col_meta}') possibly invalid. Using derived ('{x_col}', '{y_col}').")

        if not x_col or not y_col:
            raise ValueError("Could not determine valid x and y columns for line chart")
            
        # Attempt to convert x-column to datetime if it looks like a date/time string
        try:
            if pd.api.types.is_string_dtype(df[x_col]):
                df[x_col] = pd.to_datetime(df[x_col])
                df = df.sort_values(by=x_col) # Sort by time for line chart
                logger.debug(f"Converted column '{x_col}' to datetime for line chart.")
        except Exception as date_conv_err:
            logger.warning(f"Could not convert x-column '{x_col}' to datetime: {date_conv_err}. Plotting as is.")

        logger.debug(f"Rendering line chart with x='{x_col}', y='{y_col}', hue='{hue_col}'")
        try:
            sns.lineplot(data=df, x=x_col, y=y_col, hue=hue_col, marker='o', ax=ax)
        except Exception as plot_err:
             logger.error(f"Error during seaborn lineplot rendering: {plot_err}", exc_info=True)
             ax.text(0.5, 0.5, f"Error rendering line chart:\n{plot_err}", ha='center', va='center', wrap=True)

    def _render_scatter_chart(self, ax: plt.Axes, df: pd.DataFrame, metadata: Dict[str, Any]):
        """Render a scatter chart using Seaborn on the provided Axes."""
        x_col_meta = metadata.get("x_column")
        y_col_meta = metadata.get("y_column")
        hue_col_meta = metadata.get("color_column")
        
        actual_cols = df.columns.tolist()
        x_col = x_col_meta if x_col_meta in actual_cols else actual_cols[0] if len(actual_cols) > 0 else None
        y_col = y_col_meta if y_col_meta in actual_cols else actual_cols[1] if len(actual_cols) > 1 else None
        hue_col = hue_col_meta if hue_col_meta and hue_col_meta in actual_cols else None

        if x_col != x_col_meta or y_col != y_col_meta:
             logger.warning(f"Metadata columns ('{x_col_meta}', '{y_col_meta}') possibly invalid. Using derived ('{x_col}', '{y_col}').")

        if not x_col or not y_col:
            raise ValueError("Could not determine valid x and y columns for scatter chart")
            
        logger.debug(f"Rendering scatter chart with x='{x_col}', y='{y_col}', hue='{hue_col}'")
        try:
            sns.scatterplot(data=df, x=x_col, y=y_col, hue=hue_col, ax=ax)
        except Exception as plot_err:
             logger.error(f"Error during seaborn scatterplot rendering: {plot_err}", exc_info=True)
             ax.text(0.5, 0.5, f"Error rendering scatter chart:\n{plot_err}", ha='center', va='center', wrap=True)

    def _run(
        self,
        query: Optional[str] = None,
        data: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        log_prefix = "[ChartRendererTool] "
        logger.info(f"{log_prefix}Executing. Data: {data is not None}, Metadata: {metadata is not None}")
        
        pie_switch_message = None
        facet_message = None

        if data is None:
            logger.error(f"{log_prefix}Missing required 'data' argument.")
            return {"error": "Missing required 'data' argument", "visualization": None}
            
        # Use provided metadata or generate if needed (and query is provided)
        # Note: Current agent logic explicitly provides metadata, so generation is less likely needed here
        if metadata is None:
            logger.warning(f"{log_prefix}Metadata not provided. Applying fallback.")
            metadata = {} # Start with empty metadata
        elif not isinstance(metadata, dict):
             logger.error(f"{log_prefix}Invalid metadata format.")
             return {"error": "Invalid metadata format provided", "visualization": None}
        
        try:
            # 1. Create DataFrame
            df = self._create_dataframe(data)

            # 2. Apply Metadata Fallback/Defaults (updates metadata dict)
            chart_type = metadata.get("chart_type", "bar").lower()
            metadata['chart_type'] = chart_type # Ensure it's set

            if not metadata.get("x_column") or metadata["x_column"] not in df.columns or \
               not metadata.get("y_column") or metadata["y_column"] not in df.columns:
                 logger.warning(f"{log_prefix}Essential metadata missing/invalid. Applying fallback.")
                 fallback_applied = False
                 fallback_config = DEFAULT_FALLBACK_METADATA.get(chart_type)
                 if fallback_config:
                     # Try to dynamically assign columns for fallback
                     if not metadata.get("x_column") or metadata["x_column"] not in df.columns:
                          # Guess first non-numeric as X
                          non_numeric_cols = df.select_dtypes(exclude=['number']).columns.tolist()
                          if non_numeric_cols:
                               metadata["x_column"] = non_numeric_cols[0]
                               logger.info(f"{log_prefix}Applying fallback x_column: '{metadata['x_column']}'")
                               fallback_applied = True
                     if not metadata.get("y_column") or metadata["y_column"] not in df.columns:
                          # Guess first numeric as Y
                          numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                          if numeric_cols:
                               metadata["y_column"] = numeric_cols[0]
                               logger.info(f"{log_prefix}Applying fallback y_column: '{metadata['y_column']}'")
                               fallback_applied = True
                     # Apply other fallback defaults if missing
                     metadata["title"] = metadata.get("title") or fallback_config.get("title", "Chart")
                     metadata["x_label"] = metadata.get("x_label") or fallback_config.get("x_label") or metadata.get("x_column")
                     metadata["y_label"] = metadata.get("y_label") or fallback_config.get("y_label") or metadata.get("y_column")
                     
                 if not fallback_applied:
                     logger.error(f"{log_prefix}Fallback failed.")
                     # Fallback failed, we cannot proceed reasonably
                     return {"error": "Could not determine valid x/y columns for chart from metadata or fallback", "visualization": None}
            
            # 3. Handle Pie Chart Overcrowding (may change chart_type in metadata)
            if metadata['chart_type'] == "pie":
                label_col = metadata.get("x_column") # Use final determined x_col
                if label_col and label_col in df.columns:
                    n_categories = df[label_col].nunique()
                    if n_categories > MAX_PIE_CATEGORIES:
                        logger.warning(f"{log_prefix}Too many categories ({n_categories}) for pie. Switching to bar.")
                        metadata['chart_type'] = "bar" # Update metadata
                        chart_type = "bar" # Update local var for flow control
                        pie_switch_message = f"Note: Switched from pie to bar chart ({n_categories} cats > {MAX_PIE_CATEGORIES} max)."
                else:
                    logger.warning("Could not check pie chart categories as label column is invalid.")

            # 4. Apply Wide-to-Long Transformation (updates df and metadata if bar/hue)
            if metadata['chart_type'] == "bar" and metadata.get("color_column"):
                # Pass the *current* metadata for transformation
                df, metadata = self._transform_for_comparison_bar(df, metadata)
                # Update local chart_type if metadata changed (unlikely here)
                chart_type = metadata.get("chart_type", chart_type)

            # 5. Prepare Final Rendering Details (using potentially updated metadata)
            title = metadata.get("title", "Generated Chart")
            # Get final intended columns from metadata for reporting
            final_x_col = metadata.get("x_column")
            final_y_col = metadata.get("y_column")
            final_hue_col = metadata.get("color_column")
            x_label = metadata.get("x_label", final_x_col)
            y_label = metadata.get("y_label", final_y_col)

            # 6. Decide on Faceting for Bar Charts
            plot_object = None
            use_facet = False
            if chart_type == "bar" and final_hue_col:
                 # Check if essential cols for scale check exist
                 if final_y_col and final_y_col in df.columns and final_hue_col in df.columns:
                     try:
                         grouped_max = df.groupby(final_hue_col)[final_y_col].max()
                         max_val = grouped_max.max()
                         min_val = grouped_max[grouped_max > 0].min()
                         if pd.notna(min_val) and min_val > 0 and (max_val / min_val) > SCALE_RATIO_THRESHOLD:
                             use_facet = True
                             logger.warning(f"{log_prefix}Large scale difference. Using faceted bar chart.")
                             facet_message = f"Note: Used faceted bar chart due to large metric scale differences (Ratio > {SCALE_RATIO_THRESHOLD})."
                     except Exception as scale_err: logger.warning(f"{log_prefix}Scale ratio check failed: {scale_err}")
                 else:
                      logger.warning(f"{log_prefix}Skipping scale check for faceting: Y column ('{final_y_col}') or Hue column ('{final_hue_col}') not valid.")

            # 7. Render Plot
            if chart_type == "bar" and use_facet:
                # Render Faceted Bar Plot
                logger.debug(f"{log_prefix}Rendering faceted bar chart: col='{final_hue_col}', hue='{final_hue_col}'")
                try:
                    g = sns.catplot(data=df, x=final_x_col, y=final_y_col, col=final_hue_col,
                                    hue=final_hue_col,
                                    kind="bar", sharey=False, height=5, aspect=1.2,
                                    errorbar=None, palette=metadata.get("color_mapping"))
                    g.set_axis_labels(x_label, y_label)
                    g.set_titles(col_template="{col_name}")
                    for ax_facet in g.axes.flat: ax_facet.tick_params(axis='x', rotation=45, labelleft=True); # Ensure Y ticks are on
                    g.fig.suptitle(title, y=1.03)
                    # Potentially remove automatic legend if hue adds one redundantly
                    # if g.legend: g.legend.remove()
                    plot_object = g # Store FacetGrid
                except Exception as plot_err:
                    logger.error(f"{log_prefix}Catplot error: {plot_err}", exc_info=True)
                    # Create error plot on standard figure
                    fig, ax = plt.subplots(); ax.text(0.5, 0.5, f"Faceted plot error:\n{plot_err}", ha='center', va='center'); plot_object = fig
            else:
                # Render Standard Plot (non-faceted bar, pie, line, scatter)
                fig, ax = plt.subplots(figsize=(12, 7) if chart_type != 'pie' else (10, 8)) # Adjust size
                render_func = None
                if chart_type == "bar": render_func = self._render_bar_chart
                elif chart_type == "pie": render_func = self._render_pie_chart
                elif chart_type == "line": render_func = self._render_line_chart
                elif chart_type == "scatter": render_func = self._render_scatter_chart
                else: raise ValueError(f"Unsupported chart_type: {chart_type}")

                try:
                    render_func(ax, df, metadata) # Pass full metadata to helper
                    # Apply final adjustments to the single Axes object
                    ax.set_title(title, fontsize=16)
                    # Only set x/y labels if not a pie chart
                    if chart_type != 'pie':
                         ax.set_xlabel(x_label, fontsize=12)
                         ax.set_ylabel(y_label, fontsize=12)
                         ax.tick_params(axis='x', rotation=45)
                    # Apply legend title if hue was intended
                    if final_hue_col and ax.get_legend() is not None:
                        ax.legend(title=final_hue_col)
                    plt.tight_layout()
                except Exception as render_err:
                     logger.error(f"{log_prefix}Error during standard {chart_type} rendering: {render_err}", exc_info=True)
                     # Clear and add error text if render func failed internally
                     ax.cla()
                     ax.text(0.5, 0.5, f"Error rendering {chart_type} chart:\n{render_err}", ha='center', va='center')
                plot_object = fig # Store Figure

            if plot_object is None: raise ValueError("Plot object creation failed.")

            # 8. Save Plot
            figure_to_save = plot_object.fig if isinstance(plot_object, sns.FacetGrid) else plot_object
            if isinstance(plot_object, sns.FacetGrid): plot_object.tight_layout()
            # ... (saving logic: chart_id, save_path, savefig, close, logger.info) ...
            chart_id = uuid.uuid4()
            save_filename = f"chart_{chart_id}.png"
            save_path = Path(settings.CHART_DIR) / save_filename
            logger.debug(f"{log_prefix}Attempting to save chart to: {save_path}")
            figure_to_save.savefig(save_path, format='png', bbox_inches='tight')
            plt.close(figure_to_save)
            logger.info(f"{log_prefix}Chart saved successfully to {save_path}")
            chart_url = f"{settings.CHART_URL_BASE.rstrip('/')}/{save_filename}"

            # 9. Construct Return Value
            return_data = {
                "visualization": {
                    "type": metadata.get('chart_type'), # Report final type used
                    "title": title,
                    "description": metadata.get("description", f"{metadata.get('chart_type', 'chart').capitalize()} chart of {y_label} vs {x_label}"),
                    "image_url": chart_url,
                    "data_columns_used": {
                         "x": final_x_col,
                         "y": final_y_col,
                         "color": final_hue_col # Report final columns determined/used
                    }
                }
            }
            final_message = f"{pie_switch_message or ''} {facet_message or ''}".strip()
            if final_message: return_data["message"] = final_message
            return return_data

        except ValueError as ve:
            logger.error(f"{log_prefix}Chart config/data error: {ve}", exc_info=False)
            plt.close(plt.gcf())
            return {"error": f"Chart configuration error: {ve}", "visualization": None}
        except Exception as e:
            logger.error(f"{log_prefix}Chart generation failed unexpectedly: {e}", exc_info=True)
            plt.close(plt.gcf())
            return {"error": f"Failed to generate chart: {e}", "visualization": None}

    async def _arun(
        self,
        query: Optional[str] = None,
        data: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Async wrapper for _run using run_in_executor."""
        log_prefix = "[ChartRendererTool] "
        logger.debug(f"{log_prefix}Executing async run via executor...")
        loop = asyncio.get_event_loop()
        try:
            result = await loop.run_in_executor(
                None, # Use default executor
                self._run, # Target function
                query, data, metadata # Arguments for _run
            )
            return result
        except Exception as e:
             logger.error(f"{log_prefix}Async execution failed: {str(e)}", exc_info=True)
             # Propagate error as a dictionary suitable for ToolMessage content
             # Allows agent to potentially see the error reason
             return {"error": f"Chart rendering failed: {str(e)}"}

# Optional: Define a Pydantic model for stricter input validation if desired
# from pydantic import BaseModel, Field
# class ChartRendererInput(BaseModel):
#     query: Optional[str] = Field(default=None, description="User query or description")
#     data: Dict[str, Any] = Field(description="Data in SQL Tool format {'columns': [...], 'rows': [[...]]}")
#     metadata: Optional[Dict[str, Any]] = Field(default=None, description="Optional chart metadata")

# class ChartRendererTool(BaseTool):
#     name: str = "chart_renderer"
#     description: str = "..."
#     args_schema: Type[BaseModel] = ChartRendererInput
#     # ... rest of the class, _run would receive validated args ...