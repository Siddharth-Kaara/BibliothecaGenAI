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

logger = logging.getLogger(__name__)

# Set a default Seaborn style
sns.set_theme(style="whitegrid")

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
            ax.set_xlabel(metadata.get("x_label", x_col)) # Use label from metadata if available
            ax.set_ylabel(metadata.get("y_label", y_col)) # Use label from metadata if available
            if hue_col: # Add legend if hue is used
                ax.legend(title=hue_col) # Use the hue column name as legend title

        except Exception as e:
            logger.error(f"Error rendering bar chart with Seaborn: {e}", exc_info=True)
            raise ValueError(f"Failed to render bar chart: {e}")
    
    def _render_pie_chart(self, ax: plt.Axes, df: pd.DataFrame, metadata: Dict[str, Any]):
        """Render a pie chart using Matplotlib on the provided Axes."""
        labels_col_meta = metadata.get("x_column") # Map x -> labels
        values_col_meta = metadata.get("y_column") # Map y -> values

        # Get color mapping from metadata if provided
        color_mapping = metadata.get("color_mapping")
        colors = None
        if isinstance(color_mapping, dict) and color_mapping:
            # For pie charts, create a list of colors in the order of the labels
            ordered_colors = [color_mapping.get(label) for label in df[labels_col_meta]]
            # Filter out None values if some labels weren't in the mapping
            if any(c is not None for c in ordered_colors):
                # Use mapped colors where available, None otherwise (matplotlib will default)
                colors = [c if c is not None else None for c in ordered_colors]
                logger.debug(f"Using provided colors for pie chart segments (partial matches allowed).")
            else:
                logger.warning("Provided color_mapping keys do not match data labels. Using default colors.")

        actual_cols = df.columns.tolist()
        labels_col = labels_col_meta if labels_col_meta in actual_cols else actual_cols[0] if len(actual_cols) > 0 else None
        values_col = values_col_meta if values_col_meta in actual_cols else actual_cols[1] if len(actual_cols) > 1 else None

        if labels_col != labels_col_meta or values_col != values_col_meta:
             logger.warning(f"Metadata columns ('{labels_col_meta}', '{values_col_meta}') invalid. Falling back to ('{labels_col}', '{values_col}').")

        if not labels_col or not values_col:
            raise ValueError("Could not determine valid labels and values columns for pie chart")
        
        # Handle potential non-numeric data in values column
        try:
            pie_data = pd.to_numeric(df[values_col], errors='coerce').fillna(0)
            if (pie_data < 0).any():
                 logger.warning(f"Pie chart values column '{values_col}' contains negative values. Taking absolute values.")
                 pie_data = pie_data.abs()
        except KeyError:
             raise ValueError(f"Values column '{values_col}' not found for pie chart.")
        except Exception as e:
             logger.error(f"Error processing pie chart values: {e}", exc_info=True)
             raise ValueError(f"Invalid data in values column '{values_col}' for pie chart.")

        logger.debug(f"Rendering pie chart with labels='{labels_col}', values='{values_col}', colors='{bool(colors)}'")
        try:
            # Pass the colors list to ax.pie
            wedges, texts, autotexts = ax.pie(pie_data, labels=df[labels_col], autopct='%1.1f%%', startangle=90, colors=colors)
            # ax.axis('equal') # Equal aspect ratio ensures that pie is drawn as a circle.
            plt.setp(autotexts, size=8, weight="bold", color="white") # Improve autopct visibility
        except Exception as e:
            logger.error(f"Error rendering pie chart with Matplotlib: {e}", exc_info=True)
            raise ValueError(f"Failed to render pie chart: {e}")
    
    def _render_line_chart(self, ax: plt.Axes, df: pd.DataFrame, metadata: Dict[str, Any]):
        """Render a line chart using Seaborn on the provided Axes."""
        x_col_meta = metadata.get("x_column")
        y_col_meta = metadata.get("y_column")
        hue_col_meta = metadata.get("color_column")

        # Get color mapping from metadata if provided (used if hue_col is set)
        color_mapping = metadata.get("color_mapping")
        palette = None
        if isinstance(color_mapping, dict) and color_mapping and hue_col_meta:
             # Ensure mapping keys exist in the hue column data
             valid_mapping = {k: v for k, v in color_mapping.items() if k in df[hue_col_meta].unique()}
             if valid_mapping:
                 palette = valid_mapping
                 logger.debug(f"Using provided color mapping (palette) for hue: {palette}")
             else:
                 logger.warning(f"Provided color_mapping keys do not match hue categories ('{hue_col_meta}'). Using default palette.")

        actual_cols = df.columns.tolist()
        x_col = x_col_meta if x_col_meta in actual_cols else actual_cols[0] if len(actual_cols) > 0 else None
        y_col = y_col_meta if y_col_meta in actual_cols else actual_cols[1] if len(actual_cols) > 1 else None
        hue_col = hue_col_meta if hue_col_meta and hue_col_meta in actual_cols else None

        if x_col != x_col_meta or y_col != y_col_meta:
             logger.warning(f"Metadata columns ('{x_col_meta}', '{y_col_meta}') invalid. Falling back to ('{x_col}', '{y_col}').")

        if not x_col or not y_col:
            raise ValueError("Could not determine valid x and y columns for line chart")

        logger.debug(f"Rendering line chart with x='{x_col}', y='{y_col}', hue='{hue_col}', palette='{bool(palette)}'")
        try:
            # Convert x_col to numeric or datetime if possible for better plotting
            try:
                df[x_col] = pd.to_datetime(df[x_col], errors='ignore')
            except Exception: pass # Ignore if conversion fails
            try:
                df[x_col] = pd.to_numeric(df[x_col], errors='ignore')
            except Exception: pass 
            
            # Pass palette to lineplot
            sns.lineplot(data=df, x=x_col, y=y_col, hue=hue_col, marker='o', ax=ax, palette=palette)
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
        except Exception as e:
            logger.error(f"Error rendering line chart with Seaborn: {e}", exc_info=True)
            raise ValueError(f"Failed to render line chart: {e}")
    
    def _render_scatter_chart(self, ax: plt.Axes, df: pd.DataFrame, metadata: Dict[str, Any]):
        """Render a scatter chart using Seaborn on the provided Axes."""
        x_col_meta = metadata.get("x_column")
        y_col_meta = metadata.get("y_column")
        hue_col_meta = metadata.get("color_column") # Use hue for color

        # Get color mapping from metadata if provided (used if hue_col is set)
        color_mapping = metadata.get("color_mapping")
        palette = None
        if isinstance(color_mapping, dict) and color_mapping and hue_col_meta:
             # Ensure mapping keys exist in the hue column data
             valid_mapping = {k: v for k, v in color_mapping.items() if k in df[hue_col_meta].unique()}
             if valid_mapping:
                 palette = valid_mapping
                 logger.debug(f"Using provided color mapping (palette) for hue: {palette}")
             else:
                 logger.warning(f"Provided color_mapping keys do not match hue categories ('{hue_col_meta}'). Using default palette.")

        actual_cols = df.columns.tolist()
        x_col = x_col_meta if x_col_meta in actual_cols else actual_cols[0] if len(actual_cols) > 0 else None
        y_col = y_col_meta if y_col_meta in actual_cols else actual_cols[1] if len(actual_cols) > 1 else None
        hue_col = hue_col_meta if hue_col_meta and hue_col_meta in actual_cols else None

        if x_col != x_col_meta or y_col != y_col_meta:
             logger.warning(f"Metadata columns ('{x_col_meta}', '{y_col_meta}') invalid. Falling back to ('{x_col}', '{y_col}').")

        if not x_col or not y_col:
            raise ValueError("Could not determine valid x and y columns for scatter chart")

        logger.debug(f"Rendering scatter chart with x='{x_col}', y='{y_col}', hue='{hue_col}', palette='{bool(palette)}'")
        try:
            # Pass palette to scatterplot
            sns.scatterplot(data=df, x=x_col, y=y_col, hue=hue_col, ax=ax, palette=palette)
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
        except Exception as e:
            logger.error(f"Error rendering scatter chart with Seaborn: {e}", exc_info=True)
            raise ValueError(f"Failed to render scatter chart: {e}")
    
    def _run(
        self,
        query: Optional[str] = None, # Keep query for potential future internal use or logging
        data: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Generate chart from provided data and metadata."""
        logger.info("Executing chart renderer tool (Matplotlib/Seaborn)")
        
        # --- Input Validation --- 
        if not isinstance(data, dict) or "columns" not in data or "rows" not in data:
            logger.error(f"Invalid 'data' format received. Expected dict with 'columns' and 'rows'. Got: {type(data)}")
            raise ValueError("Invalid 'data' format. Must contain 'columns' and 'rows'.")
        
        if not isinstance(metadata, dict):
            logger.error(f"Invalid or missing 'metadata'. Expected dict. Got: {type(metadata)}")
            # Attempt internal generation if query is available?
            # For now, let's rely on agent providing it as per new prompt
            raise ValueError("Missing or invalid 'metadata' dictionary.")

        # --- Data Preparation --- 
        try:
            df = self._create_dataframe(data)
            if df.empty:
                 raise ValueError("Cannot generate chart from empty data.")
        except ValueError as e:
             logger.error(f"Data processing error: {e}", exc_info=True)
             # Return error structure that async_tools_node_handler expects?
             # For now, re-raise, agent should get failure message.
             raise

        # --- Automatic Transformation for Comparative Bar Charts --- 
        final_df = df
        final_metadata = metadata.copy()
        chart_type = final_metadata.get("chart_type", "bar").lower()
        if chart_type == 'bar':
            # Check if data/metadata suggests a multi-metric comparison
            if final_metadata.get("color_column") or len([col for col in df.columns if col != final_metadata.get("x_column") and pd.api.types.is_numeric_dtype(df[col])]) > 1:
                logger.debug("Comparison bar chart detected, attempting transformation...")
                final_df, final_metadata = self._transform_for_comparison_bar(df, final_metadata)
        
        # --- Chart Rendering --- 
        fig, ax = plt.subplots(figsize=(12, 7)) # Adjusted size for better readability
        chart_type = final_metadata.get("chart_type", "bar").lower()
        logger.debug(f"Selected chart type: {chart_type}")

        render_func = None
        if chart_type == "bar":
            render_func = self._render_bar_chart
        elif chart_type == "pie":
            render_func = self._render_pie_chart
        elif chart_type == "line":
            render_func = self._render_line_chart
        elif chart_type == "scatter":
            render_func = self._render_scatter_chart
        else:
            logger.warning(f"Unsupported chart type '{chart_type}'. Defaulting to bar chart.")
            render_func = self._render_bar_chart
            final_metadata["chart_type"] = "bar" # Update metadata

        try:
            render_func(ax, final_df, final_metadata)
            
            # Apply common styling
            ax.set_title(final_metadata.get("title", "Generated Chart"), fontsize=16, weight='bold')
            # Use labels from metadata if provided, otherwise use column names
            ax.set_xlabel(final_metadata.get("x_label", final_metadata.get("x_column")), fontsize=12)
            ax.set_ylabel(final_metadata.get("y_label", final_metadata.get("y_column")), fontsize=12)
            plt.xticks(rotation=45, ha='right') # Rotate x-axis labels for readability
            plt.tight_layout() # Adjust layout

            # --- Save Figure --- 
            charts_dir = Path(settings.CHART_DIR)
            charts_dir.mkdir(parents=True, exist_ok=True)
            chart_filename = f"chart_{uuid.uuid4()}.png"
            chart_path = charts_dir / chart_filename
            
            logger.debug("Adjusting layout and saving figure...")
            plt.savefig(chart_path, format='png', bbox_inches='tight')
            logger.info(f"Chart saved successfully to {chart_path}")
            
            chart_url = f"{settings.CHART_URL_BASE}/{chart_filename}"
            
            return {
                "visualization": {
                    "type": final_metadata.get("chart_type", "unknown"),
                    "title": final_metadata.get("title", "Generated Chart"),
                    "image_url": chart_url,
                    "description": final_metadata.get("description", "")
                }
            }

        except ValueError as e:
            logger.error(f"Error rendering chart: {e}", exc_info=True)
            # Return a structured error message?
            # For now, re-raise the error to be caught by the tool handler
            raise
        finally:
            # Ensure the plot is closed to free memory
            plt.close(fig)
            logger.debug("Closed Matplotlib figure.")

    async def _arun(
        self,
        query: Optional[str] = None,
        data: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Async wrapper for _run using run_in_executor."""
        logger.debug("Chart renderer _arun called, invoking _run...")
        loop = asyncio.get_event_loop()
        try:
            result = await loop.run_in_executor(
                None, # Use default executor
                self._run, # Target function
                query, data, metadata # Arguments for _run
            )
            return result
        except Exception as e:
             logger.error(f"Chart renderer tool failed: {str(e)}", exc_info=True)
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