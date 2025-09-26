"""
Filtering stage - implements the original stellar filtering functionality.
"""
import pandas as pd
from ..base_simple import PipelineStage
from core.exceptions import PipelineError
from . import import_helpers  # This sets up the archive imports
from filtering import filter_stellar_data
from plotting import plot_color_histogram, plot_color_magnitude_diagram


class FilteringStage(PipelineStage):
    """Pipeline stage for filtering stellar data."""
    
    def process(self, data):
        """Filter stellar data and create color-magnitude diagrams."""
        self.logger.info("Starting stellar data filtering")
        
        try:
            # Apply stellar filters
            self.logger.info("Applying stellar filters")
            df_filtered = filter_stellar_data(data, self.config.get_stellar_filters_dict())
            
            # Create color histogram
            self.logger.info("Creating color histogram")
            plot_color_histogram(
                df_filtered,
                output_path=f'{self.config.paths.figures_dir}/color_histogram.png',
                show_plot=False
            )
            
            # Create color-magnitude diagram
            self.logger.info("Creating color-magnitude diagram")
            combined_df = plot_color_magnitude_diagram(
                df_filtered,
                output_path=f'{self.config.paths.figures_dir}/color_magnitude_diagram.png',
                show_plot=False
            )
            
            self.logger.info(f"Filtering completed. {len(combined_df)} stars passed filters")
            return combined_df
            
        except Exception as e:
            self.logger.error(f"Filtering failed: {e}")
            raise PipelineError(f"Filtering failed: {e}") from e
