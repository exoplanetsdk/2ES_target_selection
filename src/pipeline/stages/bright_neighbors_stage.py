"""
Bright neighbors analysis stage - implements the original bright neighbors functionality.
"""
import pandas as pd
from ..base_simple import PipelineStage
from core.exceptions import PipelineError
from data_processing import analyze_bright_neighbors
from utils import execute_gaia_query


class BrightNeighborsStage(PipelineStage):
    """Pipeline stage for analyzing bright neighboring stars."""
    
    def process(self, data):
        """Analyze stars for bright neighbors and filter them out."""
        self.logger.info("Starting bright neighbors analysis")
        
        try:
            # Analyze bright neighbors
            self.logger.info("Analyzing bright neighbors")
            df_with_bright_neighbors, df_without_bright_neighbors = analyze_bright_neighbors(
                merged_df=data,
                search_radius=self.config.query_params.search_radius,
                execute_gaia_query_func=execute_gaia_query
            )
            
            # Use stars without bright neighbors
            df_clean = df_without_bright_neighbors.copy()
            
            self.logger.info(f"Bright neighbors analysis completed. {len(df_clean)} stars without bright neighbors")
            return df_clean
            
        except Exception as e:
            self.logger.error(f"Bright neighbors analysis failed: {e}")
            raise PipelineError(f"Bright neighbors analysis failed: {e}") from e
