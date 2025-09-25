"""
Data cleaning stage - implements the original data cleaning functionality.
"""
import pandas as pd
from ..base_simple import PipelineStage
from core.exceptions import PipelineError
from data_processing import clean_merged_results


class DataCleaningStage(PipelineStage):
    """Pipeline stage for cleaning merged Gaia data."""
    
    def process(self, data):
        """Clean the merged Gaia data by removing duplicates."""
        self.logger.info("Starting data cleaning")
        
        try:
            # Clean merged results (remove duplicates)
            clean_results = clean_merged_results(data)
            
            self.logger.info(f"Data cleaning completed. {len(clean_results)} stars remaining")
            return clean_results
            
        except Exception as e:
            self.logger.error(f"Data cleaning failed: {e}")
            raise PipelineError(f"Data cleaning failed: {e}") from e
