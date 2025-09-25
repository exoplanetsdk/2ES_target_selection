"""
Data consolidation stage for the pipeline.
"""
import pandas as pd
from ..base import PipelineStage
from data.processing.data_consolidator import DataConsolidator
from core.exceptions import DataValidationError


class DataConsolidationStage(PipelineStage):
    """Pipeline stage for consolidating Gaia data."""
    
    def __init__(self, config):
        super().__init__(config, "DataConsolidation")
        self.consolidator = DataConsolidator(config)
    
    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Consolidate Gaia DR2/DR3 data.
        
        Args:
            data: Merged DataFrame from Gaia queries
            
        Returns:
            Consolidated DataFrame
        """
        try:
            # Run consolidation
            consolidated_data = self.consolidator.consolidate_data(data)
            
            # Get and log statistics
            stats = self.consolidator.get_consolidation_stats(consolidated_data)
            self.logger.info(f"Consolidation stats: {stats}")
            
            # Save intermediate results
            output_file = f"{self.config.paths.results_dir}/consolidated_results.xlsx"
            consolidated_data.to_excel(output_file, index=False)
            self.logger.info(f"Saved consolidated data to {output_file}")
            
            return consolidated_data
            
        except Exception as e:
            self.logger.error(f"Data consolidation failed: {str(e)}")
            raise DataValidationError(f"Data consolidation failed: {str(e)}") from e
    
    def validate_input(self, data: pd.DataFrame) -> bool:
        """Validate input for consolidation stage."""
        if not super().validate_input(data):
            return False
        
        # Check for required columns
        required_columns = ['source_id_dr2', 'source_id_dr3']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            self.logger.error(f"Missing required columns for consolidation: {missing_columns}")
            return False
        
        return True
