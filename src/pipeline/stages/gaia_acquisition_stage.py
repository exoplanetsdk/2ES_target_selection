"""
Gaia data acquisition stage - implements the original Gaia query functionality.
"""
import pandas as pd
from ..base_simple import PipelineStage
from core.exceptions import PipelineError
from . import import_helpers  # This sets up the archive imports
from gaia_queries import get_dr2_query, get_dr3_query, get_crossmatch_query
from utils import execute_gaia_query


class GaiaAcquisitionStage(PipelineStage):
    """Pipeline stage for acquiring Gaia DR2 and DR3 data."""
    
    def validate_input(self, data):
        """Allow None input for initial data acquisition stage."""
        return True  # This stage creates initial data, so None input is valid
    
    def process(self, data):
        """Execute Gaia queries and return merged results."""
        self.logger.info("Starting Gaia data acquisition")
        
        try:
            # Execute DR2 query
            self.logger.info("Querying Gaia DR2 data")
            df_dr2 = execute_gaia_query(
                get_dr2_query(),
                str_columns=['source_id'],
                output_file=f"{self.config.paths.results_dir}/dr2_results.xlsx"
            )
            
            # Get DR2 source IDs for crossmatch
            dr2_source_ids = tuple(df_dr2['source_id'])
            
            # Execute crossmatch query
            self.logger.info("Crossmatching DR2 and DR3 data")
            df_crossmatch = execute_gaia_query(
                get_crossmatch_query(dr2_source_ids),
                str_columns=['dr2_source_id', 'dr3_source_id'],
                output_file=f"{self.config.paths.results_dir}/dr2_dr3_crossmatch.xlsx"
            )
            
            # Execute DR3 query
            self.logger.info("Querying Gaia DR3 data")
            df_dr3 = execute_gaia_query(
                get_dr3_query(),
                str_columns=['source_id'],
                output_file=f"{self.config.paths.results_dir}/dr3_results.xlsx"
            )
            
            # Process and merge the data
            from data_processing import process_gaia_data
            merged_results = process_gaia_data(df_dr2, df_dr3, df_crossmatch)
            
            self.logger.info(f"Gaia acquisition completed. Retrieved {len(merged_results)} stars")
            return merged_results
            
        except Exception as e:
            self.logger.error(f"Gaia acquisition failed: {e}")
            raise PipelineError(f"Gaia acquisition failed: {e}") from e
