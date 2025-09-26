"""
Ralf comparison stage - implements the original Ralf comparison functionality.
"""
import pandas as pd
from ..base_simple import PipelineStage
from core.exceptions import PipelineError
from . import import_helpers  # This sets up the archive imports
from data_processing import merge_and_format_stellar_data
from plotting import (
    plot_scatter_with_options,
    plot_RV_precision_HZ_detection_limit_vs_temperature
)


class RalfComparisonStage(PipelineStage):
    """Pipeline stage for comparing results with Ralf's data."""
    
    def process(self, data):
        """Compare results with Ralf's data and create comparison plots."""
        self.logger.info("Starting Ralf comparison")
        
        try:
            # Merge with Ralf's data
            self.logger.info("Merging with Ralf's results")
            merged_RJ, df_Ralf = merge_and_format_stellar_data(
                df_main=data,
                ralf_file_path=self.config.file_paths.ralf_file
            )
            
            # Calculate HZ Rmid
            merged_RJ['HZ Rmid'] = (merged_RJ['HZ Rin'] + merged_RJ['HZ Rout']) / 2
            
            # Create comparison plots
            self.logger.info("Creating comparison plots")
            plot_scatter_with_options(merged_RJ, 'magV     ', 'V_mag', min_value=3, max_value=10)
            plot_scatter_with_options(merged_RJ, 'mass ', 'Mass [M_Sun]', min_value=0.5, max_value=1.5)
            plot_scatter_with_options(merged_RJ, 'HZ Rmid', 'HZ_limit [AU]', min_value=0.1, max_value=2, label=True)
            plot_scatter_with_options(merged_RJ, 'logg', 'logg_gaia', min_value=1.9, max_value=5, label=True)
            plot_scatter_with_options(merged_RJ, 'RV_Prec(390-870) 30m', 'σ_photon [m/s]', min_value=0, max_value=1.6, label=True)
            plot_scatter_with_options(merged_RJ, 'mdl(hz) 30min', 'HZ Detection Limit [M_Earth]', min_value=0, max_value=3, label=True)
            
            # Create RV precision vs temperature plot
            plot_RV_precision_HZ_detection_limit_vs_temperature(data, df_Ralf)
            
            # Print comparison statistics
            self.logger.info("Ralf's results comparison:")
            self.logger.info(f"Number of stars: {len(merged_RJ)}")
            for detection_limit in self.config.detection_limits:
                if detection_limit is not None:
                    count_ralf = (merged_RJ['mdl(hz) 30min'] < detection_limit).sum()
                    self.logger.info(f"Number with HZ Detection Limit [M_Earth] < {detection_limit}: {count_ralf}")
            
            self.logger.info(f"Ralf comparison completed. {len(merged_RJ)} stars compared")
            return data  # Return original data, not merged_RJ
            
        except Exception as e:
            self.logger.error(f"Ralf comparison failed: {e}")
            raise PipelineError(f"Ralf comparison failed: {e}") from e
