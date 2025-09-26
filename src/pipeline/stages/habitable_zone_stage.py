"""
Habitable zone and noise modeling stage - implements the original habitable zone calculations.
"""
import pandas as pd
from ..base_simple import PipelineStage
from core.exceptions import PipelineError
from . import import_helpers  # This sets up the archive imports
from data_processing import (
    calculate_and_insert_habitable_zone,
    calculate_and_insert_photon_noise,
    calculate_hz_orbital_period,
    add_granulation_to_dataframe,
    add_pmode_rms_to_dataframe,
    calculate_and_insert_RV_noise,
    calculate_K,
    calculate_and_insert_hz_detection_limit
)


class HabitableZoneStage(PipelineStage):
    """Pipeline stage for habitable zone and noise modeling calculations."""
    
    def process(self, data):
        """Calculate habitable zones, noise models, and detection limits."""
        self.logger.info("Starting habitable zone and noise modeling")
        
        try:
            df = data.copy()
            
            # Calculate habitable zone
            self.logger.info("Calculating habitable zones")
            df = calculate_and_insert_habitable_zone(df)
            
            # Calculate photon noise
            self.logger.info("Calculating photon noise")
            df = calculate_and_insert_photon_noise(df)
            
            # Calculate orbital period
            self.logger.info("Calculating orbital periods")
            df = calculate_hz_orbital_period(df)
            
            # Add granulation noise
            self.logger.info("Adding granulation noise")
            df = add_granulation_to_dataframe(df)
            
            # Add p-mode noise
            self.logger.info("Adding p-mode noise")
            df = add_pmode_rms_to_dataframe(df)
            
            # Calculate total RV noise
            self.logger.info("Calculating total RV noise")
            df = calculate_and_insert_RV_noise(df)
            
            # Calculate K values
            self.logger.info("Calculating K values")
            df = calculate_K(df)
            df = calculate_K(df, sigma_rv_col='σ_RV,total [m/s]')
            
            # Calculate detection limits
            self.logger.info("Calculating detection limits")
            df = calculate_and_insert_hz_detection_limit(
                df,
                semi_amplitude_col='K_σ_photon [m/s]'
            )
            df = calculate_and_insert_hz_detection_limit(
                df,
                semi_amplitude_col='K_σ_RV_total [m/s]'
            )
            
            self.logger.info(f"Habitable zone calculations completed. {len(df)} stars processed")
            return df
            
        except Exception as e:
            self.logger.error(f"Habitable zone calculations failed: {e}")
            raise PipelineError(f"Habitable zone calculations failed: {e}") from e
