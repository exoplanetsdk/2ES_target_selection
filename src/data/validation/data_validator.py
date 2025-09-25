"""
Data validation utilities for the 2ES target selection pipeline.
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from core.exceptions import DataValidationError


class DataValidator:
    """Class for validating data at various stages of the pipeline."""
    
    @staticmethod
    def validate_stellar_data(df: pd.DataFrame) -> bool:
        """
        Validate that DataFrame contains required stellar data columns.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            True if valid, False otherwise
        """
        required_columns = [
            'T_eff [K]', 'Mass [M_Sun]', 'Luminosity [L_Sun]', 
            'Radius [R_Sun]', 'source_id'
        ]
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise DataValidationError(f"Missing required columns: {missing_columns}")
        
        return True
    
    @staticmethod
    def validate_temperature_range(df: pd.DataFrame) -> bool:
        """
        Validate that temperatures are within reasonable range.
        
        Args:
            df: DataFrame with temperature data
            
        Returns:
            True if valid, False otherwise
        """
        if 'T_eff [K]' not in df.columns:
            raise DataValidationError("Temperature column not found")
        
        temps = pd.to_numeric(df['T_eff [K]'], errors='coerce')
        valid_temps = temps.dropna()
        
        if len(valid_temps) == 0:
            raise DataValidationError("No valid temperature data found")
        
        if (valid_temps < 2000).any() or (valid_temps > 10000).any():
            raise DataValidationError(
                f"Temperature values outside reasonable range (2000-10000 K): "
                f"min={valid_temps.min():.0f}, max={valid_temps.max():.0f}"
            )
        
        return True
    
    @staticmethod
    def validate_magnitude_range(df: pd.DataFrame) -> bool:
        """
        Validate that magnitudes are within reasonable range.
        
        Args:
            df: DataFrame with magnitude data
            
        Returns:
            True if valid, False otherwise
        """
        magnitude_columns = ['Phot G Mean Mag', 'V_mag']
        
        for col in magnitude_columns:
            if col in df.columns:
                mags = pd.to_numeric(df[col], errors='coerce')
                valid_mags = mags.dropna()
                
                if len(valid_mags) > 0:
                    if (valid_mags < -5).any() or (valid_mags > 20).any():
                        raise DataValidationError(
                            f"Magnitude values outside reasonable range (-5 to 20): "
                            f"min={valid_mags.min():.2f}, max={valid_mags.max():.2f}"
                        )
        
        return True
    
    @staticmethod
    def validate_parallax_data(df: pd.DataFrame) -> bool:
        """
        Validate parallax data for distance calculations.
        
        Args:
            df: DataFrame with parallax data
            
        Returns:
            True if valid, False otherwise
        """
        if 'Parallax' not in df.columns:
            return True  # Parallax not required for all analyses
        
        parallax = pd.to_numeric(df['Parallax'], errors='coerce')
        valid_parallax = parallax.dropna()
        
        if len(valid_parallax) > 0:
            # Check for negative parallax (unphysical)
            negative_count = (valid_parallax < 0).sum()
            if negative_count > 0:
                print(f"Warning: {negative_count} stars have negative parallax values")
            
            # Check for very small parallax (very distant stars)
            very_small = (valid_parallax < 0.1).sum()
            if very_small > 0:
                print(f"Warning: {very_small} stars have parallax < 0.1 mas (very distant)")
        
        return True
    
    @staticmethod
    def validate_coordinates(df: pd.DataFrame) -> bool:
        """
        Validate RA and DEC coordinates.
        
        Args:
            df: DataFrame with coordinate data
            
        Returns:
            True if valid, False otherwise
        """
        coord_columns = ['RA', 'DEC']
        
        for col in coord_columns:
            if col in df.columns:
                coords = pd.to_numeric(df[col], errors='coerce')
                valid_coords = coords.dropna()
                
                if len(valid_coords) > 0:
                    if col == 'RA':
                        if (valid_coords < 0).any() or (valid_coords > 360).any():
                            raise DataValidationError(
                                f"RA values outside valid range (0-360): "
                                f"min={valid_coords.min():.2f}, max={valid_coords.max():.2f}"
                            )
                    elif col == 'DEC':
                        if (valid_coords < -90).any() or (valid_coords > 90).any():
                            raise DataValidationError(
                                f"DEC values outside valid range (-90 to 90): "
                                f"min={valid_coords.min():.2f}, max={valid_coords.max():.2f}"
                            )
        
        return True
    
    @classmethod
    def validate_complete_dataset(cls, df: pd.DataFrame) -> Dict[str, bool]:
        """
        Run all validation checks on a dataset.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Dictionary with validation results
        """
        results = {}
        
        try:
            results['stellar_data'] = cls.validate_stellar_data(df)
        except DataValidationError as e:
            results['stellar_data'] = False
            print(f"Stellar data validation failed: {e}")
        
        try:
            results['temperature_range'] = cls.validate_temperature_range(df)
        except DataValidationError as e:
            results['temperature_range'] = False
            print(f"Temperature validation failed: {e}")
        
        try:
            results['magnitude_range'] = cls.validate_magnitude_range(df)
        except DataValidationError as e:
            results['magnitude_range'] = False
            print(f"Magnitude validation failed: {e}")
        
        try:
            results['parallax_data'] = cls.validate_parallax_data(df)
        except DataValidationError as e:
            results['parallax_data'] = False
            print(f"Parallax validation failed: {e}")
        
        try:
            results['coordinates'] = cls.validate_coordinates(df)
        except DataValidationError as e:
            results['coordinates'] = False
            print(f"Coordinate validation failed: {e}")
        
        return results
