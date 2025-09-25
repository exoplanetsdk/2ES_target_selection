"""
Data consolidation utilities for merging and cleaning Gaia data.
"""
import pandas as pd
import numpy as np
from typing import Tuple, List
from tqdm import tqdm
from core.exceptions import DataValidationError
from core.logging_config import get_logger

logger = get_logger(__name__)


class DataConsolidator:
    """Class for consolidating and merging Gaia DR2/DR3 data."""
    
    def __init__(self, config):
        self.config = config
    
    def consolidate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Main method to consolidate Gaia DR2 and DR3 data.
        
        Args:
            df: Merged DataFrame from DR2/DR3
            
        Returns:
            Consolidated DataFrame with cleaned and merged data
        """
        logger.info("Starting data consolidation process")
        
        # Step 1: Merge DR2 and DR3 columns
        df_merged = self._merge_gaia_columns(df)
        
        # Step 2: Add source identifiers
        df_with_ids = self._add_source_identifiers(df_merged)
        
        # Step 3: Add SIMBAD information
        df_with_simbad = self._add_simbad_info(df_with_ids)
        
        # Step 4: Filter object types
        df_filtered = self._filter_object_types(df_with_simbad)
        
        # Step 5: Rename columns
        df_final = self._rename_columns(df_filtered)
        
        # Step 6: Convert numeric columns
        df_final = self._convert_numeric_columns(df_final)
        
        # Step 7: Select only final columns (drop intermediate DR2/DR3 columns)
        df_final = self._select_final_columns(df_final)
        
        logger.info(f"Data consolidation completed. Final dataset: {len(df_final)} stars")
        return df_final
    
    def _merge_gaia_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Merge DR2 and DR3 columns, preferring DR3 when available."""
        logger.info("Merging Gaia DR2 and DR3 columns")
        
        def choose_value(row, col_name: str) -> any:
            """Choose DR3 value if available, otherwise use DR2."""
            dr3_col = f'{col_name}_dr3'
            dr2_col = f'{col_name}_dr2'
            return row[dr3_col] if pd.notnull(row[dr3_col]) else row[dr2_col]
        
        # List of columns to process
        columns_to_process = [
            'ra', 'dec', 'phot_g_mean_mag', 'phot_bp_mean_mag', 
            'phot_rp_mean_mag', 'parallax', 'logg_gaia', 'spectraltype_esphs'
        ]
        
        logger.info(f"Merging columns: {', '.join(columns_to_process)}")
        
        for col in columns_to_process:
            df[col] = df.apply(lambda row: choose_value(row, col), axis=1)
        
        # Special handling for temperature
        logger.info("Processing temperature data")
        df['T_eff [K]'] = df.apply(
            lambda row: row['teff_gspphot'] if pd.notnull(row['teff_gspphot']) 
            else row['teff_val'], axis=1
        )
        
        # Process other columns
        other_columns = ['mass_flame', 'lum_flame', 'radius_flame', 'spectraltype_esphs']
        logger.info(f"Processing additional columns: {', '.join(other_columns)}")
        
        for col in other_columns:
            df[col] = df.apply(lambda row: choose_value(row, col), axis=1)
        
        # Add bp_rp column from DR3 if available
        logger.info("Adding bp_rp color column")
        df['bp_rp'] = df['bp_rp'].fillna(df['phot_bp_mean_mag'] - df['phot_rp_mean_mag'])
        
        return df
    
    def _add_source_identifiers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Gaia source ID column."""
        logger.info("Adding Gaia source identifiers")
        
        df['source_id'] = np.where(
            pd.notna(df['source_id_dr3']),
            'Gaia DR3 ' + df['source_id_dr3'].astype('Int64').astype(str),
            'Gaia DR2 ' + df['source_id_dr2'].astype('Int64').astype(str)
        )
        
        return df
    
    def _add_simbad_info(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add SIMBAD information (HD, GJ, HIP numbers and object type)."""
        logger.info("Retrieving SIMBAD information for stellar identifiers")
        
        # Import here to avoid circular imports
        from stellar_properties import get_simbad_info_with_retry
        
        # Create new DataFrame for SIMBAD data
        new_columns = ['HD Number', 'GJ Number', 'HIP Number', 'Object Type']
        df_new = pd.DataFrame(columns=new_columns)
        
        # Process each star
        for index, row in tqdm(df.iterrows(), total=len(df), 
                              desc="Retrieving SIMBAD data", ncols=100):
            simbad_info = get_simbad_info_with_retry(row['source_id'])
            if simbad_info:
                df_new.loc[index, 'HD Number'] = simbad_info['HD Number']
                df_new.loc[index, 'GJ Number'] = simbad_info['GJ Number']
                df_new.loc[index, 'HIP Number'] = simbad_info['HIP Number']
                df_new.loc[index, 'Object Type'] = simbad_info['Object Type']
        
        # Combine with original DataFrame
        df_combined = pd.concat([df, df_new[new_columns]], axis=1)
        
        return df_combined
    
    def _filter_object_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter to keep only stellar objects."""
        logger.info("Filtering object types")
        
        allowed_types = ['*', '**', 'MS*', 'SB*', 'PM*']
        original_count = len(df)
        
        df_filtered = df[df['Object Type'].isin(allowed_types)]
        
        filtered_count = len(df_filtered)
        logger.info(f"Filtered from {original_count} to {filtered_count} stellar objects")
        
        return df_filtered
    
    def _rename_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rename columns to standard format."""
        logger.info("Renaming columns to standard format")
        
        column_mapping = {
            'mass_flame': 'Mass [M_Sun]',
            'lum_flame': 'Luminosity [L_Sun]',
            'radius_flame': 'Radius [R_Sun]',
            'phot_g_mean_mag': 'Phot G Mean Mag',
            'phot_bp_mean_mag': 'Phot BP Mean Mag',
            'phot_rp_mean_mag': 'Phot RP Mean Mag',
            'bp_rp': 'BP-RP',
            'parallax': 'Parallax',
            'ra': 'RA',
            'dec': 'DEC',
            'spectraltype_esphs': 'Gaia Spectral type'
        }
        
        df_renamed = df.rename(columns=column_mapping)
        
        return df_renamed
    
    def _convert_numeric_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert columns to numeric types."""
        logger.info("Converting columns to numeric types")
        
        numeric_columns = ['T_eff [K]', 'Mass [M_Sun]', 'Luminosity [L_Sun]', 'Radius [R_Sun]']
        
        for column in numeric_columns:
            if column in df.columns:
                df[column] = pd.to_numeric(df[column], errors='coerce')
                logger.info(f"Converted {column} to numeric")
        
        return df
    
    def _select_final_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Select only the final consolidated columns, dropping intermediate DR2/DR3 columns.
        This matches the original consolidate_data function behavior.
        """
        logger.info("Selecting final consolidated columns")
        
        # Define the final columns that should be kept (matching original consolidate_data)
        final_columns = [
            'source_id', 'source_id_dr2', 'source_id_dr3', 
            'RA', 'DEC', 'Phot G Mean Mag', 'Phot BP Mean Mag', 'Phot RP Mean Mag', 
            'BP-RP', 'Parallax', 'T_eff [K]', 'Mass [M_Sun]', 'Luminosity [L_Sun]', 
            'Radius [R_Sun]', 'logg_gaia', 'Gaia Spectral type',
            'HD Number', 'GJ Number', 'HIP Number', 'Object Type'
        ]
        
        # Keep only columns that exist in the dataframe
        existing_columns = [col for col in final_columns if col in df.columns]
        missing_columns = [col for col in final_columns if col not in df.columns]
        
        if missing_columns:
            logger.warning(f"Missing expected columns: {missing_columns}")
        
        # Select only the final columns
        df_final = df[existing_columns].copy()
        
        logger.info(f"Selected {len(existing_columns)} final columns, dropped {len(df.columns) - len(existing_columns)} intermediate columns")
        
        return df_final
    
    def get_consolidation_stats(self, df: pd.DataFrame) -> dict:
        """Get statistics about the consolidation process."""
        stats = {
            'total_stars': len(df),
            'dr3_stars': df['source_id_dr3'].notna().sum(),
            'dr2_only_stars': df['source_id_dr3'].isna().sum(),
            'hd_numbers': df['HD Number'].notna().sum(),
            'gj_numbers': df['GJ Number'].notna().sum(),
            'hip_numbers': df['HIP Number'].notna().sum()
        }
        
        return stats
