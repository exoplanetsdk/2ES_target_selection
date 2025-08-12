import pandas as pd
from config import RESULTS_DIRECTORY
from utils import adjust_column_widths

def filter_stellar_data(df, config):
    print("\nFiltering stars based on stellar parameters")
    """
    Filter stellar data based on various physical parameters and thresholds.
    
    Args:
        df (pd.DataFrame): Input DataFrame containing stellar data
        config (dict): Configuration dictionary containing threshold values:
            {
                'temp_min': Minimum effective temperature in K
                'temp_max': Maximum effective temperature in K
                'lum_min': Minimum luminosity in solar units
                'lum_max': Maximum luminosity in solar units
                'density_min': Minimum density in solar units
                'density_max': Maximum density in solar units
                'logg_min': Minimum log g value
                'log_rhk_max': Maximum log R'HK value (for stellar activity)
            }
        output_directory (str, optional): Directory to save filtered results
            
    Returns:
        pd.DataFrame: Filtered DataFrame
        dict: Statistics about the filtering process
    """
    # Create a copy to avoid modifying the original
    df_filtered = df.copy()

    columns_to_convert = ['T_eff [K]', 'Mass [M_Sun]', 'Luminosity [L_Sun]', 'Radius [R_Sun]', 'logg_gaia', 'log_rhk']
    for column in columns_to_convert:
        if column in df_filtered.columns:
            df_filtered[column] = pd.to_numeric(df_filtered[column], errors='coerce')

    # Initial statistics for data availability
    stats_dict = {
        'total_initial': len(df),
        'has_temp_and_lum': 0,
        'has_all_params': 0,
        'has_log_rhk': 0
    }
    
    # Filter out stars with missing mass or luminosity information
    df_filtered = df_filtered[
        (df_filtered['Luminosity [L_Sun]'].notna()) & 
        (df_filtered['T_eff [K]'].notna())
    ]
    stats_dict['has_temp_and_lum'] = len(df_filtered)
    
    # Check for rows with all parameters available
    non_empty_rows = df[
        df['Luminosity [L_Sun]'].notna() &
        df['Radius [R_Sun]'].notna() &
        df['T_eff [K]'].notna() &
        df['Mass [M_Sun]'].notna()
    ]
    stats_dict['has_all_params'] = len(non_empty_rows)
    
    # Check for rows with log_rhk data
    if 'log_rhk' in df.columns:
        log_rhk_rows = df[df['log_rhk'].notna()]
        stats_dict['has_log_rhk'] = len(log_rhk_rows)
    
    # Apply temperature filter
    df_filtered = df_filtered[
        (df_filtered['T_eff [K]'] >= config['temp_min']) & 
        (df_filtered['T_eff [K]'] <= config['temp_max'])
    ]
    
    # Apply luminosity filter
    df_filtered = df_filtered[
        (df_filtered['Luminosity [L_Sun]'] <= config['lum_max']) &
        (df_filtered['Luminosity [L_Sun]'] >= config['lum_min'])
    ]
    
    # Apply density filter
    df_filtered = df_filtered[
        ((df_filtered['Density [Solar unit]'] >= config['density_min']) & 
         (df_filtered['Density [Solar unit]'] <= config['density_max'])) |
        df_filtered['Radius [R_Sun]'].isna()
    ]
    
    # Apply log g filter
    df_filtered = df_filtered[
        (df_filtered['logg_gaia'] >= config['logg_min']) | 
        (df_filtered['logg_gaia'].isna())
    ]
    
    # Apply log R'HK maximum filter (exclude overly active stars)
    if 'log_rhk' in df_filtered.columns and 'log_rhk_max' in config:
        print(f"Applying log R'HK filter: log_rhk <= {config['log_rhk_max']} (excluding overly active stars)")
        before_log_rhk = len(df_filtered)
        
        # Keep stars with log_rhk <= max OR missing log_rhk data
        df_filtered = df_filtered[
            (df_filtered['log_rhk'] <= config['log_rhk_max']) |
            df_filtered['log_rhk'].isna()
        ]
        
        after_log_rhk = len(df_filtered)
        removed_by_activity = before_log_rhk - after_log_rhk
        print(f"Log R'HK filter removed {removed_by_activity} overly active stars (log R'HK > {config['log_rhk_max']})")
    elif 'log_rhk_max' in config:
        print("Warning: 'log_rhk' column not found in data. Skipping log R'HK filter.")
    
    # Find removed entries
    df_removed = df[~df['source_id'].isin(df_filtered['source_id'])]
    
    # Update statistics
    stats_dict.update({
        'kept': len(df_filtered),
        'removed': len(df_removed)
    })

    # Save filtered data
    filtered_path = f"{RESULTS_DIRECTORY}consolidated_results_kept.xlsx"
    df_filtered.to_excel(filtered_path, index=False)
    adjust_column_widths(filtered_path)
    
    # Save removed data
    removed_path = f"{RESULTS_DIRECTORY}consolidated_results_removed.xlsx"
    df_removed.to_excel(removed_path, index=False)
    adjust_column_widths(removed_path)

    for key, value in stats_dict.items():
        print(f"{key.replace('_', ' ').capitalize()}: {value}")
    
    return df_filtered

