import pandas as pd
from config import RESULTS_DIRECTORY


def filter_stellar_data(df, config):
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
            }
        output_directory (str, optional): Directory to save filtered results
            
    Returns:
        pd.DataFrame: Filtered DataFrame
        dict: Statistics about the filtering process
    """
    # Create a copy to avoid modifying the original
    df_filtered = df.copy()

    columns_to_convert = ['T_eff [K]', 'Mass [M_Sun]', 'Luminosity [L_Sun]', 'Radius [R_Sun]', 'logg_gaia']
    for column in columns_to_convert:
        df_filtered[column] = pd.to_numeric(df_filtered[column], errors='coerce')

    # Initial statistics for data availability
    stats_dict = {
        'total_initial': len(df),
        'has_temp_and_lum': 0,
        'has_all_params': 0
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

    display(stats_dict)
    
    return df_filtered, df_removed, stats_dict

