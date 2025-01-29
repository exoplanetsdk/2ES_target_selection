import pandas as pd
import numpy as np
from config import *
from utils import adjust_column_widths
from stellar_properties import get_simbad_info_with_retry
from stellar_calculations import calculate_habitable_zone
def process_gaia_data(df_dr2, df_dr3, df_crossmatch):
    # Merge DR2 and DR3 results
    merged_dr2_crossmatch = pd.merge(df_dr2, df_crossmatch, 
                                    left_on='source_id', 
                                    right_on='dr2_source_id', 
                                    how='left')
    
    merged_results = pd.merge(merged_dr2_crossmatch, df_dr3, 
                            left_on='dr3_source_id', 
                            right_on='source_id', 
                            suffixes=('_dr2', '_dr3'), 
                            how='outer')
    
    merged_results.to_excel(RESULTS_DIRECTORY+'merged_results.xlsx', index=False)
    adjust_column_widths(RESULTS_DIRECTORY+'merged_results.xlsx')

    return merged_results


#------------------------------------------------------------------------------------------------

def check_dr3_availability(row):
    '''
    Function to check if any DR3 data is available for a given row.
    '''
    dr3_columns = [col for col in row.index if col.endswith('_dr3')]
    return not row[dr3_columns].isnull().all()

#------------------------------------------------------------------------------------------------

def process_repeated_group(group):
    '''
    Function to process a group of repeated entries with the same dr2_source_id.
    '''
    if len(group) != 2:
        # If there are more than 2 entries, keep only the first one
        return group.iloc[1:].index.tolist()  # Return indices of rows to remove
    
    row1, row2 = group.iloc[0], group.iloc[1]
    dr3_available1 = check_dr3_availability(row1)
    dr3_available2 = check_dr3_availability(row2)
    
    if not dr3_available1 and not dr3_available2:
        # If both rows have no DR3 data, remove the second one
        return [group.index[1]]
    elif dr3_available1 and not dr3_available2:
        # If only the first row has DR3 data, remove the second one
        return [group.index[1]]
    elif not dr3_available1 and dr3_available2:
        # If only the second row has DR3 data, remove the first one
        return [group.index[0]]
    else:
        # If both rows have DR3 data, remove the second one
        return [group.index[1]]

#------------------------------------------------------------------------------------------------

def clean_merged_results(merged_results):
    # Step 1: Identify repeated dr2_source_id entries
    non_empty_dr2 = merged_results[merged_results['dr2_source_id'].notna() & (merged_results['dr2_source_id'] != '')]
    repeated_dr2_ids = non_empty_dr2[non_empty_dr2.duplicated('dr2_source_id', keep=False)]['dr2_source_id'].unique()
    repeated_entries = merged_results[merged_results['dr2_source_id'].isin(repeated_dr2_ids)]

    # Save repeated entries to Excel
    repeated_entries.to_excel(RESULTS_DIRECTORY + 'repeated_entries.xlsx', index=False)
    adjust_column_widths(RESULTS_DIRECTORY + 'repeated_entries.xlsx')

    # Process repeated entries and get indices of rows to remove
    rows_to_remove_indices = repeated_entries.groupby('dr2_source_id').apply(process_repeated_group).sum()

    # Get the rows to be removed
    rows_to_remove = repeated_entries.loc[rows_to_remove_indices]

    # Remove the identified rows from merged_results
    clean_merged_results = merged_results[~merged_results.index.isin(rows_to_remove.index)]

    # Reset index of clean_merged_results
    clean_merged_results = clean_merged_results.reset_index(drop=True)
    clean_merged_results.to_excel(RESULTS_DIRECTORY + 'clean_merged_results.xlsx', index=False)
    adjust_column_widths(RESULTS_DIRECTORY + 'clean_merged_results.xlsx')

    rows_to_remove.to_excel(RESULTS_DIRECTORY + 'removed_rows.xlsx', index=False)
    adjust_column_widths(RESULTS_DIRECTORY + 'removed_rows.xlsx')

    print(f"Original shape of merged_results: {merged_results.shape}")
    print(f"Shape after removing duplicates: {clean_merged_results.shape}")
    print(f"Number of rows removed: {merged_results.shape[0] - clean_merged_results.shape[0]}")

    remaining_duplicates = clean_merged_results[clean_merged_results.duplicated('dr2_source_id', keep=False)]
    print(f"\nNumber of remaining duplicate dr2_source_id: {len(remaining_duplicates['dr2_source_id'].unique())}")

    if not remaining_duplicates.empty:
        print("\nExample of remaining duplicates:")
        print(remaining_duplicates.groupby('dr2_source_id').first().head())
    else:
        print("\nNo remaining duplicates found.")
    
    return clean_merged_results

#------------------------------------------------------------------------------------------------

def consolidate_data(df):
    def choose_value(row, col_name):
        dr3_col = f'{col_name}_dr3'
        dr2_col = f'{col_name}_dr2'
        return row[dr3_col] if pd.notnull(row[dr3_col]) else row[dr2_col]

    # List of columns to process
    columns_to_process = ['ra', 'dec', 'phot_g_mean_mag', 'phot_bp_mean_mag', 'phot_rp_mean_mag', 
                        'parallax', 'logg_gaia', 'spectraltype_esphs']

    # Process each column
    for col in columns_to_process:
        df[col] = df.apply(lambda row: choose_value(row, col), axis=1)

    if isinstance(df, tuple):
        df = df[0]  # Take the first element if it's a tuple

    # Special handling for temperature
    df['T_eff [K]'] = df.apply(lambda row: row['teff_gspphot'] if pd.notnull(row['teff_gspphot']) 
                              else row['teff_val'], axis=1)

    # For other columns
    other_columns = ['mass_flame', 'lum_flame', 'radius_flame', 'spectraltype_esphs']
    for col in other_columns:
        df[col] = df.apply(lambda row: choose_value(row, col), axis=1)

    # Add bp_rp column from DR3 if available
    df['bp_rp'] = df['bp_rp'].fillna(df['phot_bp_mean_mag'] - df['phot_rp_mean_mag'])

    # Add the new source_id column
    df['source_id'] = df['source_id_dr3'].fillna(df['source_id_dr2'])

    # Update the final columns list to include the new source_id column
    final_columns = ['source_id', 'source_id_dr2', 'source_id_dr3', 'ra', 'dec', 'phot_g_mean_mag', 'phot_bp_mean_mag', 'phot_rp_mean_mag', 
                    'bp_rp', 'parallax', 'T_eff [K]', 'mass_flame', 'lum_flame', 'radius_flame', 'logg_gaia', 'spectraltype_esphs']

    # Create the final dataframe
    df_consolidated = df[final_columns]
    # Create a new DataFrame instead of modifying the existing one
    new_columns = ['source_id', 'HD Number', 'GJ Number', 'HIP Number', 'Object Type']
    df_new = pd.DataFrame(columns=new_columns)

    # Populate the new DataFrame
    df_new['source_id'] = df_consolidated['source_id_dr3'].fillna(df_consolidated['source_id_dr2'])

    for index, row in df_new.iterrows():
        simbad_info = get_simbad_info_with_retry(row['source_id'])
        if simbad_info:
            df_new.loc[index, 'HD Number'] = simbad_info['HD Number']
            df_new.loc[index, 'GJ Number'] = simbad_info['GJ Number']
            df_new.loc[index, 'HIP Number'] = simbad_info['HIP Number']
            df_new.loc[index, 'Object Type'] = simbad_info['Object Type']


    # Combine the new DataFrame with the original one
    df_consolidated = pd.concat([df_consolidated, df_new[['HD Number', 'GJ Number', 'HIP Number', 'Object Type']]], axis=1)

    # Update the final columns list to include the new columns
    final_columns.extend(['HD Number', 'GJ Number', 'HIP Number', 'Object Type'])

    # Create the final dataframe with the updated column list
    df_consolidated = df_consolidated[final_columns]

    # Rename the columns
    df_consolidated = df_consolidated.rename(columns={
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
    })

    # Convert each column to numeric, coercing errors
    columns_to_convert = ['T_eff [K]', 'Mass [M_Sun]', 'Luminosity [L_Sun]', 'Radius [R_Sun]']

    for column in columns_to_convert:
        df_consolidated[column] = pd.to_numeric(df_consolidated[column], errors='coerce')

    # Save the result to a new Excel file
    df_consolidated.to_excel(RESULTS_DIRECTORY + 'consolidated_results.xlsx', index=False)
    adjust_column_widths(RESULTS_DIRECTORY + 'consolidated_results.xlsx')

    # Display some statistics
    print(f"Total number of stars: {len(df_consolidated)}")
    print(f"Number of stars with DR3 source_id: {df_consolidated['source_id_dr3'].notna().sum()}")
    print(f"Number of stars with only DR2 source_id: {df_consolidated['source_id_dr3'].isna().sum()}")
    print(f"Number of stars with HD Number: {df_consolidated['HD Number'].notna().sum()}")
    print(f"Number of stars with GJ Number: {df_consolidated['GJ Number'].notna().sum()}")
    print(f"Number of stars with HIP Number: {df_consolidated['HIP Number'].notna().sum()}")   

    return df_consolidated

#------------------------------------------------------------------------------------------------

def calculate_and_insert_stellar_density(df, mass_col='Mass [M_Sun]', radius_col='Radius [R_Sun]', 
                                       density_col='Density [Solar unit]'):
    """
    Calculate stellar density in solar units and insert it after the radius column.
    
    Args:
        df (pd.DataFrame): Input DataFrame containing stellar data
        mass_col (str): Name of the mass column in solar mass units
        radius_col (str): Name of the radius column in solar radius units
        density_col (str): Name of the new density column to be created
        
    Returns:
        pd.DataFrame: DataFrame with density column added and reordered
    """
    # Create a copy to avoid modifying the original DataFrame
    df = df.copy()
    
    # Initialize density column
    df[density_col] = None
    
    # Calculate density where both mass and radius are available
    mask = df[mass_col].notna() & df[radius_col].notna()
    df.loc[mask, density_col] = df[mass_col] / (df[radius_col] ** 3)
    
    # Reorder columns to place density after radius
    cols = df.columns.tolist()
    density_index = cols.index(density_col)
    radius_index = cols.index(radius_col)
    cols.insert(radius_index + 1, cols.pop(density_index))
    
    return df[cols]

#------------------------------------------------------------------------------------------------

def calculate_and_insert_habitable_zone(df, directory, output_filename='combined_query.xlsx'):
    """
    Process stellar data by calculating habitable zone limits, sorting by temperature,
    and saving to Excel with formatted columns.
    
    Args:
        df (pd.DataFrame): DataFrame containing stellar data
        directory (str): Directory path where the Excel file will be saved
        output_filename (str, optional): Name of the output Excel file. 
            Defaults to 'combined_query.xlsx'
            
    Returns:
        pd.DataFrame: Processed DataFrame with added HZ limits and sorted by temperature
    """
    # Create a copy to avoid modifying the original
    processed_df = df.copy()
    
    # Calculate and add HZ limits if temperature data is available
    if 'T_eff [K]' in processed_df.columns:
        # Calculate the HZ_limit
        hz_limits = processed_df.apply(
            lambda row: calculate_habitable_zone(
                row['T_eff [K]'], 
                row['Luminosity [L_Sun]']
            ), 
            axis=1
        )

        # Add HZ_limit column if it doesn't exist
        if 'HZ_limit [AU]' not in processed_df.columns:
            # Find the index of the 'Radius [R_Sun]' column
            radius_index = processed_df.columns.get_loc('Radius [R_Sun]')
            # Insert the new column after 'Radius [R_Sun]'        
            processed_df.insert(radius_index + 1, 'HZ_limit [AU]', hz_limits)
            
        # Sort the DataFrame by temperature (Teff)
        processed_df = processed_df.sort_values('T_eff [K]')    

    # Save the result to a new Excel file
    output_path = RESULTS_DIRECTORY + 'consolidated_results.xlsx'
    processed_df.to_excel(output_path, index=False)
    adjust_column_widths(output_path)
    print(f"Results saved to {output_path}")

    return processed_df
























def process_background_stars(merged_df):
    from concurrent.futures import ThreadPoolExecutor
    from tqdm import tqdm

    def create_neighbor_query(source_id, ra, dec, neighbor_g_mag_limit, search_radius, data_release):
        query = f"""
        SELECT 
            source_id, ra, dec, phot_g_mean_mag
        FROM 
            {data_release}.gaia_source
        WHERE 
            1=CONTAINS(
                POINT('ICRS', {ra}, {dec}),
                CIRCLE('ICRS', ra, dec, {search_radius})
            )
            AND phot_g_mean_mag < {neighbor_g_mag_limit}
            AND source_id != {source_id}
        """
        return query    
    
    def process_row_with_retry(row, max_retries=3, delay=5):
        attempt = 0
        while attempt < max_retries:
            try:
                if not pd.isna(row['source_id_dr3']):
                    query = create_neighbor_query(
                        source_id=row['source_id_dr3'],
                        ra=row['RA'],
                        dec=row['DEC'],
                        neighbor_g_mag_limit=row['Phot G Mean Mag']+3,
                        search_radius=SEARCH_RADIUS,
                        data_release='gaiadr3'
                    )
                else:
                    query = create_neighbor_query(
                        source_id=row['source_id_dr2'],
                        ra=row['RA'],
                        dec=row['DEC'],
                        neighbor_g_mag_limit=row['Phot G Mean Mag']+3,
                        search_radius=SEARCH_RADIUS,
                        data_release='gaiadr2'
                    )
                
                # Execute the query
                neighbors_df = execute_gaia_query(query)
                
                # Check if bright neighbors exist
                if neighbors_df is not None and not neighbors_df.empty:
                    return (row, True)
                else:
                    return (row, False)
            except Exception as e:
                print(f"An error occurred: {e}")
                attempt += 1
                if attempt < max_retries:
                    print(f"Retrying in {delay} seconds... (Attempt {attempt}/{max_retries})")
                    time.sleep(delay)
                else:
                    print("Max retries reached. Skipping this row.")
                    return (row, False)

    rows_with_bright_neighbors = []
    rows_without_bright_neighbors = []

    with ThreadPoolExecutor() as executor:
        results = list(tqdm(executor.map(process_row_with_retry, 
                                       [row for idx, row in merged_df.iterrows()]), 
                          total=len(merged_df), 
                          desc="Processing rows"))
        
        for row, has_bright_neighbors in results:
            if has_bright_neighbors:
                rows_with_bright_neighbors.append(row)
            else:
                rows_without_bright_neighbors.append(row)

    return pd.DataFrame(rows_with_bright_neighbors), pd.DataFrame(rows_without_bright_neighbors)