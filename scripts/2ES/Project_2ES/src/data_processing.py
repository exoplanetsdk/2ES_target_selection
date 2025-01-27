import pandas as pd
import numpy as np
from config import *

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
    
    return merged_results

def clean_merged_results(merged_results):
    # Your existing cleaning logic here
    # ...

def consolidate_data(clean_merged_results):
    # Your existing consolidation logic here
    # ...





#------------------------------------------------------------------------------------------------

def check_dr3_availability(row):
    '''
    Function to check if any DR3 data is available for a given row.
    '''
    dr3_columns = [col for col in row.index if col.endswith('_dr3')]
    return not row[dr3_columns].isnull().all()

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

def clean_merged_results(merged_results):
    # Step 1: Identify repeated dr2_source_id entries
    non_empty_dr2 = merged_results[merged_results['dr2_source_id'].notna() & (merged_results['dr2_source_id'] != '')]
    repeated_dr2_ids = non_empty_dr2[non_empty_dr2.duplicated('dr2_source_id', keep=False)]['dr2_source_id'].unique()
    repeated_entries = merged_results[merged_results['dr2_source_id'].isin(repeated_dr2_ids)]

    # Save repeated entries to Excel
    repeated_entries.to_excel(RESULTS_DIRECTORY + 'repeated_entries.xlsx', index=False)

    # Process repeated entries and get indices of rows to remove
    rows_to_remove_indices = repeated_entries.groupby('dr2_source_id').apply(process_repeated_group).sum()

    # Get the rows to be removed
    rows_to_remove = repeated_entries.loc[rows_to_remove_indices]

    # Remove the identified rows from merged_results
    clean_merged_results = merged_results[~merged_results.index.isin(rows_to_remove.index)]

    # Reset index of clean_merged_results
    clean_merged_results = clean_merged_results.reset_index(drop=True)

    return clean_merged_results, rows_to_remove

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

    return df



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