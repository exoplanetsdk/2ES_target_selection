import time
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import os
import json

from config import *
from utils import *
from rv_prec import calculate_rv_precision
from stellar_properties import get_simbad_info_with_retry
from stellar_calculations import *

#------------------------------------------------------------------------------------------------
def process_gaia_data(df_dr2, df_dr3, df_crossmatch):
    print("\nMerging Gaia DR2 and DR3 data with crossmatch information")
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
    
    save_and_adjust_column_widths(merged_results, RESULTS_DIRECTORY+'merged_results.xlsx')

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
    print("\nCleaning merged results by removing duplicate dr2_source_id entries")
    # Step 1: Identify repeated dr2_source_id entries
    non_empty_dr2 = merged_results[merged_results['dr2_source_id'].notna() & (merged_results['dr2_source_id'] != '')]
    repeated_dr2_ids = non_empty_dr2[non_empty_dr2.duplicated('dr2_source_id', keep=False)]['dr2_source_id'].unique()
    repeated_entries = merged_results[merged_results['dr2_source_id'].isin(repeated_dr2_ids)]

    # Save repeated entries to Excel
    save_and_adjust_column_widths(repeated_entries, RESULTS_DIRECTORY + 'repeated_entries.xlsx')

    # Process repeated entries and get indices of rows to remove
    print(f"\nProcessing repeated entries with the same dr2_source_id")    
    rows_to_remove_indices = repeated_entries.groupby('dr2_source_id').apply(process_repeated_group).sum()

    # Get the rows to be removed
    rows_to_remove = repeated_entries.loc[rows_to_remove_indices]

    # Remove the identified rows from merged_results
    clean_merged_results = merged_results[~merged_results.index.isin(rows_to_remove.index)]

    # Reset index of clean_merged_results
    clean_merged_results = clean_merged_results.reset_index(drop=True)
    save_and_adjust_column_widths(clean_merged_results, RESULTS_DIRECTORY + 'clean_merged_results.xlsx')

    save_and_adjust_column_widths(rows_to_remove, RESULTS_DIRECTORY + 'removed_rows.xlsx')

    print(f"Original shape of merged_results: {merged_results.shape}")
    print(f"Shape after removing duplicates: {clean_merged_results.shape}")
    print(f"Number of rows removed: {merged_results.shape[0] - clean_merged_results.shape[0]}")

    # remaining_duplicates = clean_merged_results[clean_merged_results.duplicated('dr2_source_id', keep=False)]
    # print(f"\nNumber of remaining duplicate dr2_source_id: {len(remaining_duplicates['dr2_source_id'].unique())}")

    # if not remaining_duplicates.empty:
    #     print("\nExample of remaining duplicates:")
    #     print(remaining_duplicates.groupby('dr2_source_id').first().head())
    # else:
    #     print("\nNo remaining duplicates found.")
    
    return clean_merged_results

#------------------------------------------------------------------------------------------------

def consolidate_data(df):
    print("\nMerging and consolidating Gaia DR2 and DR3 data columns...")
    def choose_value(row, col_name):
        dr3_col = f'{col_name}_dr3'
        dr2_col = f'{col_name}_dr2'
        return row[dr3_col] if pd.notnull(row[dr3_col]) else row[dr2_col]

    # List of columns to process
    columns_to_process = ['ra', 'dec', 'phot_g_mean_mag', 'phot_bp_mean_mag', 'phot_rp_mean_mag', 
                        'parallax', 'logg_gaia', 'spectraltype_esphs']

    # Process each column
    print(f"Merging the following columns between DR2 and DR3: {', '.join(columns_to_process)}")
    for col in columns_to_process:
        df[col] = df.apply(lambda row: choose_value(row, col), axis=1)

    if isinstance(df, tuple):
        df = df[0]  # Take the first element if it's a tuple

    # Special handling for temperature
    print("Processing temperature")
    df['T_eff [K]'] = df.apply(lambda row: row['teff_gspphot'] if pd.notnull(row['teff_gspphot']) 
                              else row['teff_val'], axis=1)

    # For other columns
    other_columns = ['mass_flame', 'lum_flame', 'radius_flame', 'spectraltype_esphs']
    print(f"Processing the following columns: {', '.join(other_columns)}")
    for col in other_columns:
        df[col] = df.apply(lambda row: choose_value(row, col), axis=1)

    # Add bp_rp column from DR3 if available
    print("Adding bp_rp column")
    df['bp_rp'] = df['bp_rp'].fillna(df['phot_bp_mean_mag'] - df['phot_rp_mean_mag'])

    # Add the new source_id column
    print("Adding Gaia ID column")
    df['source_id'] = np.where(
        pd.notna(df['source_id_dr3']),
        'Gaia DR3 ' + df['source_id_dr3'].astype('Int64').astype(str),
        'Gaia DR2 ' + df['source_id_dr2'].astype('Int64').astype(str)
    )    

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

    for index, row in tqdm(df_consolidated.iterrows(), total=df_new.shape[0], 
                           desc="Retrieving HD, GJ and HIP numbers and object type from Simbad based on Gaia identifiers"):
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
    print("Renaming columns")
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
    save_and_adjust_column_widths(df_consolidated, RESULTS_DIRECTORY + 'consolidated_results.xlsx')

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
    print("\nCalculating and inserting stellar density")
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

def calculate_and_insert_habitable_zone(df):
    print("\nCalculating and inserting habitable zone limits")
    """
    Process stellar data by calculating habitable zone limits, sorting by temperature,
    and saving to Excel with formatted columns.
    
    Args:
        df (pd.DataFrame): DataFrame containing stellar data

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

    save_and_adjust_column_widths(processed_df, f"{RESULTS_DIRECTORY}consolidated_results.xlsx")

    return processed_df

#------------------------------------------------------------------------------------------------

def calculate_and_insert_photon_noise(df):
    print("\nCalculating and inserting RV precision for each star")
    """
    Calculate RV precision for each star in the DataFrame and insert the results.
    Excludes White Dwarfs from the final output and saves to Excel.
    
    Args:
        df (pd.DataFrame): DataFrame containing stellar data with T_eff and V_mag columns
        
    Returns:
        pd.DataFrame: Processed DataFrame with added RV precision and filtered objects
    """
    # Create a copy to avoid modifying the original
    processed_df = df.copy()
    
    # Calculate RV precision for each star
    rv_precisions = []
    for i in range(len(processed_df)):
        result, rv_precision = calculate_rv_precision(
            Temp=processed_df.iloc[i]['T_eff [K]'],
            Vmag=processed_df.iloc[i]['V_mag']
        )
        rv_precisions.append(rv_precision)

    # Insert RV precision values after HZ_limit column
    if 'HZ_limit [AU]' in processed_df.columns:
        hz_limit_index = processed_df.columns.get_loc('HZ_limit [AU]')
        processed_df.insert(hz_limit_index + 1, 'σ_photon [m/s]', rv_precisions)
    else:
        # If HZ_limit column doesn't exist, add to the end
        processed_df['σ_photon [m/s]'] = rv_precisions

    # Filter out White Dwarfs
    processed_df = processed_df[processed_df['Object Type'] != 'WhiteDwarf']

    # Save the result to a new Excel file
    save_and_adjust_column_widths(processed_df, f"{RESULTS_DIRECTORY}combined_query_with_RV_precision.xlsx")

    return processed_df

#------------------------------------------------------------------------------------------------
def calculate_hz_orbital_period(df):
    """
    Calculate orbital period for habitable zone planets using Kepler's third law,
    and insert the result as a new column 'P_days' immediately after 'HZ_limit [AU]'.

    Uses Kepler's third law: P² = (4π²/GM) * a³
    Where:
    - P = orbital period (seconds)
    - G = gravitational constant
    - M = stellar mass (kg)
    - a = semi-major axis (meters)

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing columns:
        - 'Mass [M_Sun]': Stellar mass in solar masses
        - 'HZ_limit [AU]': Habitable zone distance in AU

    Returns:
    --------
    pandas.DataFrame
        DataFrame with 'P_days' column inserted after 'HZ_limit [AU]'
    """
    # Physical constants
    G = 6.67430e-11  # m³ kg⁻¹ s⁻²
    M_sun = 1.989e30  # kg
    AU = 1.496e11    # meters

    # Get stellar mass in kg
    M_star_kg = df['Mass [M_Sun]'] * M_sun

    # Get HZ distance in meters
    a_hz_m = df['HZ_limit [AU]'] * AU

    # Calculate orbital period using Kepler's third law
    # P² = (4π²/GM) * a³
    P_squared = (4 * np.pi**2 / (G * M_star_kg)) * (a_hz_m**3)
    P_seconds = np.sqrt(P_squared)

    # Convert to days
    P_days = P_seconds / (24 * 3600)

    # Insert 'P_days' after 'HZ_limit [AU]'
    df_new = df.copy()
    hz_limit_idx = df_new.columns.get_loc('HZ_limit [AU]')
    df_new.insert(hz_limit_idx + 1, 'Orbital Period [days]', P_days)

    return df_new

#------------------------------------------------------------------------------------------------

def calculate_K(
    df,
    N=1000,
    tau_years=5.0,
    alpha=6.0,
    sigma_rv_col='σ_photon [m/s]',
    output_col=None
):
    """
    Calculate minimum detectable RV semi-amplitude K for DataFrame using Zechmeister et al. (2009)
    and insert it into the DataFrame after the 'Orbital Period [days]' column.

    This function implements Equation 5 from Zechmeister et al. (2009):
    K = α * (σ_RV / √N) * √(1 + 10^((P/τ - 1.5)²))

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing RV noise and orbital period columns.
    N : int
        Number of RV observations (default: 1000)
    tau_years : float
        Total observational baseline in years (default: 5.0)
    alpha : float
        Detection threshold parameter (default: 6.0 for ~50% detection probability)
    sigma_rv_col : str
        Name of the column to use for σ_RV (e.g., 'σ_photon [m/s]', 'σ_RV,total [m/s]')
    output_col : str or None
        Name of the output column. If None, will use 'Semi-Amplitude_{sigma_rv_col} [m/s]'

    Returns:
    --------
    pandas.DataFrame
        DataFrame with added semi-amplitude column after 'Orbital Period [days]'
    """
    # Validate sigma_rv_col exists
    if sigma_rv_col not in df.columns:
        raise ValueError(f"Column '{sigma_rv_col}' not found in DataFrame.")

    # Use output_col or construct default
    if output_col is None:
        output_col = f"Semi-Amplitude_{sigma_rv_col.replace('[m/s]', '').replace(' ', '').replace(',', '_')} [m/s]"

    sigma_rv = df[sigma_rv_col]
    P_days = df['Orbital Period [days]']

    # Calculate K using equation 5
    period_scale_factor = np.sqrt(1 + (10 ** ((P_days / 365.25 / tau_years) - 1.5)) ** 2)
    K = sigma_rv * alpha / np.sqrt(N) * period_scale_factor

    # Insert K after 'Orbital Period [days]'
    df_new = df.copy()
    period_idx = df_new.columns.get_loc('Orbital Period [days]')
    df_new.insert(period_idx + 1, output_col, K)

    return df_new

#------------------------------------------------------------------------------------------------

def calculate_and_insert_hz_detection_limit(
    df,
    semi_amplitude_col='Semi-Amplitude_σ_photon [m/s]'
):
    """
    Calculate and insert the habitable zone detection limit for each star in the DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing stellar data with RV precision, Mass, and HZ_limit columns
        semi_amplitude_col (str): Column name to use for the semi-amplitude, e.g.,
            'Semi-Amplitude_σ_photon [m/s]' or 'Semi-Amplitude_σ_RV_total [m/s]'

    Returns:
        pd.DataFrame: Processed DataFrame with added HZ detection limit
    """
    # Create a copy to avoid modifying the original
    processed_df = df.copy()

    # Check if the specified semi_amplitude_col exists
    if semi_amplitude_col not in processed_df.columns:
        raise ValueError(f"Column '{semi_amplitude_col}' not found in DataFrame.")

    # Determine output column name according to the rule
    if semi_amplitude_col == 'Semi-Amplitude_σ_RV_total [m/s]':
        output_col = 'HZ Detection Limit [M_Earth]'
    else:
        if semi_amplitude_col.startswith('Semi-Amplitude_'):
            concise_name = semi_amplitude_col.replace('Semi-Amplitude_', '').replace(' [m/s]', '')
        else:
            concise_name = semi_amplitude_col.replace(' [m/s]', '')
        output_col = f"HZ Detection Limit ({concise_name}) [M_Earth]"

    print(f"\nCalculating and inserting habitable zone detection limits using '{semi_amplitude_col}' -> '{output_col}'")

    # Calculate HZ detection limit for each star
    processed_df[output_col] = processed_df.apply(
        lambda row: calculate_hz_detection_limit_simplify(
            row[semi_amplitude_col],
            row['Mass [M_Sun]'],
            row['HZ_limit [AU]']
        ),
        axis=1
    )

    # Count and print the number of NaN values
    nan_count = processed_df[output_col].isna().sum()
    print(f"Number of NaN values in {output_col}: {nan_count}")

    # Reorder columns to place the new column next to the chosen semi-amplitude column
    cols = processed_df.columns.tolist()
    try:
        sa_index = cols.index(semi_amplitude_col)
        cols.insert(sa_index + 1, cols.pop(cols.index(output_col)))
        processed_df = processed_df[cols]
    except Exception as e:
        print(f"Warning: Could not reorder columns due to: {e}")

    # Save the updated DataFrame
    save_and_adjust_column_widths(
        processed_df,
        f"{RESULTS_DIRECTORY}combined_query_with_mass_detection_limit.xlsx"
    )

    return processed_df

#------------------------------------------------------------------------------------------------

def analyze_bright_neighbors(merged_df, search_radius, execute_gaia_query_func, max_retries=3, delay=5):
    print("\nAnalyzing stars to identify those with bright neighboring stars")
    """
    Analyze stars in the input DataFrame to identify those with bright neighboring stars.
    
    Args:
        merged_df (pd.DataFrame): Input DataFrame containing stellar data
        search_radius (float): Search radius for finding neighbors
        execute_gaia_query_func (callable): Function to execute Gaia queries
        max_retries (int, optional): Maximum number of retry attempts for failed queries. Defaults to 3
        delay (int, optional): Delay in seconds between retry attempts. Defaults to 5
        
    Returns:
        tuple: (DataFrame with bright neighbors, DataFrame without bright neighbors)
    """
    def create_neighbor_query(source_id, ra, dec, neighbor_g_mag_limit, search_radius, data_release):
        """Create a Gaia query to find neighboring stars."""
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

    def process_row_with_retry(row):
        """Process a single row with retry logic for failed queries."""
        attempt = 0
        while attempt < max_retries:
            try:
                if not pd.isna(row['source_id_dr3']):
                    query = create_neighbor_query(
                        source_id=row['source_id_dr3'],
                        ra=row['RA'],
                        dec=row['DEC'],
                        neighbor_g_mag_limit=row['Phot G Mean Mag']+3,
                        search_radius=search_radius,
                        data_release='gaiadr3'
                    )
                else:
                    query = create_neighbor_query(
                        source_id=row['source_id_dr2'],
                        ra=row['RA'],
                        dec=row['DEC'],
                        neighbor_g_mag_limit=row['Phot G Mean Mag']+3,
                        search_radius=search_radius,
                        data_release='gaiadr2'
                    )
                
                # Execute the query
                neighbors_df = execute_gaia_query_func(query)
                
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

    # Initialize lists to store results
    rows_with_bright_neighbors = []
    rows_without_bright_neighbors = []

    # Process rows in parallel with progress tracking
    with ThreadPoolExecutor() as executor:
        results = list(tqdm(
            executor.map(process_row_with_retry, [row for idx, row in merged_df.iterrows()]),
            total=len(merged_df),
            desc="Parallel processing for detecting bright neighbors",
            ncols=100
        ))
        
        # Sort results into appropriate lists
        for row, has_bright_neighbors in results:
            if has_bright_neighbors:
                rows_with_bright_neighbors.append(row)
            else:
                rows_without_bright_neighbors.append(row)

    # Create DataFrames from results
    df_with_bright_neighbors = pd.DataFrame(rows_with_bright_neighbors)
    df_without_bright_neighbors = pd.DataFrame(rows_without_bright_neighbors)

    print(f"Stars with bright neighbors: {len(df_with_bright_neighbors)}")
    print(f"Stars without bright neighbors: {len(df_without_bright_neighbors)}")
    
    # Save the result to a new Excel file
    save_and_adjust_column_widths(df_with_bright_neighbors, f"{RESULTS_DIRECTORY}stars_with_bright_neighbors.xlsx")
    save_and_adjust_column_widths(df_without_bright_neighbors, f"{RESULTS_DIRECTORY}stars_without_bright_neighbors.xlsx")

    return df_with_bright_neighbors, df_without_bright_neighbors

#------------------------------------------------------------------------------------------------   

def merge_and_format_stellar_data(df_main, ralf_file_path):
    print("\nCrossmatching with Ralf's target list")
    """
    Merge stellar data with Ralf's target list and format the output Excel file.
    
    Args:
        df_main (pd.DataFrame): Main DataFrame containing stellar data
        ralf_file_path (str): Path to Ralf's Excel file        
    Returns:
        pd.DataFrame: Merged and formatted DataFrame
    """

    from datetime import datetime
    
    # Read Ralf's data
    df_Ralf = pd.read_excel(ralf_file_path, engine='openpyxl', header=1)
    df_Ralf = df_Ralf[df_Ralf['prio'] != 3]
    
    # Create a copy of the main DataFrame to avoid modifying the original
    merged_df = df_main.copy()
    
    # Process HD Numbers
    # merged_df[['HD Number 1', 'HD Number 2']] = merged_df['HD Number'].str.split(', ', expand=True, n=1)
    # merged_df['HD Number 1'] = merged_df['HD Number 1'].str.replace(r'HD\s+', 'HD', regex=True)
    # merged_df['HD Number 2'] = merged_df['HD Number 2'].fillna('').str.replace(r'HD\s+', 'HD', regex=True)
    
    # Process HIP Numbers
    merged_df['HIP Number'] = merged_df['HIP Number'].apply(
        lambda x: f'HIP{x}' if pd.notna(x) and x != '' and not str(x).startswith('HIP') else x
    )
    
    # Process GJ Numbers
    # merged_df[['GJ Number 1', 'GJ Number 2']] = merged_df['GJ Number'].str.split(', ', expand=True, n=1)
    # merged_df['GJ Number 1'] = merged_df['GJ Number 1'].str.replace(r'\s+', '', regex=True)
    # merged_df['GJ Number 2'] = merged_df['GJ Number 2'].fillna('').str.replace(r'\s+', '', regex=True)
    
    # Merge DataFrames
    # merge_keys = ['HD Number 1', 'HD Number 2', 'HIP Number', 'GJ Number 1', 'GJ Number 2']
    merge_keys = ['HD Number', 'HIP Number', 'GJ Number']
    merged_RJ = pd.concat([
        df_Ralf.merge(merged_df, left_on='star_ID  ', right_on=key, how='left') 
        for key in merge_keys
    ])
    
    # Clean up merged DataFrame
    merged_RJ.sort_values(by='source_id', ascending=False, inplace=True)
    merged_RJ.drop_duplicates(subset='star_ID  ', keep='first', inplace=True)
    merged_RJ.reset_index(drop=True, inplace=True)
    
    # Sort by priority and HD Number
    merged_RJ.sort_values(by=['prio', 'HD Number'], ascending=[True, True], inplace=True)

    date_str = datetime.now().strftime('%Y.%m.%d')
    output_path = f'{RESULTS_DIRECTORY}merged_RJ_{date_str}.xlsx'
    
    # Define color formatting function
    def apply_color_formatting(workbook, worksheet):
        dark_green_format = workbook.add_format({'font_color': '#006400'})
        orange_format = workbook.add_format({'font_color': '#FFA500'})
        yellow_format = workbook.add_format({'bg_color': '#FFFF00'})
        
        for row_num, (prio_value, source_id) in enumerate(
            zip(merged_RJ['prio'], merged_RJ['source_id']), start=1
        ):
            if pd.isna(source_id):
                worksheet.set_row(row_num, None, yellow_format)
            elif prio_value == 0:
                worksheet.set_row(row_num, None, dark_green_format)
            elif prio_value == 1:
                worksheet.set_row(row_num, None, orange_format)
    
    # Save with formatting
    try:
        with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
            merged_RJ.to_excel(writer, index=False)
            workbook = writer.book
            worksheet = writer.sheets['Sheet1']
            apply_color_formatting(workbook, worksheet)
    except ModuleNotFoundError:
        print("xlsxwriter module not found. Please install it using 'pip install xlsxwriter'")
        
    adjust_column_widths(output_path)
    
    return merged_RJ, df_Ralf

#------------------------------------------------------------------------------------------------   

def add_granulation_to_dataframe(df, t_eff_col='T_eff [K]', mass_col='Mass [M_Sun]', 
                                luminosity_col='Luminosity [L_Sun]'):
    """
    Add granulation noise column to a pandas DataFrame after 'σ_photon [m/s]' column.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing stellar parameters
    t_eff_col : str
        Column name for effective temperature
    mass_col : str  
        Column name for stellar mass
    luminosity_col : str
        Column name for stellar luminosity
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with added 'σ_granulation [m/s]' column after 'σ_photon [m/s]'
    """
    df = df.copy()
    
    # Check if required columns exist
    required_cols = [t_eff_col, mass_col, luminosity_col]
    missing_cols = [col for col in required_cols if col not in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"Warning: Missing columns: {missing_cols}")
        return df
    
    # Calculate granulation noise for each star
    granulation_values = []
    for idx, row in df.iterrows():
        try:
            t_eff = float(row[t_eff_col])
            mass = float(row[mass_col]) 
            luminosity = float(row[luminosity_col])
            
            gran_noise = calculate_granulation_noise(t_eff, mass, luminosity)
            granulation_values.append(gran_noise)
            
        except (ValueError, TypeError):
            granulation_values.append(np.nan)
    
    # Find the position of 'σ_photon [m/s]' column
    rv_precision_col = 'σ_photon [m/s]'
    gran_col_name = 'σ_granulation [m/s]'
    if rv_precision_col not in df.columns:
        print(f"Warning: '{rv_precision_col}' column not found. Adding granulation column at the end.")
        df[gran_col_name] = granulation_values
        return df
    
    # Get the position of RV precision column
    rv_precision_idx = df.columns.get_loc(rv_precision_col)
    
    # Insert the granulation noise column right after RV precision
    insert_position = rv_precision_idx + 1
    
    # Create new column order
    columns = df.columns.tolist()
    columns.insert(insert_position, gran_col_name)
    
    # Add the granulation noise data
    df[gran_col_name] = granulation_values
    
    # Reorder columns
    df = df[columns]
    
    return df

#------------------------------------------------------------------------------------------------

def add_pmode_rms_to_dataframe(df, t_eff_col='T_eff [K]', mass_col='Mass [M_Sun]', 
                              luminosity_col='Luminosity [L_Sun]', 
                              alpha=0.63, beta=0.47, gamma=-0.45, delta=0.57):
    """
    Add p-mode oscillation RMS column to a pandas DataFrame after 'σ_granulation [m/s]' column.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing stellar parameters
    t_eff_col : str
        Column name for effective temperature
    mass_col : str  
        Column name for stellar mass
    luminosity_col : str
        Column name for stellar luminosity
    alpha : float, optional
        Scaling coefficient (default: 0.63 m/s)
    beta : float, optional
        Luminosity exponent (default: 0.47)
    gamma : float, optional
        Mass exponent (default: -0.45)
    delta : float, optional
        Temperature exponent (default: 0.57)
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with added 'σ_p-mode [m/s]' column after 'σ_granulation [m/s]'
    """
    df = df.copy()
    
    # Check if required columns exist
    required_cols = [t_eff_col, mass_col, luminosity_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"Warning: Missing columns: {missing_cols}")
        return df
    
    # Calculate p-mode RMS for each star
    pmode_values = []
    for idx, row in df.iterrows():
        try:
            t_eff = float(row[t_eff_col])
            mass = float(row[mass_col]) 
            luminosity = float(row[luminosity_col])
            
            pmode_rms = calculate_pmode_rms(t_eff, mass, luminosity, alpha, beta, gamma, delta)
            pmode_values.append(pmode_rms)
            
        except (ValueError, TypeError):
            pmode_values.append(np.nan)
    
    # Find the position of 'σ_granulation [m/s]' column
    granulation_col = 'σ_granulation [m/s]'
    pmode_col_name = 'σ_p-mode [m/s]'
    
    if granulation_col not in df.columns:
        print(f"Warning: '{granulation_col}' column not found. Adding p-mode column at the end.")
        df[pmode_col_name] = pmode_values
        return df
    
    # Get the position of granulation column
    granulation_idx = df.columns.get_loc(granulation_col)
    
    # Insert the p-mode RMS column right after granulation
    insert_position = granulation_idx + 1
    
    # Create new column order
    columns = df.columns.tolist()
    columns.insert(insert_position, pmode_col_name)
    
    # Add the p-mode RMS data
    df[pmode_col_name] = pmode_values
    
    # Reorder columns
    df = df[columns]
    
    return df


#------------------------------------------------------------------------------------------------

def calculate_and_insert_RV_noise(df):
    """
    Calculate the total RV noise as the quadrature sum of σ_photon [m/s], σ_granulation [m/s], and σ_p-mode [m/s],
    and insert it as a new column 'σ_RV,total [m/s]' immediately after the 'σ_p-mode [m/s]' column.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the relevant noise columns.

    Returns
    -------
    pandas.DataFrame
        DataFrame with the new 'σ_RV,total [m/s]' column inserted.
    """
    df = df.copy()
    photon_col = 'σ_photon [m/s]'
    gran_col = 'σ_granulation [m/s]'
    pmode_col = 'σ_p-mode [m/s]'
    total_col = 'σ_RV,total [m/s]'

    # Calculate the quadrature sum
    df[total_col] = (
        INSTRUMENTAL_NOISE**2 +
        df[photon_col]**2 +
        (df[gran_col] * RESIDUAL_GRANULATION_FRACTION) **2 +
        (df[pmode_col] * RESIDUAL_P_MODE_FRACTION) **2
    ) ** 0.5

    # Insert after σ_p-mode [m/s]
    pmode_idx = df.columns.get_loc(pmode_col)
    cols = df.columns.tolist()
    # Remove and re-insert at the right position
    cols.insert(pmode_idx + 1, cols.pop(cols.index(total_col)))
    df = df[cols]

    return df