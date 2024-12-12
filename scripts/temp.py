    AND NOT EXISTS (
        SELECT 1
        FROM gaiadr2.gaia_source AS neighbors
        WHERE 
            1 = CONTAINS(
                POINT('ICRS', gs.ra, gs.dec),
                CIRCLE('ICRS', neighbors.ra, neighbors.dec, 2/3600.0)
            )
            AND neighbors.phot_g_mean_mag < {NEIGHBOR_G_MAG_LIMIT}
            AND gs.source_id != neighbors.source_id
    )


#----------------------------------------------------------------



    AND NOT EXISTS (
        SELECT 1
        FROM gaiadr3.gaia_source AS neighbors
        WHERE 
            1 = CONTAINS(
                POINT('ICRS', gs.ra, gs.dec),
                CIRCLE('ICRS', neighbors.ra, neighbors.dec, 2/3600.0)
            )
            AND neighbors.phot_g_mean_mag < {NEIGHBOR_G_MAG_LIMIT}
            AND gs.source_id != neighbors.source_id
    )


# -------------------------------------------------------------
# Add SIMBAD information (HD, GJ, HIP numbers and object type)
# -------------------------------------------------------------
# Customize Simbad to include the HD, GJ, and HIP identifiers and object type
custom_simbad = Simbad()
custom_simbad.add_votable_fields('ids', 'otype')

# Function to get additional information from SIMBAD
def get_simbad_info(source_id):
    result_table = custom_simbad.query_object(f"Gaia DR3 {source_id}")
    hd_numbers = []
    gj_numbers = []
    hip_numbers = []
    object_type = None
    
    if result_table is not None:
        ids = result_table['IDS'][0].split('|')
        hd_numbers = [id.strip() for id in ids if id.startswith('HD')]
        gj_numbers = [id.strip() for id in ids if id.startswith('GJ')]
        hip_numbers = [id.strip() for id in ids if id.startswith('HIP')]
        object_type = result_table['OTYPE'][0]
    
    return {
        'HD Number': ', '.join(hd_numbers) if hd_numbers else None,
        'GJ Number': ', '.join(gj_numbers) if gj_numbers else None,
        'HIP Number': ', '.join(hip_numbers) if hip_numbers else None,
        'Object Type': object_type
    }


# Fetch SIMBAD information for each star
for index, row in df_new.iterrows():
    simbad_info = get_simbad_info(row['source_id'])
    df_new.loc[index, 'HD Number'] = simbad_info['HD Number']
    df_new.loc[index, 'GJ Number'] = simbad_info['GJ Number']
    df_new.loc[index, 'HIP Number'] = simbad_info['HIP Number']
    df_new.loc[index, 'Object Type'] = simbad_info['Object Type']


#----------------------------------------------------------------
# Batch query for Gaia DR2 in parallel
#----------------------------------------------------------------

import pandas as pd
import time

from concurrent.futures import ThreadPoolExecutor, as_completed

def execute_gaia_query_segment(query, str_columns=None):
    """
    Execute a Gaia query segment and return results as a DataFrame.
    
    Parameters:
    -----------
    query : str
        The ADQL query segment to execute
    str_columns : list, optional
        List of column names to convert to string type
        
    Returns:
    --------
    pandas.DataFrame
        Query results as a dataframe, or an empty dataframe if the query fails
    """
    try:
        # Execute query
        job = Gaia.launch_job_async(query)
        df = job.get_results().to_pandas()
        
        # Convert specified columns to string
        if str_columns:
            for col in str_columns:
                if col in df.columns:
                    df[col] = df[col].astype(str)
        
        print(f"Number of results: {len(df)}")
        return df
        
    except requests.exceptions.HTTPError as e:
        print(f"An HTTP error occurred: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")
    
    return pd.DataFrame()  # Return an empty DataFrame if the query fails

def execute_gaia_query_in_batches(query_template, ra_ranges, str_columns=None):
    """
    Execute Gaia queries in batches based on RA ranges and combine results.
    
    Parameters:
    -----------
    query_template : str
        Template for the ADQL query with placeholders for RA range.
    ra_ranges : list of tuples
        List of (min_ra, max_ra) tuples to query in batches.
    str_columns : list, optional
        List of column names to convert to string type
        
    Returns:
    --------
    pandas.DataFrame
        Combined query results as a dataframe
    """
    all_results = []
    
    def query_ra_range(min_ra, max_ra):
        # Format the query with the current RA range
        query = query_template.format(min_ra=min_ra, max_ra=max_ra)
        # Execute the query segment and return results
        return execute_gaia_query_segment(query, str_columns=str_columns)
    
    # Use ThreadPoolExecutor to run queries in parallel
    with ThreadPoolExecutor() as executor:
        future_to_ra = {executor.submit(query_ra_range, min_ra, max_ra): (min_ra, max_ra) for min_ra, max_ra in ra_ranges}
        for future in as_completed(future_to_ra):
            ra_range = future_to_ra[future]
            try:
                df = future.result()
                if not df.empty:
                    all_results.append(df)
            except Exception as e:
                print(f"An error occurred for RA range {ra_range}: {e}")
    
    # Combine all results into a single DataFrame
    combined_df = pd.concat(all_results, ignore_index=True)
    return combined_df

# Define RA ranges for batch processing
ra_ranges = [(i, i + 1) for i in range(0, 360)]

# Adjust query template to include RA range
query_template = query_dr2.replace("WHERE", "WHERE gs.ra BETWEEN {min_ra} AND {max_ra} AND")

# Execute queries in batches and combine results
df_dr2_combined = execute_gaia_query_in_batches(
    query_template,
    ra_ranges,
    str_columns=['source_id']
)

# Save combined results to Excel
if not df_dr2_combined.empty:
    df_dr2_combined.to_excel(directory + 'dr2_results_combined.xlsx', index=False)
    adjust_column_widths(directory + 'dr2_results_combined.xlsx')


#----------------------------------------------------------------
# Batch query for Gaia DR2 in truncated serial
#----------------------------------------------------------------   

# Note: it's taking too long to run (over 24 hours and still running)

import pandas as pd
import time

def execute_gaia_query_segment(query, str_columns=None):
    """
    Execute a Gaia query segment and return results as a DataFrame.
    
    Parameters:
    -----------
    query : str
        The ADQL query segment to execute
    str_columns : list, optional
        List of column names to convert to string type
        
    Returns:
    --------
    pandas.DataFrame
        Query results as a dataframe, or an empty dataframe if the query fails
    """
    try:
        # Execute query
        job = Gaia.launch_job_async(query)
        df = job.get_results().to_pandas()
        
        # Convert specified columns to string
        if str_columns:
            for col in str_columns:
                if col in df.columns:
                    df[col] = df[col].astype(str)
        
        print(f"Number of results: {len(df)}")
        return df
        
    except requests.exceptions.HTTPError as e:
        print(f"An HTTP error occurred: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")
    
    return pd.DataFrame()  # Return an empty DataFrame if the query fails

def execute_gaia_query_in_batches(query_template, ra_ranges, str_columns=None):
    """
    Execute Gaia queries in batches based on RA ranges and combine results.
    
    Parameters:
    -----------
    query_template : str
        Template for the ADQL query with placeholders for RA range.
    ra_ranges : list of tuples
        List of (min_ra, max_ra) tuples to query in batches.
    str_columns : list, optional
        List of column names to convert to string type
        
    Returns:
    --------
    pandas.DataFrame
        Combined query results as a dataframe
    """
    all_results = []
    
    for min_ra, max_ra in ra_ranges:
        # Format the query with the current RA range
        query = query_template.format(min_ra=min_ra, max_ra=max_ra)
        
        # Execute the query segment and collect results
        df = execute_gaia_query_segment(query, str_columns=str_columns)
        
        if not df.empty:
            all_results.append(df)
    
    # Combine all results into a single DataFrame
    combined_df = pd.concat(all_results, ignore_index=True)
    return combined_df

# Define RA ranges for batch processing
ra_ranges = [(0, 60), (60, 120), (120, 180), (180, 240), (240, 300), (300, 360)]

# Adjust query template to include RA range
query_template = query_dr2.replace("WHERE", "WHERE gs.ra BETWEEN {min_ra} AND {max_ra} AND")

# Execute queries in batches and combine results
df_dr2_combined = execute_gaia_query_in_batches(
    query_template,
    ra_ranges,
    str_columns=['source_id']
)

# Save combined results to Excel
if not df_dr2_combined.empty:
    df_dr2_combined.to_excel(directory + 'dr2_results_combined.xlsx', index=False)
    adjust_column_widths(directory + 'dr2_results_combined.xlsx')











from astroquery.vizier import Vizier
import time

df_consolidated_HD = df_consolidated_HIP.copy()

# Define the Vizier catalog and set a limit for results
catalog = "V/117A"
Vizier.ROW_LIMIT = 1  # Limit to one result to avoid large datasets


# Iterate over the DataFrame and fill missing values
for index, row in df_consolidated_HD.iterrows():
    if pd.isna(row["T_eff [K]"]) or pd.isna(row["Luminosity [L_Sun]"]):
        hd_number = clean_hd_number(row["HD Number"])
        
        if hd_number:
            temperature, luminosity = get_star_properties(hd_number)
            mass = extract_mass(hd_number)
            
            if pd.isna(row["T_eff [K]"]) and temperature is not None:
                df_consolidated_HD.at[index, "T_eff [K]"] = temperature
            if pd.isna(row["Luminosity [L_Sun]"]) and luminosity is not None:
                df_consolidated_HD.at[index, "Luminosity [L_Sun]"] = luminosity
            if pd.isna(row["Mass [M_Sun]"]) and mass is not None:
                df_consolidated_HD.at[index, "Mass [M_Sun]"] = mass                # 

# Save the updated DataFrame to an Excel file
df_consolidated_HD.to_excel(directory + 'consolidated_HD_results.xlsx', index=False)
adjust_column_widths(directory + 'consolidated_HD_results.xlsx')







import matplotlib.pyplot as plt
import numpy as np

# Create a scatter plot with color mapping
plt.figure(figsize=(8, 6), dpi=150)
scatter = plt.scatter(
    merged_RJ['magV     '], 
    merged_RJ['V_mag'], 
    c=merged_RJ['T_eff [K]'], 
    cmap='autumn', 
    edgecolor='k', 
    alpha=0.7
)

# # Add labels to each point
# for i, name in enumerate(merged_result['other name']):
#     plt.text(merged_result['Vmag'][i], merged_result['V_mag'][i], name, fontsize=8, ha='right')

# Add titles and labels
plt.title('Selection Crossmatch', fontsize=14)
plt.xlabel('V mag (Ralf)', fontsize=12)
plt.ylabel('V mag (Jinglin)', fontsize=12)

# Add a color bar
cbar = plt.colorbar(scatter)
cbar.set_label('T_eff [K]', fontsize=12)

# Add grid for better readability
plt.grid(True, linestyle='--', alpha=0.7)

# Overplot the suitable points from df_Melissa
suitable_indices = df_Melissa['suitable'] == 1
name_suitable = df_Melissa.loc[suitable_indices, 'other name']

x_suitable = []
y_suitable = []

# for name in name_suitable:
#     x_value = df_Melissa.loc[df_Melissa['other name'] == name, 'Vmag'].values[0]
#     if name in merged_result['other name'].values:        
#         y_value = merged_result.loc[merged_result['other name'] == name, 'V_mag'].values[0]
#     else:
#         y_value = 0
#     x_suitable.append(x_value)
#     y_suitable.append(y_value)

#     plt.text(x_value, y_value, name, fontsize=8, ha='right', color='green')

# Plot the suitable points
# plt.scatter(x_suitable, y_suitable, edgecolor='g', facecolors='none', label='Suitable = 1')

# Plot the x = y line
min_value = min(min(merged_RJ['magV     ']), min(merged_RJ['V_mag']))
max_value = max(max(merged_RJ['magV     ']), max(merged_RJ['V_mag']))
plt.plot([min_value, max_value], [min_value, max_value], color='gray', linestyle='--', )

# Add legend
legend = plt.legend()
for text in legend.get_texts():
    text.set_color('green')

# Show the plot
plt.tight_layout()
plt.show()



# -------------------------------------------------------------
# Batch query for Gaia DR2 in parallel
import pandas as pd
from astroquery.gaia import Gaia
import time
from concurrent.futures import ThreadPoolExecutor

# Function to create a query for nearby stars
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

# Function to process each row
def process_row(row):
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

# DataFrames: merged_df (input sources)
rows_with_bright_neighbors = []
rows_without_bright_neighbors = []

# Use ThreadPoolExecutor to process rows in parallel
with ThreadPoolExecutor() as executor:
    results = executor.map(process_row, [row for idx, row in merged_df.iterrows()])
    for row, has_bright_neighbors in results:
        if has_bright_neighbors:
            rows_with_bright_neighbors.append(row)
        else:
            rows_without_bright_neighbors.append(row)

# Create DataFrames for rows with and without bright neighbors
bright_neighbors_df = pd.DataFrame(rows_with_bright_neighbors)
rows_without_bright_neighbors_df = pd.DataFrame(rows_without_bright_neighbors)

# Output the results
print(f"Rows with bright neighbors: {len(bright_neighbors_df)}")
print(f"Rows without bright neighbors: {len(rows_without_bright_neighbors_df)}")


# -------------------------------------------------------------
# Batch query for Gaia DR2 in serial
# ------------------------------------------------------------- 

import pandas as pd
from astroquery.gaia import Gaia
import time

# Function to create a query for nearby stars
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

# DataFrames: merged_df (input sources)
rows_with_bright_neighbors = []
rows_without_bright_neighbors = []

# Iterate through each row in merged_df
# for i, (idx, row) in enumerate(merged_df.iloc[0:10].iterrows()):
for i, (idx, row) in enumerate(merged_df.iterrows()):
    print(i, row['source_id_dr2'], row['source_id_dr3'])
    
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
        # Add the row to the bright neighbors list
        rows_with_bright_neighbors.append(row)
    else:
        # Add the row to the no bright neighbors list
        rows_without_bright_neighbors.append(row)

# Create DataFrames for rows with and without bright neighbors
bright_neighbors_df = pd.DataFrame(rows_with_bright_neighbors)
rows_without_bright_neighbors_df = pd.DataFrame(rows_without_bright_neighbors)

# Output the results
print(f"Rows with bright neighbors: {len(bright_neighbors_df)}")
print(f"Rows without bright neighbors: {len(rows_without_bright_neighbors_df)}")
