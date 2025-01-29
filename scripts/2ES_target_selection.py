#!/usr/bin/env python
# coding: utf-8

# Possible improvements:
# 
# - Use the crosscheck table to merge DR2 and DR3
# 
# - Check the mass: some stars do not have mass information? 
# 
# 
# note: adding the last constrint "ap.mass_flame IS NOT NULL" decreases the number of results from 408 to 218
# 
# 

# In[1]:


from astroquery.gaia import Gaia
# Gaia.MAIN_GAIA_TABLE = "gaiadr2.gaia_source"  # Select Data Release 2
Gaia.MAIN_GAIA_TABLE = "gaiadr3.gaia_source"  # Reselect Data Release 3, default

import numpy as np
import matplotlib.pyplot as plt
from astroquery.simbad import Simbad
import pandas as pd

from stellar_functions import *
from query import *
from get_stellar_info import *


# In[2]:





# In[3]:










#-------------------------------------------------------------
# Merge DR2 and DR3 results 
#-------------------------------------------------------------
# merged_dr2_crossmatch = pd.merge(df_dr2, df_crossmatch, left_on='source_id', right_on='dr2_source_id', how='left')
# merged_results = pd.merge(merged_dr2_crossmatch, df_dr3, left_on='dr3_source_id', right_on='source_id', suffixes=('_dr2', '_dr3'), how='outer')
# merged_results.to_excel(directory+'merged_results.xlsx', index=False)
# adjust_column_widths(directory+'merged_results.xlsx')


# In[4]:


#-------------------------------------------------------------
# Data Cleaning
#-------------------------------------------------------------

# ***** Step 1: Identify repeated dr2_source_id entries *****
# non_empty_dr2 = merged_results[merged_results['dr2_source_id'].notna() & (merged_results['dr2_source_id'] != '')]
# repeated_dr2_ids = non_empty_dr2[non_empty_dr2.duplicated('dr2_source_id', keep=False)]['dr2_source_id'].unique()
# repeated_entries = merged_results[merged_results['dr2_source_id'].isin(repeated_dr2_ids)]

# Save repeated entries to Excel
# repeated_entries.to_excel(directory + 'repeated_entries.xlsx', index=False)
# adjust_column_widths(directory + 'repeated_entries.xlsx')

# # ***** Step 2: Clean repeated entries *****
# def check_dr3_availability(row):
#     '''
#     Function to check if any DR3 data is available for a given row.
#     '''
#     dr3_columns = [col for col in row.index if col.endswith('_dr3')]
#     return not row[dr3_columns].isnull().all()

# def process_repeated_group(group):
#     '''
#     Function to process a group of repeated entries with the same dr2_source_id.
#     '''
#     if len(group) != 2:
#         # If there are more than 2 entries, keep only the first one
#         return group.iloc[1:].index.tolist()  # Return indices of rows to remove
    
#     row1, row2 = group.iloc[0], group.iloc[1]
#     dr3_available1 = check_dr3_availability(row1)
#     dr3_available2 = check_dr3_availability(row2)
    
#     if not dr3_available1 and not dr3_available2:
#         # If both rows have no DR3 data, remove the second one
#         return [group.index[1]]
#     elif dr3_available1 and not dr3_available2:
#         # If only the first row has DR3 data, remove the second one
#         return [group.index[1]]
#     elif not dr3_available1 and dr3_available2:
#         # If only the second row has DR3 data, remove the first one
#         return [group.index[0]]
#     else:
#         # If both rows have DR3 data, remove the second one
#         return [group.index[1]]

# Process repeated entries and get indices of rows to remove
# rows_to_remove_indices = repeated_entries.groupby('dr2_source_id').apply(process_repeated_group).sum()

# # Get the rows to be removed
# rows_to_remove = repeated_entries.loc[rows_to_remove_indices]

# # Remove the identified rows from merged_results
# clean_merged_results = merged_results[~merged_results.index.isin(rows_to_remove.index)]

# # Reset index of clean_merged_results
# clean_merged_results = clean_merged_results.reset_index(drop=True)

# Print some information about the results
# print(f"Original shape of merged_results: {merged_results.shape}")
# print(f"Shape after removing duplicates: {clean_merged_results.shape}")
# print(f"Number of rows removed: {merged_results.shape[0] - clean_merged_results.shape[0]}")

# Save clean_merged_results to Excel
# clean_merged_results.to_excel(directory + 'clean_merged_results.xlsx', index=False)
# adjust_column_widths(directory + 'clean_merged_results.xlsx')

# Save rows_to_remove to Excel
# rows_to_remove.to_excel(directory + 'removed_rows.xlsx', index=False)
# adjust_column_widths(directory + 'removed_rows.xlsx')

# Check if there are still any duplicates
# remaining_duplicates = clean_merged_results[clean_merged_results.duplicated('dr2_source_id', keep=False)]
# print(f"\nNumber of remaining duplicate dr2_source_id: {len(remaining_duplicates['dr2_source_id'].unique())}")

# if not remaining_duplicates.empty:
#     print("\nExample of remaining duplicates:")
#     print(remaining_duplicates.groupby('dr2_source_id').first().head())
# else:
#     print("\nNo remaining duplicates found.")

# ***** Step 3: Consolidate the data *****
# Create a new DataFrame to store the final consolidated data
df = clean_merged_results.copy()

# Function to choose between DR3 and DR2 values
# def choose_value(row, col_name):
#     dr3_col = f'{col_name}_dr3'
#     dr2_col = f'{col_name}_dr2'
#     return row[dr3_col] if pd.notnull(row[dr3_col]) else row[dr2_col]

# List of columns to process
# columns_to_process = ['ra', 'dec', 'phot_g_mean_mag', 'phot_bp_mean_mag', 'phot_rp_mean_mag', 'parallax', 'logg_gaia', 'spectraltype_esphs']

# # Process each column
# for col in columns_to_process:
#     df[col] = df.apply(lambda row: choose_value(row, col), axis=1)

# # Special handling for temperature
# df['T_eff [K]'] = df.apply(lambda row: row['teff_gspphot'] if pd.notnull(row['teff_gspphot']) else row['teff_val'], axis=1)

# # For other columns
# other_columns = ['mass_flame', 'lum_flame', 'radius_flame', 'spectraltype_esphs']
# for col in other_columns:
#     df[col] = df.apply(lambda row: choose_value(row, col), axis=1)

# # Add bp_rp column from DR3 if available
# df['bp_rp'] = df['bp_rp'].fillna(df['phot_bp_mean_mag'] - df['phot_rp_mean_mag'])

# # Add the new source_id column
# df['source_id'] = df['source_id_dr3'].fillna(df['source_id_dr2'])

# Update the final columns list to include the new source_id column
# final_columns = ['source_id', 'source_id_dr2', 'source_id_dr3', 'ra', 'dec', 'phot_g_mean_mag', 'phot_bp_mean_mag', 'phot_rp_mean_mag', 
#                 'bp_rp', 'parallax', 'T_eff [K]', 'mass_flame', 'lum_flame', 'radius_flame', 'logg_gaia', 'spectraltype_esphs']

# # Create the final dataframe
# df_consolidated = df[final_columns]

# # Create a new DataFrame instead of modifying the existing one
# new_columns = ['source_id', 'HD Number', 'GJ Number', 'HIP Number', 'Object Type']
# df_new = pd.DataFrame(columns=new_columns)

# # Populate the new DataFrame
# df_new['source_id'] = df_consolidated['source_id_dr3'].fillna(df_consolidated['source_id_dr2'])

# for index, row in df_new.iterrows():
#     simbad_info = get_simbad_info_with_retry(row['source_id'])
#     if simbad_info:
#         df_new.loc[index, 'HD Number'] = simbad_info['HD Number']
#         df_new.loc[index, 'GJ Number'] = simbad_info['GJ Number']
#         df_new.loc[index, 'HIP Number'] = simbad_info['HIP Number']
#         df_new.loc[index, 'Object Type'] = simbad_info['Object Type']


# # Combine the new DataFrame with the original one
# df_consolidated = pd.concat([df_consolidated, df_new[['HD Number', 'GJ Number', 'HIP Number', 'Object Type']]], axis=1)

# # Update the final columns list to include the new columns
# final_columns.extend(['HD Number', 'GJ Number', 'HIP Number', 'Object Type'])

# # Create the final dataframe with the updated column list
# df_consolidated = df_consolidated[final_columns]

# # Rename the columns
# df_consolidated = df_consolidated.rename(columns={
#     'mass_flame': 'Mass [M_Sun]',
#     'lum_flame': 'Luminosity [L_Sun]',
#     'radius_flame': 'Radius [R_Sun]',
#     'phot_g_mean_mag': 'Phot G Mean Mag',
#     'phot_bp_mean_mag': 'Phot BP Mean Mag',
#     'phot_rp_mean_mag': 'Phot RP Mean Mag',
#     'bp_rp': 'BP-RP',
#     'parallax': 'Parallax',
#     'ra': 'RA',
#     'dec': 'DEC',
#     'spectraltype_esphs': 'Gaia Spectral type'
# })

# # Convert each column to numeric, coercing errors
# columns_to_convert = ['T_eff [K]', 'Mass [M_Sun]', 'Luminosity [L_Sun]', 'Radius [R_Sun]']

# for column in columns_to_convert:
#     df_consolidated[column] = pd.to_numeric(df_consolidated[column], errors='coerce')

# Save the result to a new Excel file
# df_consolidated.to_excel(directory + 'consolidated_results.xlsx', index=False)
# adjust_column_widths(directory + 'consolidated_results.xlsx')

# # Display some statistics
# print(f"Total number of stars: {len(df_consolidated)}")
# print(f"Number of stars with DR3 source_id: {df_consolidated['source_id_dr3'].notna().sum()}")
# print(f"Number of stars with only DR2 source_id: {df_consolidated['source_id_dr3'].isna().sum()}")
# print(f"Number of stars with HD Number: {df_consolidated['HD Number'].notna().sum()}")
# print(f"Number of stars with GJ Number: {df_consolidated['GJ Number'].notna().sum()}")
# print(f"Number of stars with HIP Number: {df_consolidated['HIP Number'].notna().sum()}")


# In[5]:


# from audio import *

# if 1:
#     # Display the stop button
#     display(stop_button)
#     # Automatically start the audio process
#     start_audio_process()


# # #### HIP -- add temperature and luminosity if missing

# # In[12]:


# len(df_consolidated)


# # In[7]:


# # Define the column specifications based on the byte positions
# colspecs = [
#     (39, 45),  # Num    
#     (10, 15),  # Teff
#     (16, 28),  # Lum
# ]
# # Define the column names
# column_names = [
#     "HIP Number",
#     "T_eff [K]",
#     "Luminosity [L_Sun]"
# ]

# # Read the data into a DataFrame
# df_CELESTA = pd.read_fwf("../data/Catalogue_CELESTA.txt", colspecs=colspecs, names=column_names, skiprows=28)

# # Display the first few rows of the DataFrame
# df_CELESTA.head()


# # In[8]:


# # Step 1: Extract numeric HIP numbers from df_consolidated
# df_consolidated['HIP Number'] = df_consolidated['HIP Number'].str.extract(r'HIP\s*(\d+)')
# df_CELESTA['HIP Number'] = df_CELESTA['HIP Number'].astype(str)

# # Step 2: Merge the dataframes on the HIP Number
# merged_df = pd.merge(df_consolidated, df_CELESTA[['HIP Number', 'T_eff [K]', 'Luminosity [L_Sun]']],
#                      on='HIP Number', suffixes=('', '_CELESTA'), how='left')

# # Step 3: Fill missing T_eff [K] values
# merged_df['T_eff [K]'] = merged_df['T_eff [K]'].fillna(merged_df['T_eff [K]_CELESTA'])

# # Step 4: Fill missing Luminosity [L_Sun] values
# merged_df['Luminosity [L_Sun]'] = merged_df['Luminosity [L_Sun]'].fillna(merged_df['Luminosity [L_Sun]_CELESTA'])

# # Step 5: Drop the extra columns from df_CELESTA
# df_consolidated_HIP = merged_df.drop(columns=['T_eff [K]_CELESTA', 'Luminosity [L_Sun]_CELESTA'])


# df_consolidated_HIP.to_excel(directory + 'consolidated_HIP_results.xlsx', index=False)
# adjust_column_widths(directory + 'consolidated_HIP_results.xlsx')

# df_consolidated_HIP.head()



# # #### HD

# # In[9]:


# # Example usage
# temperature, luminosity = get_star_properties("10700")
# print("Temperature [K]:", temperature if temperature else "N/A")
# print("Estimated Luminosity [L_sun]:", luminosity if luminosity else "N/A")


# # In[17]:


# # Load the stellar catalog file and store it in memory

# with open('../data/Catalogue_V_117A_table1.txt', 'r') as file:
#     STELLAR_CATALOG = file.readlines()

# def extract_mass(hd_number):
#     """
#     Extract mass for a given HD number using the pre-loaded catalog.
#     """
#     for line in STELLAR_CATALOG:
#         # Extract the HD number from the line
#         line_hd_number = line[7:18].strip()
#         # Check if the line contains the desired HD number
#         if line_hd_number == f"HD {hd_number}":
#             # Extract mass information
#             mass = line[130:134].strip()
#             # Return the mass as a float if available
#             return float(mass) if mass else None


# # In[22]:


# df_consolidated_HD = df_consolidated_HIP.copy()
            
# from tqdm import tqdm

# # Iterate over the DataFrame and fill missing values with a progress bar
# for index, row in tqdm(df_consolidated_HD.iterrows(), 
#                     total=df_consolidated_HD.shape[0], 
#                     desc="Filling missing T_eff, Luminosity and Mass"):
#     # Check for missing temperature or luminosity
#     if pd.isna(row["T_eff [K]"]) or pd.isna(row["Luminosity [L_Sun]"]) or pd.isna(row["Mass [M_Sun]"]):
#         # Extract and clean HD number
#         hd_number = clean_hd_number(row["HD Number"])
        
#         if hd_number:  # If a valid HD number was extracted
#             # Retrieve properties
#             temperature, luminosity = get_star_properties_with_retries(hd_number)
#             mass = extract_mass(hd_number)
            
#             # Fill missing values
#             if pd.isna(row["T_eff [K]"]) and temperature is not None:
#                 df_consolidated_HD.at[index, "T_eff [K]"] = temperature
#             if pd.isna(row["Luminosity [L_Sun]"]) and luminosity is not None:
#                 df_consolidated_HD.at[index, "Luminosity [L_Sun]"] = luminosity
#             if pd.isna(row["Mass [M_Sun]"]) and mass is not None:
#                 df_consolidated_HD.at[index, "Mass [M_Sun]"] = mass

# df_consolidated_HD.to_excel(directory + 'consolidated_HD_results.xlsx', index=False)

# adjust_column_widths(directory + 'consolidated_HD_results.xlsx')

# df_consolidated_HD


# # In[24]:


# if 0: # parallel 
#     from multiprocessing import Pool
#     from functools import partial

#     def process_row(row, df):
#         """Process a single row and return the updated values"""
#         if pd.isna(row["T_eff [K]"]) or pd.isna(row["Luminosity [L_Sun]"]) or pd.isna(row["Mass [M_Sun]"]):
#             hd_number = clean_hd_number(row["HD Number"])
            
#             if hd_number:
#                 temperature, luminosity = get_star_properties_with_retries(hd_number)
#                 mass = extract_mass(hd_number)
                
#                 # Return updated values
#                 return {
#                     'index': row.name,
#                     'T_eff [K]': temperature if pd.isna(row["T_eff [K]"]) else row["T_eff [K]"],
#                     'Luminosity [L_Sun]': luminosity if pd.isna(row["Luminosity [L_Sun]"]) else row["Luminosity [L_Sun]"],
#                     'Mass [M_Sun]': mass if pd.isna(row["Mass [M_Sun]"]) else row["Mass [M_Sun]"]
#                 }
        
#         # Return original values if no updates needed
#         return {
#             'index': row.name,
#             'T_eff [K]': row["T_eff [K]"],
#             'Luminosity [L_Sun]': row["Luminosity [L_Sun]"],
#             'Mass [M_Sun]': row["Mass [M_Sun]"]
#         }

#     def parallel_process_dataframe(df):
#         df_consolidated_HD = df.copy()
        
#         # Create a pool of workers
#         with Pool() as pool:
#             # Process rows in parallel with progress bar
#             results = list(tqdm(
#                 pool.imap(partial(process_row, df=df_consolidated_HD), 
#                 [row for _, row in df_consolidated_HD.iterrows()]),
#                 total=len(df_consolidated_HD),
#                 desc="Processing rows in parallel"
#             ))
        
#         # Update DataFrame with results
#         for result in results:
#             idx = result['index']
#             df_consolidated_HD.at[idx, 'T_eff [K]'] = result['T_eff [K]']
#             df_consolidated_HD.at[idx, 'Luminosity [L_Sun]'] = result['Luminosity [L_Sun]']
#             df_consolidated_HD.at[idx, 'Mass [M_Sun]'] = result['Mass [M_Sun]']
        
#         return df_consolidated_HD

#     # Use the parallel processing function
#     df_consolidated_HD = parallel_process_dataframe(df_consolidated_HIP)

#     # Save results
#     df_consolidated_HD.to_excel(directory + 'consolidated_HD_results.xlsx', index=False)
#     adjust_column_widths(directory + 'consolidated_HD_results.xlsx')


# # #### Simbad

# # In[25]:


# len(df_consolidated_HD)


# # In[26]:


# # Example usage
# if 1:
#     gaia_dr3_id = "4072260704719970944"
#     original, processed = get_stellar_type_dr3(gaia_dr3_id)
#     print(f"Original: {original}, Processed: {processed}")

# # Example usage for DR2
# if 0:
#     # gaia_dr2_id = "1234567890123456789"
#     gaia_dr2_id = "25488745411919360"
#     original, processed = get_stellar_type_dr2(gaia_dr2_id)
#     print(f"Original: {original}, Processed: {processed}")
# # 


# # In[29]:


# if 0:
#     df_consolidated_HD = pd.read_excel(directory + 'consolidated_HD_results.xlsx', dtype={'source_id': str, 'source_id_dr2': str, 'source_id_dr3': str, 'HIP Number': str})
#     df_consolidated_HD


# # In[30]:


# list(df_consolidated_HD.columns)


# # In[51]:


# # Update the DataFrame with stellar properties 
# df_consolidated_simbad = get_stellar_properties_from_gaia(df_consolidated_HD)

# # Save to Excel
# filename = directory + 'consolidated_Simbad_results.xlsx'
# df_consolidated_simbad.to_excel(filename, index=False)
# adjust_column_widths(filename)


# # In[54]:


# if 0:
#     df_consolidated_simbad = pd.read_excel(directory + 'consolidated_Simbad_results.xlsx', dtype={'source_id': str, 'source_id_dr2': str, 'source_id_dr3': str, 'HIP Number': str})
#     df_consolidated_simbad


# # In[55]:


# len(df_consolidated_simbad) 


# In[56]:


# df_consolidated = df_consolidated_simbad.copy()

# df_consolidated['Density [Solar unit]'] = None
# df_consolidated.loc[df_consolidated['Mass [M_Sun]'].notna() & df_consolidated['Radius [R_Sun]'].notna(), 'Density [Solar unit]'] = df_consolidated['Mass [M_Sun]'] / (df_consolidated['Radius [R_Sun]'] ** 3)

# # Reorder columns to place 'Density [Solar unit]' after 'Radius [R_Sun]'
# cols = df_consolidated.columns.tolist()
# density_index = cols.index('Density [Solar unit]')
# radius_index = cols.index('Radius [R_Sun]')
# cols.insert(radius_index + 1, cols.pop(density_index))
# df_consolidated = df_consolidated[cols]


# In[57]:


# plt.figure(figsize=(10, 6))

# colors = np.where(df_consolidated['Density [Solar unit]'] < 0.1, 'red', 'blue')

# plt.scatter(df_consolidated['logg_gaia'], df_consolidated['Density [Solar unit]'], alpha=0.6, edgecolors='w', s=20, c=colors)
# plt.xlabel('logg_gaia', fontsize=14)
# plt.ylabel('Density [Solar unit]', fontsize=14)
# plt.ylim(-0.1, 6)
# plt.xlim(2, 5)
# plt.grid(True, linestyle='--', alpha=0.7)
# plt.show()


# # In[58]:


# plt.figure(figsize=(10, 6))

# colors = np.where(df_consolidated['Density [Solar unit]'] < 0.1, 'red', 'blue')

# plt.scatter(df_consolidated['logg_gaia'], df_consolidated['Density [Solar unit]'], alpha=0.6, edgecolors='w', s=20, c=colors)
# plt.xlabel('logg_gaia', fontsize=14)
# plt.ylabel('Density [Solar unit]', fontsize=14)
# plt.ylim(-0.01, 0.3)
# plt.xlim(2, 4.5)
# plt.grid(True, linestyle='--', alpha=0.7)
# plt.show()


# In[65]:


# #-------------------------------------------------------------
# # Extract data with mass and luminosity information available
# #-------------------------------------------------------------
# df_filtered = df_consolidated.copy()

# # Filter out stars with missing mass or luminosity information
# df_filtered = df_filtered[(df_filtered['Luminosity [L_Sun]'].notna()) & (df_filtered['T_eff [K]'].notna())]

# print('Number of stars with both T_eff and Luminosity available:', len(df_filtered))

# # Check for non-empty rows in the specified columns
# non_empty_rows = df_consolidated[
#     df_consolidated['Luminosity [L_Sun]'].notna() &
#     df_consolidated['Radius [R_Sun]'].notna() &
#     df_consolidated['T_eff [K]'].notna() &
#     df_consolidated['Mass [M_Sun]'].notna()
# ]

# # Count the number of such rows
# num_non_empty_rows = len(non_empty_rows)

# print(f"Number of stars with all of 'Luminosity [L_Sun]', 'T_eff [K]', 'Mass [M_Sun]' and 'Radius [R_Sun]' available: {num_non_empty_rows}")

# # Filter out stars with temperature < 3500K or > 7000K
# df_filtered = df_filtered[
#     (df_filtered['T_eff [K]'] >= 3800) & 
#     (df_filtered['T_eff [K]'] <= 7000)
# ]
# # Filter out stars with luminosity < 0.01 or > 4 L_Sun
# df_filtered = df_filtered[
#     (df_filtered['Luminosity [L_Sun]'] < 5.2) & # 5--> 5.2, to include HD23754 (F5IV-V)
#     (df_filtered['Luminosity [L_Sun]'] > 0.05)
# ]
# # df_filtered = df_filtered[df_filtered['Radius [R_Sun]'] < 2] #--> redundant 

# # Filter out stars with density < 0.5 or > 5 solor unit
# if 1:
#     df_filtered = df_filtered[
#         ((df_filtered['Density [Solar unit]'] >= 0.1) & 
#             (df_filtered['Density [Solar unit]'] < 5)) |
#         df_filtered['Radius [R_Sun]'].isna()
#     ]

# # Filter out stars with logg < 3.9 (i.e. sub-giants / giants)    
# df_filtered['logg_gaia'] = pd.to_numeric(df_filtered['logg_gaia'], errors='coerce')
# df_filtered = df_filtered[
#     (df_filtered['logg_gaia'] > 3.8) | 
#     (df_filtered['logg_gaia'].isna())
# ]

# # Save the filtered consolidated data to a new Excel file
# df_filtered.to_excel(directory + 'consolidated_results_kept.xlsx', index=False)
# adjust_column_widths(directory + 'consolidated_results_kept.xlsx')
# display(df_filtered)


# # In[66]:


# # Find entries that were removed during filtering
# df_removed = df_consolidated[~df_consolidated['source_id'].isin(df_filtered['source_id'])]

# # Save the removed entries to Excel
# output_path = directory + 'consolidated_results_removed.xlsx'
# df_removed.to_excel(output_path, index=False)
# adjust_column_widths(output_path)

# # Print some statistics
# print(f"Total entries in df_consolidated: {len(df_consolidated)}")
# print(f"Entries kept in df_filtered: {len(df_filtered)}")
# print(f"Entries removed and saved to file: {len(df_removed)}")


# In[67]:


# combined_df = df_filtered.copy()


# # In[68]:


# len(combined_df)


# # In[69]:


# color = combined_df['Phot BP Mean Mag'] - combined_df['Phot RP Mean Mag']


# # Create a high-resolution plot
# plt.figure(figsize=(6, 4), dpi=150)  # Set the dpi to 300 for high resolution
# plt.hist(color, bins=50, color='skyblue', edgecolor='black')
# plt.xlabel('Color (B - R)')
# plt.ylabel('Frequency')
# plt.title('Color (B - R) Histogram')
# plt.savefig('../figures/color_histogram.png', dpi=300)  # Save the plot as a PNG file
# plt.show()


# # In[70]:


# # Assuming 'color' is a column in combined_df or an array of the same length as the DataFrame
# # For example, if 'color' is the difference between 'Phot BP Mean Mag' and 'Phot RP Mean Mag'
# color = combined_df['Phot BP Mean Mag'] - combined_df['Phot RP Mean Mag']

# # Calculate the conversion factor for colors between 1 and 4
# conv = 0.20220 + 0.02489 * color

# # Use np.where to apply the conversion conditionally
# V_mag = np.where((color >= 1) & (color <= 4),
#                  combined_df['Phot BP Mean Mag'] - conv,
#                  combined_df['Phot G Mean Mag'])

# # Create a high-resolution plot
# plt.figure(figsize=(6, 4), dpi=150)  # Set the dpi to 300 for high resolution

# # Plot the diagram using the color array for point colors
# plt.scatter(combined_df['Phot G Mean Mag'], V_mag, c=color, cmap='viridis', edgecolor='k', s=50, alpha=0.5)
# plt.xlabel('G Magnitude')
# plt.ylabel('V Magnitude')
# # plt.title('Color-Magnitude Diagram')
# plt.colorbar(label='Color $(G_{BP} - G_{RP})$')
# plt.grid()
# plt.savefig('../figures/color_magnitude_diagram.png', dpi=300)
# plt.show()


# In[74]:


# # List of columns to convert to numeric
# columns_to_convert = ['T_eff [K]', 'Mass [M_Sun]', 'Luminosity [L_Sun]', 'Radius [R_Sun]']

# # Convert specified columns to numeric, coercing errors to NaN
# for column in columns_to_convert:
#     combined_df[column] = pd.to_numeric(combined_df[column], errors='coerce')

# # Display the first few rows to verify the conversion
# display(combined_df.head())


# In[81]:


# # Insert the 'V_mag' column right after the 'DEC' column
# if 'V_mag' not in combined_df.columns:
#     combined_df.insert(combined_df.columns.get_loc('DEC') + 1, 'V_mag', V_mag)

if 'T_eff [K]' in combined_df.columns:

    # Calculate the HZ_limit
    hz_limits = combined_df.apply(lambda row: calculate_habitable_zone(row['T_eff [K]'], row['Luminosity [L_Sun]']), axis=1)

    if 'HZ_limit [AU]' not in combined_df.columns:
        # Find the index of the 'Radius [R_Sun]' column
        radius_index = combined_df.columns.get_loc('Radius [R_Sun]')
        # Insert the new column after 'Radius [R_Sun]'        
        combined_df.insert(radius_index + 1, 'HZ_limit [AU]', hz_limits)
    # Sort the DataFrame by temperature (Teff)
    combined_df = combined_df.sort_values('T_eff [K]')    

# Export the combined DataFrame to an Excel file
combined_excel_file = 'combined_query.xlsx'
output_path = directory + combined_excel_file
combined_df.to_excel(output_path, index=False)

print(f"Combined results saved to {output_path}")
display(combined_df.head())

# Adjust the column widths
adjust_column_widths(output_path)


# In[82]:


from rv_prec import calculate_rv_precision

rv_precisions = []

for i in range(len(combined_df)):
    result, rv_precision = calculate_rv_precision(Temp=combined_df.iloc[i]['T_eff [K]'], Vmag=combined_df.iloc[i]['V_mag'])

    rv_precisions.append(rv_precision)


# In[83]:


# Find the index of 'HZ_limit [AU]' column
hz_limit_index = combined_df.columns.get_loc('HZ_limit [AU]')

# Insert the rv_precisions to the right of 'HZ_limit [AU]'
combined_df.insert(hz_limit_index + 1, 'RV precision [m/s]', rv_precisions)


# In[84]:


merged_df = combined_df.copy()

merged_df = merged_df[merged_df['Object Type'] != 'WhiteDwarf']
# Save the updated Excel file
output_path = directory + 'combined_query_with_RV_precision.xlsx'
merged_df.to_excel(output_path, index=False)

# Adjust the column widths
adjust_column_widths(output_path)


# ### Calculate the HZ mass detection limit

# In[85]:


# Calculate detection limits using both methods
merged_df['HZ Detection Limit [M_Earth]'] = merged_df.apply(
    lambda row: calculate_hz_detection_limit(
        row['RV precision [m/s]'],
        row['Mass [M_Sun]'],
        row['HZ_limit [AU]']
    ),
    axis=1
)

merged_df['HZ Detection Limit Simplified [M_Earth]'] = merged_df.apply(
    lambda row: calculate_hz_detection_limit_simplify(
        row['RV precision [m/s]'],
        row['Mass [M_Sun]'],
        row['HZ_limit [AU]']
    ),
    axis=1
)

# Plot the results
plt.figure(figsize=(10, 6))
plt.scatter(merged_df['HZ Detection Limit [M_Earth]'], merged_df['HZ Detection Limit Simplified [M_Earth]'], alpha=0.5)
# plt.plot([0, max(merged_df['HZ Detection Limit [M_Earth]'])], [0, max(merged_df['HZ Detection Limit [M_Earth]'])], 'r--')
plt.xlabel('HZ Detection Limit [M_Earth]')
plt.ylabel('HZ Detection Limit Simplified [M_Earth]')
plt.title('Comparison of HZ Detection Limits')
plt.grid(True)
plt.show()



# In[86]:


# Apply the calculation to each row
merged_df['HZ Detection Limit [M_Earth]'] = merged_df.apply(
    lambda row: calculate_hz_detection_limit(
        row['RV precision [m/s]'],
        row['Mass [M_Sun]'],
        row['HZ_limit [AU]']
    ),
    axis=1
)

# Print some statistics about the new column
print(merged_df['HZ Detection Limit [M_Earth]'].describe())

# Count and print the number of NaN values
nan_count = merged_df['HZ Detection Limit [M_Earth]'].isna().sum()
print(f"Number of NaN values: {nan_count}")

# Reorder the columns to place the new column next to 'Mass'
cols = merged_df.columns.tolist()
mass_index = cols.index('RV precision [m/s]')
cols.insert(mass_index + 1, cols.pop(cols.index('HZ Detection Limit [M_Earth]')))
merged_df = merged_df[cols]

# Save the updated DataFrame
output_path = directory + 'combined_query_with_mass_detection_limit.xlsx'
merged_df.to_excel(output_path, index=False)

# Adjust the column widths
adjust_column_widths(output_path)

print(f"Updated DataFrame saved to '{output_path}'.")


# In[87]:


len(merged_df[merged_df['HZ Detection Limit [M_Earth]'] < 4])


# In[88]:


if 0:
    merged_df = pd.read_excel('../results/combined_query_with_mass_detection_limit.xlsx', dtype={'source_id': str, 'source_id_dr2': str, 'source_id_dr3': str, 'HIP Number': str})


# In[89]:


len(merged_df)


# ### Background stars removal

# In[90]:


# ---------------------------------------------------------------
# Batch query for Gaia DR2/DR3 in parallel with progress tracking
# ---------------------------------------------------------------
import pandas as pd
from astroquery.gaia import Gaia
import time
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm  # Import tqdm for progress tracking

# Function to create a query for nearby stars
# def create_neighbor_query(source_id, ra, dec, neighbor_g_mag_limit, search_radius, data_release):
#     query = f"""
#     SELECT 
#         source_id, ra, dec, phot_g_mean_mag
#     FROM 
#         {data_release}.gaia_source
#     WHERE 
#         1=CONTAINS(
#             POINT('ICRS', {ra}, {dec}),
#             CIRCLE('ICRS', ra, dec, {search_radius})
#         )
#         AND phot_g_mean_mag < {neighbor_g_mag_limit}
#         AND source_id != {source_id}
#     """
#     return query

# Function to process each row with retry logic
# def process_row_with_retry(row, max_retries=3, delay=5):
#     attempt = 0
#     while attempt < max_retries:
#         try:
#             if not pd.isna(row['source_id_dr3']):
#                 query = create_neighbor_query(
#                     source_id=row['source_id_dr3'],
#                     ra=row['RA'],
#                     dec=row['DEC'],
#                     neighbor_g_mag_limit=row['Phot G Mean Mag']+3,
#                     search_radius=SEARCH_RADIUS,
#                     data_release='gaiadr3'
#                 )
#             else:
#                 query = create_neighbor_query(
#                     source_id=row['source_id_dr2'],
#                     ra=row['RA'],
#                     dec=row['DEC'],
#                     neighbor_g_mag_limit=row['Phot G Mean Mag']+3,
#                     search_radius=SEARCH_RADIUS,
#                     data_release='gaiadr2'
#                 )
            
#             # Execute the query
#             neighbors_df = execute_gaia_query(query)
            
#             # Check if bright neighbors exist
#             if neighbors_df is not None and not neighbors_df.empty:
#                 return (row, True)
#             else:
#                 return (row, False)
#         except Exception as e:
#             print(f"An error occurred: {e}")
#             attempt += 1
#             if attempt < max_retries:
#                 print(f"Retrying in {delay} seconds... (Attempt {attempt}/{max_retries})")
#                 time.sleep(delay)
#             else:
#                 print("Max retries reached. Skipping this row.")
#                 return (row, False)

# DataFrames: merged_df (input sources)
# rows_with_bright_neighbors = []
# rows_without_bright_neighbors = []

# # Use ThreadPoolExecutor to process rows in parallel with progress tracking
# with ThreadPoolExecutor() as executor:
#     results = list(tqdm(executor.map(process_row_with_retry, [row for idx, row in merged_df.iterrows()]), total=len(merged_df), desc="Processing rows"))
#     for row, has_bright_neighbors in results:
#         if has_bright_neighbors:
#             rows_with_bright_neighbors.append(row)
#         else:
#             rows_without_bright_neighbors.append(row)

# Create DataFrames for rows with and without bright neighbors
bright_neighbors_df = pd.DataFrame(rows_with_bright_neighbors)
rows_without_bright_neighbors_df = pd.DataFrame(rows_without_bright_neighbors)


# In[91]:


print(f"Stars with bright neighbors: {len(bright_neighbors_df)}")
output_path = directory + 'stars_with_bright_neighbors.xlsx'
bright_neighbors_df.to_excel(output_path, index=False)
adjust_column_widths(output_path)

print(f"Stars without bright neighbors: {len(rows_without_bright_neighbors_df)}")
output_path = directory + 'stars_without_bright_neighbors.xlsx'
rows_without_bright_neighbors_df.to_excel(output_path, index=False)
adjust_column_widths(output_path)


# In[92]:


merged_df = rows_without_bright_neighbors_df.copy()


# ### Statistics and plots

# In[93]:


print(merged_df.columns.tolist())


# In[94]:


import seaborn as sns

def plot_scatter(x, y, data, xlabel, ylabel, xlim=None, ylim=None, filename=None, color=None, alpha=0.7, size=60, invert_xaxis=False, x2=None, y2=None, data2=None, color2=None, alpha2=0.7, size2=60):
    """Creates and saves a scatter plot with the given parameters, with an option to add a second group of data."""
    plt.figure(figsize=(6, 4), dpi=150)
    
    # Plot the first group of data
    if color is not None:
        sns.scatterplot(x=x, y=y, data=data, color=color, alpha=alpha, s=size)
    else:
        sns.scatterplot(x=x, y=y, data=data, alpha=alpha, s=size)

    # Plot the second group of data if provided
    if x2 is not None and y2 is not None and data2 is not None:
        if color2 is not None:
            sns.scatterplot(x=x2, y=y2, data=data2, color=color2, alpha=alpha2, s=size2, marker='+', linewidth=2)
        else:
            sns.scatterplot(x=x2, y=y2, data=data2, alpha=alpha2, s=size2, marker='+', linewidth=2)

    # Customize the plot
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    if xlim:
        plt.xlim(xlim)
    if ylim:
        plt.ylim(ylim)

    # Add a grid for better readability
    plt.grid(True, linestyle='--', alpha=0.7)

    # Invert x-axis if specified
    if invert_xaxis:
        plt.gca().invert_xaxis()

    # Customize tick labels
    plt.tick_params(axis='both', which='major')

    # Adjust layout and display the plot
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=300)
    plt.show()

# Use the function to create the RA vs DEC plot
plot_scatter(
    x='RA',
    y='DEC',
    data=merged_df,
    xlabel='Right Ascension (RA)',
    ylabel='Declination (DEC)',
    xlim=(0, 360),
    
    filename='../figures/ra_dec.png',
    alpha=0.6,
    invert_xaxis=True
)


# In[95]:


import numpy as np
import matplotlib.pyplot as plt

# Assuming 'merged_df' is your DataFrame

# Plotting the H-R Diagram with color transition from red to yellow
plt.figure(figsize=(10, 8), dpi=150)
plt.scatter(
    merged_df['T_eff [K]'], 
    merged_df['Luminosity [L_Sun]'], 
    c=merged_df['T_eff [K]'],  # Color by temperature
    cmap='autumn',  # Use autumn colormap for red to yellow transition
    alpha=0.99, 
    edgecolors='w',  # Use white for edges
    linewidths=0.05,  # Set edge width
    s=merged_df['Radius [R_Sun]'] * 20  # Scale the radius for visibility
)
plt.colorbar(label='Effective Temperature (K)')  # Add color bar
plt.xscale('log')
plt.yscale('log')
plt.xlim(min(merged_df['T_eff [K]'])-50, max(merged_df['T_eff [K]'])+50)  # Set the same x range
plt.ylim(min(merged_df['Luminosity [L_Sun]']), max(merged_df['Luminosity [L_Sun]'])+0.5)  # Set the same y range
plt.gca().invert_xaxis()  # Invert x-axis for temperature
plt.xlabel('Effective Temperature (K)')
plt.ylabel('Luminosity (L/L_sun)')
plt.title('Hertzsprung-Russell Diagram')
plt.grid(True, which="both", ls="--", linewidth=0.5)
plt.savefig('../figures/HR_diagram.png')
plt.show()


# In[96]:


def plot_stellar_properties_vs_temperature(merged_df, detection_limit):
    """
    Plots various stellar properties as a function of effective temperature.

    Parameters:
    merged_df (DataFrame): DataFrame containing stellar data with columns for effective temperature and other properties.
    """
    fig, axs = plt.subplots(2, 4, figsize=(12, 6), dpi=300)

    # Flatten the axs array for easier iteration
    axs = axs.flatten()

    # List of columns to plot, including the new Density column
    columns = [
        'V_mag', 'Mass [M_Sun]', 'Luminosity [L_Sun]', 'Radius [R_Sun]', 
        'Density [Solar unit]', 'HZ_limit [AU]', 'RV precision [m/s]', 'HZ Detection Limit [M_Earth]'
    ]

    # Define a color palette
    colors = plt.cm.viridis(np.linspace(0, 1, len(columns)))

    # Plot each column
    for i, col in enumerate(columns):
        axs[i].scatter(merged_df['T_eff [K]'], merged_df[col], alpha=0.2, color=colors[i], s=10)
        axs[i].set_xlabel('$T_{eff}$ (K)', fontsize=12)
        axs[i].set_ylabel(col, fontsize=12)
        axs[i].grid(True, linestyle='--', alpha=0.6)

    # Add a main title
    fig.suptitle('Stellar Properties as a Function of Temperature (' + str(len(merged_df)) + ' < ' + str(detection_limit) + ' M_Earth)', fontsize=14)

    # Adjust the layout and display the plot
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('../figures/stellar_properties_vs_temperature.png')
    plt.show()


# In[97]:


import numpy as np
import matplotlib.pyplot as plt

# Assuming 'merged_df' is your DataFrame

# Option to switch between original and filtered plot
use_filtered_data = True  # Set to False to use the original data

if use_filtered_data:
    # Filter the DataFrame to include only rows where 'HZ Detection Limit [M_Earth]' is less than 1.5
    Detection_Limit = 1.5
    data_to_plot = merged_df[merged_df['HZ Detection Limit [M_Earth]'] <= Detection_Limit]
    color_data = data_to_plot['HZ Detection Limit [M_Earth]']
    colorbar_label = 'HZ Detection Limit [M_Earth] (Capped at ' + str(Detection_Limit) + ')'
    filename = '../figures/HR_diagram_HZ_detection_limit_filtered.png'
    print('Number of stars with HZ Detection Limit <= ' + str(Detection_Limit) + ':', len(data_to_plot))
    plot_stellar_properties_vs_temperature(data_to_plot, Detection_Limit)
else:
    # Use the original data with capped values
    data_to_plot = merged_df
    color_data = np.minimum(merged_df['HZ Detection Limit [M_Earth]'], 4)
    colorbar_label = 'HZ Detection Limit [M_Earth] (Capped at 4)'
    filename = '../figures/HR_diagram_HZ_detection_limit.png'

# Plotting the H-R Diagram without gradient background and with color-coded circles by HZ Detection Limit
plt.figure(figsize=(10, 8), dpi=150)

# Scatter plot with circles color-coded by HZ Detection Limit
sc = plt.scatter(
    data_to_plot['T_eff [K]'], 
    data_to_plot['Luminosity [L_Sun]'], 
    c=color_data,  # Use appropriate color data
    cmap='viridis',  # Use viridis colormap for detection limit
    alpha=0.99, 
    edgecolors='grey',  # Use grey for edges
    linewidths=0.05,  # Set edge width
    s=data_to_plot['Radius [R_Sun]'] * 20  # Scale the radius for visibility
)

# Add a color bar with the same scaling
cbar = plt.colorbar(sc, label=colorbar_label)
sc.set_clim(0, 4)  # Set color bar limits to maintain the same scaling

plt.xscale('log')
plt.yscale('log')
plt.xlim(min(merged_df['T_eff [K]'])-50, max(merged_df['T_eff [K]'])+50)  # Set the same x range
plt.ylim(min(merged_df['Luminosity [L_Sun]']), max(merged_df['Luminosity [L_Sun]'])+0.5)  # Set the same y range
plt.gca().invert_xaxis()  # Invert x-axis for temperature
plt.xlabel('Effective Temperature (K)')
plt.ylabel('Luminosity (L/L_sun)')
plt.title('Hertzsprung-Russell Diagram')
plt.grid(True, which="both", ls="--", linewidth=0.5)
plt.savefig(filename)
plt.show()



# In[98]:


# Set the theme for the plot
sns.set_theme(style="whitegrid")

# Calculate distance and add it to the DataFrame
merged_df['Distance [pc]'] = 1000 / merged_df['Parallax']


columns_to_plot = [
    'V_mag', 'Phot G Mean Mag', 'Phot BP Mean Mag', 'Distance [pc]',
    'T_eff [K]', 'Luminosity [L_Sun]', 'Mass [M_Sun]', 'Radius [R_Sun]', 
    'Density [Solar unit]', 'HZ_limit [AU]', 'RV precision [m/s]', 'HZ Detection Limit [M_Earth]'
]


# Define reversed grayscale colors for each group
group_colors = {
    'Brightest': 'red',
    'Bright': 'orange',
    'Dim': 'green',
    'Dimmer': 'blue',
    'Dimmest': 'black'
}

def plot_histograms(df, title, filename):
    # Divide V_mag into 5 groups and assign reversed grayscale colors
    v_mag_bins = np.linspace(df['V_mag'].min(), df['V_mag'].max(), 6)
    df['V_mag_group'] = pd.cut(df['V_mag'], bins=v_mag_bins, labels=group_colors.keys())

    # Set up the plot
    fig, axes = plt.subplots(3, 4, figsize=(12, 8), dpi=150)
    fig.suptitle(title + ' (sample = ' + str(len(df)) + ')', fontsize=16)

    # Flatten the axes array for easy iteration
    axes = axes.flatten()

    # Create histograms with color coding
    for i, column in enumerate(columns_to_plot):
        if column in df.columns:
            sns.histplot(data=df, x=column, ax=axes[i], hue='V_mag_group', palette=group_colors, multiple='stack', legend=False)
            axes[i].set_xlabel(column)
            axes[i].set_ylabel('Count')
        else:
            axes[i].set_visible(False)  # Hide axes if column is not available

    # Adjust layout and display
    plt.tight_layout()
    plt.subplots_adjust(top=0.93, hspace=0.5, wspace=0.4, left=0.05, right=0.99)
    plt.savefig(filename)
    plt.show()

# Plot for the entire DataFrame
plot_histograms(merged_df, 'All Data', '../figures/star_properties_histograms.png')

# save the merged_df to an excel file
output_path = directory + 'Gaia_homogeneous_target_selection_Jinglin_2025.01.09.xlsx'
merged_df.sort_values(by='HZ Detection Limit Simplified [M_Earth]').to_excel(output_path, index=False)
adjust_column_widths(output_path)

# Plot for HZ Detection Limit < 4
filtered_df_4 = merged_df[merged_df['HZ Detection Limit [M_Earth]'] < 4].copy()
plot_histograms(filtered_df_4, 'HZ Detection Limit [M_Earth] < 4', '../figures/star_properties_histograms_filtered_4.png')

# save the filtered_df_4 to an excel file
output_path = directory + 'Gaia_homogeneous_target_selection_M_earth_4_Jinglin_2025.01.09.xlsx'
filtered_df_4.sort_values(by='HZ Detection Limit Simplified [M_Earth]').to_excel(output_path, index=False)
adjust_column_widths(output_path)

# Plot for HZ Detection Limit < 1.5
filtered_df_1_5 = merged_df[merged_df['HZ Detection Limit [M_Earth]'] < 1.5].copy()
plot_histograms(filtered_df_1_5, 'HZ Detection Limit [M_Earth] < 1.5', '../figures/star_properties_histograms_filtered_1_5.png')

# save the filtered_df_1_5 to an excel file
output_path = directory + 'Gaia_homogeneous_target_selection_M_earth_1.5_Jinglin_2025.01.09.xlsx'
filtered_df_1_5.sort_values(by='HZ Detection Limit Simplified [M_Earth]').to_excel(output_path, index=False)
adjust_column_widths(output_path)



# ### Comparison with Ralf's results

# In[99]:


# df_Ralf = pd.read_excel('../data/Ralf/2ES_targetlist_astrid_export_2024Nov_comments.xlsx', engine='openpyxl', header=1)
df_Ralf = pd.read_excel('../data/Ralf/2ES_targetlist_astrid_export_2024Dec_comments.xlsx', engine='openpyxl', header=1)
df_Ralf = df_Ralf[df_Ralf['prio'] != 3]

print(len(df_Ralf))
print(df_Ralf.columns)


# Merge df_Ralf and merged_df for the same stars. 

# In[100]:


# ---------------------------------------------------------------- #
#  HD
# ---------------------------------------------------------------- #
# Split 'HD Number' into two separate columns 'HD Number 1' and 'HD Number 2'
merged_df[['HD Number 1', 'HD Number 2']] = merged_df['HD Number'].str.split(', ', expand=True, n=1)
# Clean up 'HD Number 1' and 'HD Number 2' by removing extra spaces after 'HD'
merged_df['HD Number 1'] = merged_df['HD Number 1'].str.replace(r'HD\s+', 'HD', regex=True)
merged_df['HD Number 2'] = merged_df['HD Number 2'].fillna('').str.replace(r'HD\s+', 'HD', regex=True)

# ---------------------------------------------------------------- #
#  HIP
# ---------------------------------------------------------------- #
merged_df['HIP Number'] = merged_df['HIP Number'].apply(lambda x: f'HIP{x}' if pd.notna(x) and x != '' and not str(x).startswith('HIP') else x)

# ---------------------------------------------------------------- #
#  GJ
# ---------------------------------------------------------------- #    
# Split 'GJ Number' into two separate columns 'GJ Number 1' and 'GJ Number 2'
merged_df[['GJ Number 1', 'GJ Number 2']] = merged_df['GJ Number'].str.split(', ', expand=True, n=1)
# Clean up 'GJ Number 1' and 'GJ Number 2' by removing extra spaces after 'GJ'
merged_df['GJ Number 1'] = merged_df['GJ Number 1'].str.replace(r'\s+', '', regex=True)
merged_df['GJ Number 2'] = merged_df['GJ Number 2'].fillna('').str.replace(r'\s+', '', regex=True)


# ---------------------------------------------------------------- #
#  Merge
# ---------------------------------------------------------------- #
# Perform left merges on various columns and combine results
merge_keys = ['HD Number 1', 'HD Number 2', 'HIP Number', 'GJ Number 1', 'GJ Number 2']
merged_RJ = pd.concat([df_Ralf.merge(merged_df, left_on='star_ID  ', right_on=key, how='left') for key in merge_keys])

# Sort the combined DataFrame by 'source_id' to prioritize non-null values
merged_RJ.sort_values(by='source_id', ascending=False, inplace=True)

# Remove duplicate entries based on 'star_ID  ', keeping the first occurrence
merged_RJ.drop_duplicates(subset='star_ID  ', keep='first', inplace=True)

# Reset the index of the final DataFrame
merged_RJ.reset_index(drop=True, inplace=True)

# Save the final DataFrame to an Excel file
filename = '../results/merged_RJ_2025.01.09.xlsx'
merged_RJ.sort_values(by=['prio', 'HD Number'], ascending=[True, True], inplace=True)

# Apply color formatting to fonts of the whole row based on 'prio' values
# and highlight rows where 'source_id' is missing in yellow
def apply_color_formatting(workbook, worksheet):
    dark_green_format = workbook.add_format({'font_color': '#006400'})
    orange_format = workbook.add_format({'font_color': '#FFA500'})
    yellow_format = workbook.add_format({'bg_color': '#FFFF00'})  # Yellow background for missing 'source_id'
    
    for row_num, (prio_value, source_id) in enumerate(zip(merged_RJ['prio'], merged_RJ['source_id']), start=1):
        if pd.isna(source_id):
            worksheet.set_row(row_num, None, yellow_format)
        elif prio_value == 0:
            worksheet.set_row(row_num, None, dark_green_format)
        elif prio_value == 1:
            worksheet.set_row(row_num, None, orange_format)

# Save the DataFrame to an Excel file with formatting
try:
    with pd.ExcelWriter(filename, engine='xlsxwriter') as writer:
        merged_RJ.to_excel(writer, index=False)
        workbook = writer.book
        worksheet = writer.sheets['Sheet1']
        apply_color_formatting(workbook, worksheet)
except ModuleNotFoundError:
    print("xlsxwriter module not found. Please install it using 'pip install xlsxwriter'.")

adjust_column_widths(filename)


# In[104]:


def plot_scatter_with_options(df, col_x, col_y, min_value=None, max_value=None, label=False):
    # Create a scatter plot with color mapping
    plt.figure(figsize=(8, 6), dpi=150)
    scatter = plt.scatter(
        df[col_x], 
        df[col_y], 
        c=df['T_eff [K]'], 
        cmap='autumn', 
        edgecolor='k', 
        alpha=0.7
    )

    # Add titles and labels
    # plt.title('Selection Crossmatch', fontsize=14)
    plt.xlabel(f'{col_x} (Ralf)', fontsize=12)
    plt.ylabel(f'{col_y} (Jinglin)', fontsize=12)

    # Add a color bar
    cbar = plt.colorbar(scatter)
    cbar.set_label('T_eff [K]', fontsize=12)

    # Add grid for better readability
    plt.grid(True, linestyle='--', alpha=0.7)

    # Plot the x = y line
    if min_value is None:
        min_value = min(min(df[col_x]), min(df[col_y]))
    if max_value is None:
        max_value = max(max(df[col_x]), max(df[col_y]))
    plt.plot([min_value, max_value], [min_value, max_value], color='gray', linestyle='--')
    plt.xlim(min_value, max_value)
    plt.ylim(min_value, max_value)

    if label:
        # Add labels to each point
        x_range = max_value - min_value
        x_offset = x_range * 0.01
        for i, name in enumerate(df['star_ID  ']):
            if (df[col_x][i] > min_value) and (df[col_x][i] < max_value) and (df[col_y][i] > min_value) and (df[col_y][i] < max_value):
                plt.text(df[col_x][i] - x_offset, df[col_y][i], name, fontsize=5, ha='right')    

    plt.tight_layout()
    plt.savefig('../figures/crossmatch_' + col_y.strip().replace(" ", "_").replace("[", "").replace("]", "").replace("/", "") + '.png')
    # Show the plot
    plt.show()


# In[105]:


merged_RJ


# In[106]:


merged_RJ['HZ Rmid'] = (merged_RJ['HZ Rin'] + merged_RJ['HZ Rout']) / 2
plot_scatter_with_options(merged_RJ, 'magV     ', 'V_mag', min_value = 3, max_value = 10)
plot_scatter_with_options(merged_RJ, 'mass ', 'Mass [M_Sun]', min_value = 0.5, max_value = 1.4)
plot_scatter_with_options(merged_RJ, 'HZ Rmid', 'HZ_limit [AU]', min_value = 0.1, max_value = 2, label=True)
plot_scatter_with_options(merged_RJ, 'logg', 'logg_gaia', min_value = 2, max_value = 5, label=True)
plot_scatter_with_options(merged_RJ, 'RV_Prec(390-870) 30m', 'RV precision [m/s]', min_value = 0, max_value = 1.6, label=True)

plot_scatter_with_options(merged_RJ, 'mdl(hz) 30min', 'HZ Detection Limit [M_Earth]', min_value = 0, max_value = 3, label=True)



# In[107]:


i = 7
colors = plt.cm.viridis(np.linspace(0, 1, 8))

# Use the function to create the plot
plot_scatter(
    x='T_eff [K]',
    y='RV precision [m/s]',
    data=merged_df,
    xlabel='Stellar Temperature (K)',
    ylabel='RV precision [m/s]',
    xlim=(min(min(merged_df['T_eff [K]']), min(df_Ralf['Teff '])) - 100, max(max(merged_df['T_eff [K]']), max(df_Ralf['Teff '])) + 100),
    ylim=(0, 2),
    filename='../figures/RV_precision_vs_temperature.png',
    color=colors[i-1],  # Assuming 'colors' is defined and 'i' is an integer index
    x2 = 'Teff ', 
    y2 = 'RV_Prec(390-870) 30m',
    data2 = df_Ralf,
    color2 = 'red'
)

plot_scatter(
    x='T_eff [K]',
    y='HZ Detection Limit [M_Earth]',
    data=merged_df,
    xlabel='Stellar Temperature (K)',
    ylabel='HZ Detection Limit (M_Earth)',
    xlim=(min(merged_df['T_eff [K]']) - 200, 6000 + 500),
    ylim=(0, 10),
    filename='../figures/HZ_detection_limit_vs_temperature_full.png',
    color=colors[i],  # Replace with actual color if using a list
    x2 = 'Teff ', 
    y2 = 'mdl(hz) 30min',
    data2 = df_Ralf,
    color2 = 'red'
)

plot_scatter(
    x='T_eff [K]',
    y='HZ Detection Limit [M_Earth]',
    data=merged_df,
    xlabel='Stellar Temperature (K)',
    ylabel='HZ Detection Limit (M_Earth)',
    xlim=(min(merged_df['T_eff [K]']) - 200, 6000 + 100),
    ylim=(0, 1.5),
    filename='../figures/HZ_detection_limit_vs_temperature_zoomed.png',
    color=colors[i],  # Replace with actual color if using a list
    x2 = 'Teff ', 
    y2 = 'mdl(hz) 30min',
    data2 = df_Ralf,
    color2 = 'red'
)


# In[108]:


print("my results:")
print("Number of stars:", len(merged_df))
print("Number of stars with HZ Detection Limit [M_Earth] < 4:", len(merged_df[merged_df['HZ Detection Limit [M_Earth]'] < 4]))
print("Number of stars with HZ Detection Limit [M_Earth] < 1.5:", len(merged_df[merged_df['HZ Detection Limit [M_Earth]'] < 1.5]))

print("\nRalf's results:")
print("Number of stars:", len(merged_RJ))
print("Number of stars with HZ Detection Limit [M_Earth] < 4:", len(merged_RJ[merged_RJ['mdl(hz) 30min'] < 4]))
print("Number of stars with HZ Detection Limit [M_Earth] < 1.5:", len(merged_RJ[merged_RJ['mdl(hz) 30min'] < 1.5]))


