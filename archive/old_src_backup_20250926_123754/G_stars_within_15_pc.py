import pandas as pd
import numpy as np
from config import *

# Load the dataset
def load_data(file_path):
    # Read the Excel file with specified data types
    df = pd.read_excel(file_path, dtype={'source_id': str, 'source_id_dr2': str, 'source_id_dr3': str, 'HIP Number': str})
    return df

# Function to find G-type stars within a specified distance
def find_g_stars_within_distance(df, max_distance_pc=15):
    # Convert distance to numeric, handling any potential errors
    df['Distance [pc]'] = pd.to_numeric(df['Distance [pc]'], errors='coerce')
    
    # Convert temperature to numeric, handling any potential errors
    df['T_eff [K]'] = pd.to_numeric(df['T_eff [K]'], errors='coerce')
    
    # Filter for stars within the specified distance
    distance_mask = df['Distance [pc]'] <= max_distance_pc
    
    # Filter for stars with temperature between 5300 and 6000 K
    temperature_mask = (df['T_eff [K]'] >= 5300) & (df['T_eff [K]'] <= 6000)
    
    # Only filter by temperature and distance, do not use spectral type information
    g_stars_nearby = df[distance_mask & temperature_mask]
    
    return g_stars_nearby

def process_file(file_path, output_csv):
    df = load_data(file_path)
    g_stars_nearby = find_g_stars_within_distance(df, 15)
    print(f"Total number of G-type stars within 20 pc in '{file_path}': {len(g_stars_nearby)}")
    g_stars_nearby.to_csv(output_csv, index=False)
    print(f"\nResults saved to '{output_csv}'")

def main():
    files = [
        (RESULTS_DIRECTORY+"Gaia_homogeneous_target_selection_M_earth_1_5_2025.03.10.xlsx", RESULTS_DIRECTORY+"g_stars_within_15pc_M_earth_1_5.csv"),
        (RESULTS_DIRECTORY+"Gaia_homogeneous_target_selection_M_earth_2_2025.03.10.xlsx", RESULTS_DIRECTORY+"g_stars_within_15pc_M_earth_2.csv"),
        (RESULTS_DIRECTORY+"Gaia_homogeneous_target_selection_M_earth_4_2025.03.10.xlsx", RESULTS_DIRECTORY+"g_stars_within_15pc_M_earth_4.csv"),
    ]
    for file_path, output_csv in files:
        print(f"\nProcessing file: {file_path}")
        process_file(file_path, output_csv)

if __name__ == "__main__":
    main()