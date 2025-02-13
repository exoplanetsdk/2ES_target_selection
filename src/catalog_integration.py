import pandas as pd
import numpy as np
from tqdm import tqdm
from config import RESULTS_DIRECTORY
from utils import save_and_adjust_column_widths
from stellar_properties import get_star_properties_with_retries, get_empirical_stellar_parameters    
from concurrent.futures import ThreadPoolExecutor



class CatalogProcessor:
    def __init__(self, celesta_path, stellar_catalog_path):
        self.celesta_path = celesta_path
        self.stellar_catalog_path = stellar_catalog_path
        self.load_catalogs()

    def load_catalogs(self):
        """Load CELESTA and stellar catalogs"""
        # Define CELESTA column specifications
        colspecs = [
            (39, 45),  # Num    
            (10, 15),  # Teff
            (16, 28),  # Lum
        ]
        column_names = ["HIP Number", "T_eff [K]", "Luminosity [L_Sun]"]
        
        # Load CELESTA catalog
        self.df_CELESTA = pd.read_fwf(
            self.celesta_path, 
            colspecs=colspecs, 
            names=column_names, 
            skiprows=28
        )
        
        # Load stellar catalog
        with open(self.stellar_catalog_path, 'r') as file:
            self.STELLAR_CATALOG = file.readlines()

    def process_hip_data(self, df_consolidated):
        print("Filling missing T_eff and Luminosity from CELESTA catalog using HIP numbers")
        # Extract numeric HIP numbers
        df_consolidated['HIP Number'] = df_consolidated['HIP Number'].str.extract(r'HIP\s*(\d+)')
        self.df_CELESTA['HIP Number'] = self.df_CELESTA['HIP Number'].astype(str)
        
        # Merge dataframes
        merged_df = pd.merge(
            df_consolidated, 
            self.df_CELESTA[['HIP Number', 'T_eff [K]', 'Luminosity [L_Sun]']],
            on='HIP Number', 
            suffixes=('', '_CELESTA'), 
            how='left'
        )
        
        # Fill missing values
        merged_df['T_eff [K]'] = merged_df['T_eff [K]'].fillna(merged_df['T_eff [K]_CELESTA'])
        merged_df['Luminosity [L_Sun]'] = merged_df['Luminosity [L_Sun]'].fillna(merged_df['Luminosity [L_Sun]_CELESTA'])
        
        # Drop temporary columns
        df_consolidated_HIP = merged_df.drop(columns=['T_eff [K]_CELESTA', 'Luminosity [L_Sun]_CELESTA'])
        
        return df_consolidated_HIP

    def extract_mass(self, hd_number):
        """Extract mass for a given HD number from stellar catalog"""
        for line in self.STELLAR_CATALOG:
            line_hd_number = line[7:18].strip()
            if line_hd_number == f"HD {hd_number}":
                mass = line[130:134].strip()
                return float(mass) if mass else None
        return None
    

    def process_hd_data(self, df_consolidated_HIP):
        print("Filling missing T_eff, Luminosity and Mass from Vizier V/117A using HD numbers")
        df_consolidated_HD = df_consolidated_HIP.copy()

        def clean_hd_number(hd_string):
            """Extract the first HD number from a given string."""
            if pd.isna(hd_string) or not hd_string.startswith("HD"):
                return None
            # Split by comma and take the first part, then remove "HD" and whitespace
            return hd_string.split(',')[0].replace("HD", "").strip()           
        
        # Define a function to process each row
        def process_row(index, row):
            if pd.isna(row["T_eff [K]"]) or pd.isna(row["Luminosity [L_Sun]"]) or pd.isna(row["Mass [M_Sun]"]):
                hd_number = clean_hd_number(row["HD Number"])
                
                if hd_number:
                    temperature, luminosity = get_star_properties_with_retries(hd_number)
                    mass = self.extract_mass(hd_number)
                    
                    if pd.isna(row["T_eff [K]"]) and temperature is not None:
                        df_consolidated_HD.at[index, "T_eff [K]"] = temperature
                    if pd.isna(row["Luminosity [L_Sun]"]) and luminosity is not None:
                        df_consolidated_HD.at[index, "Luminosity [L_Sun]"] = luminosity
                    if pd.isna(row["Mass [M_Sun]"]) and mass is not None:
                        df_consolidated_HD.at[index, "Mass [M_Sun]"] = mass

        # Use ThreadPoolExecutor to process rows in parallel
        with ThreadPoolExecutor() as executor:
            list(tqdm(executor.map(lambda args: process_row(*args), df_consolidated_HD.iterrows()), 
                      total=df_consolidated_HD.shape[0], 
                      desc="Progress",
                      ncols=100))
        
        return df_consolidated_HD

    def calculate_density(self, df):
        """Calculate stellar density in solar units"""
        df['Density [Solar unit]'] = None
        mask = df['Mass [M_Sun]'].notna() & df['Radius [R_Sun]'].notna()
        df.loc[mask, 'Density [Solar unit]'] = df['Mass [M_Sun]'] / (df['Radius [R_Sun]'] ** 3)
        
        # Reorder columns to place density after radius
        cols = df.columns.tolist()
        density_index = cols.index('Density [Solar unit]')
        radius_index = cols.index('Radius [R_Sun]')
        cols.insert(radius_index + 1, cols.pop(density_index))
        
        return df[cols]

    def process_catalogs(self, df_consolidated):
        """Main processing pipeline"""
        # Process HIP data
        df_consolidated_HIP = self.process_hip_data(df_consolidated)
        save_and_adjust_column_widths(df_consolidated_HIP, f"{RESULTS_DIRECTORY}consolidated_HIP_results.xlsx")
        
        # Process HD data
        df_consolidated_HD = self.process_hd_data(df_consolidated_HIP)
        save_and_adjust_column_widths(df_consolidated_HD, f"{RESULTS_DIRECTORY}consolidated_HD_results.xlsx")
        
        # Process Simbad data
        df_consolidated_simbad = get_empirical_stellar_parameters(df_consolidated_HD)
        save_and_adjust_column_widths(df_consolidated_simbad, f"{RESULTS_DIRECTORY}consolidated_Simbad_results.xlsx")
        
        # Calculate density
        final_df = self.calculate_density(df_consolidated_simbad)

        return final_df