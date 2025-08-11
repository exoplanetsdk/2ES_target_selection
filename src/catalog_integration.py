import pandas as pd
import numpy as np
from tqdm import tqdm
from config import RESULTS_DIRECTORY
from utils import save_and_adjust_column_widths
from stellar_properties import get_star_properties_with_retries, get_empirical_stellar_parameters    
from concurrent.futures import ThreadPoolExecutor
import re
import warnings
warnings.filterwarnings('ignore')


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
        print("\nFilling missing T_eff and Luminosity from CELESTA catalog using HIP numbers")
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
        print("\nFilling missing T_eff, Luminosity and Mass from Vizier V/117A using HD numbers")
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
    


def load_catalog_simple(catalog_file='../data/catalog_BoroSaikia2018.dat'):
    """Load the simplified R'HK catalog"""
    print("🔍 Loading R'HK catalog...")
    
    # Read the catalog line by line and parse manually
    catalog_data = []
    
    with open(catalog_file, 'r') as f:
        for line in f:
            if line.strip():
                parts = line.split()
                if len(parts) >= 4:  # At least seq, name, sptype, rhk
                    seq = int(parts[0])
                    name = parts[1]
                    sptype = parts[2]
                    
                    # Find the R'HK value (last number, typically negative)
                    rhk_value = None
                    for part in reversed(parts):
                        try:
                            val = float(part)
                            if val < 0:  # R'HK values are typically negative
                                rhk_value = val
                                break
                        except:
                            continue
                    
                    # Try to extract coordinates if present
                    ra, dec = None, None
                    if len(parts) >= 6:
                        try:
                            ra = float(parts[-3])
                            dec = float(parts[-2])
                        except:
                            pass
                    
                    catalog_data.append({
                        'Seq': seq,
                        'Name': name,
                        'SpType': sptype,
                        'RAdeg': ra,
                        'DEdeg': dec,
                        'logRpHK': rhk_value
                    })
    
    catalog = pd.DataFrame(catalog_data)
    
    print(f"✅ Loaded {len(catalog)} catalog entries")
    print(f"✅ Found {catalog['logRpHK'].notna().sum()} stars with R'HK measurements")
    
    return catalog

def extract_hd_number(name_or_hd):
    """Extract HD number from various formats"""
    if pd.isna(name_or_hd):
        return None
    
    name_str = str(name_or_hd).strip()
    
    # Look for HD followed by numbers
    hd_match = re.search(r'HD\s*(\d+)', name_str, re.IGNORECASE)
    if hd_match:
        return int(hd_match.group(1))
    
    # If it's just a number, assume it's an HD number
    if name_str.isdigit():
        return int(name_str)
    
    return None

def find_rhk_for_star(star, catalog):
    """
    Find R'HK value for a single star - COMPLETELY FIXED VERSION
    
    This function searches for R'HK values using multiple identification methods,
    now with EXACT matching to prevent false positives.
    """
    # Initialize results
    results = {
        'log_rhk': np.nan,
        'rhk_source': ''
    }
    
    # Method 1: HD Number matching (FIXED - now uses exact matching)
    if 'HD Number' in star and pd.notna(star['HD Number']):
        hd_value = str(star['HD Number']).strip()
        
        # Extract HD number using regex
        hd_match = re.search(r'HD\s*(\d+)', hd_value, re.IGNORECASE)
        if hd_match:
            hd_number = hd_match.group(1)
            
            # FIXED: Use exact matching with regex
            matches = catalog[catalog['Name'].str.match(f'^HD\\s*{hd_number}$', case=False, na=False)]
            
            if len(matches) > 0:
                best_match = matches.iloc[0]
                results['log_rhk'] = best_match['logRpHK']
                results['rhk_source'] = 'HD_number_match'
                return results
    
    # Method 2: GJ Number matching (FIXED - now uses exact matching)
    if 'GJ Number' in star and pd.notna(star['GJ Number']):
        gj_value = str(star['GJ Number']).strip()
        
        # Extract GJ number
        gj_match = re.search(r'GJ\s*([\d.]+)', gj_value, re.IGNORECASE)
        if gj_match:
            gj_number = gj_match.group(1)
            
            # FIXED: Use exact matching with regex
            matches = catalog[catalog['Name'].str.match(f'^GJ\\s*{re.escape(gj_number)}$', case=False, na=False)]
            
            if len(matches) > 0:
                best_match = matches.iloc[0]
                results['log_rhk'] = best_match['logRpHK']
                results['rhk_source'] = 'GJ_number_match'
                return results
    
    # Method 3: HIP Number matching (FIXED - now uses exact matching)
    if 'HIP Number' in star and pd.notna(star['HIP Number']):
        hip_value = str(star['HIP Number']).strip()
        
        # The HIP number in the input is just the number, but in the catalog it's 'HIP <number>'
        # So we match 'HIP <number>' in the catalog
        if hip_value.isdigit():
            hip_number = hip_value
            matches = catalog[catalog['Name'].str.match(f'^HIP\\s*{hip_number}$', case=False, na=False)]
            
            if len(matches) > 0:
                best_match = matches.iloc[0]
                results['log_rhk'] = best_match['logRpHK']
                results['rhk_source'] = 'HIP_number_match'
                return results
    
    # Method 4: Direct name matching (FIXED - now uses exact matching)
    name_columns = ['HD Number', 'GJ Number', 'HIP Number']
    for col in name_columns:
        if col in star and pd.notna(star[col]):
            star_name = str(star[col]).strip()
            
            # FIXED: Use exact matching with proper escaping
            escaped_name = re.escape(star_name)
            matches = catalog[catalog['Name'].str.match(f'^{escaped_name}$', case=False, na=False)]
            
            if len(matches) > 0:
                best_match = matches.iloc[0]
                results['log_rhk'] = best_match['logRpHK']
                results['rhk_source'] = 'Name_match'
                return results
    
    # No match found
    results['rhk_source'] = 'Not found'
    return results

def add_rhk_to_dataframe(df, catalog_file='../data/catalog_BoroSaikia2018.dat', verbose=True):
    """
    Add R'HK values to a DataFrame containing stellar data - FIXED VERSION
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame containing stellar data. Should have columns like 'HD Number', 
        'GJ Number', 'HIP Number' for cross-matching.
    catalog_file : str
        Path to the R'HK catalog file (default: '../data/catalog_BoroSaikia2018.dat')
    verbose : bool
        Whether to print progress information (default: True)
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with added R'HK columns:
        - 'log_rhk': The R'HK value
        - 'rhk_source': How the match was found
    
    FIXED: Now uses exact matching instead of partial matching to prevent 
           false matches like HD2151 matching HD215152
    """
    
    # Make a copy to avoid modifying the original DataFrame
    result_df = df.copy()
    
    # Load the R'HK catalog
    if verbose:
        catalog = load_catalog_simple(catalog_file)
    else:
        # Silent loading (identical logic as load_catalog_simple, but without prints)
        catalog_data = []
        with open(catalog_file, 'r') as f:
            for line in f:
                if line.strip():
                    parts = line.split()
                    if len(parts) >= 4:
                        seq = int(parts[0])
                        name = parts[1]
                        sptype = parts[2]
                        
                        rhk_value = None
                        for part in reversed(parts):
                            try:
                                val = float(part)
                                if val < 0:
                                    rhk_value = val
                                    break
                            except:
                                continue
                        
                        ra, dec = None, None
                        if len(parts) >= 6:
                            try:
                                ra = float(parts[-3])
                                dec = float(parts[-2])
                            except:
                                pass
                        
                        catalog_data.append({
                            'Seq': seq,
                            'Name': name,
                            'SpType': sptype,
                            'RAdeg': ra,
                            'DEdeg': dec,
                            'logRpHK': rhk_value
                        })
        catalog = pd.DataFrame(catalog_data)
    
    # Initialize result columns
    result_df['log_rhk'] = np.nan
    result_df['rhk_source'] = ''
    
    # Process each star
    if verbose:
        print(f"\n🔍 Searching for R'HK values for {len(result_df)} stars...")
    
    found_count = 0
    
    for idx, star in result_df.iterrows():
        
        results = find_rhk_for_star(star, catalog)
        
        # Only update 'log_rhk' and 'rhk_source'
        result_df.at[idx, 'log_rhk'] = results.get('log_rhk', np.nan)
        result_df.at[idx, 'rhk_source'] = results.get('rhk_source', '')
        
        if not pd.isna(results['log_rhk']):
            found_count += 1
    
    # Move 'log_rhk' and 'rhk_source' after 'logg_gaia' if present
    if 'logg_gaia' in result_df.columns:
        cols = result_df.columns.tolist()
        # Remove if already present
        cols.remove('log_rhk')
        cols.remove('rhk_source')
        logg_idx = cols.index('logg_gaia')
        # Insert after logg_gaia
        cols = cols[:logg_idx+1] + ['log_rhk', 'rhk_source'] + cols[logg_idx+1:]
        result_df = result_df[cols]
    
    if verbose:
        print(f"\n✅ Search complete!")
        print(f"📊 Found R'HK values for {found_count}/{len(result_df)} stars ({found_count/len(result_df)*100:.1f}%)")
        
        # Show summary statistics
        found_stars = result_df.dropna(subset=['log_rhk'])
        if len(found_stars) > 0:
            print(f"\nR'HK statistics:")
            print(f"  Range: [{found_stars['log_rhk'].min():.3f}, {found_stars['log_rhk'].max():.3f}]")
            print(f"  Mean: {found_stars['log_rhk'].mean():.3f}")
            print(f"  Median: {found_stars['log_rhk'].median():.3f}")
            
            print(f"\nMatching methods used:")
            method_counts = found_stars['rhk_source'].value_counts()
            for method, count in method_counts.items():
                print(f"  {method}: {count} stars")
    
    return result_df    