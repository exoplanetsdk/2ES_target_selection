import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import warnings
from config import *
warnings.filterwarnings('ignore')

def angular_separation_vectorized(ra1, dec1, ra2_array, dec2_array):
    """Vectorized calculation of angular separation"""
    ra1, dec1 = np.radians([ra1, dec1])
    ra2_array, dec2_array = np.radians([ra2_array, dec2_array])
    
    delta_ra = ra2_array - ra1
    delta_dec = dec2_array - dec1
    
    a = np.sin(delta_dec/2)**2 + np.cos(dec1) * np.cos(dec2_array) * np.sin(delta_ra/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    return np.degrees(c)

def format_gaia_id(gaia_id):
    """Convert GAIA ID to string format without scientific notation"""
    if pd.isna(gaia_id):
        return None
    # Remove 'Gaia DR2 ' prefix if present and convert to string without scientific notation
    gaia_id_str = str(gaia_id).replace('Gaia DR2 ', '')
    # If in scientific notation, convert to regular format
    if 'e' in gaia_id_str.lower():
        return f"{float(gaia_id_str):.0f}"
    return gaia_id_str

def match_gaia_tess(gaia_file, tess_file, output_file, is_candidate=False, threshold_arcsec=2):
    """
    Match GAIA and TESS targets based on position.
    
    Parameters:
    -----------
    gaia_file : str
        Path to the GAIA Excel file
    tess_file : str
        Path to the TESS tab-separated file
    output_file : str
        Path to save the output CSV file
    is_candidate : bool, optional
        Whether the TESS file is a candidate file (default: False)
    threshold_arcsec : float, optional
        Matching threshold in arcseconds (default: 2)
    
    Returns:
    --------
    pd.DataFrame
        DataFrame containing the matches
    """
    # Read the files
    print("Reading input files...")
    gaia_df = pd.read_excel(gaia_file, dtype={'source_id_dr2': str})
    tess_df = pd.read_csv(tess_file, sep='\t', comment='#')

    # Remove duplicate stars based on file type
    if is_candidate:
        id_col = 'toi'
    else:
        id_col = 'hostname'
    
    tess_df = tess_df.drop_duplicates(subset=[id_col, 'ra', 'dec'])

    # Convert threshold to degrees
    threshold = threshold_arcsec/3600

    # Convert TESS data to numpy arrays
    tess_ra = tess_df['ra'].values
    tess_dec = tess_df['dec'].values

    matches = []
    print("Processing matches...")
    for _, gaia_row in tqdm(gaia_df.iterrows(), total=len(gaia_df), desc="Processing GAIA entries", ncols=100):
        # Calculate separations for all TESS targets at once
        seps = angular_separation_vectorized(gaia_row['RA'], gaia_row['DEC'], tess_ra, tess_dec)
        
        # Find matches within threshold
        matches_idx = np.where(seps < threshold)[0]
        
        for idx in matches_idx:
            tess_row = tess_df.iloc[idx]
            
            # Common fields
            match_info = {
                'HZ_Detection_Limit_MEarth': gaia_row['HZ Detection Limit [M_Earth]'],
                'GAIA_RA': gaia_row['RA'],
                'GAIA_DEC': gaia_row['DEC'],
                'TESS_RA': tess_row['ra'],
                'TESS_DEC': tess_row['dec'],
                'Separation_arcsec': seps[idx] * 3600,
                'GAIA_ID': gaia_row['source_id_dr2']
            }
            
            # Add fields specific to candidate or confirmed files
            if is_candidate:
                match_info.update({
                    'TOI': tess_row['toi'],
                    'TESS_mag': tess_row['st_tmag'],
                    'TESS_Teff': tess_row['st_teff'],
                    'TESS_dist': tess_row['st_dist']
                })
            else:
                tess_gaia = format_gaia_id(tess_row['gaia_id'])
                gaia_dr2 = format_gaia_id(gaia_row['source_id_dr2'])
                match_info.update({
                    'TESS_GAIA_ID': tess_gaia,
                    'Host_name': tess_row['hostname']
                })
            
            matches.append(match_info)

    # Convert to DataFrame
    matches_df = pd.DataFrame(matches)

    if len(matches) > 0:
        # Remove duplicates based on file type
        if is_candidate:
            matches_df = matches_df.drop_duplicates(subset=['TOI', 'GAIA_ID'])
        else:
            matches_df = matches_df.drop_duplicates(subset=['Host_name', 'GAIA_ID'])
            
            # Check for Gaia ID mismatches in confirmed planets
            if not is_candidate:
                mask = (matches_df['GAIA_ID'].notna() & 
                       matches_df['TESS_GAIA_ID'].notna() & 
                       (matches_df['GAIA_ID'] != matches_df['TESS_GAIA_ID']))
                gaia_id_mismatches = matches_df[mask]
                
                if len(gaia_id_mismatches) > 0:
                    print("\nWARNING: Found Gaia ID mismatches:")
                    for _, mismatch in gaia_id_mismatches.iterrows():
                        print(f"Host: {mismatch['Host_name']}, GAIA_ID: {mismatch['GAIA_ID']}, "
                              f"TESS_GAIA_ID: {mismatch['TESS_GAIA_ID']}")
        
        # Print results
        print(f"\nFound {len(matches_df)} unique position matches within {threshold_arcsec:.1f} arcseconds")
        print("\nMatches found:")
        with pd.option_context('display.float_format', lambda x: '{:.3f}'.format(x)):
            if is_candidate:
                print(matches_df[['TOI', 'Separation_arcsec', 'GAIA_ID', 'TESS_mag', 'HZ_Detection_Limit_MEarth']].to_string())
            else:
                print(matches_df[['Host_name', 'Separation_arcsec', 'GAIA_ID', 'TESS_GAIA_ID', 'HZ_Detection_Limit_MEarth']].to_string())
        
        # Clean up Gaia IDs for saving
        if not is_candidate:
            matches_df['GAIA_ID'] = matches_df['GAIA_ID'].astype(str)
            matches_df['TESS_GAIA_ID'] = matches_df['TESS_GAIA_ID'].astype(str)
            matches_df['GAIA_ID'] = matches_df['GAIA_ID'].replace('nan', '')
            matches_df['TESS_GAIA_ID'] = matches_df['TESS_GAIA_ID'].replace('nan', '')
        
        # Save results
        matches_df.to_csv(output_file, index=False)
        
        return matches_df
    else:
        print("No matches found!")
        return None

def main():
    # Process both confirmed and candidate files

    # Confirmed planets
    print("\nProcessing TOIs with confirmed TESS planets...")
    matches_confirmed = match_gaia_tess(
        GAIA_FILE,
        TESS_CONFIRMED_FILE,
        OUTPUT_CONFIRMED_FILE,
        is_candidate=False,
        threshold_arcsec=2.5
    )

    # Candidates
    print("\nProcessing TESS candidates...")
    matches_candidates = match_gaia_tess(
        GAIA_FILE,
        TESS_CANDIDATE_FILE,
        OUTPUT_CANDIDATE_FILE,
        is_candidate=True,
        threshold_arcsec=2.5
    )
    
if __name__ == "__main__":
    main()
