import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings
from config import *
warnings.filterwarnings('ignore')
from utils import *


#-------------------------------------------------------------------------------------------------- 

def create_header_mapping_from_file(file_content):
    """Extract header mappings from the file's comment section"""
    mapping = {}
    for line in file_content.split('\n'):
        if line.startswith('# COLUMN'):
            # Extract the column name and description
            parts = line.replace('# COLUMN', '').strip().split(':')
            if len(parts) == 2:
                column_name = parts[0].strip()
                description = parts[1].strip()
                mapping[column_name] = description
    return mapping


#-------------------------------------------------------------------------------------------------- 

def merge_planet_data(matches_df, is_candidate=False):
    """
    Merge multiple entries of the same planet by averaging numerical values
    and keeping the most common non-numerical values.
    """
    if is_candidate:
        group_by = 'toi'
    else:
        group_by = ['hostname', 'pl_name']
    
    columns = matches_df.columns.tolist()
    categorical_columns = ['TESS_GAIA_ID', 'GAIA_DR2_ID']
    merged_data = []
    
    for name, group in matches_df.groupby(group_by):
        merged_row = {}
        
        if isinstance(name, tuple):
            merged_row[group_by[0]] = name[0]
            merged_row[group_by[1]] = name[1]
        else:
            merged_row[group_by] = name
        
        for column in columns:
            if column in (group_by if isinstance(group_by, list) else [group_by]):
                continue
                
            values = group[column].dropna()
            if len(values) == 0:
                merged_row[column] = np.nan
                continue
            
            if column in categorical_columns:
                merged_row[column] = values.mode().iloc[0] if len(values) > 0 else np.nan
                continue
                
            try:
                numeric_values = pd.to_numeric(values)
                merged_row[column] = numeric_values.mean()
            except:
                merged_row[column] = values.mode().iloc[0] if len(values) > 0 else np.nan
        
        merged_data.append(merged_row)
    
    merged_df = pd.DataFrame(merged_data)
    merged_df = merged_df[columns]
    
    return merged_df 


#-------------------------------------------------------------------------------------------------- 

def match_gaia_tess(gaia_file, tess_file, output_file, output_merged_file, is_candidate=False, threshold_arcsec=2):
    """
    Match GAIA and TESS targets based on position.
    Returns matched_gaia_ids in addition to the existing returns.
    """
    # Read the files
    print("Reading input files...")
    gaia_df = pd.read_excel(gaia_file, dtype={'source_id': str, 'source_id_dr2': str, 'source_id_dr3': str})
    tess_df = pd.read_csv(tess_file, sep='\t', comment='#')

    threshold = threshold_arcsec / 3600
    tess_ra = tess_df['ra'].values
    tess_dec = tess_df['dec'].values

    matches = []
    matched_gaia_ids = set()  # Keep track of matched GAIA IDs

    print("Processing matches...")
    for _, gaia_row in tqdm(gaia_df.iterrows(), total=len(gaia_df), desc="Processing GAIA entries", ncols=100):
        seps = angular_separation_vectorized(gaia_row['RA'], gaia_row['DEC'], tess_ra, tess_dec)
        matches_idx = np.where(seps < threshold)[0]

        if len(matches_idx) > 0:  # If there's at least one match
            matched_gaia_ids.add(gaia_row['source_id_dr2'])  # Add the GAIA ID to our set

        for idx in matches_idx:
            tess_row = tess_df.iloc[idx]

            # Start with all TESS columns
            match_info = tess_row.to_dict()

            # Add GAIA information
            if not is_candidate:
                tess_gaia = format_gaia_id(tess_row['gaia_id'])
                gaia_dr2 = format_gaia_id(gaia_row['source_id_dr2'])
                match_info.update({
                    'TESS_GAIA_ID': tess_gaia,
                    'GAIA_DR2_ID': gaia_dr2,
                    'Host_name': tess_row['hostname'],
                    'HZ_Detection_Limit': gaia_row['HZ Detection Limit [M_Earth]'],
                    'Separation_arcsec': seps[idx] * 3600  # Convert degrees to arcseconds
                })
            else:
                match_info.update({
                    'HZ_Detection_Limit': gaia_row['HZ Detection Limit [M_Earth]'],
                    'Separation_arcsec': seps[idx] * 3600
                })

            matches.append(match_info)

    if len(matches) > 0:
        matches_df = pd.DataFrame(matches)

        # Get header mapping from TESS_candidate.tab
        with open(tess_file, 'r') as f:
            header_mapping = create_header_mapping_from_file(f.read())

        # Create merged version
        print("\nCreating merged dataset...")
        merged_df = merge_planet_data(matches_df, is_candidate)
        merged_df = merged_df.sort_values('HZ_Detection_Limit')

        # Print statistics
        print(f"\nTotal matches: {len(matches_df)}")
        print(f"Unique planets: {len(merged_df)}")

        # Apply header mapping to both DataFrames
        valid_mappings = {k: v for k, v in header_mapping.items() if k in matches_df.columns}
        matches_df = matches_df.rename(columns=valid_mappings)
        merged_df = merged_df.rename(columns=valid_mappings)

        # Save both versions
        if is_candidate:
            highlight_columns = ['Planet Radius Value [R_Earth]', 'HZ_Detection_Limit']
        else:
            highlight_columns = ['Planet Radius [Earth Radius]', 'Planet Mass [Earth Mass]', 'Planet Mass*sin(i) [Earth Mass]', 'HZ_Detection_Limit']
        save_and_adjust_column_widths(matches_df, output_file, highlight_columns=highlight_columns)
        save_and_adjust_column_widths(merged_df, output_merged_file, highlight_columns=highlight_columns)

        return matches_df, merged_df, matched_gaia_ids
    else:
        print("No matches found!")
        return None, None, set()

#-------------------------------------------------------------------------------------------------- 

def save_overlapping_stars(gaia_file, matched_gaia_ids, output_overlap_file):
    """
    Save the subset of GAIA stars that were matched to TESS targets and print detection limit statistics.
    """
    gaia_df = pd.read_excel(gaia_file, dtype={'source_id': str, 'source_id_dr2': str, 'source_id_dr3': str})
    overlapping_stars = gaia_df[gaia_df['source_id_dr2'].isin(matched_gaia_ids)]
    save_and_adjust_column_widths(overlapping_stars, output_overlap_file)
    
    # Print basic statistics
    print(f"Overlapping stars: {len(overlapping_stars)}")
    
    # Print statistics for each detection limit
    for limit in DETECTION_LIMITS:
        if limit is None:
            continue  # Skip None value
        count = len(overlapping_stars[overlapping_stars['HZ Detection Limit [M_Earth]'] <= limit])
        print(f"Stars with HZ Detection Limit ≤ {limit} M_Earth: {count}")
    
    return overlapping_stars


#-------------------------------------------------------------------------------------------------- 

def main():
    # Process confirmed planets
    print("\nProcessing confirmed TESS planets...")
    matches_confirmed, merged_confirmed, confirmed_gaia_ids = match_gaia_tess(
        GAIA_FILE,
        TESS_CONFIRMED_FILE,
        OUTPUT_CONFIRMED_FILE,
        OUTPUT_CONFIRMED_UNIQUE_PLANETS,
        is_candidate=False,
        threshold_arcsec=2.5
    )
    # Save overlapping stars for confirmed planets
    overlapping_stars = save_overlapping_stars(GAIA_FILE, confirmed_gaia_ids, OUTPUT_CONFIRMED_UNIQUE_STARS)

    # Process candidates
    print("\nProcessing TESS candidates...")
    matches_candidates, merged_candidates, candidate_gaia_ids = match_gaia_tess(
        GAIA_FILE,
        TESS_CANDIDATE_FILE,
        OUTPUT_CANDIDATE_FILE,
        OUTPUT_CANDIDATE_UNIQUE_PLANETS,
        is_candidate=True,
        threshold_arcsec=2.5
    )
    # Save overlapping stars for candidates only
    overlapping_stars = save_overlapping_stars(GAIA_FILE, candidate_gaia_ids, OUTPUT_CANDIDATE_UNIQUE_STARS)


#-------------------------------------------------------------------------------------------------- 

if __name__ == "__main__":
    main()