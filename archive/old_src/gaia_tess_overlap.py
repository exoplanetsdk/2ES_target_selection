import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings
from config import *
warnings.filterwarnings('ignore')
from utils import *

# --------------------------------------------------------------------------------------------------
def create_header_mapping_from_file(file_content):
    """Extract header mappings from the file's comment section."""
    mapping = {}
    for line in file_content.split('\n'):
        if line.startswith('# COLUMN'):
            parts = line.replace('# COLUMN', '').strip().split(':')
            if len(parts) == 2:
                column_name = parts[0].strip()
                description = parts[1].strip()
                mapping[column_name] = description
    return mapping

# --------------------------------------------------------------------------------------------------
def merge_planet_data(matches_df, is_candidate=False):
    """
    Merge multiple entries of the same planet by averaging numerical columns
    and using the mode for categorical columns.
    """
    if is_candidate:
        group_by = 'toi'
    else:
        group_by = ['hostname', 'pl_name']

    columns = matches_df.columns.tolist()
    categorical_columns = ['TESS_GAIA_ID', 'GAIA_DR2_ID']
    merged_rows = []

    for name, group in matches_df.groupby(group_by):
        merged_row = {}
        if isinstance(name, tuple):
            merged_row[group_by[0]] = name[0]
            merged_row[group_by[1]] = name[1]
        else:
            merged_row[group_by] = name

        for col in columns:
            if col in (group_by if isinstance(group_by, list) else [group_by]):
                continue

            values = group[col].dropna()
            if values.empty:
                merged_row[col] = np.nan
                continue

            if col in categorical_columns:
                merged_row[col] = values.mode().iloc[0] if not values.empty else np.nan
                continue

            try:
                numeric_values = pd.to_numeric(values)
                merged_row[col] = numeric_values.mean()
            except Exception:
                merged_row[col] = values.mode().iloc[0] if not values.empty else np.nan

        merged_rows.append(merged_row)

    merged_df = pd.DataFrame(merged_rows)
    # Reorder columns to match original
    merged_df = merged_df[[c for c in columns if c in merged_df.columns]]
    return merged_df

# --------------------------------------------------------------------------------------------------
def match_gaia_tess(df,
                    tess_file,
                    output_file,
                    output_merged_file,
                    is_candidate=False,
                    threshold_arcsec=2.0):
    """
    Cross-match GAIA targets to TESS planets/candidates by position.

    Returns:
        matches_df (per-match rows) or None
        merged_df  (per-planet aggregated rows) or None
        matched_gaia_ids (set of GAIA source_id strings)
    """
    print("Reading input files...")
    gaia_df = df.copy()
    tess_df = pd.read_csv(tess_file, sep='\t', comment='#')

    threshold_deg = threshold_arcsec / 3600.0
    tess_ra = tess_df['ra'].values
    tess_dec = tess_df['dec'].values

    matches = []
    matched_gaia_ids = set()

    print("Processing matches...")
    for _, gaia_row in tqdm(gaia_df.iterrows(),
                            total=len(gaia_df),
                            desc="Processing GAIA entries",
                            ncols=100):
        seps = angular_separation_vectorized(
            gaia_row['RA'],
            gaia_row['DEC'],
            tess_ra,
            tess_dec
        )
        match_idx = np.where(seps < threshold_deg)[0]

        if match_idx.size > 0:
            matched_gaia_ids.add(gaia_row['source_id'])

        for idx in match_idx:
            tess_row = tess_df.iloc[idx]
            match_info = tess_row.to_dict()

            if not is_candidate:
                tess_gaia = format_gaia_id(tess_row.get('gaia_id'))
                gaia_dr2 = format_gaia_id(gaia_row['source_id'])
                match_info.update({
                    'TESS_GAIA_ID': tess_gaia,
                    'GAIA_DR2_ID': gaia_dr2,
                    'Host_name': tess_row.get('hostname'),
                    'HZ_Detection_Limit': gaia_row.get('HZ Detection Limit [M_Earth]'),
                    'Separation_arcsec': seps[idx] * 3600.0
                })
            else:
                match_info.update({
                    'HZ_Detection_Limit': gaia_row.get('HZ Detection Limit [M_Earth]'),
                    'Separation_arcsec': seps[idx] * 3600.0
                })

            matches.append(match_info)

    if not matches:
        print("No matches found!")
        return None, None, set()

    matches_df = pd.DataFrame(matches)

    # Header mapping (from candidate/confirmed TESS file comments if present)
    with open(tess_file, 'r') as f:
        header_mapping = create_header_mapping_from_file(f.read())

    print("\nCreating merged dataset...")
    merged_df = merge_planet_data(matches_df, is_candidate=is_candidate)
    merged_df = merged_df.sort_values('HZ_Detection_Limit', na_position='last')

    print(f"\nTotal matches: {len(matches_df)}")
    print(f"Unique planets: {len(merged_df)}")

    valid_mappings = {k: v for k, v in header_mapping.items() if k in matches_df.columns}
    matches_df = matches_df.rename(columns=valid_mappings)
    merged_df = merged_df.rename(columns=valid_mappings)

    if is_candidate:
        highlight_columns = [
            'Planet Radius Value [R_Earth]',
            'HZ_Detection_Limit'
        ]
    else:
        highlight_columns = [
            'Planet Radius [Earth Radius]',
            'Planet Mass [Earth Mass]',
            'Planet Mass*sin(i) [Earth Mass]',
            'HZ_Detection_Limit'
        ]

    save_and_adjust_column_widths(matches_df, output_file, highlight_columns=highlight_columns)
    save_and_adjust_column_widths(merged_df, output_merged_file, highlight_columns=highlight_columns)

    return matches_df, merged_df, matched_gaia_ids

# --------------------------------------------------------------------------------------------------
def save_overlapping_stars(df, 
                           matched_gaia_ids,
                           output_overlap_file,
                           new_column='TESS_match'):
    """
    Save subset of GAIA stars that matched TESS targets.
    Adds a binary column new_column indicating matches.
    Prints detection limit statistics.
    """

    gaia_df = df.copy()
    gaia_df[new_column] = gaia_df['source_id'].isin(matched_gaia_ids).astype(int)
    overlapping = gaia_df[gaia_df[new_column] == 1]

    save_and_adjust_column_widths(overlapping, output_overlap_file)
    save_and_adjust_column_widths(gaia_df, GAIA_FILE)

    print(f"Overlapping stars: {len(overlapping)}")
    for limit in DETECTION_LIMITS:
        if limit is None:
            continue
        count = (overlapping['HZ Detection Limit [M_Earth]'] <= limit).sum()
        print(f"Stars with HZ Detection Limit ≤ {limit} M_Earth: {count}")

    return gaia_df, overlapping

# --------------------------------------------------------------------------------------------------
def run_tess_overlap_batch(df, threshold_arcsec=2.5):
    """
    Execute confirmed + candidate TESS overlap workflow.

    Parameters:
        gaia_file: Excel file containing GAIA targets (must already include
                   'HZ Detection Limit [M_Earth]' if you want that propagated).
        threshold_arcsec: positional match radius.

    Returns:
        {
          "confirmed": {"matches": DataFrame|None, "merged": DataFrame|None, "gaia_ids": set},
          "candidates": {...},
          "df": DataFrame (possibly updated with TESS match columns)
        }
    """
    print("\nProcessing confirmed TESS planets...")
    matches_confirmed, merged_confirmed, confirmed_ids = match_gaia_tess(
        df=df,
        tess_file=TESS_CONFIRMED_FILE,
        output_file=OUTPUT_CONFIRMED_FILE,
        output_merged_file=OUTPUT_CONFIRMED_UNIQUE_PLANETS,
        is_candidate=False,
        threshold_arcsec=threshold_arcsec
    )
    df_with_confirmed, _ = save_overlapping_stars(df, confirmed_ids, OUTPUT_CONFIRMED_UNIQUE_STARS, new_column='TESS_confirmed_match')

    print("\nProcessing TESS candidates...")
    matches_candidates, merged_candidates, candidate_ids = match_gaia_tess(
        df=df_with_confirmed,
        tess_file=TESS_CANDIDATE_FILE,
        output_file=OUTPUT_CANDIDATE_FILE,
        output_merged_file=OUTPUT_CANDIDATE_UNIQUE_PLANETS,
        is_candidate=True,
        threshold_arcsec=threshold_arcsec
    )
    df_with_candidates, _ = save_overlapping_stars(df_with_confirmed, candidate_ids, OUTPUT_CANDIDATE_UNIQUE_STARS, new_column='TESS_candidate_match')

    return {
        "confirmed": {
            "matches": matches_confirmed,
            "merged": merged_confirmed,
            "gaia_ids": confirmed_ids
        },
        "candidates": {
            "matches": matches_candidates,
            "merged": merged_candidates,
            "gaia_ids": candidate_ids
        },
        "df": df_with_candidates
    }

# --------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    run_tess_overlap_batch()
