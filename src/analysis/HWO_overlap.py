import pandas as pd
from datetime import datetime
from config import DATA_DIRECTORY, RESULTS_DIRECTORY
from core.utils import adjust_column_widths

def HWO_match(df):
    """
    Adds a column 'HWO_match' to df, labeling matches with the HWO target list
    based on HD, GJ, or HIP numbers.

    Parameters:
        df (pd.DataFrame): The DataFrame to annotate.

    Returns:
        pd.DataFrame: The input DataFrame with an added 'HWO_match' column.
    """

    file_path = f"{DATA_DIRECTORY}HWO_target_list_164.txt"

    # Read the file into a DataFrame
    df_HWO = pd.read_csv(file_path, sep="\t", header=36, usecols=lambda column: column != 'loc_rowid')

    # Read the column headers from the file
    with open(file_path, 'r') as file:
        lines = file.readlines()
        column_headers = [line.split(':')[1].strip() for line in lines[5:35]]

    # Add ' (HWO)' to each column header
    column_headers = [f"{header} (HWO)" for header in column_headers]

    # Assign the new column headers to the DataFrame
    df_HWO.columns = column_headers

    # Match on HD Number
    hd_match = df['HD Number'].isin(df_HWO['HD ID (HWO)'])

    # Match on GJ Number, if available
    if 'GJ Number' in df.columns and 'GJ ID (HWO)' in df_HWO.columns:
        gj_match = df['GJ Number'].astype(str).isin(df_HWO['GJ ID (HWO)'].astype(str))
    else:
        gj_match = False

    # Match on HIP Number, if available
    if 'HIP Number' in df.columns and 'HIP ID (HWO)' in df_HWO.columns:
        hip_match = df['HIP Number'].astype(str).isin(df_HWO['HIP ID (HWO)'].astype(str))
    else:
        hip_match = False

    # Combine all matches
    if isinstance(gj_match, bool) and isinstance(hip_match, bool):
        match_mask = hd_match
    else:
        match_mask = hd_match | gj_match | hip_match

    # Add the column to df
    df = df.copy()
    df['HWO_match'] = 0
    df.loc[match_mask, 'HWO_match'] = 1

    # Extract the overlapping part and save to Excel
    df_overlap = df[df['HWO_match'] == 1].copy()
    today_str = datetime.today().strftime('%Y.%m.%d')
    filename_overlap = f'{RESULTS_DIRECTORY}Overlap_with_HWO_M_earth_{today_str}.xlsx'
    df_overlap.to_excel(filename_overlap, index=False)
    adjust_column_widths(filename_overlap)

    return df
