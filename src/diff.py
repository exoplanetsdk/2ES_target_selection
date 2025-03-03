import pandas as pd
import re
from datetime import datetime
from config import RESULTS_DIRECTORY, RESULTS_DIRECTORY
from utils import save_and_adjust_column_widths

# Load the data from the two files
files = [
    f'{RESULTS_DIRECTORY}Gaia_homogeneous_target_selection_2025.02.20.xlsx',
    f'{RESULTS_DIRECTORY}Gaia_homogeneous_target_selection_2024.12.19.xlsx'
]
dfs = [pd.read_excel(file, dtype={'source_id': str, 'source_id_dr2': str, 'source_id_dr3': str}, header=0) for file in files]

# Find the extra stars and the removed stars
added_stars = dfs[0][~dfs[0]['source_id'].isin(dfs[1]['source_id'])].sort_values(by='HZ Detection Limit [M_Earth]')
removed_stars = dfs[1][~dfs[1]['source_id'].isin(dfs[0]['source_id'])].sort_values(by='HZ Detection Limit [M_Earth]')

# Save the diff files
date_strs = [re.search(r'\d{4}\.\d{2}\.\d{2}', file).group() for file in files]
save_and_adjust_column_widths(added_stars, RESULTS_DIRECTORY + f'added_stars_{date_strs[0]}_compared_to_{date_strs[1]}.xlsx')
save_and_adjust_column_widths(removed_stars, RESULTS_DIRECTORY + f'removed_stars_{date_strs[0]}_compared_to_{date_strs[1]}.xlsx')

print("Diff files generated successfully!")