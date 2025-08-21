import os
from datetime import datetime

# Directory paths
RESULTS_DIRECTORY = '../results/'
FIGURES_DIRECTORY = '../figures/'
DATA_DIRECTORY = '../data/'

# Query parameters
TARGET_G_MAG_LIMIT = 12     # Gaia G-band magnitude limit
MIN_DEC = -15               # Minimum declination in degrees
MAX_DEC = 75                # Maximum declination in degrees
MIN_PARALLAX = 20           # Minimum parallax in mas; corresponds to 50 pc
SEARCH_RADIUS = 2 / 3600.0  # Search radius in arcseconds for finding background contaminants (depends on the field of view of the telescope)
THRESHOLD_ARCSEC = 2.5      # arcseconds for GAIA-TESS overlap (already tuned to exclude false positives)

# Stellar filtering parameters
STELLAR_FILTERS = {
    'temp_min':     3800,       # Minimum effective temperature in K
    'temp_max':     7000,       # Maximum effective temperature in K
    'lum_min':      0.05,       # Minimum luminosity in solar units
    'lum_max':      5.2,        # Maximum luminosity in solar units (increased to include HD23754)
    'density_min':  0.1,        # Minimum density in solar units
    'density_max':  5,          # Maximum density in solar units
    'logg_min':     3.8,        # Minimum log g value (for filtering out sub-giants/giants)
    'log_rhk_max': -4.5,        # Maximum log R'HK value (for stellar activity corresponding to ~10 m/s RMS)
}

INSTRUMENTAL_NOISE = 0.1 # m/s (instrumental noise， optimistic)
RESIDUAL_P_MODE_FRACTION = 0.1 # p-mode noise RMS residual (tested; recommended by J. Zhao)
RESIDUAL_GRANULATION_FRACTION = 0.1 # granulation noise RMS residual (not tested; use with caution)


RALF_FILE_PATH = f'{DATA_DIRECTORY}2ES_targetlist_astrid_export_2024Dec_comments.xlsx'

# Determine the current date in the format 'YYYY.MM.DD'
today_date = datetime.today().strftime('%Y.%m.%d')

# Construct the file path for today's Gaia homogeneous target selection file
today_file = f"{RESULTS_DIRECTORY}THE_Gaia_homogeneous_target_selection_{today_date}_{int(100*RESIDUAL_GRANULATION_FRACTION)}.xlsx"

# Define the backup file path for the Gaia homogeneous target selection file
backup_file = f"{RESULTS_DIRECTORY}Gaia_homogeneous_target_selection_2025.02.17.xlsx"

# Check if today's file exists; if it does, use it as GAIA_FILE, otherwise use the backup file
# if os.path.exists(today_file):
#     GAIA_FILE = today_file
# else:
#     GAIA_FILE = backup_file

GAIA_FILE = today_file

# Define the file path for the TESS confirmed targets file
TESS_CONFIRMED_FILE = f"{DATA_DIRECTORY}TESS_confirmed.tab"
TESS_CANDIDATE_FILE = f"{DATA_DIRECTORY}TESS_candidate.tab"

# Define the output file path for the Gaia-TESS matches
OUTPUT_CONFIRMED_FILE = f"{RESULTS_DIRECTORY}GAIA_TESS_confirmed_matches.xlsx"
OUTPUT_CANDIDATE_FILE = f"{RESULTS_DIRECTORY}GAIA_TESS_candidate_matches.xlsx"

OUTPUT_CONFIRMED_UNIQUE_PLANETS = OUTPUT_CONFIRMED_FILE.replace('.xlsx', '_unique_planets.xlsx')
OUTPUT_CONFIRMED_UNIQUE_STARS   = OUTPUT_CONFIRMED_FILE.replace('.xlsx', '_unique_stars.xlsx')
OUTPUT_CANDIDATE_UNIQUE_PLANETS = OUTPUT_CANDIDATE_FILE.replace('.xlsx', '_unique_planets.xlsx')
OUTPUT_CANDIDATE_UNIQUE_STARS   = OUTPUT_CANDIDATE_FILE.replace('.xlsx', '_unique_stars.xlsx')

DETECTION_LIMITS = [None, 4, 2, 1.5] # M_Earth