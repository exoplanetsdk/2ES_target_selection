# Directory paths
RESULTS_DIRECTORY = '../results/'
FIGURES_DIRECTORY = '../figures/'
DATA_DIRECTORY = '../data/'

# Query parameters
TARGET_G_MAG_LIMIT = 12
MIN_DEC = -85
MAX_DEC = 30
MIN_PARALLAX = 20
SEARCH_RADIUS = 2 / 3600.0  # 2 arcseconds

# Stellar filtering parameters
STELLAR_FILTERS = {
    'temp_min':     3800,       # Minimum effective temperature in K
    'temp_max':     7000,       # Maximum effective temperature in K
    'lum_min':      0.05,       # Minimum luminosity in solar units
    'lum_max':      5.2,        # Maximum luminosity in solar units (increased to include HD23754)
    'density_min':  0.1,        # Minimum density in solar units
    'density_max':  5,          # Maximum density in solar units
    'logg_min':     3.8         # Minimum log g value (for filtering out sub-giants/giants)
}

RALF_FILE_PATH = f'{DATA_DIRECTORY}2ES_targetlist_astrid_export_2024Dec_comments.xlsx'

DETECTION_LIMITS = [None, 4, 2, 1.5] # M_Earth