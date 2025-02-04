# 2ES Target Selection Pipeline

A Python pipeline for generating a homogeneous stellar target list for the 2ES spectrograph, primarily utilizing Gaia DR2 and DR3 catalogs.

## Features

- Automated querying of Gaia DR2/DR3 catalogs
- Additional stellar parameters from other catalogs
- Automated bright neighbor detection
- Habitable zone and RV precision calculations
- Mass detection limits calculations
- Cross-matching with other catalogs

## Requirements

### Python Dependencies
```bash
astroquery>=0.4.6
numpy>=1.20.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
tqdm>=4.60.0
openpyxl>=3.0.7
scipy>=1.7.0
```

## Directory Structure

```
project/
├── data/                           # Input data directory
│   ├── Catalogue_CELESTA.txt      # CELESTA catalog data
│   ├── Catalogue_V_117A_table1.txt # Stellar catalog
│   ├── dataVmag.csv              # RV precision reference data
│   └── EEM_dwarf_UBVIJHK_colors_Teff.txt  # Stellar classification data
├── results/                        # Output directory
├── figures/                        # Generated plots
└── src/                           # Source code
    ├── main.py                    # Main execution script
    ├── config.py                  # Configuration parameters
    ├── catalog_integration.py     # Catalog processing
    ├── data_processing.py         # Data cleaning and merging
    ├── filtering.py              # Stellar filtering
    ├── gaia_queries.py           # Gaia query definitions
    ├── plotting.py               # Visualization functions
    ├── rv_prec.py               # RV precision calculations
    ├── stellar_calculations.py   # Stellar physics calculations
    ├── stellar_properties.py     # Stellar property handling
    └── utils.py                  # Utility functions
```

## Configuration

Key parameters in `config.py`:
```python
# Target selection criteria
TARGET_G_MAG_LIMIT = 12    # Maximum G magnitude
MIN_DEC = -85             # Minimum declination
MAX_DEC = 30              # Maximum declination
MIN_PARALLAX = 20         # Minimum parallax (mas)

# Stellar filtering parameters
STELLAR_FILTERS = {
    'temp_min': 3800,     # Minimum temperature (K)
    'temp_max': 7000,     # Maximum temperature (K)
    'lum_min': 0.05,      # Minimum luminosity (L_sun)
    'lum_max': 5.2,       # Maximum luminosity (L_sun)
    'density_min': 0.1,   # Minimum density (solar units)
    'density_max': 5,     # Maximum density (solar units)
    'logg_min': 3.8       # Minimum log g
}
```

## Pipeline Steps

1. **Data Collection**
   - Query Gaia DR2/DR3 catalogs
   - Cross-match between catalogs
   - Retrieve additional identifiers (HD, GJ, HIP)

2. **Data Processing**
   - Clean and merge catalog data
   - Calculate stellar parameters
   - Validate physical properties
   - Detect bright neighboring stars

3. **Analysis**
   - Calculate habitable zones
   - Determine RV precision estimates
   - Compute detection limits
   - Generate visualization plots

4. **Output Generation**
   - Excel files with processed data
   - Diagnostic plots and figures
   - Cross-matched catalog information

## Usage

1. Set up the environment:
```bash
pip install -r requirements.txt
```

2. Configure parameters in `config.py`

3. Run the pipeline:
```bash
python main.py
```

## Output Files

The pipeline generates several key outputs in the `results/` directory:

- `consolidated_results.xlsx`: Complete stellar parameters
- `consolidated_HIP_results.xlsx`: HIP catalog cross-matches
- `stars_with_bright_neighbors.xlsx`: Stars with nearby bright companions
- `combined_query_with_mass_detection_limit.xlsx`: Final target list with detection limits

## Visualization

Generated plots in `figures/` include:
- HR diagrams with detection limits
- RV precision vs. temperature
- Stellar parameter distributions
- Target sky distribution (RA/DEC)

## Notes

- Prioritizes Gaia DR3 data over DR2 when available
- Implements multiple validation steps for stellar parameters
- Includes bright neighbor detection within 2 arcseconds
- Calculates planet detection limits in habitable zones
