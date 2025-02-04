# Stellar Analysis Pipeline

A Python-based pipeline for analyzing stellar data from Gaia DR2 and DR3 catalogs, with cross-matching and enrichment from Simbad and other stellar catalogs.

## Overview

This pipeline performs the following operations:
1. Queries Gaia DR2 and DR3 catalogs for stars matching specific criteria
2. Cross-matches results between DR2 and DR3
3. Enriches data with Simbad information (HD, GJ, and HIP numbers)
4. Calculates and validates stellar properties
5. Generates consolidated datasets and visualizations

## Requirements

- Python 3.7+
- Required packages:
  ```
  astroquery
  numpy
  pandas
  matplotlib
  tqdm
  openpyxl
  ```

## Project Structure

```
project/
├── main.py                  # Main execution script
├── config.py                # Configuration parameters
├── queries.py               # Gaia query definitions
├── data_cleaning.py         # Data cleaning and merging
├── stellar_processing.py    # Stellar property calculations
├── plotting.py             # Visualization functions
├── utils.py                # Utility functions
└── data/                   # Input data directory
    └── Catalogue_CELESTA.txt
    └── Catalogue_V_117A_table1.txt
```

## Configuration

Key parameters in `config.py`:
```python
TARGET_G_MAG_LIMIT = 12
MIN_DEC = -85
MAX_DEC = 30
MIN_PARALLAX = 20
SEARCH_RADIUS = 2 / 3600.0  # 2 arcseconds in degrees
```

## Usage

1. Run the main script:
   ```bash
   python main.py
   ```

2. The pipeline will generate several output files in the `results/` directory:
   - `dr2_results.xlsx`: Raw DR2 query results
   - `dr3_results.xlsx`: Raw DR3 query results
   - `dr2_dr3_crossmatch.xlsx`: Cross-matching results
   - `merged_results.xlsx`: Initial merged dataset
   - `clean_merged_results.xlsx`: Cleaned dataset
   - `consolidated_results.xlsx`: Final consolidated results
   - `consolidated_HIP_results.xlsx`: Results enriched with HIP catalog data

## Data Processing Steps

1. **Query Execution**
   - Retrieves stars with G magnitude < 12
   - Declination between -85° and 30°
   - Parallax ≥ 20 mas

2. **Data Cleaning**
   - Removes duplicate entries
   - Handles conflicting DR2/DR3 data
   - Consolidates stellar parameters

3. **Data Enrichment**
   - Adds Simbad identifiers (HD, GJ, HIP numbers)
   - Incorporates additional stellar properties
   - Cross-references with CELESTA catalog

## Output Data

The final consolidated dataset includes:
- Gaia source IDs (DR2 and DR3)
- Stellar coordinates (RA, DEC)
- Photometric data (G, BP, RP magnitudes)
- Physical parameters (temperature, mass, luminosity, radius)
- Cross-matched catalog numbers (HD, GJ, HIP)
- Spectral classifications

## Notes

- The pipeline prioritizes DR3 data over DR2 when available
- Missing values are filled using cross-referenced catalogs where possible
- All coordinate transformations use J2000 epoch