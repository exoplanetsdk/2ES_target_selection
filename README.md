# 2ES Target Selection Pipeline

A Python pipeline for generating a homogeneous stellar target list for the 2ES spectrograph, primarily utilizing Gaia DR2 and DR3 catalogs.


## Highlights

- Automated querying of Gaia DR2/DR3 catalogs
- Additional stellar parameters from other catalogs (CELESTA, Vizier V/117A)
- Bright neighbor detection
- **Automated RV precision calculations**
- Habitable zone and mass detection limits calculations
- Cross-matching with other catalogs


## Directory Structure

```
2ES_target_selection/
├── data/                           # Input data directory
│   ├── Catalogue_CELESTA.txt       # CELESTA catalog data
│   ├── Catalogue_V_117A_table1.txt # V_117A catalog
│   ├── dataVmag.csv                # Empirical RV precision reference data
│   └── EEM_dwarf_UBVIJHK_colors_Teff.txt  # Stellar color and effective temperature sequence
├── results/                        # Output directory
├── figures/                        # Plots directory
└── src/                            # Source code
    ├── main.py                     # Main execution script
    ├── config.py                   # Configuration parameters
    ├── catalog_integration.py      # Catalog processing
    ├── data_processing.py          # Data cleaning and merging
    ├── filtering.py                # Stellar filtering
    ├── G_stars_within_15_pc.py     # G star selection within 15 pc
    ├── gaia_queries.py             # Gaia queries
    ├── plotting.py                 # Visualization functions
    ├── rv_prec.py                  # RV precision calculations
    ├── stellar_calculations.py     # Habitable zone and mass detection limits calculations
    ├── stellar_properties.py       # Stellar property handling
    ├── utils.py                    # Utility functions
    ├── gaia_tess_overlap.py        # Matches GAIA and TESS targets based on position
    └── plato_lops2.py              # Plots the PLATO LOPS2 field with GAIA stars
```

## Configuration

Key parameters in `config.py`:
```python
# Target selection criteria
TARGET_G_MAG_LIMIT = 12   # Maximum G magnitude
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
    'logg_min': 3.8,      # Minimum log g
    'log_rhk_max': -4.5   # maximum log_rhk
}
```

## Pipeline Steps

1. **Data Collection**
   - Query Gaia DR2/DR3 catalogs (prioritizes DR3 over DR2 when available)
   - Retrieve additional identifiers (HD, GJ, HIP)
   - Obtain stellar types from SIMBAD

2. **Data Processing**
   - Clean and merge DR2/DR3 catalog data
   - Retrieve stellar parameters from DR2/DR3
   - Retrieve additional stellar parameters from other catalogs (currently [CELESTA](https://ui.adsabs.harvard.edu/abs/2016yCat..51510059C/abstract) and [Vizier V/117A](https://vizier.cds.unistra.fr/viz-bin/VizieR?-source=V/117A))
   - Derive stellar parameters empirically based on stellar types if not available
   - Detect and remove targets with bright neighboring stars 

3. **Analysis**
   - Calculate habitable zones --> orbital radius
   - Determine RV precision estimates --> RV amplitude
   - Compute mass detection limits in habitable zones
   - Generate visualization plots

4. **Output Generation**
   - Excel files with processed data
   - Diagnostic plots and figures
   - Cross-matched catalog information


## Environment Setup

If you are using a conda environment, run the following commands to create and set up a virtual environment (e.g., named `2ES`):

```bash
conda create -n 2ES
conda activate 2ES
conda install pip
pip install -r requirements.txt
```

## Usage

1. Configure parameters in `config.py`

2. Run the pipeline:
```bash
cd src && python 2ES.py
```

## Output Files

The pipeline generates several key outputs in the `results/` directory:

- `consolidated_Simbad_results.xlsx`: Complete stellar parameters
- `stars_without_bright_neighbors.xlsx`: Stars after removing nearby bright companions
- `Gaia_homogeneous_target_selection_M_earth_1_5.xlsx`: Target list with detection limit less than 1.5 M_earth
- `merged_RJ.xlsx`: Crossmatch with Ralf's results

## Visualization

Generated plots in `figures/` include:
- HR diagrams with detection limits
- RV precision vs. temperature
- Stellar parameter distributions
- Target sky distribution (RA/DEC)

## Interactive 2ES Target List Explorer

Explore and analyze the Gaia target list interactively in your browser. You can launch the interactive histogram explorer (powered by Binder and Voilà) using the button below:

[![Launch Interactive Target List (Binder + Voilà)](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/exoplanetsdk/2ES_target_selection/main?urlpath=voila/render/notebooks/Interactive_2ES_Targets_Explorer.ipynb)

Alternatively, open the explorer in a standard Jupyter notebook interface [here](https://mybinder.org/v2/gh/exoplanetsdk/2ES_target_selection/HEAD?urlpath=%2Fdoc%2Ftree%2Fnotebooks%2FInteractive_2ES_Targets_Explorer.ipynb).

*Note: The first launch may take a few minutes to initialize.*

<img src="figures/histogram.png" alt="Interactive Gaia Explorer"/>

## Radial Velocity precision calculator

The photon-limited RV precision calculations, available in the [Radial Velocity precision calculator](http://www.astro.physik.uni-goettingen.de/research/rvprecision/) (Reiners & Zechmeister, 2020), have been automated in the `rv_prec.py` script. This automation eliminates the need for the online calculator, significantly speeding up the pipeline and allowing it to scale efficiently for a large number of stars.
![Radial Velocity precision calculator](figures/RV_tool.png)

**Example Usage**

To calculate the Radial Velocity (RV) precision for a star with a temperature of 5000 K and a V magnitude of 10, use the following code:

```python
from rv_prec import calculate_rv_precision
result, custom_rv_precision = calculate_rv_precision(5000, 10)
custom_rv_precision
```
returning an RV precision of 0.563624937587981 m/s. The telescope parameters are tailored for 2ES and are predefined in the `get_manual_values` function. 

## Recent Major Maintenance

- **2025-08-18**: Included p-mode and granulation RV noise for calculating the detection limit; integrated log R'HK activity metric for estimating stellar noise floor; introduced score for cross-matching HWO/TESS/PLATO targets; enabled interactive histograms.

- **2025-07-22**: Due to changes in SIMBAD query, the Gaia data release (e.g., Gaia DR3) needs to be specified to ensure stars are retrieved correctly and stellar identifiers (e.g. HD, GJ, HIP) are extracted. Updated column name handling to accommodate SIMBAD's revised naming conventions for stellar types. Updated the crossmatching logic. 

## Troubleshooting: SIMBAD Connection Issues

Occasionally, the pipeline may be interrupted by an error similar to the following:

```
pyvo.dal.exceptions.DALServiceError: Unable to access the capabilities endpoint at:
- https://simbad.cds.unistra.fr/simbad/sim-tap/capabilities: Connection failed (Possible causes: incorrect URL, DNS issue, or service is down)

This could mean:
1. The service URL is incorrect
2. The service is temporarily unavailable
3. The service doesn't support this protocol
4. If a 503 was encountered, retry after the suggested delay.
```

This issue is related to the SIMBAD service connection and is not caused by the pipeline code itself. If you encounter this error, it is likely due to a temporary outage or network issue with the SIMBAD service. If this happens, simply wait and try running the code again at a later time. 