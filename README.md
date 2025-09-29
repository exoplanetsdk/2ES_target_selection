# 2ES Target Selection Pipeline

A professional, modular pipeline for selecting optimal targets for the 2ES (The Second Earth Spectrograph) exoplanet detection mission. This pipeline processes Gaia data, applies stellar filters, calculates habitable zones, and performs crossmatching with external catalogs to identify the best targets for exoplanet detection.

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Conda environment with required packages
- Access to Gaia database

### Installation
```bash
# Clone the repository
git clone <repository-url>
cd 2ES_target_selection

# Activate the 2ES conda environment
conda activate 2ES

# Install dependencies (if not already installed)
pip install -r requirements.txt
```

### Running the Pipeline
```bash
cd src
python 2ES.py
```

## 📁 Project Structure

```
2ES_target_selection/
├── src/                           # Main source code
│   ├── 2ES.py                     # Main pipeline script
│   ├── config.py                  # Configuration settings
│   ├── core/                      # Core processing modules
│   │   ├── data_processing.py     # Main data processing (940 lines)
│   │   ├── gaia_queries.py        # Gaia database queries
│   │   └── utils.py               # Utility functions
│   ├── analysis/                  # Analysis modules
│   │   ├── catalog_integration.py # External catalog processing
│   │   ├── filtering.py           # Stellar filtering
│   │   ├── plotting.py            # All plotting functions (658 lines)
│   │   ├── HWO_overlap.py         # HWO matching
│   │   ├── plato_lops2.py         # PLATO matching
│   │   └── gaia_tess_overlap.py   # TESS matching
│   ├── calculations/              # Stellar calculations
│   │   ├── rv_prec.py             # RV precision calculations
│   │   ├── stellar_calculations.py # Stellar math
│   │   └── stellar_properties.py   # Stellar properties
│   ├── utilities/                 # Utility scripts
│   │   ├── diff.py                # Data comparison
│   │   └── G_stars_within_15_pc.py # G-star finder
│   └── original_codes/             # Complete backup of original files
├── data/                          # Input data files
├── results/                       # Output files
├── figures/                       # Generated plots
├── notebooks/                     # Jupyter notebooks
├── archive/                       # Archived original files
└── requirements.txt               # Python dependencies
```

## 🔄 Pipeline Workflow

The pipeline processes data through the following stages:

1. **Gaia Data Acquisition** - Queries Gaia DR2/DR3 databases and crossmatches
2. **Data Processing** - Merges DR2/DR3 data, removes duplicates, and consolidates
3. **Catalog Enrichment** - Adds external catalog data (CELESTA, Vizier, R'HK)
4. **Stellar Filtering** - Applies stellar parameter filters and activity metrics
5. **Bright Neighbors Analysis** - Identifies and filters stars with bright neighbors
6. **Habitable Zone Calculations** - Calculates habitable zones and noise models
7. **RV Precision & Detection Limits** - Computes RV precision and detection limits
8. **Visualization** - Creates plots and diagrams
9. **Crossmatching** - Crossmatches with HWO, PLATO, and TESS catalogs
10. **Final Analysis** - Generates comprehensive target selection results

## ⚙️ Configuration

The pipeline uses a centralized configuration system in `src/config.py`:

```python
from config import *

# Access configuration parameters
print(STELLAR_FILTERS['temp_min'])
print(TARGET_G_MAG_LIMIT)
```

### Key Configuration Parameters

- **Stellar Filters**: Temperature, luminosity, density, logg ranges
- **Query Parameters**: Magnitude limits, parallax thresholds, search radius
- **Noise Parameters**: Instrumental noise, granulation, p-mode fractions
- **Detection Limits**: HZ detection limits in Earth masses

## 📊 Output Files

The pipeline generates several output files in the `results/` directory:

- **`Gaia_homogeneous_target_selection_YYYY.MM.DD_100_granulation.xlsx`** - Main results file
- **`GAIA_TESS_confirmed_matches.xlsx`** - TESS confirmed planet matches
- **`GAIA_TESS_candidate_matches.xlsx`** - TESS candidate matches
- **`stars_without_bright_neighbors.xlsx`** - Filtered star list

## 📈 Generated Plots

The pipeline creates various plots in the `figures/` directory:

- **HR Diagrams** - Hertzsprung-Russell diagrams with detection limits
- **Color-Magnitude Diagrams** - Stellar color vs magnitude plots
- **Density vs logg** - Stellar density analysis
- **RA/Dec Maps** - Sky distribution of targets
- **Crossmatch Comparisons** - Validation against external catalogs

## 🛠️ Development

### Code Organization

The codebase is organized into logical modules:

- **`core/`** - Core processing modules (data processing, Gaia queries, utilities)
- **`analysis/`** - Analysis modules (plotting, filtering, catalog integration)
- **`calculations/`** - Stellar calculations (RV precision, stellar properties)
- **`utilities/`** - Utility scripts (data comparison, G-star finder)

### Modifying Configuration

Edit `src/config.py` to change default parameters:

```python
# Stellar filtering parameters
STELLAR_FILTERS = {
    'temp_min': 4000,       # Minimum temperature
    'temp_max': 6500,       # Maximum temperature
    # ... other parameters
}
```

### Adding New Modules

1. Create your module in the appropriate directory (`core/`, `analysis/`, `calculations/`)
2. Add a symlink in the root `src/` directory for backward compatibility
3. Import and use in your main script

```bash
# Example: Adding a new analysis module
ln -sf analysis/my_new_module.py src/my_new_module.py
```


---
## Interactive 2ES Target List Explorer

Explore and analyze the Gaia target list interactively in your browser. You can launch the interactive histogram explorer (powered by Binder and Voilà) using the button below:

[![Launch Interactive Target List (Binder + Voilà)](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/exoplanetsdk/2ES_target_selection/THE?urlpath=voila/render/notebooks/Interactive_2ES_Targets_Explorer.ipynb)

Alternatively, open the explorer in a standard Jupyter notebook interface [here](https://mybinder.org/v2/gh/exoplanetsdk/2ES_target_selection/THE?urlpath=%2Fdoc%2Ftree%2Fnotebooks%2FInteractive_2ES_Targets_Explorer.ipynb).

*Note: The first launch may take a few minutes to initialize.*

<img src="figures/histogram.png" alt="Interactive Gaia Explorer"/>

## Radial Velocity precision calculator

The photon-limited RV precision calculations, available in the [Radial Velocity precision calculator](http://www.astro.physik.uni-goettingen.de/research/rvprecision/) (Reiners & Zechmeister, 2020), have been automated in the `rv_prec.py` script. This automation eliminates the need for the online calculator, significantly speeding up the pipeline and allowing it to scale efficiently for a large number of stars.
![Radial Velocity precision calculator](figures/RV_tool.png)

**Example Usage**

To compute the photon-limited Radial Velocity (RV) precision for a star (e.g., Teff = 5000 K, Vmag = 8, 10-minute exposure by default), simply run:

```python
from rv_prec import calculate_rv_precision
result, custom_rv_precision = calculate_rv_precision(5000, 8)
custom_rv_precision
```
returning an RV precision of 0.3886429800604213 m/s. The telescope parameters are tailored for 2ES and are predefined in the `get_manual_values` function. 


## Recent Major Maintenance

- **2025-09**: Complete code restructuring and cleanup. Organized code into logical modules (`core/`, `analysis/`, `calculations/`, `utilities/`). The main script `2ES.py` now runs with a clean, organized structure while maintaining full functionality. Implemented [TACS](https://github.com/MichaelCretignier/TACS) from Michael Cretignier to calculate stellar visibility throughout the year.


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