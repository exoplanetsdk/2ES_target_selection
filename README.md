# 2ES Target Selection Pipeline

A professional, modular pipeline for selecting optimal targets for the 2ES (Two Earth-Sized) exoplanet detection mission. This pipeline processes Gaia data, applies stellar filters, calculates habitable zones, and performs crossmatching with external catalogs to identify the best targets for exoplanet detection.

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
python main_2ES_pipeline.py
```

## 📁 Project Structure

```
2ES_target_selection/
├── src/                           # Main source code
│   ├── core/                      # Core infrastructure
│   │   ├── config.py             # Centralized configuration
│   │   ├── exceptions.py         # Custom exception handling
│   │   └── logging_config.py     # Professional logging
│   ├── data/                      # Data processing modules
│   │   ├── validation/           # Data validation
│   │   └── processing/            # Data processing
│   ├── pipeline/                  # Pipeline architecture
│   │   ├── base_simple.py        # Base pipeline classes
│   │   └── stages/                # Individual pipeline stages
│   │       ├── gaia_acquisition_stage.py
│   │       ├── data_cleaning_stage.py
│   │       ├── data_consolidation_stage.py
│   │       ├── catalog_enrichment_stage.py
│   │       ├── filtering_stage.py
│   │       ├── bright_neighbors_stage.py
│   │       ├── habitable_zone_stage.py
│   │       ├── visualization_stage.py
│   │       ├── ralf_comparison_stage.py
│   │       └── crossmatching_stage.py
│   └── main_2ES_pipeline.py       # Main pipeline script
├── data/                          # Input data files
├── results/                       # Output files
├── figures/                       # Generated plots
├── logs/                          # Pipeline logs
├── archive/                       # Archived original files
└── requirements.txt               # Python dependencies
```

## 🔄 Pipeline Stages

The pipeline consists of 10 modular stages:

1. **Gaia Acquisition** - Queries Gaia DR2/DR3 databases
2. **Data Cleaning** - Removes duplicates and cleans data
3. **Data Consolidation** - Merges DR2/DR3 data and adds identifiers
4. **Catalog Enrichment** - Adds external catalog data (CELESTA, Vizier, R'HK)
5. **Filtering** - Applies stellar parameter filters
6. **Bright Neighbors** - Identifies and filters stars with bright neighbors
7. **Habitable Zone** - Calculates habitable zones and noise models
8. **Visualization** - Creates plots and diagrams
9. **Ralf Comparison** - Compares with Ralf's target list
10. **Crossmatching** - Crossmatches with HWO, PLATO, and TESS catalogs

## ⚙️ Configuration

The pipeline uses a centralized configuration system in `src/core/config.py`:

```python
from core.config import Config

config = Config()
# Access configuration parameters
print(config.stellar_filters.temp_min)
print(config.query_params.target_g_mag_limit)
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

### Adding New Pipeline Stages

1. Create a new stage class in `src/pipeline/stages/`
2. Inherit from `PipelineStage`
3. Implement the `process()` method
4. Add to the main pipeline in `src/main_2ES_pipeline.py`

```python
from pipeline.base_simple import PipelineStage

class MyNewStage(PipelineStage):
    def process(self, data):
        # Your processing logic here
        return processed_data
```

### Modifying Configuration

Edit `src/core/config.py` to change default parameters:

```python
@dataclass
class StellarFilters:
    temp_min: float = 3800      # Minimum temperature
    temp_max: float = 6500      # Maximum temperature
    # ... other parameters
```

## 📝 Logging

The pipeline includes comprehensive logging:

- **File Logging**: All logs saved to `logs/pipeline.log`
- **Console Output**: Real-time progress updates
- **Stage Tracking**: Detailed progress through each pipeline stage
- **Error Handling**: Robust error reporting and recovery

## 🔍 Data Validation

The pipeline includes extensive data validation:

- **Coordinate Validation**: RA/Dec range checks
- **Magnitude Validation**: Photometric data quality
- **Parallax Validation**: Distance calculation accuracy
- **Temperature Validation**: Stellar parameter ranges

## 📚 Dependencies

Key Python packages required:

- `pandas` - Data manipulation
- `numpy` - Numerical computations
- `matplotlib` - Plotting
- `astropy` - Astronomical calculations
- `astroquery` - Database queries
- `tqdm` - Progress bars

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📞 Support

For questions or issues:
1. Check the logs in `logs/pipeline.log`
2. Review the configuration in `src/core/config.py`
3. Open an issue on the repository

## 🎯 Mission Goals

The 2ES target selection pipeline is designed to identify the best stellar targets for detecting Earth-sized exoplanets in the habitable zone. The pipeline optimizes for:

- **High-precision RV measurements** - Stars with low noise characteristics
- **Habitable zone accessibility** - Detectable Earth-sized planets
- **Observational efficiency** - Bright, nearby stars
- **Scientific value** - Targets with known planetary systems

---

**Last Updated**: 2025-09-25  
**Version**: 2.0 (Modular Pipeline)  
**Status**: Production Ready ✅