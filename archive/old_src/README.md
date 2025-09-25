# Archived Original Source Files

This directory contains the original source files from the 2ES target selection project that have been replaced by the new modular pipeline architecture.

## Archived Files

### Main Scripts
- **`2ES.py`** - Original main script (replaced by `src/main_2ES_pipeline.py`)
- **`config.py`** - Original configuration (replaced by `src/core/config.py`)

### Data Processing
- **`data_processing.py`** - Original data processing functions (replaced by `src/data/processing/`)
- **`catalog_integration.py`** - Catalog integration functions (now in pipeline stages)
- **`filtering.py`** - Stellar filtering functions (now in `src/pipeline/stages/filtering_stage.py`)

### Plotting and Visualization
- **`plotting.py`** - Original plotting functions (now in `src/pipeline/stages/visualization_stage.py`)

### Stellar Calculations
- **`stellar_calculations.py`** - Stellar calculation functions (now in pipeline stages)
- **`stellar_properties.py`** - Stellar property calculations (now in pipeline stages)
- **`rv_prec.py`** - RV precision calculations (now in pipeline stages)

### External Data Sources
- **`gaia_queries.py`** - Gaia query functions (now in `src/pipeline/stages/gaia_acquisition_stage.py`)
- **`gaia_tess_overlap.py`** - TESS crossmatching (now in `src/pipeline/stages/crossmatching_stage.py`)
- **`HWO_overlap.py`** - HWO crossmatching (now in `src/pipeline/stages/crossmatching_stage.py`)
- **`plato_lops2.py`** - PLATO crossmatching (now in `src/pipeline/stages/crossmatching_stage.py`)

### Utilities
- **`utils.py`** - Utility functions (now distributed across pipeline stages)
- **`G_stars_within_15_pc.py`** - G-star analysis (now in pipeline stages)
- **`diff.py`** - Comparison utilities (now in pipeline stages)

## Migration Notes

All functionality from these original files has been preserved and reorganized into the new modular pipeline structure:

- **`src/core/`** - Configuration, logging, and exception handling
- **`src/data/`** - Data validation and processing modules
- **`src/pipeline/`** - Pipeline stages and orchestration
- **`src/main_2ES_pipeline.py`** - New main script

## Benefits of New Structure

1. **Modularity**: Each component has a single responsibility
2. **Maintainability**: Easy to modify and extend individual components
3. **Testability**: Each stage can be tested independently
4. **Professional Logging**: Comprehensive logging and error handling
5. **Configuration Management**: Centralized configuration system
6. **Documentation**: Clear separation of concerns

## Usage

To run the new pipeline, use:
```bash
cd src
python main_2ES_pipeline.py
```

The new pipeline produces identical results to the original `2ES.py` but with much better organization and maintainability.

---
*Archived on: 2025-09-25*
*Original files preserved for reference and rollback if needed*
