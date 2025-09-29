"""
Configuration settings for the 2ES Target Selection Pipeline.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from pathlib import Path


@dataclass
class QueryParameters:
    """Parameters for Gaia database queries."""
    target_g_mag_limit: int = 8
    min_dec: float = -80
    max_dec: float = 20
    min_parallax: float = 20
    search_radius: float = 2 / 3600.0
    threshold_arcsec: float = 2.5


@dataclass
class StellarFilters:
    """Parameters for stellar filtering."""
    temp_min: float = 4000
    temp_max: float = 6500
    lum_min: float = 0.05
    lum_max: float = 5.2
    density_min: float = 0.1
    density_max: float = 5
    logg_min: float = 3.8
    log_rhk_max: float = -4.5


@dataclass
class NoiseParameters:
    """Parameters for noise modeling."""
    instrumental_noise: float = 0.1  # m/s
    residual_p_mode_fraction: float = 0.1
    residual_granulation_fraction: float = 1.0


@dataclass
class FilePaths:
    """File paths for external catalogs and data."""
    celesta_catalog: str = '../data/Catalogue_CELESTA.txt'
    vizier_catalog: str = '../data/Catalogue_V_117A_table1.txt'
    rhk_catalog: str = '../data/catalog_BoroSaikia2018.dat'
    tess_confirmed: str = '../data/TESS_confirmed.tab'
    tess_candidate: str = '../data/TESS_candidate.tab'
    hwo_target_list: str = '../data/HWO_target_list_164.xlsx'


@dataclass
class Paths:
    """Directory paths."""
    results_dir: Path = field(default_factory=lambda: Path('../results'))
    figures_dir: Path = field(default_factory=lambda: Path('../figures'))
    data_dir: Path = field(default_factory=lambda: Path('../data'))


@dataclass
class Config:
    """Main configuration class."""
    query_params: QueryParameters = field(default_factory=QueryParameters)
    stellar_filters: StellarFilters = field(default_factory=StellarFilters)
    noise_params: NoiseParameters = field(default_factory=NoiseParameters)
    file_paths: FilePaths = field(default_factory=FilePaths)
    paths: Paths = field(default_factory=Paths)
    
    # Detection limits for analysis
    detection_limits: List[Optional[float]] = field(default_factory=lambda: [None, 4, 2, 1.5])
    
    def get_stellar_filters_dict(self) -> Dict[str, float]:
        """Convert stellar filters to dictionary format for backward compatibility."""
        return {
            'temp_min': self.stellar_filters.temp_min,
            'temp_max': self.stellar_filters.temp_max,
            'lum_min': self.stellar_filters.lum_min,
            'lum_max': self.stellar_filters.lum_max,
            'density_min': self.stellar_filters.density_min,
            'density_max': self.stellar_filters.density_max,
            'logg_min': self.stellar_filters.logg_min,
            'log_rhk_max': self.stellar_filters.log_rhk_max,
        }
