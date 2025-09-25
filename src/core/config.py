"""
Centralized configuration management for 2ES target selection pipeline.
"""
from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path
from datetime import datetime


@dataclass
class Paths:
    """Configuration for file and directory paths."""
    data_dir: Path = field(default_factory=lambda: Path("../data"))
    results_dir: Path = field(default_factory=lambda: Path("../results"))
    figures_dir: Path = field(default_factory=lambda: Path("../figures"))
    log_dir: Path = field(default_factory=lambda: Path("../logs"))
    
    def __post_init__(self):
        """Create directories if they don't exist."""
        for path in [self.data_dir, self.results_dir, self.figures_dir, self.log_dir]:
            path.mkdir(parents=True, exist_ok=True)


@dataclass
class QueryParameters:
    """Configuration for Gaia query parameters."""
    target_g_mag_limit: int = 8
    min_dec: float = -80
    max_dec: float = 20
    min_parallax: float = 20
    search_radius: float = 2 / 3600.0
    threshold_arcsec: float = 2.5


@dataclass
class StellarFilters:
    """Configuration for stellar filtering parameters."""
    temp_min: float = 3800
    temp_max: float = 6500
    lum_min: float = 0.05
    lum_max: float = 5.2
    density_min: float = 0.1
    density_max: float = 5
    logg_min: float = 3.8
    log_rhk_max: float = -4.5


@dataclass
class NoiseParameters:
    """Configuration for noise modeling parameters."""
    instrumental_noise: float = 0.1  # m/s
    residual_p_mode_fraction: float = 0.1
    residual_granulation_fraction: float = 1.0


@dataclass
class FilePaths:
    """Configuration for specific file paths."""
    ralf_file: str = "../data/2ES_targetlist_astrid_export_2024Dec_comments.xlsx"
    tess_confirmed: str = "../data/TESS_confirmed.tab"
    tess_candidate: str = "../data/TESS_candidate.tab"
    hwo_target_list: str = "../data/HWO_target_list_164.txt"
    celesta_catalog: str = "../data/Catalogue_CELESTA.txt"
    stellar_catalog: str = "../data/Catalogue_V_117A_table1.txt"
    rhk_catalog: str = "../data/catalog_BoroSaikia2018.dat"


@dataclass
class Config:
    """Main configuration class that combines all configuration sections."""
    paths: Paths = field(default_factory=Paths)
    query_params: QueryParameters = field(default_factory=QueryParameters)
    stellar_filters: StellarFilters = field(default_factory=StellarFilters)
    noise_params: NoiseParameters = field(default_factory=NoiseParameters)
    file_paths: FilePaths = field(default_factory=FilePaths)
    
    # Detection limits for analysis
    detection_limits: List[Optional[float]] = field(default_factory=lambda: [None, 4, 2, 1.5])
    
    def __post_init__(self):
        """Initialize default values after object creation."""
        # Generate today's date for file naming
        self.today_date = datetime.today().strftime('%Y.%m.%d')
        
        # Generate output file paths
        self.gaia_file = (
            f"{self.paths.results_dir}/Gaia_homogeneous_target_selection_"
            f"{self.today_date}_{int(100*self.noise_params.residual_granulation_fraction)}_granulation.xlsx"
        )
        
        # TESS output files
        self.tess_confirmed_output = f"{self.paths.results_dir}/GAIA_TESS_confirmed_matches.xlsx"
        self.tess_candidate_output = f"{self.paths.results_dir}/GAIA_TESS_candidate_matches.xlsx"
        
        # Unique files
        self.tess_confirmed_unique_planets = self.tess_confirmed_output.replace('.xlsx', '_unique_planets.xlsx')
        self.tess_confirmed_unique_stars = self.tess_confirmed_output.replace('.xlsx', '_unique_stars.xlsx')
        self.tess_candidate_unique_planets = self.tess_candidate_output.replace('.xlsx', '_unique_planets.xlsx')
        self.tess_candidate_unique_stars = self.tess_candidate_output.replace('.xlsx', '_unique_stars.xlsx')
    
    def get_stellar_filters_dict(self) -> dict:
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


# Global configuration instance
config = Config()
