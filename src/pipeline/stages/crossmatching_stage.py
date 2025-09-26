"""
Crossmatching stage - implements the original crossmatching functionality.
"""
import pandas as pd
from ..base_simple import PipelineStage
from core.exceptions import PipelineError
from . import import_helpers  # This sets up the archive imports
from HWO_overlap import HWO_match
from plato_lops2 import plato_lops2_match
from gaia_tess_overlap import run_tess_overlap_batch


class CrossmatchingStage(PipelineStage):
    """Pipeline stage for crossmatching with external catalogs."""
    
    def process(self, data):
        """Crossmatch with HWO, PLATO, and TESS catalogs."""
        self.logger.info("Starting crossmatching with external catalogs")
        
        try:
            df = data.copy()
            
            # HWO crossmatching
            self.logger.info("Crossmatching with HWO catalog")
            df = HWO_match(df)
            
            # PLATO LOPS2 crossmatching
            self.logger.info("Crossmatching with PLATO LOPS2 catalog")
            df = plato_lops2_match(df)
            
            # TESS crossmatching
            self.logger.info("Crossmatching with TESS catalogs")
            tess_results = run_tess_overlap_batch(df)
            confirmed_gaia_ids = tess_results["confirmed"]["gaia_ids"]
            candidate_gaia_ids = tess_results["candidates"]["gaia_ids"]
            df = tess_results["df"]
            
            self.logger.info(f"TESS confirmed matches: {len(confirmed_gaia_ids)}")
            self.logger.info(f"TESS candidate matches: {len(candidate_gaia_ids)}")
            
            # Calculate sum score
            self.logger.info("Calculating sum scores")
            df['sum_score'] = (
                df['HWO_match'] * 1 +
                df['LOPS2_match'] * 1 +
                df['TESS_confirmed_match'] * 1 +
                df['TESS_candidate_match'] * 0.5
            )
            
            # Sort by sum score and detection limit
            df = df.sort_values(['sum_score', 'HZ Detection Limit [M_Earth]'], ascending=[False, True])
            
            # Filter by detection limit
            df = df[df['HZ Detection Limit [M_Earth]'] <= 2]
            
            self.logger.info(f"Crossmatching completed. {len(df)} stars with crossmatches")
            return df
            
        except Exception as e:
            self.logger.error(f"Crossmatching failed: {e}")
            raise PipelineError(f"Crossmatching failed: {e}") from e
