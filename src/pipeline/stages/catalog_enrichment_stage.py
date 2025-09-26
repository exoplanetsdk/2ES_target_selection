"""
Catalog enrichment stage - implements the original catalog processing functionality.
"""
import pandas as pd
from ..base_simple import PipelineStage
from core.exceptions import PipelineError
from . import import_helpers  # This sets up the archive imports
from catalog_integration import CatalogProcessor, add_rhk_to_dataframe
from plotting import plot_density_vs_logg
from utils import save_and_adjust_column_widths


class CatalogEnrichmentStage(PipelineStage):
    """Pipeline stage for enriching data with external catalogs."""
    
    def process(self, data):
        """Enrich data with external catalogs (CELESTA, Vizier, R'HK)."""
        self.logger.info("Starting catalog enrichment")
        
        try:
            # Initialize catalog processor
            processor = CatalogProcessor(
                celesta_path=self.config.file_paths.celesta_catalog,
                stellar_catalog_path=self.config.file_paths.stellar_catalog
            )
            
            # Process catalogs
            self.logger.info("Processing CELESTA and Vizier catalogs")
            df_enriched = processor.process_catalogs(data)
            
            # Create density vs logg plot
            self.logger.info("Creating density vs logg plot")
            plot_density_vs_logg(
                df_enriched,
                output_path=f'{self.config.paths.figures_dir}/density_vs_logg.png',
                show_plot=False
            )
            
            # Add R'HK data
            self.logger.info("Adding R'HK activity data")
            df_with_rhk = add_rhk_to_dataframe(df_enriched)
            
            # Save intermediate results
            save_and_adjust_column_widths(
                df_with_rhk, 
                f'{self.config.paths.results_dir}/merged_df_with_rhk.xlsx'
            )
            
            self.logger.info(f"Catalog enrichment completed. {len(df_with_rhk)} stars enriched")
            return df_with_rhk
            
        except Exception as e:
            self.logger.error(f"Catalog enrichment failed: {e}")
            raise PipelineError(f"Catalog enrichment failed: {e}") from e
