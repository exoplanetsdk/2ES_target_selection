"""
Main 2ES Pipeline - implements the complete original 2ES.py functionality using the new modular structure.
"""
from core.config import Config
from core.logging_config import get_logger
from core.exceptions import PipelineError
from pipeline.base_simple import PipelineRunner
from pipeline.stages.gaia_acquisition_stage import GaiaAcquisitionStage
from pipeline.stages.data_cleaning_stage import DataCleaningStage
from pipeline.stages.data_consolidation_stage import DataConsolidationStage
from pipeline.stages.catalog_enrichment_stage import CatalogEnrichmentStage
from pipeline.stages.filtering_stage import FilteringStage
from pipeline.stages.bright_neighbors_stage import BrightNeighborsStage
from pipeline.stages.habitable_zone_stage import HabitableZoneStage
from pipeline.stages.visualization_stage import VisualizationStage
from pipeline.stages.ralf_comparison_stage import RalfComparisonStage
from pipeline.stages.crossmatching_stage import CrossmatchingStage
from utils import save_and_adjust_column_widths

logger = get_logger(__name__)


def main():
    """Main 2ES pipeline function - equivalent to the original main() function."""
    logger.info("Initializing 2ES Target Selection Pipeline...")
    
    try:
        # Load configuration
        config = Config()
        logger.info("Configuration loaded successfully")
        
        # Create pipeline stages (in the same order as original 2ES.py)
        stages = [
            GaiaAcquisitionStage(config),
            DataCleaningStage(config),
            DataConsolidationStage(config),
            CatalogEnrichmentStage(config),
            FilteringStage(config),
            BrightNeighborsStage(config),
            HabitableZoneStage(config),
            VisualizationStage(config),
            RalfComparisonStage(config),
            CrossmatchingStage(config)
        ]
        
        # Create and run pipeline
        pipeline = PipelineRunner(stages, config)
        logger.info(f"Starting pipeline with {len(stages)} stages")
        
        # Run the pipeline
        final_results = pipeline.run(initial_data=None)
        
        # Save final results
        logger.info("Saving final results")
        save_and_adjust_column_widths(final_results, config.gaia_file)
        
        logger.info("2ES Pipeline completed successfully!")
        logger.info(f"Final results saved to: {config.gaia_file}")
        logger.info(f"Total stars processed: {len(final_results)}")
        
        return final_results
        
    except PipelineError as e:
        logger.error(f"Pipeline failed: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in main pipeline: {e}")
        raise PipelineError(f"Main pipeline failed: {e}") from e


if __name__ == "__main__":
    main()
