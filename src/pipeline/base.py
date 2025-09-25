"""
Base classes for the 2ES target selection pipeline.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from core.exceptions import PipelineError
from core.logging_config import get_logger

# Import pandas only when needed to avoid version conflicts
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    # Create a dummy pd class for type hints
    class pd:
        class DataFrame:
            pass


class PipelineStage(ABC):
    """Base class for pipeline stages."""
    
    def __init__(self, config: Any, name: Optional[str] = None):
        self.config = config
        self.name = name or self.__class__.__name__
        self.logger = get_logger(self.name)
    
    @abstractmethod
    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Process the data and return modified DataFrame.
        
        Args:
            data: Input DataFrame
            
        Returns:
            Processed DataFrame
        """
        pass
    
    def validate_input(self, data: pd.DataFrame) -> bool:
        """
        Validate input data for this stage.
        
        Args:
            data: Input DataFrame
            
        Returns:
            True if valid, False otherwise
        """
        if data is None or data.empty:
            self.logger.error("Input data is None or empty")
            return False
        return True
    
    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Run the pipeline stage with validation and error handling.
        
        Args:
            data: Input DataFrame
            
        Returns:
            Processed DataFrame
            
        Raises:
            PipelineError: If processing fails
        """
        try:
            self.logger.info(f"Starting {self.name}")
            
            if not self.validate_input(data):
                raise PipelineError(f"Input validation failed for {self.name}")
            
            result = self.process(data)
            
            self.logger.info(f"Completed {self.name}. Output shape: {result.shape}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error in {self.name}: {str(e)}")
            raise PipelineError(f"Stage {self.name} failed: {str(e)}") from e


class PipelineRunner:
    """Runs a sequence of pipeline stages."""
    
    def __init__(self, stages: list, config: Any):
        self.stages = stages
        self.config = config
        self.logger = get_logger('PipelineRunner')
    
    def run(self, initial_data: pd.DataFrame) -> pd.DataFrame:
        """
        Run all pipeline stages in sequence.
        
        Args:
            initial_data: Starting DataFrame
            
        Returns:
            Final processed DataFrame
            
        Raises:
            PipelineError: If any stage fails
        """
        self.logger.info(f"Starting pipeline with {len(self.stages)} stages")
        
        data = initial_data.copy()
        
        for i, stage in enumerate(self.stages):
            self.logger.info(f"Running stage {i+1}/{len(self.stages)}: {stage.name}")
            data = stage.run(data)
        
        self.logger.info("Pipeline completed successfully")
        return data
    
    def run_stage(self, stage_index: int, data: pd.DataFrame) -> pd.DataFrame:
        """
        Run a specific stage by index.
        
        Args:
            stage_index: Index of stage to run
            data: Input DataFrame
            
        Returns:
            Processed DataFrame
        """
        if stage_index >= len(self.stages):
            raise PipelineError(f"Stage index {stage_index} out of range")
        
        stage = self.stages[stage_index]
        return stage.run(data)
