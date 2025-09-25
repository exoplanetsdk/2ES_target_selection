"""
Simplified base classes for the 2ES target selection pipeline.
This version avoids pandas imports to prevent version conflicts.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from core.exceptions import PipelineError
from core.logging_config import get_logger


class PipelineStage(ABC):
    """Base class for pipeline stages."""
    
    def __init__(self, config: Any, name: Optional[str] = None):
        self.config = config
        self.name = name or self.__class__.__name__
        self.logger = get_logger(self.name)
    
    @abstractmethod
    def process(self, data: Any) -> Any:
        """
        Process the data and return modified data.
        
        Args:
            data: Input data (DataFrame or other)
            
        Returns:
            Processed data
        """
        pass
    
    def validate_input(self, data: Any) -> bool:
        """
        Validate input data for this stage.
        
        Args:
            data: Input data
            
        Returns:
            True if valid, False otherwise
        """
        if data is None:
            self.logger.error("Input data is None")
            return False
        return True
    
    def run(self, data: Any) -> Any:
        """
        Run the pipeline stage with validation and error handling.
        
        Args:
            data: Input data
            
        Returns:
            Processed data
            
        Raises:
            PipelineError: If processing fails
        """
        try:
            self.logger.info(f"Starting {self.name}")
            
            if not self.validate_input(data):
                raise PipelineError(f"Input validation failed for {self.name}")
            
            result = self.process(data)
            
            self.logger.info(f"Completed {self.name}")
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
    
    def run(self, initial_data: Any) -> Any:
        """
        Run all pipeline stages in sequence.
        
        Args:
            initial_data: Starting data
            
        Returns:
            Final processed data
            
        Raises:
            PipelineError: If any stage fails
        """
        self.logger.info(f"Starting pipeline with {len(self.stages)} stages")
        
        data = initial_data
        
        for i, stage in enumerate(self.stages):
            self.logger.info(f"Running stage {i+1}/{len(self.stages)}: {stage.name}")
            data = stage.run(data)
        
        self.logger.info("Pipeline completed successfully")
        return data
