"""
Custom exception classes for the 2ES target selection pipeline.
"""


class PipelineError(Exception):
    """Base exception for pipeline errors."""
    pass


class DataValidationError(PipelineError):
    """Raised when data validation fails."""
    pass


class QueryError(PipelineError):
    """Raised when external queries (Gaia, SIMBAD, etc.) fail."""
    pass


class FileError(PipelineError):
    """Raised when file operations fail."""
    pass


class ConfigurationError(PipelineError):
    """Raised when configuration is invalid."""
    pass


class CalculationError(PipelineError):
    """Raised when calculations fail."""
    pass
