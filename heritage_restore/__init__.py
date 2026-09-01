"""HeritageGuard: classical OpenCV restoration and damage analysis."""

from .config import ProcessingConfig
from .pipeline import ProcessingResult, RestorationPipeline

__all__ = ["ProcessingConfig", "ProcessingResult", "RestorationPipeline"]
__version__ = "1.0.0"
