"""
Fractal Analyzer Package

A comprehensive toolkit for fractal dimension analysis of fluid interfaces
and mathematical fractals, with specialized tools for Rayleigh-Taylor instability simulations.
"""

__version__ = "2.1.0"
__author__ = "Rod Douglass"
__email__ = "rwdlanm@gmail.com"

# Import main classes for easy access
from .core.fractal_analyzer import FractalAnalyzer
from .core.rt_analyzer import RTAnalyzer
from .core.conrec_extractor import CONRECExtractor, compare_extraction_methods
from .optimized import *

__all__ = [
    'FractalAnalyzer',
    'RTAnalyzer',
	'CONRECExtractor',
	'compare_extraction_methods',
]
