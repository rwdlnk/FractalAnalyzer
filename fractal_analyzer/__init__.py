"""
Fractal Analyzer Package

A comprehensive toolkit for fractal dimension analysis of fluid interfaces
and mathematical fractals, with specialized tools for Rayleigh-Taylor instability simulations.
"""

__version__ = "1.0.0"
__author__ = "Rod Douglass"
__email__ = "rwdlanm@gmail.com"

# Import main classes for easy access
from .core.fractal_analyzer import FractalAnalyzer
from .core.rt_analyzer import RTAnalyzer

__all__ = [
    'FractalAnalyzer',
    'RTAnalyzer'
]
