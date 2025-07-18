"""
FractalAnalyzer: Rayleigh-Taylor instability analysis tools.

A comprehensive package for analyzing Rayleigh-Taylor instabilities with 
fractal dimension analysis, temporal evolution studies, and validation 
against experimental data.

Two-Tier Analysis System:
- Tier 1: General RT Analysis (analysis module)
- Tier 2: Dalziel Validation (validation module)
"""

__version__ = "2.0.0"
__author__ = "Your Name"

# Core components (main building blocks)
try:
    from .core.rt_analyzer import RTAnalyzer
    from .core.fractal_analyzer import FractalAnalyzer
except ImportError:
    print("Warning: Could not import core analyzers")
    RTAnalyzer = None
    FractalAnalyzer = None

# Main analysis functions
try:
    from .analysis.enhanced_analyzer import determine_analysis_mode, find_timestep_files_for_resolution
except ImportError:
    print("Warning: Could not import analysis functions")

# Dalziel validation functions  
try:
    from .validation.dalziel_comparison import compare_with_dalziel_experiments
except ImportError:
    print("Warning: Could not import validation functions")

__all__ = [
    'RTAnalyzer',
    'FractalAnalyzer',
    'determine_analysis_mode',
    'find_timestep_files_for_resolution',
    'compare_with_dalziel_experiments'
]
