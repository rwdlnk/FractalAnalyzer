"""
Main analysis tools for general RT research.

Tier 1: General RT Analysis
- Enhanced analyzer for temporal evolution and convergence studies
- Parallel processing capabilities
- Performance optimization tools

Tier 2: Advanced Fractal Analysis
- Multifractal spectrum analysis
- Advanced interface characterization
"""

# Import main functions from enhanced_analyzer
from .enhanced_analyzer import (
    parse_grid_resolution,
    format_grid_resolution, 
    validate_grid_resolution_input,
    find_timestep_files_for_resolution,
    determine_analysis_mode
)

# Import utility classes
from .fast_vtk_reader import FastVTKReader
from .grid_cache_manager import GridCacheManager

# Import multifractal analysis (when available)
try:
    from .multifractal_analyzer import MultifractalAnalyzer
    _MULTIFRACTAL_AVAILABLE = True
except ImportError:
    _MULTIFRACTAL_AVAILABLE = False
    MultifractalAnalyzer = None

__all__ = [
    'parse_grid_resolution',
    'format_grid_resolution',
    'validate_grid_resolution_input', 
    'find_timestep_files_for_resolution',
    'determine_analysis_mode',
    'FastVTKReader',
    'GridCacheManager'
]

# Add MultifractalAnalyzer to exports if available
if _MULTIFRACTAL_AVAILABLE:
    __all__.append('MultifractalAnalyzer')
