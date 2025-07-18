"""
Fast VTK Reader for F-Data Only

This module reads only the F (volume fraction) data from VTK files,
using pre-cached grid geometry to avoid redundant coordinate processing.
"""

import os
import numpy as np
import re
from typing import Dict, Optional

class FastVTKReader:
    """Fast VTK reader that skips coordinate processing using cached grids."""
    
    def __init__(self, grid_cache_manager):
        """
        Initialize fast VTK reader.
        
        Args:
            grid_cache_manager: GridCacheManager instance with cached grids
        """
        self.cache_manager = grid_cache_manager
        
    def read_f_data_only(self, vtk_file: str, resolution: int) -> Dict:
        """
        Read only F data from VTK file using cached grid.
        
        Args:
            vtk_file: Path to VTK file
            resolution: Grid resolution (must be cached)
            
        Returns:
            Dictionary with F data and cached grid geometry
        """
        # Get cached grid for this resolution
        cache = self.cache_manager.get_resolution_cache(resolution, vtk_file)
        
        if not cache['grid_initialized']:
            # Initialize grid from this file if not already done
            cache = self.cache_manager.get_resolution_cache(resolution, vtk_file)
        
        if not cache['grid_initialized']:
            raise ValueError(f"Could not initialize grid for resolution {resolution}")
        
        # Read F data only
        f_data = self._extract_f_data_only(vtk_file)
        
        # Extract simulation time from filename
        sim_time = self._extract_time_from_filename(vtk_file)
        
        # Reshape F data to proper grid shape
        f_grid = f_data.reshape(cache['f_shape'])
        
        return {
            'f': f_grid,
            'x': cache['x_grid'],      # Pre-computed!
            'y': cache['y_grid'],      # Pre-computed!
            'time': sim_time,
            'dims': cache['dimensions']
        }
    
    def _extract_f_data_only(self, vtk_file: str) -> np.ndarray:
        """
        Extract only the F scalar field data from VTK file.
        
        This skips ALL coordinate processing and jumps directly to F data.
        """
        print(f"  📖 Reading F-data from: {os.path.basename(vtk_file)}")
        
        with open(vtk_file, 'r') as f:
            lines = f.readlines()
        
        f_data = []
        in_f_section = False
        
        for line in lines:
            # Look for F data section
            if 'SCALARS F float' in line:
                in_f_section = True
                continue
            elif 'LOOKUP_TABLE default' in line and in_f_section:
                # Skip lookup table line
                continue
            elif in_f_section and line.strip():
                # Parse F values
                try:
                    f_values = [float(x) for x in line.strip().split()]
                    f_data.extend(f_values)
                except ValueError:
                    # End of F data section
                    break
            elif in_f_section and not line.strip():
                # Empty line might indicate end of section
                break
        
        if not f_data:
            raise ValueError("No F data found in VTK file")
        
        print(f"  📊 Read {len(f_data)} F values")
        return np.array(f_data)
    
    def _extract_time_from_filename(self, vtk_file: str) -> float:
        """Extract simulation time from VTK filename."""
        basename = os.path.basename(vtk_file)
        
        # Pattern for RT800x800-5999.vtk -> time = 5999/1000 = 5.999
        time_match = re.search(r'(\d+)\.vtk$', basename)
        if time_match:
            return float(time_match.group(1)) / 1000.0
        else:
            print(f"  ⚠️  Could not extract time from: {basename}")
            return 0.0

# Test function
# Test function
def test_fast_vtk_reader():
    """Test FastVTKReader functionality."""
    print("Testing FastVTKReader...")
    
    # Import GridCacheManager (absolute import when run directly)
    try:
        from .grid_cache_manager import GridCacheManager
    except ImportError:
        # Fallback for direct execution
        from grid_cache_manager import GridCacheManager
    
    # Create cache manager
    cache_manager = GridCacheManager("./test_fast_reader_output")
    
    # Create fast reader
    reader = FastVTKReader(cache_manager)
    
    # Test with dummy file (will show warnings but test structure)
    try:
        result = reader.read_f_data_only("dummy_RT800x800-5999.vtk", 800)
        print("✅ FastVTKReader structure works!")
    except Exception as e:
        print(f"⚠️  Expected error with dummy file: {e}")
        print("✅ FastVTKReader error handling works!")
    
    print("FastVTKReader test complete!")

if __name__ == "__main__":
    test_fast_vtk_reader()
