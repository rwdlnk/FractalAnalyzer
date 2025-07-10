"""
Grid Cache Manager for Resolution-Based Optimization

This module caches VTK grid geometry once per resolution to avoid
redundant coordinate reading and cell-center calculations.
"""

import os
import numpy as np
from typing import Dict, Tuple, Optional

class GridCacheManager:
    """Manages grid geometry caching across RT parameter studies."""
    
    def __init__(self, base_output_dir: str):
        """
        Initialize grid cache manager.
        
        Args:
            base_output_dir: Base directory for organizing results by resolution
        """
        self.base_dir = base_output_dir
        self.caches = {}  # resolution -> cache_data
        os.makedirs(base_output_dir, exist_ok=True)
        
    def get_resolution_cache(self, resolution: int, sample_vtk_file: str) -> Dict:
        """
        Get or create grid cache for specified resolution.
        
        Args:
            resolution: Grid resolution (e.g., 800 for 800x800)
            sample_vtk_file: Sample VTK file to extract grid from
            
        Returns:
            Dictionary containing cached grid data and output directory
        """
        if resolution not in self.caches:
            print(f"🔧 Initializing {resolution}×{resolution} grid cache...")
            self._create_resolution_cache(resolution, sample_vtk_file)
            
        return self.caches[resolution]

    def _create_resolution_cache(self, resolution: int, sample_vtk_file: str):
        """Create and cache grid geometry for resolution."""
        res_dir = os.path.join(self.base_dir, f"{resolution}x{resolution}")
        os.makedirs(res_dir, exist_ok=True)
    
        # Extract grid geometry once
        if os.path.exists(sample_vtk_file):
            try:
                grid_data = self._extract_grid_geometry(sample_vtk_file)
                grid_initialized = True
                print(f"  ✅ Grid extracted: {grid_data['x_grid'].shape}")
            except Exception as e:
                print(f"  ❌ Grid extraction failed: {e}")
                grid_data = {}
                grid_initialized = False
        else:
            print(f"  ⚠️  Sample file not found: {sample_vtk_file}")
            grid_data = {}
            grid_initialized = False
    
        # Cache everything
        self.caches[resolution] = {
            'resolution': resolution,
            'output_dir': res_dir,
            'sample_file': sample_vtk_file,
            'grid_initialized': grid_initialized,  # This was the issue!
            **grid_data  # Unpack grid data into cache
        }
    
        print(f"  📁 Output directory: {res_dir}")
        if grid_initialized:
            print(f"  📐 Grid cached: {resolution}×{resolution}")

    def _extract_grid_geometry(self, vtk_file: str) -> Dict:
        """
        Extract grid coordinates and compute cell centers ONCE.

        This replicates the logic from rt_analyzer.py read_vtk_file()
        but only reads coordinates, not F data.
        """
        print(f"  📖 Reading grid from: {os.path.basename(vtk_file)}")

        with open(vtk_file, 'r') as f:
            lines = f.readlines()

        # Extract dimensions
        nx = ny = nz = None
        for line in lines:
            if "DIMENSIONS" in line:
                parts = line.strip().split()
                nx, ny, nz = int(parts[1]), int(parts[2]), int(parts[3])
                break

        if nx is None:
            raise ValueError("Could not find DIMENSIONS in VTK file")

        # Extract coordinates
        x_coords = self._extract_coordinates(lines, "X_COORDINATES")
        y_coords = self._extract_coordinates(lines, "Y_COORDINATES")

        # Check for cell-centered data
        is_cell_data = any("CELL_DATA" in line for line in lines)

        if is_cell_data:
            # Cell-centered data - compute cell centers
            nx_cells, ny_cells = nx-1, ny-1

            # Create cell-centered coordinates
            x_cell = 0.5 * (x_coords[:-1] + x_coords[1:])
            y_cell = 0.5 * (y_coords[:-1] + y_coords[1:])

            # Create 2D meshgrid
            x_grid, y_grid = np.meshgrid(x_cell, y_cell)
            x_grid = x_grid.T  # Transpose to match data ordering
            y_grid = y_grid.T

            f_shape = (nx_cells, ny_cells)  # Shape for F data

        else:
            # Point data - use coordinates directly
            x_grid, y_grid = np.meshgrid(x_coords, y_coords)
            x_grid = x_grid.T
            y_grid = y_grid.T

            f_shape = (nx, ny)  # Shape for F data

        print(f"  📊 Cell data: {is_cell_data}, F shape: {f_shape}")

        return {
            'x_coords': x_coords,
            'y_coords': y_coords,
            'x_grid': x_grid,
            'y_grid': y_grid,
            'f_shape': f_shape,
            'is_cell_data': is_cell_data,
            'dimensions': (nx, ny, nz)
        }

    def _extract_coordinates(self, lines, coord_type):
        """Extract X_COORDINATES or Y_COORDINATES from VTK lines."""
        coords = []

        for i, line in enumerate(lines):
            if coord_type in line:
                parts = line.strip().split()
                n_coords = int(parts[1])

                # Read coordinate data
                j = i + 1
                while len(coords) < n_coords and j < len(lines):
                    coords.extend([float(x) for x in lines[j].strip().split()])
                    j += 1
                break

        return np.array(coords)

# Enhanced test function
def test_grid_cache_manager():
    """Test basic functionality."""
    print("Testing GridCacheManager...")

    # Create manager
    manager = GridCacheManager("./test_cache_output")

    # Test with dummy files (will show warning but still work)
    cache_800 = manager.get_resolution_cache(800, "dummy_file.vtk")
    cache_400 = manager.get_resolution_cache(400, "dummy_file.vtk")

    print(f"✅ Cache 800: {cache_800['output_dir']}")
    print(f"✅ Cache 400: {cache_400['output_dir']}")
    print("GridCacheManager test complete!")

if __name__ == "__main__":
    test_grid_cache_manager()

