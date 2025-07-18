#!/usr/bin/env python3
"""
CONREC Interface Extractor - Production Version

Python port of the CONREC contouring algorithm by Paul D. Bourke,
optimized for fractal dimension analysis of fluid interfaces.

Original Fortran code by Paul D. Bourke, modified by Rod Douglass.
Python port for integration with RT fractal analysis toolkit.
"""

import numpy as np
from typing import List, Tuple, Optional

class CONRECExtractor:
    """
    Python implementation of the CONREC contouring algorithm.
    
    Designed for precise interface extraction in fractal analysis applications.
    Preserves all geometric detail needed for accurate fractal dimension calculation.
    """
    
    def __init__(self):
        """Initialize the CONREC extractor."""
        self.segments = []
        self.domain_bounds = None
        
        # Lookup table for contour cases (from original Fortran)
        self.castab = np.array([
            [0, 0, 9],
            [0, 1, 5], 
            [7, 4, 8],
            [0, 3, 6],
            [2, 3, 2],
            [6, 3, 0],
            [8, 4, 7],
            [5, 1, 0],
            [9, 0, 0]
        ]).reshape(3, 3, 3)
        
        print("CONREC extractor initialized - using precision marching squares")
    
    def xsect(self, p1: int, p2: int, h: np.ndarray, xh: np.ndarray) -> float:
        """Calculate x-intersection point between two triangle vertices."""
        if abs(h[p2] - h[p1]) < 1e-10:  # Avoid division by zero
            return xh[p1]
        return (h[p2] * xh[p1] - h[p1] * xh[p2]) / (h[p2] - h[p1])
    
    def ysect(self, p1: int, p2: int, h: np.ndarray, yh: np.ndarray) -> float:
        """Calculate y-intersection point between two triangle vertices."""
        if abs(h[p2] - h[p1]) < 1e-10:  # Avoid division by zero
            return yh[p1]
        return (h[p2] * yh[p1] - h[p1] * yh[p2]) / (h[p2] - h[p1])
    
    def extract_interface_conrec(self, f_grid: np.ndarray, x_grid: np.ndarray, 
                                y_grid: np.ndarray, contour_level: float = 0.5) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
        """
        Extract interface contours using CONREC algorithm.
        
        Args:
            f_grid: 2D scalar field (volume fraction)
            x_grid: 2D x-coordinate array
            y_grid: 2D y-coordinate array  
            contour_level: Contour level to extract (default: 0.5 for interface)
            
        Returns:
            List of line segments as ((x1, y1), (x2, y2)) tuples
        """
        # Initialize storage
        self.segments = []
        
        # Get grid dimensions
        if len(f_grid.shape) != 2:
            raise ValueError("f_grid must be 2D array")
        
        ny, nx = f_grid.shape
        
        # Initialize domain bounds tracking
        x_min = np.inf
        x_max = -np.inf
        y_min = np.inf  
        y_max = -np.inf
        
        segment_count = 0

        # Main CONREC algorithm - scan grid top-down, left-right
        for j in range(ny - 1, 0, -1):  # Top to bottom (Fortran style)
            for i in range(nx - 1):     # Left to right
        
                # Get data values at the four corners of this cell
                cell_values = np.array([
                    f_grid[j-1, i],     # Bottom-left
                    f_grid[j-1, i+1],   # Bottom-right  
                    f_grid[j, i+1],     # Top-right
                    f_grid[j, i]        # Top-left
                ])
        
                # Get coordinate values at corners
                cell_x = np.array([
                    x_grid[j-1, i],     # Bottom-left
                    x_grid[j-1, i+1],   # Bottom-right
                    x_grid[j, i+1],     # Top-right  
                    x_grid[j, i]        # Top-left
                ])
        
                cell_y = np.array([
                    y_grid[j-1, i],     # Bottom-left
                    y_grid[j-1, i+1],   # Bottom-right
                    y_grid[j, i+1],     # Top-right
                    y_grid[j, i]        # Top-left
                ])
        
                # Check if contour level passes through this cell
                dmin = np.min(cell_values)
                dmax = np.max(cell_values)
        
                if dmax >= contour_level and dmin <= contour_level:
                    # Contour passes through this cell - process all 4 triangles
                    for m in range(4):
                        # Set up triangle vertices (using original Fortran indexing)
                        m1 = m + 1         # Current vertex (1-based)
                        m2 = 0             # Center point (0-based)
                        m3 = (m + 1) % 4 + 1  # Next vertex (1-based)
                        
                        # Initialize arrays for this triangle
                        h = np.zeros(5)      # Height differences from contour level
                        xh = np.zeros(5)     # X coordinates
                        yh = np.zeros(5)     # Y coordinates
                        sh = np.zeros(5)     # Signs of height differences
                        
                        # Set corner values (1-based indexing for corners)
                        for vertex in range(4):
                            h[vertex + 1] = cell_values[vertex] - contour_level
                            xh[vertex + 1] = cell_x[vertex]
                            yh[vertex + 1] = cell_y[vertex]
                        
                        # Calculate center point (average of 4 corners)
                        h[0] = 0.25 * (h[1] + h[2] + h[3] + h[4])
                        xh[0] = 0.25 * (xh[1] + xh[2] + xh[3] + xh[4])  # True center
                        yh[0] = 0.25 * (yh[1] + yh[2] + yh[3] + yh[4])  # True center
                        
                        # Calculate signs
                        for vertex in range(5):
                            if h[vertex] > 0.0:
                                sh[vertex] = 1
                            elif h[vertex] < 0.0:
                                sh[vertex] = -1
                            else:
                                sh[vertex] = 0
                        
                        # Look up contour case from table
                        try:
                            # Convert signs to integers for array indexing
                            idx1 = int(sh[m1]) + 1
                            idx2 = int(sh[m2]) + 1
                            idx3 = int(sh[m3]) + 1
                            case = self.castab[idx1, idx2, idx3]

                            if case != 0:
                                x1, y1, x2, y2 = self._extract_segment_for_case(case, m1, m2, m3, h, xh, yh)

                                if x1 is not None:  # Valid segment found
                                    # Update domain bounds
                                    x_min = min(x_min, x1, x2)
                                    x_max = max(x_max, x1, x2)
                                    y_min = min(y_min, y1, y2)
                                    y_max = max(y_max, y1, y2)
                                    
                                    # Add to segments list
                                    self.segments.append(((x1, y1), (x2, y2)))
                                    segment_count += 1
                                    
                        except IndexError:
                            continue  # Skip invalid cases

        # Store domain bounds
        if segment_count > 0:
            self.domain_bounds = {
                'x_min': x_min,
                'x_max': x_max, 
                'y_min': y_min,
                'y_max': y_max
            }
        else:
            # Fallback to grid bounds
            self.domain_bounds = {
                'x_min': np.min(x_grid),
                'x_max': np.max(x_grid),
                'y_min': np.min(y_grid), 
                'y_max': np.max(y_grid)
            }
        
        return self.segments
    
    def _extract_segment_for_case(self, case: int, m1: int, m2: int, m3: int,
                                 h: np.ndarray, xh: np.ndarray, yh: np.ndarray) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
        """
        Extract line segment coordinates for a given contour case.
        
        This implements the 9 different cases from the original CONREC algorithm.
        """
        
        try:
            if case == 1:
                # Case 1 - Line between vertices 1 and 2
                x1, y1 = xh[m1], yh[m1]
                x2, y2 = xh[m2], yh[m2]
                
            elif case == 2:
                # Case 2 - Line between vertices 2 and 3
                x1, y1 = xh[m2], yh[m2]
                x2, y2 = xh[m3], yh[m3]
                
            elif case == 3:
                # Case 3 - Line between vertices 3 and 1
                x1, y1 = xh[m3], yh[m3]
                x2, y2 = xh[m1], yh[m1]
                
            elif case == 4:
                # Case 4 - Line between vertex 1 and side 2-3
                x1, y1 = xh[m1], yh[m1]
                x2 = self.xsect(m2, m3, h, xh)
                y2 = self.ysect(m2, m3, h, yh)
                
            elif case == 5:
                # Case 5 - Line between vertex 2 and side 3-1
                x1, y1 = xh[m2], yh[m2]
                x2 = self.xsect(m3, m1, h, xh)
                y2 = self.ysect(m3, m1, h, yh)
                
            elif case == 6:
                # Case 6 - Line between vertex 3 and side 1-2
                x1, y1 = xh[m3], yh[m3]
                x2 = self.xsect(m1, m2, h, xh)
                y2 = self.ysect(m1, m2, h, yh)
                
            elif case == 7:
                # Case 7 - Line between sides 1-2 and 2-3
                x1 = self.xsect(m1, m2, h, xh)
                y1 = self.ysect(m1, m2, h, yh)
                x2 = self.xsect(m2, m3, h, xh)
                y2 = self.ysect(m2, m3, h, yh)
                
            elif case == 8:
                # Case 8 - Line between sides 2-3 and 3-1
                x1 = self.xsect(m2, m3, h, xh)
                y1 = self.ysect(m2, m3, h, yh)
                x2 = self.xsect(m3, m1, h, xh)
                y2 = self.ysect(m3, m1, h, yh)
                
            elif case == 9:
                # Case 9 - Line between sides 3-1 and 1-2
                x1 = self.xsect(m3, m1, h, xh)
                y1 = self.ysect(m3, m1, h, yh)
                x2 = self.xsect(m1, m2, h, xh)
                y2 = self.ysect(m1, m2, h, yh)
                
            else:
                # Invalid case
                return None, None, None, None
            
            # Validate segment (check for NaN or infinite values)
            if any(val is None for val in [x1, y1, x2, y2]):
                return None, None, None, None
                
            if any(np.isnan([x1, y1, x2, y2])) or any(np.isinf([x1, y1, x2, y2])):
                return None, None, None, None
                
            return x1, y1, x2, y2
            
        except Exception:
            return None, None, None, None
    
    def get_segments(self) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
        """Get the extracted segments."""
        return self.segments
    
    def get_domain_bounds(self) -> Optional[dict]:
        """Get the domain bounds from the last extraction."""
        return self.domain_bounds
    
    def extract_multiple_levels(self, f_grid: np.ndarray, x_grid: np.ndarray, 
                               y_grid: np.ndarray, 
                               levels: List[float] = [0.05, 0.5, 0.95]) -> dict:
        """
        Extract multiple contour levels for mixing zone analysis.
        
        Args:
            f_grid: 2D scalar field
            x_grid: 2D x-coordinate array
            y_grid: 2D y-coordinate array
            levels: List of contour levels to extract
            
        Returns:
            Dictionary with level names as keys and segment lists as values
        """
        level_names = ['lower_boundary', 'interface', 'upper_boundary']
        results = {}
        
        for i, level in enumerate(levels):
            level_name = level_names[i] if i < len(level_names) else f'level_{level:.2f}'
            segments = self.extract_interface_conrec(f_grid, x_grid, y_grid, level)
            results[level_name] = segments
        
        return results


def compare_extraction_methods(f_grid: np.ndarray, x_grid: np.ndarray, y_grid: np.ndarray,
                              contour_level: float = 0.5) -> dict:
    """
    Compare CONREC extraction with scikit-image method.
    
    Useful for validation and debugging.
    """
    from skimage import measure
    
    print(f"🔍 COMPARISON: CONREC vs scikit-image at level {contour_level:.3f}")
    
    # CONREC method
    conrec = CONRECExtractor()
    conrec_segments = conrec.extract_interface_conrec(f_grid, x_grid, y_grid, contour_level)
    
    # Scikit-image method
    try:
        skimage_contours = measure.find_contours(f_grid.T, contour_level)
        
        # Convert to segments
        skimage_segments = []
        for contour in skimage_contours:
            if len(contour) > 1:
                # Convert indices to physical coordinates
                x_physical = np.interp(contour[:, 1], np.arange(f_grid.shape[0]), x_grid[:, 0])
                y_physical = np.interp(contour[:, 0], np.arange(f_grid.shape[1]), y_grid[0, :])
                
                # Convert to segments
                for i in range(len(contour) - 1):
                    x1, y1 = x_physical[i], y_physical[i]
                    x2, y2 = x_physical[i+1], y_physical[i+1]
                    skimage_segments.append(((x1, y1), (x2, y2)))
                    
    except Exception as e:
        print(f"   Scikit-image failed: {e}")
        skimage_segments = []
    
    print(f"   CONREC segments: {len(conrec_segments)}")
    print(f"   Scikit-image segments: {len(skimage_segments)}")
    
    return {
        'conrec_segments': conrec_segments,
        'skimage_segments': skimage_segments,
        'conrec_count': len(conrec_segments),
        'skimage_count': len(skimage_segments),
        'improvement_factor': len(conrec_segments) / max(1, len(skimage_segments))
    }
