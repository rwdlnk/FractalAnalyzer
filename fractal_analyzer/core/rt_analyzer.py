# rt_analyzer.py - REBUILT VERSION with Complete Optimization
import numpy as np
import pandas as pd
from scipy import stats, ndimage
import matplotlib.pyplot as plt
import os
import re
import time
import glob
import argparse
from typing import Tuple, List, Dict, Optional
from skimage import measure
try:
    from .conrec_extractor import CONRECExtractor, compare_extraction_methods
    from .plic_extractor import PLICExtractor, AdvancedPLICExtractor, compare_plic_vs_conrec
except ImportError:
    # Fallback for direct script execution
    from conrec_extractor import CONRECExtractor, compare_extraction_methods
    from plic_extractor import PLICExtractor, AdvancedPLICExtractor, compare_plic_vs_conrec
from scipy import fft

# =================================================================
# OPTIMIZATION: Interface Cache System
# =================================================================

class InterfaceCache:
    """
    OPTIMIZATION: Comprehensive cache for interface extraction results.
    
    Eliminates redundant interface extractions and spatial index creation
    by storing all interface-related data in a single, reusable object.
    """
    
    def __init__(self, primary_segments=None, contours=None, spatial_index=None, 
                 bounding_box=None, grid_spacing=None, extraction_method='scikit-image'):
        """Initialize interface cache with extracted data."""
        self.primary_segments = primary_segments or []
        self.contours = contours or {}
        self.spatial_index = spatial_index
        self.bounding_box = bounding_box
        self.grid_spacing = grid_spacing
        self.extraction_method = extraction_method
        
        # Metadata for diagnostics and optimization tracking
        self.extraction_time = time.time()
        self.segment_count = len(self.primary_segments)
        self.total_contour_points = self._count_contour_points()
        
    def _count_contour_points(self):
        """Count total points across all contour levels."""
        total = 0
        for level_contours in self.contours.values():
            if level_contours:
                for contour in level_contours:
                    if hasattr(contour, 'shape'):
                        total += contour.shape[0]  # numpy array
                    else:
                        total += len(contour)  # list
        return total
    
    def get_segments_for_fractal(self):
        """Get segments optimized for fractal analysis."""
        return self.primary_segments
    
    def get_contours_for_mixing(self, method='dalziel'):
        """Get contours optimized for specific mixing thickness calculation."""
        if method == 'dalziel':
            # Dalziel needs 0.05 and 0.95 level contours
            return {
                'lower_boundary': self.contours.get('lower_boundary', []),
                'upper_boundary': self.contours.get('upper_boundary', []),
                'interface': self.contours.get('interface', [])  # For fallback
            }
        elif method == 'geometric':
            # Geometric needs 0.5 level contours
            return {
                'interface': self.contours.get('interface', [])
            }
        else:
            # Statistical method doesn't need contours (uses grid data)
            return {}
    
    def has_spatial_index(self):
        """Check if spatial index is available for optimized box counting."""
        return self.spatial_index is not None
    
    def has_segments(self):
        """Check if primary segments are available for fractal analysis."""
        return len(self.primary_segments) > 0
    
    def has_contours(self, level='interface'):
        """Check if contours are available for specified level."""
        return level in self.contours and len(self.contours[level]) > 0

class RTAnalyzer:
    """Complete Rayleigh-Taylor simulation analyzer with fractal dimension calculation."""

    def __init__(self, output_dir="./rt_analysis", use_grid_optimization=False, no_titles=False,
                 use_conrec=False, use_plic=False, debug=False):
        """Initialize the RT analyzer."""
        self.output_dir = output_dir
        self.use_grid_optimization = use_grid_optimization
        self.no_titles = no_titles
        self.use_conrec = use_conrec
        self.use_plic = use_plic
        self.debug = debug
        os.makedirs(output_dir, exist_ok=True)

        # Add rectangular grid support
        self.grid_shape = None  # Will store (ny, nx) or (n, n) for square
        self.is_rectangular = False

        # Initialize interface cache for optimization
        self.interface_cache = {}
        self.interface_value = 0.5  # Default interface level

        # Create fractal analyzer instance
        try:
            from fractal_analyzer import FractalAnalyzer
            self.fractal_analyzer = FractalAnalyzer(no_titles=no_titles)
            print(f"Fractal analyzer initialized (grid optimization: {'ENABLED' if use_grid_optimization else 'DISABLED'})")
        except ImportError as e:
            print(f"Warning: fractal_analyzer module not found: {str(e)}")
            print("Make sure fractal_analyzer.py is in the same directory")
            self.fractal_analyzer = None
        
        # Initialize PLIC extractor if requested
        if self.use_plic:
            try:
                from .plic_extractor import AdvancedPLICExtractor
                self.plic_extractor = AdvancedPLICExtractor(debug=debug)
                print(f"PLIC extractor initialized - theoretical VOF interface reconstruction enabled")
            except ImportError as e:
                print(f"ERROR: Could not import PLIC extractor: {e}")
                print("Make sure plic_extractor.py is in the fractal_analyzer/core/ directory")
                self.plic_extractor = None
                self.use_plic = False
        else:
            self.plic_extractor = None

        # Initialize CONREC extractor if requested (and PLIC not enabled)
        if self.use_conrec and not self.use_plic:
            try:
                self.conrec_extractor = CONRECExtractor(debug=debug)
                print(f"CONREC extractor initialized - precision interface extraction enabled")
            except ImportError as e:
                print(f"ERROR: Could not import CONREC extractor: {e}")
                self.conrec_extractor = None
                self.use_conrec = False
        else:
            self.conrec_extractor = None

        # Validation: Don't allow both CONREC and PLIC simultaneously
        if self.use_conrec and self.use_plic:
            print("WARNING: Both CONREC and PLIC enabled. PLIC will take precedence.")
            self.use_conrec = False
            print("CONREC disabled. Using PLIC for interface extraction.")
    
    def enable_multifractal_analysis(self):
        """Lazy loading of multifractal capabilities."""
        if not hasattr(self, 'multifractal_analyzer'):
            from ..analysis.multifractal_analyzer import MultifractalAnalyzer
            self.multifractal_analyzer = MultifractalAnalyzer(debug=self.debug)
            print("Multifractal analyzer enabled")

    def auto_detect_resolution_from_vtk_filename(self, vtk_file):
        """Enhanced resolution detection for both square and rectangular grids."""
        import re
        import os

        basename = os.path.basename(vtk_file)

        # Pattern for rectangular RT###x###-*.vtk files
        rect_patterns = [
            r'RT(\d+)x(\d+)',      # RT160x200-1234.vtk
            r'(\d+)x(\d+)',        # 160x200-1234.vtk
            r'RT(\d+)_(\d+)',      # RT160_200-1234.vtk
        ]

        # Try rectangular patterns first
        for pattern in rect_patterns:
            match = re.search(pattern, basename)
            if match:
                nx = int(match.group(1))
                ny = int(match.group(2))
                if nx != ny:  # Rectangular
                    print(f"  Auto-detected rectangular resolution: {nx}×{ny}")
                    self.grid_shape = (ny, nx)  # Store as (ny, nx) for array indexing
                    self.is_rectangular = True
                    return nx, ny
                else:  # Square but in rectangular format
                    print(f"  Auto-detected square resolution: {nx}×{ny}")
                    self.grid_shape = (nx, nx)
                    self.is_rectangular = False
                    return nx, ny

        # Pattern for square RT###-*.vtk files (backward compatibility)
        square_pattern = r'RT(\d+)-'
        match = re.search(square_pattern, basename)
        if match:
            n = int(match.group(1))
            print(f"  Auto-detected square resolution from legacy format: {n}×{n}")
            self.grid_shape = (n, n)
            self.is_rectangular = False
            return n, n

        # Try to extract from directory path as fallback
        dir_patterns = [r'(\d+)x(\d+)', r'(\d+)_(\d+)']
        for pattern in dir_patterns:
            dir_match = re.search(pattern, vtk_file)
            if dir_match:
                nx = int(dir_match.group(1))
                ny = int(dir_match.group(2))
                print(f"  Auto-detected resolution from path: {nx}×{ny}")
                self.grid_shape = (ny, nx)
                self.is_rectangular = (nx != ny)
                return nx, ny

        print(f"  Could not auto-detect resolution from: {basename}")
        return None, None

    def read_vtk_file(self, vtk_file):
        """Enhanced VTK reader to extract velocity components AND VOF data."""
        with open(vtk_file, 'r') as f:
            lines = f.readlines()
        
        # Extract dimensions
        for i, line in enumerate(lines):
            if "DIMENSIONS" in line:
                parts = line.strip().split()
                nx, ny, nz = int(parts[1]), int(parts[2]), int(parts[3])
                break
        
        # Extract coordinates
        x_coords = []
        y_coords = []
        
        for i, line in enumerate(lines):
            if "X_COORDINATES" in line:
                parts = line.strip().split()
                n_coords = int(parts[1])
                coords_data = []
                j = i + 1
                while len(coords_data) < n_coords:
                    coords_data.extend(list(map(float, lines[j].strip().split())))
                    j += 1
                x_coords = np.array(coords_data)
            
            if "Y_COORDINATES" in line:
                parts = line.strip().split()
                n_coords = int(parts[1])
                coords_data = []
                j = i + 1
                while len(coords_data) < n_coords:
                    coords_data.extend(list(map(float, lines[j].strip().split())))
                    j += 1
                y_coords = np.array(coords_data)
        
        # Check if this is cell-centered data
        is_cell_data = any("CELL_DATA" in line for line in lines)
        
        # Extract ALL scalar fields (F, u, v, and any others)
        scalar_data = {}
        
        for i, line in enumerate(lines):
            if line.strip().startswith("SCALARS"):
                parts = line.strip().split()
                if len(parts) >= 2:
                    field_name = parts[1]  # F, u, v, etc.
                    
                    # Read the data values
                    data_values = []
                    j = i + 2  # Skip the LOOKUP_TABLE line
                    while j < len(lines) and not lines[j].strip().startswith("SCALARS"):
                        if lines[j].strip():  # Skip empty lines
                            data_values.extend(list(map(float, lines[j].strip().split())))
                        j += 1
                    
                    if data_values:
                        scalar_data[field_name] = np.array(data_values)

        # Handle coordinate system (cell vs node centered)
        if is_cell_data:
            nx_cells, ny_cells = nx-1, ny-1
            x_cell = 0.5 * (x_coords[:-1] + x_coords[1:])
            y_cell = 0.5 * (y_coords[:-1] + y_coords[1:])
            x_grid, y_grid = np.meshgrid(x_cell, y_cell)
            x_grid, y_grid = x_grid.T, y_grid.T
            grid_shape = (nx_cells, ny_cells)
        else:
            x_grid, y_grid = np.meshgrid(x_coords, y_coords)
            x_grid, y_grid = x_grid.T, y_grid.T
            grid_shape = (nx, ny)
        
        # Reshape all scalar fields to 2D grids
        reshaped_data = {}
        for field_name, data in scalar_data.items():
            if len(data) == grid_shape[0] * grid_shape[1]:
                reshaped_data[field_name] = data.reshape(grid_shape[1], grid_shape[0]).T
            else:
                print(f"   WARNING: {field_name} size mismatch: {len(data)} vs expected {grid_shape[0] * grid_shape[1]}")
        
        # Extract simulation time
        time_match = re.search(r'(\d+)\.vtk$', os.path.basename(vtk_file))
        sim_time = float(time_match.group(1))/1000.0 if time_match else 0.0
        
        # Create comprehensive output dictionary
        result = {
            'x': x_grid,
            'y': y_grid,
            'dims': (nx, ny, nz),
            'time': sim_time,
            'is_cell_data': is_cell_data
        }
        
        # Add all available fields
        result.update(reshaped_data)
        
        # Ensure we have at least F field for backward compatibility
        if 'F' not in result and 'f' not in result:
            print("   WARNING: No VOF field (F) found in VTK file")
        else:
            # Standardize field name for backward compatibility
            if 'F' in result and 'f' not in result:
                result['f'] = result['F']
        
        return result
    # Step 2
    def extract_interface(self, f_grid, x_grid, y_grid, level=0.5, extract_all_levels=True):
        """
        ENHANCED: Extract interface contour(s) with PLIC, CONREC, or scikit-image methods.
        FIXED: Proper method priority handling.
        """
        if self.use_plic and self.plic_extractor is not None:
            print(f"   Interface extraction method: PLIC")
            return self._extract_interface_plic(f_grid, x_grid, y_grid, level, extract_all_levels)
        elif self.use_conrec and self.conrec_extractor is not None:
            print(f"   Interface extraction method: CONREC")
            return self._extract_interface_conrec(f_grid, x_grid, y_grid, level, extract_all_levels)
        else:
            print(f"   Interface extraction method: scikit-image")
            return self._extract_interface_skimage(f_grid, x_grid, y_grid, level, extract_all_levels)

    def _extract_interface_plic(self, f_grid, x_grid, y_grid, level=0.5, extract_all_levels=True):
        """Extract interface using PLIC algorithm with enhanced debugging."""
        print(f"     PLIC: Starting interface extraction...")
        if self.debug:
            print(f"     DEBUG: F-field stats - min={np.min(f_grid):.3f}, max={np.max(f_grid):.3f}")
            print(f"     DEBUG: Grid shape: {f_grid.shape}")

        # Check if PLIC extractor is available
        if self.plic_extractor is None:
            print(f"     ERROR: PLIC extractor is None!")
            raise RuntimeError("PLIC extractor not initialized")

        try:
            if extract_all_levels:
                # PLIC doesn't have built-in multi-level support, so we'll extract each level separately
                mixing_levels = {
                    'lower_boundary': 0.05,   # Lower mixing zone boundary
                    'interface': 0.5,         # Primary interface
                    'upper_boundary': 0.95    # Upper mixing zone boundary
                }

                all_contours = {}
                total_segments = 0

                for level_name, level_value in mixing_levels.items():
                    try:
                        print(f"     PLIC: Extracting {level_name} (F={level_value:.2f})")

                        # For PLIC, we need to modify the volume fraction field to center around each level
                        if level_value == 0.5:
                            # Use original field for interface
                            f_for_plic = f_grid
                        else:
                            # Rescale field to center around the desired level
                            f_centered = f_grid - 0.5 + level_value
                            f_for_plic = np.clip(f_centered, 0.0, 1.0)

                        # Extract using PLIC with error handling
                        segments = self.plic_extractor.extract_interface_plic(f_for_plic, x_grid, y_grid)
                        print(f"       PLIC extraction successful, got {len(segments)} segments")

                        # Convert segments to contour format for compatibility
                        contour_paths = self._segments_to_contour_paths(segments)
                        all_contours[level_name] = contour_paths

                        segment_count = len(segments)
                        total_segments += segment_count
                        print(f"     {level_name}: {segment_count} segments → {len(contour_paths)} paths")

                    except Exception as level_error:
                        print(f"     PLIC ERROR for {level_name}: {level_error}")
                        all_contours[level_name] = []

                print(f"     PLIC total: {total_segments} segments across all levels")
                return all_contours

            else:
                # Extract single level
                print(f"     PLIC: Extracting single level F={level:.3f}")
                segments = self.plic_extractor.extract_interface_plic(f_grid, x_grid, y_grid)
                print(f"       PLIC extraction successful, got {len(segments)} segments")

                # Convert to contour format for compatibility
                contour_paths = self._segments_to_contour_paths(segments)
                print(f"     PLIC: {len(segments)} segments → {len(contour_paths)} paths")
                return contour_paths

        except Exception as outer_error:
            print(f"     OUTER PLIC ERROR: {outer_error}")
            print(f"     Falling back to scikit-image method...")
            return self._extract_interface_skimage(f_grid, x_grid, y_grid, level if not extract_all_levels else 0.5, extract_all_levels)

    def _extract_interface_conrec(self, f_grid, x_grid, y_grid, level=0.5, extract_all_levels=True):
        """Extract interface using CONREC algorithm."""
        f_for_contour = f_grid

        if extract_all_levels:
            # Extract all three mixing zone levels
            mixing_levels = [0.05, 0.5, 0.95]  # lower_boundary, interface, upper_boundary

            print(f"     CONREC: Extracting multiple levels: {mixing_levels}")
            level_results = self.conrec_extractor.extract_multiple_levels(
                f_for_contour, x_grid, y_grid, mixing_levels
            )

            # Convert segments to contour format for compatibility
            all_contours = {}
            total_segments = 0

            for level_name, segments in level_results.items():
                # Convert segments back to contour paths for compatibility
                contour_paths = self._segments_to_contour_paths(segments)
                all_contours[level_name] = contour_paths

                segment_count = len(segments)
                total_segments += segment_count
                print(f"     {level_name}: {segment_count} segments → {len(contour_paths)} paths")

            print(f"     CONREC total: {total_segments} segments across all levels")
            return all_contours

        else:
            # Extract single level
            print(f"     CONREC: Extracting single level F={level:.3f}")
            segments = self.conrec_extractor.extract_interface_conrec(
                f_for_contour, x_grid, y_grid, level
            )

            # Convert to contour format for compatibility
            contour_paths = self._segments_to_contour_paths(segments)

            print(f"     CONREC: {len(segments)} segments → {len(contour_paths)} paths")
            return contour_paths

    def _extract_interface_skimage(self, f_grid, x_grid, y_grid, level=0.5, extract_all_levels=True):
        """Extract interface using scikit-image (original method) - COMPLETE VERSION."""
        # Check for binary data
        unique_vals = np.unique(f_grid)
        is_binary = len(unique_vals) <= 10

        if is_binary:
            print(f"     Binary VOF detected: {unique_vals}")
            print(f"     Applying smoothing for contour interpolation...")

            # For binary VOF data, apply gentle smoothing to create interpolation zones
            f_smoothed = ndimage.gaussian_filter(f_grid.astype(float), sigma=0.8)
            print(f"     After smoothing: min={np.min(f_smoothed):.3f}, max={np.max(f_smoothed):.3f}")
            f_for_contour = f_smoothed
        else:
            print(f"     Continuous F-field detected, using direct contouring")
            f_for_contour = f_grid

        if extract_all_levels:
            # Extract all three mixing zone levels
            mixing_levels = {
                'lower_boundary': 0.05,   # Lower mixing zone boundary
                'interface': 0.5,         # Primary interface
                'upper_boundary': 0.95    # Upper mixing zone boundary
            }

            all_contours = {}
            total_segments = 0

            for level_name, level_value in mixing_levels.items():
                try:
                    contours = measure.find_contours(f_for_contour.T, level_value)

                    # Convert to physical coordinates
                    physical_contours = []
                    for contour in contours:
                        if len(contour) > 1:  # Skip single-point contours
                            x_physical = np.interp(contour[:, 1], np.arange(f_grid.shape[0]), x_grid[:, 0])
                            y_physical = np.interp(contour[:, 0], np.arange(f_grid.shape[1]), y_grid[0, :])
                            physical_contours.append(np.column_stack([x_physical, y_physical]))

                    all_contours[level_name] = physical_contours

                    # Count segments for this level
                    level_segments = sum(len(contour) - 1 for contour in physical_contours if len(contour) > 1)
                    total_segments += level_segments

                    print(f"     F={level_value:.2f} ({level_name}): {len(physical_contours)} paths, {level_segments} segments")

                except Exception as e:
                    print(f"     F={level_value:.2f} ({level_name}): ERROR - {e}")
                    all_contours[level_name] = []

            print(f"     Total segments across all levels: {total_segments}")

            # If primary interface (F=0.5) failed, try alternative approaches
            if len(all_contours.get('interface', [])) == 0:
                print(f"     Primary interface (F=0.5) failed, trying adaptive approach...")
                all_contours['interface'] = self._extract_interface_adaptive(f_for_contour, x_grid, y_grid)

            return all_contours

        else:
            # Extract single level (backward compatibility)
            try:
                contours = measure.find_contours(f_for_contour.T, level)
                print(f"     Found {len(contours)} contour paths for F={level:.2f}")

                # Convert to physical coordinates
                physical_contours = []
                total_segments = 0

                for i, contour in enumerate(contours):
                    if len(contour) > 1:
                        x_physical = np.interp(contour[:, 1], np.arange(f_grid.shape[0]), x_grid[:, 0])
                        y_physical = np.interp(contour[:, 0], np.arange(f_grid.shape[1]), y_grid[0, :])
                        physical_contours.append(np.column_stack([x_physical, y_physical]))

                        segments_in_path = len(contour) - 1
                        total_segments += segments_in_path

                print(f"     Total segments: {total_segments}")

                # If we got very few segments, try adaptive approach
                if total_segments < 10:
                    print(f"     Too few segments ({total_segments}), trying adaptive approach...")
                    adaptive_contours = self._extract_interface_adaptive(f_for_contour, x_grid, y_grid)
                    if adaptive_contours:
                        return adaptive_contours

                return physical_contours

            except Exception as e:
                print(f"     ERROR in find_contours: {e}")
                print(f"     Falling back to adaptive method...")
                return self._extract_interface_adaptive(f_for_contour, x_grid, y_grid)

    def _extract_interface_adaptive(self, f_grid, x_grid, y_grid):
        """Adaptive interface extraction when standard contouring fails."""
        print(f"       Trying adaptive interface extraction...")

        # Method 1: Try multiple contour levels
        test_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        best_contours = []
        best_count = 0
        best_level = 0.5

        for test_level in test_levels:
            try:
                contours = measure.find_contours(f_grid.T, test_level)

                # Convert to physical coordinates
                physical_contours = []
                for contour in contours:
                    if len(contour) > 1:
                        x_physical = np.interp(contour[:, 1], np.arange(f_grid.shape[0]), x_grid[:, 0])
                        y_physical = np.interp(contour[:, 0], np.arange(f_grid.shape[1]), y_grid[0, :])
                        physical_contours.append(np.column_stack([x_physical, y_physical]))

                # Count total segments
                total_segments = sum(len(contour) - 1 for contour in physical_contours if len(contour) > 1)

                if total_segments > best_count:
                    best_count = total_segments
                    best_contours = physical_contours
                    best_level = test_level

            except:
                continue

        if best_count > 0:
            print(f"       Best adaptive level: F={best_level:.1f} with {best_count} segments")
            return best_contours

        print(f"       All adaptive methods failed")
        return []

    def _segments_to_contour_paths(self, segments):
        """Convert line segments back to contour paths for compatibility."""
        if not segments:
            return []

        # For now, convert each segment to a 2-point contour
        # More sophisticated path reconstruction could be added later
        contour_paths = []

        for (x1, y1), (x2, y2) in segments:
            contour_path = np.array([[x1, y1], [x2, y2]])
            contour_paths.append(contour_path)

        return contour_paths

    def convert_contours_to_segments(self, contours_input):
        """
        ENHANCED: Convert contours to line segments, handles both single-level and multi-level input.
        """
        if isinstance(contours_input, dict):
            # Multi-level input - use primary interface (F=0.5)
            if 'interface' in contours_input and contours_input['interface']:
                contours = contours_input['interface']
                print(f"   Using primary interface contours: {len(contours)} paths")
            else:
                # Fallback to any available level
                for level_name, level_contours in contours_input.items():
                    if level_contours:
                        contours = level_contours
                        print(f"   Using {level_name} contours as fallback: {len(contours)} paths")
                        break
                else:
                    print(f"   No contours available in any level")
                    return []
        else:
            # Single-level input (backward compatibility)
            contours = contours_input

        # Convert contours to segments
        segments = []

        for contour in contours:
            # Each contour is a numpy array of points: [[x1,y1], [x2,y2], ...]
            for i in range(len(contour) - 1):
                x1, y1 = contour[i]
                x2, y2 = contour[i+1]
                segments.append(((x1, y1), (x2, y2)))

        print(f"   Converted to {len(segments)} line segments")
        return segments

    # STEP 3
    def compute_fractal_dimension(self, data, min_box_size=None):
        """Compute fractal dimension of the interface using basic box counting."""
        if self.fractal_analyzer is None:
            print("Fractal analyzer not available. Skipping fractal dimension calculation.")
            return {
                'dimension': np.nan,
                'error': np.nan,
                'r_squared': np.nan
            }

        # Extract contours - ENHANCED VERSION
        contours = self.extract_interface(data['f'], data['x'], data['y'], extract_all_levels=False)

        # Convert to segments - ENHANCED VERSION
        segments = self.convert_contours_to_segments(contours)

        if not segments:
            print("No interface segments found.")
            return {
                'dimension': np.nan,
                'error': np.nan,
                'r_squared': np.nan
            }

        print(f"Found {len(segments)} interface segments")

        # PHYSICS-BASED AUTO-ESTIMATION
        if min_box_size is None:
            min_box_size = self.fractal_analyzer.estimate_min_box_size(segments)
            print(f"Auto-estimated min_box_size: {min_box_size:.8f}")
        else:
            print(f"Using provided min_box_size: {min_box_size:.8f}")

        try:
            # Use the basic analyze_linear_region method
            results = self.fractal_analyzer.analyze_linear_region(
                segments,
                fractal_type=None,  # No known theoretical value for RT
                plot_results=False,  # Don't create plots here
                plot_boxes=False,
                trim_boundary=0,
                box_size_factor=1.5,
                use_grid_optimization=self.use_grid_optimization,
                return_box_data=True,
                min_box_size=min_box_size
            )

            # Unpack results - analyze_linear_region returns tuple when return_box_data=True
            windows, dims, errs, r2s, optimal_window, optimal_dimension, box_sizes, box_counts, bounding_box = results

            # Get error for the optimal window
            optimal_idx = np.where(np.array(windows) == optimal_window)[0][0]
            error = errs[optimal_idx]
            r_squared = r2s[optimal_idx]

            print(f"Fractal dimension: {optimal_dimension:.6f} ± {error:.6f}, R² = {r_squared:.6f}")
            print(f"Window size: {optimal_window}")

            return {
                'dimension': optimal_dimension,
                'error': error,
                'r_squared': r_squared,
                'window_size': optimal_window,
                'box_sizes': box_sizes,
                'box_counts': box_counts,
                'segments': segments
            }

        except Exception as e:
            print(f"Error in fractal dimension calculation: {str(e)}")
            return {
                'dimension': np.nan,
                'error': np.nan,
                'r_squared': np.nan
            }

    # =================================================================
    # OPTIMIZATION METHODS - Step 3 Core Optimization
    # =================================================================

    def _generate_cache_key(self, vtk_file_path):
        """Generate a unique cache key for a VTK file."""
        import os
        import hashlib

        try:
            stat = os.stat(vtk_file_path)
            file_info = f"{vtk_file_path}_{stat.st_mtime}_{stat.st_size}_{self.interface_value}"
            cache_key = hashlib.md5(file_info.encode()).hexdigest()
            return cache_key
        except OSError:
            return hashlib.md5(vtk_file_path.encode()).hexdigest()

    def _load_vtk_data(self, vtk_file_path):
        """Load VTK data from file."""
        try:
            return self.read_vtk_file(vtk_file_path)
        except Exception as e:
            print(f"Error loading VTK file {vtk_file_path}: {str(e)}")
            return None

    def _extract_interface_contour(self, vtk_data, interface_value):
        """Extract interface contour using existing method."""
        try:
            contours = self.extract_interface(
                vtk_data['f'], vtk_data['x'], vtk_data['y'],
                level=interface_value, extract_all_levels=False
            )
            interface_points = []
            for contour in contours:
                for point in contour:
                    interface_points.append((float(point[0]), float(point[1])))
            return interface_points
        except Exception as e:
            print(f"Error extracting interface: {str(e)}")
            return []

    def _compute_curvature_statistics(self, interface_points):
        """Compute curvature statistics for interface."""
        try:
            if len(interface_points) < 3:
                return {'mean_curvature': 0, 'max_curvature': 0, 'curvature_std': 0}

            curvatures = []
            for i in range(1, len(interface_points) - 1):
                p1, p2, p3 = interface_points[i-1], interface_points[i], interface_points[i+1]
                dx1, dy1 = p2[0] - p1[0], p2[1] - p1[1]
                dx2, dy2 = p3[0] - p2[0], p3[1] - p2[1]
                cross = dx1 * dy2 - dy1 * dx2
                norm1 = (dx1**2 + dy1**2)**0.5
                norm2 = (dx2**2 + dy2**2)**0.5
                if norm1 > 0 and norm2 > 0:
                    curvature = cross / (norm1 * norm2)
                    curvatures.append(abs(curvature))

            if curvatures:
                return {
                    'mean_curvature': sum(curvatures) / len(curvatures),
                    'max_curvature': max(curvatures),
                    'curvature_std': (sum((c - sum(curvatures)/len(curvatures))**2 for c in curvatures) / len(curvatures))**0.5
                }
            else:
                return {'mean_curvature': 0, 'max_curvature': 0, 'curvature_std': 0}

        except Exception as e:
            print(f"Error computing curvature: {str(e)}")
            return {'error': str(e)}

    def _compute_wavelength_spectrum(self, interface_points):
        """Compute wavelength spectrum for interface."""
        try:
            if len(interface_points) < 4:
                return {'dominant_wavelength': 0, 'wavelength_count': 0}

            # Extract y-coordinates as function of x
            x_coords = [p[0] for p in interface_points]
            y_coords = [p[1] for p in interface_points]

            # Simple FFT analysis
            y_fft = np.fft.fft(y_coords)
            freqs = np.fft.fftfreq(len(y_coords))

            # Find dominant frequency
            power = np.abs(y_fft)**2
            dominant_freq_idx = np.argmax(power[1:len(power)//2]) + 1
            dominant_freq = freqs[dominant_freq_idx]

            if dominant_freq != 0:
                dominant_wavelength = 1.0 / abs(dominant_freq)
            else:
                dominant_wavelength = max(x_coords) - min(x_coords)

            return {
                'dominant_wavelength': dominant_wavelength,
                'wavelength_count': len([f for f in freqs[:len(freqs)//2] if abs(f) > 0.01])
            }

        except Exception as e:
            print(f"Error computing wavelength spectrum: {str(e)}")
            return {'error': str(e)}

    def _compute_box_counting_dimension(self, interface_points):
        """Compute box counting fractal dimension from interface points."""
        try:
            if not interface_points:
                return np.nan, np.nan, np.nan

            # Convert to segments format for fractal analyzer
            segments = []
            for i in range(len(interface_points) - 1):
                p1 = interface_points[i]
                p2 = interface_points[i + 1]
                segments.append((p1, p2))

            if not segments:
                return np.nan, np.nan, np.nan

            # Use fractal analyzer if available
            if self.fractal_analyzer is not None:
                results = self.fractal_analyzer.analyze_linear_region(
                    segments,
                    fractal_type=None,
                    plot_results=False,
                    plot_boxes=False,
                    trim_boundary=0,
                    box_size_factor=1.5,
                    use_grid_optimization=self.use_grid_optimization,
                    return_box_data=True,
                    min_box_size=None
                )

                windows, dims, errs, r2s, optimal_window, optimal_dimension, box_sizes, box_counts, bounding_box = results
                # Return dimension, error, and r_squared
                optimal_idx = windows.index(optimal_window)
                optimal_error = errs[optimal_idx]
                optimal_r2 = r2s[optimal_idx]
                return optimal_dimension, optimal_error, optimal_r2
            else:
                print("Warning: Fractal analyzer not available")
                return np.nan, np.nan, np.nan
        except Exception as e:
            return np.nan, np.nan, np.nan

    def _extract_interface_comprehensive(self, vtk_file_path):
        """
        Single-pass comprehensive interface extraction.
        Extracts ALL interface data needed for analysis in one CONREC call.
        """
        import time
        start_time = time.time()

        # Load VTK data
        vtk_data = self._load_vtk_data(vtk_file_path)
        if vtk_data is None:
            return None

        # Single CONREC call for base interface
        base_interface = self._extract_interface_contour(vtk_data, self.interface_value)
        if not base_interface:
            print(f"Warning: No interface found in {vtk_file_path}")
            return None

        # Calculate bounds for metadata
        x_coords = [pt[0] for pt in base_interface]
        y_coords = [pt[1] for pt in base_interface]
        bounds = (min(x_coords), max(x_coords), min(y_coords), max(y_coords))

        # Generate smoothed interface (reuse base data with smoothing)
        smoothed_interface = self._apply_smoothing_filter(base_interface)

        # Generate perturbed interfaces (if needed for analysis)
        perturbed_interfaces = []
        if hasattr(self, 'perturbation_levels') and self.perturbation_levels:
            for level in self.perturbation_levels:
                perturbed = self._apply_perturbation(base_interface, level)
                perturbed_interfaces.append(perturbed)

        extraction_time = time.time() - start_time

        # Package comprehensive results
        comprehensive_data = {
            'base_interface': base_interface,
            'smoothed_interface': smoothed_interface,
            'perturbed_interfaces': perturbed_interfaces,
            'metadata': {
                'file_path': vtk_file_path,
                'extraction_time': extraction_time,
                'point_count': len(base_interface),
                'bounds': bounds
            }
        }

        # Cache the comprehensive data
        cache_key = self._generate_cache_key(vtk_file_path)
        self.interface_cache[cache_key] = comprehensive_data

        return comprehensive_data

    def _apply_smoothing_filter(self, interface_points, window_size=5):
        """Apply smoothing filter to interface points for more stable analysis."""
        if len(interface_points) < window_size:
            return interface_points.copy()

        smoothed = []
        half_window = window_size // 2

        for i in range(len(interface_points)):
            # Define window bounds
            start_idx = max(0, i - half_window)
            end_idx = min(len(interface_points), i + half_window + 1)

            # Average coordinates in window
            window_points = interface_points[start_idx:end_idx]
            avg_x = sum(pt[0] for pt in window_points) / len(window_points)
            avg_y = sum(pt[1] for pt in window_points) / len(window_points)

            smoothed.append((avg_x, avg_y))

        return smoothed

    def _apply_perturbation(self, interface_points, perturbation_level=0.1):
        """Apply controlled perturbation to interface for sensitivity analysis."""
        import random

        # Calculate interface scale for perturbation normalization
        x_coords = [pt[0] for pt in interface_points]
        y_coords = [pt[1] for pt in interface_points]
        x_scale = max(x_coords) - min(x_coords)
        y_scale = max(y_coords) - min(y_coords)

        perturbation_amplitude = perturbation_level * min(x_scale, y_scale)

        perturbed = []
        for i, (x, y) in enumerate(interface_points):
            # Add controlled random perturbation
            dx = (random.random() - 0.5) * perturbation_amplitude
            dy = (random.random() - 0.5) * perturbation_amplitude

            perturbed.append((x + dx, y + dy))

        return perturbed

    def analyze_vtk_file(self, vtk_file_path, analysis_types=None, h0=0.5, mixing_method='dalziel',
         min_box_size=None, enable_multifractal=False, q_values=None, mf_output_dir=None):
        """
        OPTIMIZED: Single-pass analysis with comprehensive interface extraction.
        Eliminates redundant CONREC calls by extracting all needed interface data once.
        """
        import time

        if analysis_types is None:
            analysis_types = ['fractal_dim', 'curvature', 'wavelength']

        print(f"\n🚀 OPTIMIZED Analysis: {vtk_file_path}")
        analysis_start = time.time()

        # Check cache first
        cache_key = self._generate_cache_key(vtk_file_path)
        cached_data = self.interface_cache.get(cache_key)

        if cached_data:
            print(f"✅ Using cached interface data")
            interface_data = cached_data
        else:
            print(f"🔄 Extracting comprehensive interface data...")
            interface_data = self._extract_interface_comprehensive(vtk_file_path)
            if interface_data is None:
                return None

        # Initialize results with metadata
        results = {
            'file_path': vtk_file_path,
            'interface_extraction_time': interface_data['metadata']['extraction_time'],
            'interface_point_count': interface_data['metadata']['point_count'],
            'interface_bounds': interface_data['metadata']['bounds'],
            'analysis_types_performed': analysis_types
        }

        # Perform requested analyses using pre-extracted data
        base_interface = interface_data['base_interface']
        smoothed_interface = interface_data['smoothed_interface']

        if 'fractal_dim' in analysis_types:
            print("  📐 Computing fractal dimension...")
            fractal_start = time.time()

            # Use smoothed interface for more stable fractal calculation
            fractal_dim, fractal_error, fractal_r2 = self._compute_box_counting_dimension(smoothed_interface)
            results['fractal_dimension'] = fractal_dim
            results['fractal_error'] = fractal_error
            results['fractal_r_squared'] = fractal_r2
            results['fractal_computation_time'] = time.time() - fractal_start

        if 'curvature' in analysis_types:
            print("  📏 Computing curvature analysis...")
            curvature_start = time.time()

            curvature_stats = self._compute_curvature_statistics(smoothed_interface)
            results['curvature_analysis'] = curvature_stats
            results['curvature_computation_time'] = time.time() - curvature_start

        if 'wavelength' in analysis_types:
            print("  🌊 Computing wavelength analysis...")
            wavelength_start = time.time()

            wavelength_stats = self._compute_wavelength_spectrum(base_interface)
            results['wavelength_analysis'] = wavelength_stats
            results['wavelength_computation_time'] = time.time() - wavelength_start

        if 'mixing' in analysis_types:
            print("  🧪 Computing mixing analysis...")
            mixing_start = time.time()
            # Get raw VTK data for mixing analysis
            vtk_data = self._load_vtk_data(vtk_file_path)
            mixing_result = self.compute_mixing_thickness(vtk_data, h0, method=mixing_method)
            results.update(mixing_result)
            results['mixing_computation_time'] = time.time() - mixing_start

        if 'perturbation' in analysis_types and interface_data['perturbed_interfaces']:
            print("  🔀 Computing perturbation analysis...")
            perturbation_start = time.time()

            perturbation_results = []
            for i, perturbed_interface in enumerate(interface_data['perturbed_interfaces']):
                perturb_fractal, perturb_error, perturb_r2 = self._compute_box_counting_dimension(perturbed_interface)
                perturbation_results.append({
                    'perturbation_level': i,
                    'fractal_dimension': perturb_fractal,
                    'fractal_error': perturb_error,
                    'fractal_r_squared': perturb_r2
                })

            results['perturbation_analysis'] = perturbation_results
            results['perturbation_computation_time'] = time.time() - perturbation_start

        total_time = time.time() - analysis_start
        results['total_analysis_time'] = total_time

        print(f"✅ OPTIMIZED Analysis complete: {total_time:.2f}s")
        print(f"   Interface extraction: {interface_data['metadata']['extraction_time']:.2f}s")
        print(f"   Points processed: {interface_data['metadata']['point_count']}")

        # Multifractal analysis if requested
        if enable_multifractal and interface_data and interface_data.get("base_interface"):
            print(f"\n🔬 MULTIFRACTAL ANALYSIS")
            print(f"=" * 50)
            
            self.enable_multifractal_analysis()
            
            # Set output directory for multifractal results
            if mf_output_dir is None:
                mf_output_dir = os.path.join(self.output_dir, "multifractal")
            os.makedirs(mf_output_dir, exist_ok=True)
            
            # Convert interface data to segments for multifractal analysis
            segments = []
            base_interface = interface_data["base_interface"]
            for i in range(len(base_interface) - 1):
                p1 = base_interface[i]
                p2 = base_interface[i + 1]
                segments.append((p1, p2))
            
            # Extract time value for labeling
            time_value = results.get("file_path", "").split("_")[-1].replace(".vtk", "") if "file_path" in results else None
            if time_value:
                try:
                    time_value = float(time_value) / 1000.0
                except:
                    time_value = None
            
            try:
                mf_results = self.multifractal_analyzer.compute_multifractal_spectrum(
                    segments,
                    min_box_size=min_box_size,
                    q_values=q_values,
                    output_dir=mf_output_dir,
                    time_value=time_value
                )
                
                if mf_results:
                    results["multifractal"] = mf_results
                    self.multifractal_analyzer.print_multifractal_summary(mf_results)
                    print(f"Multifractal results saved to: {mf_output_dir}")
                else:
                    print("⚠️ Multifractal analysis failed")
                    results["multifractal"] = None
                    
            except Exception as e:
                print(f"❌ Multifractal analysis error: {str(e)}")
                results["multifractal"] = None
                result['multifractal'] = None

        return results

    # Step 4

    def find_initial_interface(self, data):
        """Find the initial interface position (y=1.0 for RT)."""
        f_avg = np.mean(data['f'], axis=0)
        y_values = data['y'][0, :]
        
        # Find where f crosses 0.5
        idx = np.argmin(np.abs(f_avg - 0.5))
        return y_values[idx]

    def compute_mixing_thickness(self, data, h0, method='geometric'):
        """Compute mixing layer thickness using different methods."""
        if method == 'geometric':
            # Extract interface contours
            contours = self.extract_interface(data['f'], data['x'], data['y'])
            
            # Find maximum displacement above and below initial interface
            ht = 0.0
            hb = 0.0
            
            for contour in contours:
                y_coords = contour[:, 1]
                # Calculate displacements from initial interface
                y_displacements = y_coords - h0
                
                # Upper mixing thickness (positive displacements)
                if np.any(y_displacements > 0):
                    ht = max(ht, np.max(y_displacements[y_displacements > 0]))
                
                # Lower mixing thickness (negative displacements, made positive)
                if np.any(y_displacements < 0):
                    hb = max(hb, np.max(-y_displacements[y_displacements < 0]))
            
            return {'ht': ht, 'hb': hb, 'h_total': ht + hb, 'method': 'geometric'}
        
        elif method == 'statistical':
            # Use concentration thresholds to define mixing zone
            f_avg = np.mean(data['f'], axis=0)
            y_values = data['y'][0, :]
            
            epsilon = 0.01  # Threshold for "pure" fluid
            
            # Find uppermost position where f drops below 1-epsilon
            upper_idx = np.where(f_avg < 1 - epsilon)[0]
            if len(upper_idx) > 0:
                y_upper = y_values[upper_idx[0]]
            else:
                y_upper = y_values[-1]
            
            # Find lowermost position where f rises above epsilon
            lower_idx = np.where(f_avg > epsilon)[0]
            if len(lower_idx) > 0:
                y_lower = y_values[lower_idx[-1]]
            else:
                y_lower = y_values[0]
            
            # Calculate thicknesses
            ht = max(0, y_upper - h0)
            hb = max(0, h0 - y_lower)
            
            return {'ht': ht, 'hb': hb, 'h_total': ht + hb, 'method': 'statistical'}

        elif method == 'dalziel':
            return self.compute_mixing_thickness_dalziel_correct(data, h0)

        elif method == 'three_level':
            # NEW: Use the three-level contour extraction
            contours_dict = self.extract_interface(data['f'], data['x'], data['y'], extract_all_levels=True)
            return self.get_mixing_zone_thickness(contours_dict, h0)

    def get_mixing_zone_thickness(self, contours_dict, h0=0.5):
        """
        Calculate mixing zone thickness from the three extracted contour levels.
        """
        if not isinstance(contours_dict, dict):
            print("Warning: get_mixing_zone_thickness requires multi-level contours")
            return {'ht': 0, 'hb': 0, 'h_total': 0}
        
        # Extract Y coordinates from each level
        y_coords = {}
        
        for level_name, contours in contours_dict.items():
            if contours:
                all_y = []
                for contour in contours:
                    all_y.extend(contour[:, 1])  # Y coordinates
                y_coords[level_name] = all_y
            else:
                y_coords[level_name] = []
        
        # Calculate mixing zone extent
        try:
            # Upper mixing thickness: how far upper boundary extends above h0
            if y_coords.get('upper_boundary'):
                y_upper_max = max(y_coords['upper_boundary'])
                ht = max(0, y_upper_max - h0)
            else:
                ht = 0
            
            # Lower mixing thickness: how far lower boundary extends below h0
            if y_coords.get('lower_boundary'):
                y_lower_min = min(y_coords['lower_boundary'])
                hb = max(0, h0 - y_lower_min)
            else:
                hb = 0
            
            h_total = ht + hb
            
            print(f"   Mixing zone thickness: ht={ht:.6f}, hb={hb:.6f}, total={h_total:.6f}")
            
            return {
                'ht': ht,
                'hb': hb, 
                'h_total': h_total,
                'method': 'three_level_contours'
            }
            
        except Exception as e:
            print(f"Error calculating mixing thickness: {e}")
            return {'ht': 0, 'hb': 0, 'h_total': 0}

    def find_concentration_crossing(self, f_profile, y_profile, target_concentration):
        """
        Find y-position where concentration profile crosses target value using interpolation.
        """
        # Find indices where profile crosses the target
        diff = f_profile - target_concentration
        sign_changes = np.diff(np.sign(diff))

        crossings = []
        crossing_indices = np.where(sign_changes != 0)[0]

        for i in crossing_indices:
            # Linear interpolation between points i and i+1
            y1, y2 = y_profile[i], y_profile[i+1]
            f1, f2 = f_profile[i], f_profile[i+1]
            
            # Interpolate to find exact crossing point
            if f2 != f1:  # Avoid division by zero
                y_cross = y1 + (target_concentration - f1) * (y2 - y1) / (f2 - f1)
                crossings.append(y_cross)

        return crossings

    def compute_mixing_thickness_dalziel_correct(self, data, h0):
        """
        Correct Dalziel mixing thickness implementation following JFM 1999 Equation 7.
        """
        print(f"DEBUG: Computing Dalziel mixing thickness (CORRECTED)")

        # Horizontal average (along-tank averaging) - following Dalziel notation C̄(z)
        f_avg = np.mean(data['f'], axis=0)  # Average over x (first axis)
        y_values = data['y'][0, :]  # y-coordinates along first row

        print(f"DEBUG: f_avg shape: {f_avg.shape}")
        print(f"DEBUG: y_values shape: {y_values.shape}")
        print(f"DEBUG: f_avg range: [{np.min(f_avg):.3f}, {np.max(f_avg):.3f}]")
        print(f"DEBUG: y_values range: [{np.min(y_values):.3f}, {np.max(y_values):.3f}]")
        print(f"DEBUG: h0 = {h0:.6f}")

        # Dalziel thresholds
        lower_threshold = 0.05  # 5% threshold for h_{1,0}
        upper_threshold = 0.95  # 95% threshold for h_{1,1}

        # Find exact crossing points using interpolation
        try:
            crossings_005 = self.find_concentration_crossing(f_avg, y_values, lower_threshold)
            crossings_095 = self.find_concentration_crossing(f_avg, y_values, upper_threshold)
            
            print(f"DEBUG: Found {len(crossings_005)} crossings at f=0.05: {crossings_005}")
            print(f"DEBUG: Found {len(crossings_095)} crossings at f=0.95: {crossings_095}")
            
        except Exception as e:
            print(f"ERROR in crossing detection: {e}")
            return {
                'ht': 0, 'hb': 0, 'h_total': 0,
                'h_10': None, 'h_11': None,
                'method': 'dalziel_error',
                'error': str(e)
            }

        h_10 = None  # Lower boundary position
        h_11 = None  # Upper boundary position

        # Select appropriate crossings
        if crossings_005:
            if len(crossings_005) == 1:
                h_10 = crossings_005[0]
            else:
                # Multiple crossings - select the one that makes physical sense
                valid_crossings = [c for c in crossings_005 if c <= h0]  # Below initial interface
                if valid_crossings:
                    h_10 = max(valid_crossings)  # Highest crossing below h0
                else:
                    h_10 = min(crossings_005)  # Fallback to lowest crossing
            
            print(f"DEBUG: Selected h_10 = {h_10:.6f} (f=0.05 crossing)")

        if crossings_095:
            if len(crossings_095) == 1:
                h_11 = crossings_095[0]
            else:
                # Multiple crossings - select the one that makes physical sense
                valid_crossings = [c for c in crossings_095 if c >= h0]  # Above initial interface
                if valid_crossings:
                    h_11 = min(valid_crossings)  # Lowest crossing above h0
                else:
                    h_11 = max(crossings_095)  # Fallback to highest crossing
            
            print(f"DEBUG: Selected h_11 = {h_11:.6f} (f=0.95 crossing)")

        # Calculate mixing thicknesses according to Dalziel methodology
        if h_10 is not None and h_11 is not None:
            # Upper mixing thickness: how far 95% contour extends above initial interface
            ht = max(0, h_11 - h0)
            
            # Lower mixing thickness: how far 5% contour extends below initial interface  
            hb = max(0, h0 - h_10)
            
            # Total mixing thickness
            h_total = ht + hb
            
            print(f"DEBUG: Dalziel mixing thickness calculation:")
            print(f"  h_10 (5% crossing): {h_10:.6f}")
            print(f"  h_11 (95% crossing): {h_11:.6f}")
            print(f"  h0 (initial interface): {h0:.6f}")
            print(f"  ht = max(0, {h_11:.6f} - {h0:.6f}) = {ht:.6f}")
            print(f"  hb = max(0, {h0:.6f} - {h_10:.6f}) = {hb:.6f}")
            print(f"  h_total = {ht:.6f} + {hb:.6f} = {h_total:.6f}")
            
            # Additional Dalziel-specific diagnostics
            mixing_zone_center = (h_10 + h_11) / 2
            mixing_zone_width = h_11 - h_10
            interface_offset = mixing_zone_center - h0
            
            # Mixing efficiency (fraction of domain that is mixed)
            mixing_region = (f_avg >= lower_threshold) & (f_avg <= upper_threshold)
            mixing_fraction = np.sum(mixing_region) / len(f_avg)
            
            return {
                'ht': ht,
                'hb': hb, 
                'h_total': h_total,
                'h_10': h_10,  # Position where C̄ = 0.05 (Dalziel h_{1,0})
                'h_11': h_11,  # Position where C̄ = 0.95 (Dalziel h_{1,1})
                'mixing_zone_center': mixing_zone_center,
                'mixing_zone_width': mixing_zone_width,
                'interface_offset': interface_offset,
                'mixing_fraction': mixing_fraction,
                'lower_threshold': lower_threshold,
                'upper_threshold': upper_threshold,
                'method': 'dalziel_corrected',
                'crossings_005': crossings_005,  # All 5% crossings found
                'crossings_095': crossings_095   # All 95% crossings found
            }

        else:
            # Handle case where crossings are not found
            print(f"WARNING: Could not find required concentration crossings")
            print(f"  5% crossings found: {len(crossings_005) if crossings_005 else 0}")
            print(f"  95% crossings found: {len(crossings_095) if crossings_095 else 0}")
            
            # Provide fallback using simple thresholding
            mixed_indices = np.where((f_avg >= lower_threshold) & (f_avg <= upper_threshold))[0]
            
            if len(mixed_indices) > 0:
                # Fallback to extent-based calculation
                mixed_y_min = y_values[mixed_indices[0]]
                mixed_y_max = y_values[mixed_indices[-1]]
                
                ht_fallback = max(0, mixed_y_max - h0)
                hb_fallback = max(0, h0 - mixed_y_min)
                h_total_fallback = ht_fallback + hb_fallback
                
                print(f"  Using fallback extent method:")
                print(f"    ht = {ht_fallback:.6f}, hb = {hb_fallback:.6f}")
                
                return {
                    'ht': ht_fallback,
                    'hb': hb_fallback,
                    'h_total': h_total_fallback,
                    'h_10': mixed_y_min,  # Approximate
                    'h_11': mixed_y_max,  # Approximate
                    'mixing_fraction': len(mixed_indices) / len(f_avg),
                    'lower_threshold': lower_threshold,
                    'upper_threshold': upper_threshold,
                    'method': 'dalziel_fallback',
                    'warning': 'Used extent-based fallback due to missing crossings'
                }
            else:
                # No mixing detected at all
                print(f"  No mixing zone detected between {lower_threshold} and {upper_threshold}")
                return {
                    'ht': 0,
                    'hb': 0,
                    'h_total': 0,
                    'h_10': None,
                    'h_11': None,
                    'mixing_fraction': 0,
                    'lower_threshold': lower_threshold,
                    'upper_threshold': upper_threshold,
                    'method': 'dalziel_no_mixing'
                }

    def calculate_physics_based_min_box_size(self, resolution_or_shape, domain_size=1.0, safety_factor=4):
        """Enhanced min box size calculation for rectangular grids."""
        # Handle both square and rectangular inputs
        if isinstance(resolution_or_shape, (tuple, list)) and len(resolution_or_shape) == 2:
            nx, ny = resolution_or_shape
            resolution = max(nx, ny)  # Use finer dimension for safety
            grid_spacing = domain_size / resolution
            print(f"  Rectangular grid: {nx}×{ny}, using resolution={resolution} for min_box_size")
        else:
            resolution = resolution_or_shape
            grid_spacing = domain_size / resolution
            print(f"  Square grid: {resolution}×{resolution}")
        
        if resolution is None:
            return None
        
        max_box_size = domain_size / 2  # Typical maximum
        
        # ADAPTIVE SAFETY FACTOR for low resolutions
        if resolution <= 128:
            adaptive_safety_factor = 2
            print(f"  Low resolution detected: using adaptive safety factor {adaptive_safety_factor}")
        else:
            adaptive_safety_factor = safety_factor
        
        min_box_size = adaptive_safety_factor * grid_spacing
        
        # VALIDATE SCALING RANGE
        scaling_ratio = min_box_size / max_box_size
        min_scaling_ratio = 0.005  # Need at least ~2.3 decades
        
        if scaling_ratio > min_scaling_ratio:
            adjusted_min_box_size = max_box_size * min_scaling_ratio
            print(f"  Scaling range too limited (ratio: {scaling_ratio:.4f})")
            print(f"  Adjusting min_box_size from {min_box_size:.8f} to {adjusted_min_box_size:.8f}")
            min_box_size = adjusted_min_box_size
            effective_safety_factor = min_box_size / grid_spacing
            print(f"  Effective safety factor: {effective_safety_factor:.1f}×Δx")
        else:
            effective_safety_factor = adaptive_safety_factor
        
        print(f"  Physics-based box sizing:")
        if isinstance(resolution_or_shape, (tuple, list)):
            print(f"    Resolution: {resolution_or_shape[0]}×{resolution_or_shape[1]}")
        else:
            print(f"    Resolution: {resolution}×{resolution}")
        print(f"    Grid spacing (Δx): {grid_spacing:.8f}")
        print(f"    Safety factor: {effective_safety_factor:.1f}")
        print(f"    Min box size: {min_box_size:.8f} ({effective_safety_factor:.1f}×Δx)")
        print(f"    Max box size: {max_box_size:.8f}")
        print(f"    Scaling ratio: {min_box_size/max_box_size:.6f}")
        print(f"    Expected decades: {np.log10(max_box_size/min_box_size):.2f}")
        
        return min_box_size

    def determine_optimal_min_box_size(self, vtk_file, segments, user_min_box_size=None):
        """Enhanced optimal min box size determination for rectangular grids."""
        print(f"Determining optimal min_box_size for: {os.path.basename(vtk_file)}")
        
        # User override always takes precedence
        if user_min_box_size is not None:
            print(f"  Using user-specified min_box_size: {user_min_box_size:.8f}")
            return user_min_box_size
        
        # Try physics-based approach first
        nx, ny = self.auto_detect_resolution_from_vtk_filename(vtk_file)
        
        if nx is not None and ny is not None:
            # Physics-based calculation
            if nx == ny:
                physics_min_box_size = self.calculate_physics_based_min_box_size(nx)
            else:
                physics_min_box_size = self.calculate_physics_based_min_box_size((nx, ny))
            
            # Validate against actual segment data
            if segments:
                lengths = []
                for (x1, y1), (x2, y2) in segments:
                    length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                    if length > 0:
                        lengths.append(length)
                
                if lengths:
                    min_segment = np.min(lengths)
                    median_segment = np.median(lengths)
                    
                    print(f"  Validation against interface segments:")
                    print(f"    Min segment length: {min_segment:.8f}")
                    print(f"    Median segment length: {median_segment:.8f}")
                    
                    # Ensure we're not going below reasonable limits
                    if physics_min_box_size < min_segment * 0.1:
                        adjusted = min_segment * 0.5  # More conservative
                        print(f"    → Physics size too small, adjusting to: {adjusted:.8f}")
                        return adjusted
            
            return physics_min_box_size
        
        else:
            # Fallback to robust statistical approach
            print(f"  Resolution not detected, using robust statistical approach")
            return self.calculate_robust_min_box_size(segments)

    def calculate_robust_min_box_size(self, segments, percentile=10):
        """Robust statistical approach for when resolution is unknown."""
        if not segments:
            print("    Warning: No segments for statistical analysis")
            return 0.001
        
        lengths = []
        for (x1, y1), (x2, y2) in segments:
            length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            if length > 0:
                lengths.append(length)
        
        if not lengths:
            print("    Warning: No valid segment lengths")
            return 0.001
        
        lengths = np.array(lengths)
        
        # Use percentile instead of minimum to avoid noise
        robust_length = np.percentile(lengths, percentile)
        
        # Calculate domain extent for scaling validation
        min_x = min(min(s[0][0], s[1][0]) for s in segments)
        max_x = max(max(s[0][0], s[1][0]) for s in segments)
        min_y = min(min(s[0][1], s[1][1]) for s in segments)
        max_y = max(max(s[0][1], s[1][1]) for s in segments)
        extent = max(max_x - min_x, max_y - min_y)
        
        print(f"    Statistical analysis:")
        print(f"      {percentile}th percentile length: {robust_length:.8f}")
        print(f"      Domain extent: {extent:.6f}")
        
        # Use conservative multiplier
        min_box_size = robust_length * 0.8  # Slightly smaller than percentile
        
        # Ensure sufficient scaling range (at least 2 decades)
        max_box_size = extent / 2
        if min_box_size / max_box_size > 0.01:
            adjusted = max_box_size * 0.005  # Force ~2.3 decades
            print(f"      → Adjusting for scaling range: {adjusted:.8f}")
            min_box_size = adjusted
        
        print(f"    Final robust min_box_size: {min_box_size:.8f}")
        return min_box_size

    # STEP 5
    def analyze_vtk_file_legacy(self, vtk_file, subdir, mixing_method='dalziel', h0=None, min_box_size=None):
        """
        Legacy analyze_vtk_file method for backward compatibility.
        This is the original method that performs complete analysis of a single VTK file.
        """
        print(f"🔍 Analyzing: {os.path.basename(vtk_file)}")

        # Read VTK data
        data = self.read_vtk_file(vtk_file)

        # Auto-detect resolution
        nx, ny = self.auto_detect_resolution_from_vtk_filename(vtk_file)

        # Find initial interface position
        if h0 is None:
            h0 = self.find_initial_interface(data)

        print(f"Initial interface position: h0 = {h0:.6f}")

        # Calculate mixing thickness using specified method
        mixing = self.compute_mixing_thickness(data, h0, method=mixing_method)

        # Calculate fractal dimension
        fd_results = self.compute_fractal_dimension(data, min_box_size=min_box_size)

        # Create results dictionary
        result = {
            'file': vtk_file,
            'time': data['time'],
            'h0': h0,
            'ht': mixing['ht'],
            'hb': mixing['hb'],
            'h_total': mixing['h_total'],
            'fractal_dim': fd_results['dimension'],
            'fd_error': fd_results['error'],
            'fd_r_squared': fd_results['r_squared'],
            'mixing_method': mixing_method,
            'resolution': (nx, ny) if nx and ny else None
        }

        # Add method-specific results
        if mixing_method == 'dalziel':
            if 'mixing_zone_center' in mixing:
                result['y_center'] = mixing['mixing_zone_center']
                result['mixing_fraction'] = mixing.get('mixing_fraction', 0.0)
                result['h_10'] = mixing.get('h_10')
                result['h_11'] = mixing.get('h_11')

        # Save interface data
        interface_file = os.path.join(subdir, f"interface_t{data['time']:.6f}.dat")
        self._save_interface_data_simple(interface_file, data, mixing_method, nx or 0, ny or 0)

        print(f"  Results: ht={mixing['ht']:.6f}, hb={mixing['hb']:.6f}, D={fd_results['dimension']:.6f}")

        return result

    def _save_interface_data_simple(self, interface_file, data, mixing_method, nx, ny):
        """Simple interface data saving method."""
        try:
            # Extract interface contours
            contours = self.extract_interface(data['f'], data['x'], data['y'], extract_all_levels=False)
            segments = self.convert_contours_to_segments(contours)

            with open(interface_file, 'w') as f:
                f.write(f"# Interface data for t = {data['time']:.6f}\n")
                f.write(f"# Method: {mixing_method}\n")
                f.write(f"# Grid: {nx}×{ny}\n")

                # Write segments
                for (x1, y1), (x2, y2) in segments:
                    f.write(f"{x1:.7f},{y1:.7f} {x2:.7f},{y2:.7f}\n")

                f.write(f"# Found {len(segments)} interface segments\n")

            print(f"Interface saved to {interface_file} ({len(segments)} segments)")

        except Exception as e:
            print(f"Error saving interface data: {e}")

    def process_vtk_series(self, vtk_pattern, resolution=None, mixing_method='dalziel'):
        """Process a series of VTK files matching the given pattern."""
        # Find all matching VTK files
        vtk_files = sorted(glob.glob(vtk_pattern))

        if not vtk_files:
            raise ValueError(f"No VTK files found matching pattern: {vtk_pattern}")

        print(f"Found {len(vtk_files)} VTK files matching {vtk_pattern}")
        print(f"Using mixing method: {mixing_method}")

        # Create subdirectory for this resolution if provided
        if resolution:
            subdir = f"res_{resolution}_{mixing_method}"
        else:
            subdir = f"results_{mixing_method}"

        results_dir = os.path.join(self.output_dir, subdir)
        os.makedirs(results_dir, exist_ok=True)

        # Process each file
        results = []

        for i, vtk_file in enumerate(vtk_files):
            print(f"\nProcessing file {i+1}/{len(vtk_files)}: {vtk_file}")

            try:
                # Analyze this file
                result = self.analyze_vtk_file_legacy(vtk_file, results_dir, mixing_method=mixing_method)
                results.append(result)

                # Print progress
                print(f"Completed {i+1}/{len(vtk_files)} files")

            except Exception as e:
                print(f"Error processing {vtk_file}: {str(e)}")
                import traceback
                traceback.print_exc()

        # Create summary dataframe
        if results:
            df = pd.DataFrame(results)

            # Save results
            csv_file = os.path.join(results_dir, f'results_summary_{mixing_method}.csv')
            df.to_csv(csv_file, index=False)
            print(f"Results saved to {csv_file}")

            # Create summary plots
            self.create_summary_plots(df, results_dir, mixing_method)

            return df
        else:
            print("No results to summarize")
            return None

    def create_summary_plots(self, df, output_dir, mixing_method):
        """Create summary plots of the time series results."""
        # Plot mixing layer evolution
        plt.figure(figsize=(10, 6))
        plt.plot(df['time'], df['h_total'], 'b-', label='Total', linewidth=2)
        plt.plot(df['time'], df['ht'], 'r--', label='Upper', linewidth=2)
        plt.plot(df['time'], df['hb'], 'g--', label='Lower', linewidth=2)
        plt.xlabel('Time')
        plt.ylabel('Mixing Layer Thickness')
        # Only add title if not disabled
        if not self.no_titles:
            plt.title(f'Mixing Layer Evolution ({mixing_method} method)')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, f'mixing_evolution_{mixing_method}.png'), dpi=300)
        plt.close()

        # Plot fractal dimension evolution
        plt.figure(figsize=(10, 6))
        plt.errorbar(df['time'], df['fractal_dim'], yerr=df['fd_error'],
                   fmt='ko-', capsize=3, linewidth=2, markersize=5)
        plt.fill_between(df['time'],
                   df['fractal_dim'] - df['fd_error'],
                   df['fractal_dim'] + df['fd_error'],
                   alpha=0.3, color='gray')
        plt.xlabel('Time')
        plt.ylabel('Fractal Dimension')
        # Only add title if not disabled
        if not self.no_titles:
            plt.title(f'Fractal Dimension Evolution ({mixing_method} method)')
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, f'dimension_evolution_{mixing_method}.png'), dpi=300)
        plt.close()

def main():
    """Main function to run RT analyzer from command line."""
    parser = argparse.ArgumentParser(
        description='Rayleigh-Taylor Simulation Analyzer with Fractal Dimension Calculation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze with standard method
  python rt_analyzer.py --file RT_0009000.vtk

  # Analyze with CONREC precision extraction
  python rt_analyzer.py --file RT_0009000.vtk --use-conrec

  # Analyze with PLIC theoretical reconstruction
  python rt_analyzer.py --file RT_0009000.vtk --use-plic

  # Process time series with optimization
  python rt_analyzer.py --pattern "RT_*.vtk" --mixing_method dalziel

  # Test the new optimization
  python rt_analyzer.py --file RT_0009000.vtk --test-optimization

  # Multifractal analysis
  python rt_analyzer.py --file RT_0009000.vtk --multifractal

  # Custom multifractal analysis with specific q-values
  python rt_analyzer.py --file RT_0009000.vtk --multifractal --q-values -3 -2 -1 0 1 2 3
""")

    parser.add_argument('--file', help='Single VTK file to analyze')
    parser.add_argument('--pattern', help='Pattern for VTK files (e.g., "RT_*.vtk")')
    parser.add_argument('--output_dir', default='./rt_analysis',
                   help='Output directory for results (default: ./rt_analysis)')
    parser.add_argument('--mixing_method', choices=['geometric', 'statistical', 'dalziel'],
                   default='dalziel', help='Method for computing mixing thickness')
    parser.add_argument('--h0', type=float, help='Initial interface position (auto-detected if not provided)')
    parser.add_argument('--min_box_size', type=float,
                   help='Minimum box size for fractal analysis (auto-estimated if not provided)')
    parser.add_argument('--use_grid_optimization', action='store_true',
                   help='Use grid optimization for fractal dimension calculation')
    parser.add_argument('--no_titles', action='store_true',
                   help='Disable plot titles for journal submissions')
    parser.add_argument('--use-conrec', action='store_true',
                   help='Use CONREC algorithm for precision interface extraction')
    parser.add_argument('--use-plic', action='store_true',
                   help='Use PLIC algorithm for theoretical VOF interface reconstruction')
    parser.add_argument('--test-optimization', action='store_true',
                   help='Test the new optimization methods')
    parser.add_argument('--debug', action='store_true',
                   help='Enable debug output')
    parser.add_argument('--multifractal', action='store_true',
                       help='Enable multifractal analysis')
    parser.add_argument('--q-values', nargs='+', type=float, default=None,
                       help='Q values for multifractal analysis (default: -5 to 5 in 0.5 steps)')
    parser.add_argument('--mf-output-dir', default=None,
                       help='Output directory for multifractal results (default: same as --output_dir)')
    args = parser.parse_args()

    # Validate arguments
    if not any([args.file, args.pattern]):
        print("Error: Must specify --file or --pattern")
        parser.print_help()
        return

    # Create analyzer instance
    analyzer = RTAnalyzer(
        output_dir=args.output_dir,
        use_grid_optimization=args.use_grid_optimization,
        no_titles=args.no_titles,
        use_conrec=getattr(args, 'use_conrec', False),
        use_plic=getattr(args, 'use_plic', False),
        debug=args.debug
    )

    print(f"RT Analyzer initialized")
    # Determine active extraction method for display
    if getattr(args, 'use_plic', False):
        extraction_method = "PLIC (theoretical reconstruction)"
    elif getattr(args, 'use_conrec', False):
        extraction_method = "CONREC (precision)"
    else:
        extraction_method = "scikit-image (standard)"

    print(f"Interface extraction: {extraction_method}")
    print(f"Output directory: {args.output_dir}")
    print(f"Mixing method: {args.mixing_method}")
    print(f"Grid optimization: {'ENABLED' if args.use_grid_optimization else 'DISABLED'}")

    try:
        if args.file:
            print(f"\nAnalyzing single file: {args.file}")

            if args.test_optimization:
                # Test the new optimization
                print("🚀 Testing optimization methods...")
                result = analyzer.analyze_vtk_file(args.file, 
                    analysis_types=['fractal_dim', 'curvature', 'wavelength'],
                    enable_multifractal=args.multifractal,
                    q_values=getattr(args, 'q_values', None),
                    mf_output_dir=getattr(args, 'mf_output_dir', None)
                )
                if result:
                    print(f"✅ Optimization test successful!")
                    print(f"  File: {result['file_path']}")
                    print(f"  Interface extraction time: {result['interface_extraction_time']:.3f}s")
                    print(f"  Interface points: {result['interface_point_count']}")
                    print(f"  Total analysis time: {result['total_analysis_time']:.3f}s")
                    if 'fractal_dimension' in result:
                        print(f"  Fractal dimension: {result['fractal_dimension']:.6f}")
                else:
                    print("❌ Optimization test failed")
            else:
                # Use legacy analysis
                subdir = os.path.join(args.output_dir, "single_file_analysis")
                os.makedirs(subdir, exist_ok=True)

                result = analyzer.analyze_vtk_file(
                    args.file,
                    analysis_types=['fractal_dim', 'mixing'],
                    h0=args.h0,
                    mixing_method=args.mixing_method,
                    min_box_size=args.min_box_size,
                    enable_multifractal=args.multifractal,
                    q_values=getattr(args, 'q_values', None),
                    mf_output_dir=getattr(args, 'mf_output_dir', None)
                )

                print(f"\nResults for {args.file}:")
                print(f"  Time: {result['time']:.6f}")
                print(f"  Mixing thickness: {result['h_total']:.6f}")
                print(f"  Fractal dimension: {result['fractal_dim']:.6f} ± {result['fd_error']:.6f}")
                print(f"  R²: {result['fd_r_squared']:.6f}")

        elif args.pattern:
            # Process time series
            print(f"\nProcessing time series with pattern: {args.pattern}")
            df = analyzer.process_vtk_series(
                args.pattern,
                mixing_method=args.mixing_method
            )

            if df is not None:
                print(f"\nTime series analysis complete:")
                print(f"  Processed {len(df)} files")
                print(f"  Time range: {df['time'].min():.3f} to {df['time'].max():.3f}")
                print(f"  Final mixing thickness: {df['h_total'].iloc[-1]:.6f}")
                print(f"  Final fractal dimension: {df['fractal_dim'].iloc[-1]:.6f}")

    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

    print(f"\nAnalysis complete. Results saved to: {args.output_dir}")
    return 0

if __name__ == "__main__":
    exit(main())
