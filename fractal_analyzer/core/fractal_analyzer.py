# Step 1
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
from scipy import stats
import argparse
from matplotlib.ticker import LogLocator, FuncFormatter
import time
import math
from collections import defaultdict
from numba import jit, prange
import os
import logging
from typing import Tuple, List, Dict, Optional


class FractalAnalyzer:
    """Universal fractal dimension analysis tool with optimized algorithms."""
    
    # ================ Class Constants ================
    # Physical and mathematical constants
    MARGIN_FACTOR = 0.01
    DEFAULT_SAFETY_FACTOR = 4
    MIN_WINDOW_SIZE = 3
    SPATIAL_CELL_MULTIPLIER = 2
    DEFAULT_BOX_SIZE_FACTOR = 1.5
    DEFAULT_PERCENTILE = 10
    MIN_SCALING_DECADES = 1.5
    DEFAULT_DOMAIN_SIZE = 1.0
    
    # Grid validation thresholds
    HIGH_ASPECT_RATIO_THRESHOLD = 3.0
    MIN_R_SQUARED_THRESHOLD = 0.99
    SLOPE_DEVIATION_THRESHOLD = 0.12
    
    # Performance thresholds
    LARGE_DATASET_THRESHOLD = 20000
    VERY_LARGE_DATASET_THRESHOLD = 50000
    MAX_GRID_CELLS = 1000000
    
    # Default plot settings
    DEFAULT_FIGURE_SIZE = (12, 10)
    DEFAULT_DPI = 300
    
    def __init__(self, fractal_type: Optional[str] = None, no_titles: bool = False, 
                 eps_plots: bool = False, log_level: str = 'INFO'):
        """
        Initialize the fractal analyzer.
        
        Args:
            fractal_type: Type of fractal if known (koch, sierpinski, etc.)
            no_titles: If True, disable plot titles for journal submissions
            eps_plots: If True, save plots in EPS format for publication
            log_level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
        """
        # Theoretical dimensions lookup
        self.theoretical_dimensions = {
            'koch': np.log(4) / np.log(3),      # ≈ 1.2619
            'sierpinski': np.log(3) / np.log(2), # ≈ 1.5850
            'minkowski': 1.5,                   # Exact value
            'hilbert': 2.0,                     # Space-filling
            'dragon': 1.5236                    # Approximate
        }
        
        self.fractal_type = fractal_type
        self.no_titles = no_titles
        self.eps_plots = eps_plots
        
        # Setup logging
        self._setup_logging(log_level)
        
        # Cache for expensive operations
        self._spatial_index_cache = {}
        self._segment_length_cache = {}
        
        # Configure matplotlib for EPS output if requested
        if self.eps_plots:
            self._configure_matplotlib_for_eps()
    
    def _setup_logging(self, log_level: str):
        """Setup logging configuration."""
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Reduce matplotlib logging noise
        logging.getLogger('matplotlib').setLevel(logging.WARNING)
    
    def _configure_matplotlib_for_eps(self):
        """Streamlined matplotlib configuration for publication-quality EPS."""
        import matplotlib
        matplotlib.use('Agg')
        
        # Organized configuration dictionary
        config = {
            # Font settings
            'font.size': 14,
            'font.family': 'serif',
            'font.serif': ['Times New Roman', 'Liberation Serif', 'DejaVu Serif', 'Times'],
            'text.usetex': False,
            
            # Axes and labels  
            'axes.linewidth': 1.2,
            'axes.labelsize': 16,
            'axes.titlesize': 18,
            'axes.labelweight': 'bold',
            
            # Ticks
            'xtick.labelsize': 14,
            'ytick.labelsize': 14,
            'xtick.major.width': 1.2,
            'ytick.major.width': 1.2,
            'xtick.minor.width': 0.8,
            'ytick.minor.width': 0.8,
            
            # Legend
            'legend.fontsize': 12,
            'legend.frameon': True,
            'legend.fancybox': True,
            'legend.shadow': True,
            'legend.framealpha': 0.9,
            
            # Figure settings
            'figure.figsize': list(self.DEFAULT_FIGURE_SIZE),
            'figure.dpi': self.DEFAULT_DPI,
            'savefig.dpi': self.DEFAULT_DPI,
            'savefig.format': 'eps',
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.2,
            
            # Lines and markers
            'lines.linewidth': 2.0,
            'lines.markersize': 8,
            'lines.markeredgewidth': 1.0,
            
            # Grid
            'grid.linewidth': 0.8,
            'grid.alpha': 0.6,
            
            # Math text
            'mathtext.fontset': 'stix',
            'mathtext.default': 'regular'
        }
        
        plt.rcParams.update(config)
        self.logger.info("Configured matplotlib for publication-quality EPS output")
    
    def _get_plot_extension(self) -> str:
        """Get the appropriate file extension for plots."""
        return '.eps' if self.eps_plots else '.png'
    
    def _get_plot_dpi(self) -> int:
        """Get the appropriate DPI for plots."""
        return self.DEFAULT_DPI
    
    def _save_plot(self, filename_base: str, custom_dpi: Optional[int] = None):
        """Save plot with appropriate format and settings."""
        extension = self._get_plot_extension()
        filename = filename_base + extension
        dpi = custom_dpi if custom_dpi else self._get_plot_dpi()
        
        try:
            if self.eps_plots:
                plt.savefig(filename, format='eps', dpi=dpi, bbox_inches='tight', 
                           pad_inches=0.1, facecolor='white', edgecolor='none')
                self.logger.info(f"Saved EPS plot: {filename}")
            else:
                plt.savefig(filename, dpi=dpi, bbox_inches='tight', 
                           facecolor='white', edgecolor='none')
                self.logger.info(f"Saved PNG plot: {filename}")
                
        except Exception as e:
            self.logger.error(f"Error saving plot {filename}: {str(e)}")
            try:
                plt.savefig(filename, format='eps' if self.eps_plots else 'png', dpi=150)
                self.logger.info(f"Saved simplified plot: {filename}")
            except Exception as e2:
                self.logger.error(f"Failed to save plot completely: {str(e2)}")

    # Step 2
    # ================ Utility and Validation Methods ================
    
    def _validate_segments(self, segments: List) -> bool:
        """Validate segment data quality."""
        if not segments:
            self.logger.error("No segments provided")
            return False
            
        if len(segments) < 10:
            self.logger.warning(f"Very few segments ({len(segments)}) - results may be unreliable")
            
        # Check for degenerate segments
        valid_segments = 0
        for (x1, y1), (x2, y2) in segments:
            if (x2 - x1)**2 + (y2 - y1)**2 > 1e-12:  # Non-zero length
                valid_segments += 1
                
        if valid_segments < len(segments) * 0.9:
            self.logger.warning(f"Many degenerate segments detected: {len(segments) - valid_segments}/{len(segments)}")
            
        return True
    
    def _calculate_segment_lengths(self, segments: List) -> np.ndarray:
        """Calculate all segment lengths with caching."""
        # Create cache key from segment data
        cache_key = hash(tuple(tuple(seg) for seg in segments[:100]))  # Hash first 100 for speed
        
        if cache_key in self._segment_length_cache:
            return self._segment_length_cache[cache_key]
            
        lengths = []
        for (x1, y1), (x2, y2) in segments:
            length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            if length > 0:
                lengths.append(length)
                
        result = np.array(lengths) if lengths else np.array([])
        self._segment_length_cache[cache_key] = result
        return result
    
    def _calculate_domain_extent(self, segments: List) -> float:
        """Calculate domain extent."""
        if not segments:
            return self.DEFAULT_DOMAIN_SIZE
    
        min_x = min(min(s[0][0], s[1][0]) for s in segments)
        max_x = max(max(s[0][0], s[1][0]) for s in segments)
        min_y = min(min(s[0][1], s[1][1]) for s in segments)
        max_y = max(max(s[0][1], s[1][1]) for s in segments)
    
        return max(max_x - min_x, max_y - min_y)
    
    def clean_cache(self):
        """Clean internal caches to free memory."""
        self._spatial_index_cache.clear()
        self._segment_length_cache.clear()
        self.logger.debug("Cleaned internal caches")
    
    def auto_detect_resolution_from_filename(self, filename: str) -> Tuple[Optional[int], Optional[int]]:
        """
        Enhanced resolution detection supporting both square and rectangular grids.
        
        Returns:
            tuple: (nx, ny) for grid dimensions, or (None, None) if not found
        """
        import re
        import os
        
        basename = os.path.basename(filename)
        
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
                self.logger.info(f"Auto-detected grid resolution: {nx}×{ny}")
                return nx, ny
        
        # Pattern for square RT###-*.vtk files (backward compatibility)
        square_pattern = r'RT(\d+)-'
        match = re.search(square_pattern, basename)
        if match:
            n = int(match.group(1))
            self.logger.info(f"Auto-detected square resolution: {n}×{n}")
            return n, n
        
        # Try to extract from directory path as fallback
        dir_patterns = [r'(\d+)x(\d+)', r'(\d+)_(\d+)']
        for pattern in dir_patterns:
            dir_match = re.search(pattern, filename)
            if dir_match:
                nx = int(dir_match.group(1))
                ny = int(dir_match.group(2))
                self.logger.info(f"Auto-detected resolution from path: {nx}×{ny}")
                return nx, ny
        
        self.logger.info("Could not auto-detect resolution from filename")
        return None, None
    
    def validate_and_adjust_box_sizes(self, min_box_size: float, max_box_size: float, 
                                     segments: List) -> Tuple[float, float, str]:
        """
        Validate and adjust box sizes to ensure min_box_size < max_box_size.
        
        Returns:
            Tuple of (adjusted_min_box_size, max_box_size, warning_message)
        """
        warning_msg = ""
        
        # Check if min_box_size >= max_box_size
        if min_box_size >= max_box_size:
            warning_msg = (f"WARNING: min_box_size ({min_box_size:.6f}) >= "
                          f"max_box_size ({max_box_size:.6f})")
            self.logger.warning(warning_msg)
            
            # Strategy 1: Use fraction of max_box_size
            fallback_min = max_box_size * 0.01
            
            # Strategy 2: Use smallest segment length
            if segments:
                lengths = self._calculate_segment_lengths(segments)
                if len(lengths) > 0:
                    min_segment_length = np.min(lengths[lengths > 0])
                    adjusted_min = min(fallback_min, min_segment_length)
                else:
                    adjusted_min = fallback_min
            else:
                adjusted_min = fallback_min
            
            # Ensure sufficient scaling range
            ratio = adjusted_min / max_box_size
            if ratio > 0.5:
                adjusted_min = max_box_size * 0.001
                self.logger.info(f"Further reducing min_box_size for sufficient scaling range")
            
            self.logger.info(f"Adjusted min_box_size to {adjusted_min:.6f}")
            self.logger.info(f"min/max ratio = {adjusted_min/max_box_size:.4f}")
            
            return adjusted_min, max_box_size, warning_msg
        
        # Check for limited scaling range
        ratio = min_box_size / max_box_size
        if ratio > 0.1:
            warning_msg = f"WARNING: Limited scaling range - min/max ratio = {ratio:.4f}"
            self.logger.warning(warning_msg)
            self.logger.info("Consider using smaller min_box_size for better accuracy")
        
        return min_box_size, max_box_size, warning_msg

    # Step 3
    # ================ Unified Min Box Size Estimation ================

    def estimate_min_box_size(self, segments: List, method: str = 'auto',
                             resolution: Optional[int] = None, nx: Optional[int] = None,
                             ny: Optional[int] = None, domain_size: float = None,
                             safety_factor: float = None, percentile: float = None,
                             **kwargs) -> float:
        """
        Unified minimum box size estimation with multiple methods.

        Args:
            segments: List of line segments
            method: 'physics', 'statistical', or 'auto'
            resolution: Square grid resolution
            nx, ny: Rectangular grid dimensions
            domain_size: Physical domain size
            safety_factor: Physics-based safety multiplier
            percentile: Statistical percentile to use

        Returns:
            Estimated minimum box size
        """
        # Set defaults
        domain_size = domain_size or self.DEFAULT_DOMAIN_SIZE
        safety_factor = safety_factor or self.DEFAULT_SAFETY_FACTOR
        percentile = percentile or self.DEFAULT_PERCENTILE

        self.logger.info("=== UNIFIED BOX SIZE ESTIMATION ===")

        # Auto-select method if not specified
        if method == 'auto':
            if nx is not None and ny is not None:
                method = 'physics'
            elif resolution is not None:
                method = 'physics'
            else:
                method = 'statistical'

        self.logger.info(f"Using method: {method}")

        if method == 'physics':
            return self._estimate_physics_based(segments, resolution, nx, ny,
                                              domain_size, safety_factor)
        elif method == 'statistical':
            return self._estimate_statistical(segments, percentile, **kwargs)
        else:
            raise ValueError(f"Unknown method: {method}")

    def _estimate_physics_based(self, segments: List, resolution: Optional[int],
                               nx: Optional[int], ny: Optional[int],
                               domain_size: float, safety_factor: float) -> float:
        """Physics-based estimation unified for square and rectangular grids."""
        # Determine grid configuration
        if nx is not None and ny is not None:
            effective_resolution = max(nx, ny)
            is_rectangular = (nx != ny)
            aspect_ratio = max(nx, ny) / min(nx, ny) if min(nx, ny) > 0 else 1.0
            self.logger.info(f"Grid: {nx}×{ny} ({'rectangular' if is_rectangular else 'square'})")

            # Apply additional safety for high aspect ratios
            if is_rectangular and aspect_ratio > 2.0:
                self.logger.info(f"High aspect ratio ({aspect_ratio:.2f}) - applying additional safety")
                safety_factor *= 1.2

        elif resolution is not None:
            effective_resolution = resolution
            nx = ny = resolution
            is_rectangular = False
            self.logger.info(f"Grid: {resolution}×{resolution} (square)")
        else:
            self.logger.info("No grid information - falling back to statistical method")
            return self._estimate_statistical(segments, self.DEFAULT_PERCENTILE)

        # Calculate minimum grid spacing
        min_grid_spacing = domain_size / effective_resolution
        min_box_size = safety_factor * min_grid_spacing

        self.logger.info(f"Grid spacing: {min_grid_spacing:.8f}")
        self.logger.info(f"Safety factor: {safety_factor}")
        self.logger.info(f"Initial min_box_size: {min_box_size:.8f}")

        # Validate against segment data
        if segments:
            lengths = self._calculate_segment_lengths(segments)
            if len(lengths) > 0:
                min_segment = np.min(lengths[lengths > 0])
                median_segment = np.median(lengths)

                self.logger.info(f"Min segment length: {min_segment:.8f}")
                self.logger.info(f"Median segment length: {median_segment:.8f}")

                # Ensure we're not smaller than the finest details
                if min_box_size < min_segment * 0.1:
                    adjusted = min_segment * 0.1
                    self.logger.info(f"Adjusting up to avoid sub-grid noise: {adjusted:.8f}")
                    min_box_size = adjusted

        # Check scaling range
        extent = self._calculate_domain_extent(segments)
        max_box_size = extent / 2
        scaling_decades = np.log10(max_box_size / min_box_size)

        if scaling_decades < self.MIN_SCALING_DECADES:
            self.logger.warning(f"Limited scaling range: {scaling_decades:.2f} decades")
            adjusted = max_box_size * 0.01  # Force at least 2 decades
            self.logger.info(f"Adjusting for better scaling: {adjusted:.8f}")
            min_box_size = adjusted

        self.logger.info(f"Final physics-based min_box_size: {min_box_size:.8f}")
        return min_box_size

    def _estimate_statistical(self, segments: List, percentile: float,
                             multiplier: float = 1.0, target_decades: float = 2.0,
                             min_box_sizes: int = 12) -> float:
        """Statistical estimation method."""
        if not segments:
            self.logger.warning("No segments provided for statistical estimation")
            return 0.001

        lengths = self._calculate_segment_lengths(segments)
        if len(lengths) == 0:
            self.logger.warning("No valid segment lengths")
            return 0.001

        # Use percentile instead of minimum to avoid noise
        robust_length = np.percentile(lengths, percentile)
        extent = self._calculate_domain_extent(segments)

        self.logger.info(f"Statistical estimation:")
        self.logger.info(f"  {percentile}th percentile length: {robust_length:.8f}")
        self.logger.info(f"  Domain extent: {extent:.6f}")
        self.logger.info(f"  Multiplier: {multiplier}")

        # Initial estimate
        min_box_size = robust_length * multiplier

        # Ensure sufficient scaling range
        max_box_size = extent / 2
        expected_steps = int(np.log(max_box_size/min_box_size) / np.log(self.DEFAULT_BOX_SIZE_FACTOR))
        scaling_decades = np.log10(max_box_size/min_box_size)

        self.logger.info(f"  Expected box sizes: {expected_steps}")
        self.logger.info(f"  Scaling decades: {scaling_decades:.2f}")

        # Adjust if insufficient scaling range
        if expected_steps < min_box_sizes or scaling_decades < target_decades:
            self.logger.info("Insufficient scaling range - adjusting")
            target_from_steps = max_box_size / (self.DEFAULT_BOX_SIZE_FACTOR ** min_box_sizes)
            target_from_decades = max_box_size / (10 ** target_decades)

            adjusted = min(target_from_steps, target_from_decades)
            # Don't go below minimum segment length
            min_segment_length = np.min(lengths) if len(lengths) > 0 else adjusted
            min_box_size = max(adjusted, min_segment_length * 0.1)

            final_steps = int(np.log(max_box_size/min_box_size) / np.log(self.DEFAULT_BOX_SIZE_FACTOR))
            final_decades = np.log10(max_box_size/min_box_size)
            self.logger.info(f"  Adjusted min_box_size: {min_box_size:.8f}")
            self.logger.info(f"  Final box sizes: {final_steps}")
            self.logger.info(f"  Final scaling decades: {final_decades:.2f}")

        return min_box_size

    # Step 4
    # ================ File I/O Methods ================
    
    def read_line_segments(self, filename: str) -> List:
        """Read line segments from a file."""
        start_time = time.time()
        self.logger.info(f"Reading segments from {filename}...")
        
        segments = []
        try:
            with open(filename, 'r') as file:
                for line_num, line in enumerate(file, 1):
                    if not line.strip() or line.strip().startswith('#'):
                        continue
                    
                    try:
                        coords = [float(x) for x in line.replace(',', ' ').split()]
                        if len(coords) == 4:
                            segments.append(((coords[0], coords[1]), (coords[2], coords[3])))
                        else:
                            self.logger.warning(f"Line {line_num}: Expected 4 coordinates, got {len(coords)}")
                    except ValueError as e:
                        self.logger.warning(f"Line {line_num}: Could not parse coordinates: {e}")
        except IOError as e:
            self.logger.error(f"Error reading file {filename}: {e}")
            return []
        
        elapsed = time.time() - start_time
        self.logger.info(f"Read {len(segments)} segments in {elapsed:.2f} seconds")
        
        # Validate the segments
        self._validate_segments(segments)
        return segments
    
    def write_segments_to_file(self, segments: List, filename: str):
        """Write line segments to a file."""
        start_time = time.time()
        self.logger.info(f"Writing {len(segments)} segments to file {filename}...")
        
        try:
            with open(filename, 'w') as file:
                file.write(f"# Line segments file - {len(segments)} segments\n")
                file.write(f"# Format: x1 y1 x2 y2\n")
                for (x1, y1), (x2, y2) in segments:
                    file.write(f"{x1} {y1} {x2} {y2}\n")
        except IOError as e:
            self.logger.error(f"Error writing to file {filename}: {e}")
            return
        
        elapsed = time.time() - start_time
        self.logger.info(f"File writing completed in {elapsed:.2f} seconds")
    
    # ================ Spatial Indexing and Intersection Methods ================
    
    def liang_barsky_line_box_intersection(self, x1, y1, x2, y2, xmin, ymin, xmax, ymax):
        """Determine if a line segment intersects with a box using Liang-Barsky algorithm."""
        dx = x2 - x1
        dy = y2 - y1
        
        p = [-dx, dx, -dy, dy]
        q = [x1 - xmin, xmax - x1, y1 - ymin, ymax - y1]
        
        if dx == 0 and (x1 < xmin or x1 > xmax):
            return False
        if dy == 0 and (y1 < ymin or y1 > ymax):
            return False
        
        if dx == 0 and dy == 0:
            return xmin <= x1 <= xmax and ymin <= y1 <= ymax
        
        t_min = 0.0
        t_max = 1.0
        
        for i in range(4):
            if p[i] == 0:
                if q[i] < 0:
                    return False
            else:
                t = q[i] / p[i]
                if p[i] < 0:
                    t_min = max(t_min, t)
                else:
                    t_max = min(t_max, t)
        
        return t_min <= t_max
    
    def create_spatial_index(self, segments, min_x, min_y, max_x, max_y, cell_size):
        """Create a spatial index to speed up intersection checks."""
        start_time = time.time()
        self.logger.info("Creating spatial index...")

        # Calculate grid dimensions
        grid_width = max(1, int(np.ceil((max_x - min_x) / cell_size)))
        grid_height = max(1, int(np.ceil((max_y - min_y) / cell_size)))

        self.logger.debug(f"Grid dimensions: {grid_width} x {grid_height}")
        self.logger.debug(f"Total cells: {grid_width * grid_height}")
        self.logger.debug(f"Cell size: {cell_size}")

        # Check for extremely large grids
        total_cells = grid_width * grid_height
        if total_cells > self.MAX_GRID_CELLS:
            self.logger.warning(f"Very large grid ({total_cells} cells). This may take a while...")

        # Create the spatial index
        segment_grid = defaultdict(list)

        # Add progress reporting for large datasets
        segment_count = len(segments)
        report_interval = max(1, segment_count // 10)  # Report every 10%

        for i, ((x1, y1), (x2, y2)) in enumerate(segments):
            if i % report_interval == 0 and segment_count > self.LARGE_DATASET_THRESHOLD:
                self.logger.debug(f"Progress: {i}/{segment_count} segments processed ({i*100//segment_count}%)")

            # Determine which grid cells this segment might intersect
            min_cell_x = max(0, int((min(x1, x2) - min_x) / cell_size))
            max_cell_x = min(grid_width - 1, int((max(x1, x2) - min_x) / cell_size))
            min_cell_y = max(0, int((min(y1, y2) - min_y) / cell_size))
            max_cell_y = min(grid_height - 1, int((max(y1, y2) - min_y) / cell_size))

            # Add segment to all relevant grid cells
            for cell_x in range(min_cell_x, max_cell_x + 1):
                for cell_y in range(min_cell_y, max_cell_y + 1):
                    segment_grid[(cell_x, cell_y)].append(i)

        elapsed = time.time() - start_time
        self.logger.info(f"Spatial index created in {elapsed:.2f} seconds")
        self.logger.info(f"Grid cells with segments: {len(segment_grid)}")

        return segment_grid, grid_width, grid_height
    
    def _get_or_create_spatial_index(self, segments: List, min_x: float, min_y: float, 
                                   max_x: float, max_y: float, cell_size: float):
        """Get spatial index from cache or create new one."""
        # Create cache key
        cache_key = (len(segments), min_x, min_y, max_x, max_y, cell_size)
        
        if cache_key in self._spatial_index_cache:
            self.logger.debug("Using cached spatial index")
            return self._spatial_index_cache[cache_key]
        
        # Create new spatial index
        result = self.create_spatial_index(segments, min_x, min_y, max_x, max_y, cell_size)
        
        # Cache it (but limit cache size)
        if len(self._spatial_index_cache) > 5:  # Limit cache size
            oldest_key = next(iter(self._spatial_index_cache))
            del self._spatial_index_cache[oldest_key]
        
        self._spatial_index_cache[cache_key] = result
        return result

    # Step 5
    # ================ Unified Box Counting Methods ================
    
    def box_counting_unified(self, segments: List, min_box_size: float, max_box_size: float, 
                           box_size_factor: float = None, use_grid_optimization: bool = True) -> Tuple[np.ndarray, np.ndarray, tuple]:
        """
        Unified box counting with optional grid optimization.
        
        Args:
            segments: List of line segments
            min_box_size: Minimum box size
            max_box_size: Maximum box size  
            box_size_factor: Factor by which to reduce box size
            use_grid_optimization: Whether to use grid optimization
            
        Returns:
            Tuple of (box_sizes, box_counts, bounding_box)
        """
        box_size_factor = box_size_factor or self.DEFAULT_BOX_SIZE_FACTOR
        
        total_start_time = time.time()
        self.logger.info(f"Starting unified box counting (optimization: {'ON' if use_grid_optimization else 'OFF'})")
        
        # Find bounding box and add margin
        min_x = min(min(s[0][0], s[1][0]) for s in segments)
        max_x = max(max(s[0][0], s[1][0]) for s in segments)
        min_y = min(min(s[0][1], s[1][1]) for s in segments)
        max_y = max(max(s[0][1], s[1][1]) for s in segments)
        
        margin = max(max_x - min_x, max_y - min_y) * self.MARGIN_FACTOR
        min_x -= margin; max_x += margin
        min_y -= margin; max_y += margin
        
        box_sizes, box_counts = [], []
        current_box_size = max_box_size
        
        # Create spatial index once (with caching)
        spatial_cell_size = min_box_size * self.SPATIAL_CELL_MULTIPLIER
        segment_grid, grid_width, grid_height = self._get_or_create_spatial_index(
            segments, min_x, min_y, max_x, max_y, spatial_cell_size)
        
        # Debug header
        if use_grid_optimization:
            self.logger.info("Box counting with grid optimization:")
            self.logger.info("  Box size  |  Min count |  Grid tests        | Improv | Time (s)")
            self.logger.info("-----------------------------------------------------")
        else:
            self.logger.info("Standard box counting:")
            self.logger.info("  Box size  |  Box count  |  Time (s)")
            self.logger.info("------------------------------------------")
        
        while current_box_size >= min_box_size:
            box_start_time = time.time()
            
            if use_grid_optimization:
                min_count, max_count, grid_tests = self._count_boxes_with_grid_optimization(
                    segments, current_box_size, min_x, min_y, max_x, max_y,
                    segment_grid, grid_width, grid_height, spatial_cell_size)
                
                improvement = (max_count - min_count) / max_count * 100 if max_count > 0 else 0
                elapsed = time.time() - box_start_time
                self.logger.info(f"  {current_box_size:.6f}  |  {min_count:8d}  |  {grid_tests:10d}  |  {improvement:5.1f}%  |  {elapsed:.2f}")
                actual_count = min_count
            else:
                actual_count = self._count_boxes_standard(
                    segments, current_box_size, min_x, min_y, max_x, max_y,
                    segment_grid, grid_width, grid_height, spatial_cell_size)
                
                elapsed = time.time() - box_start_time
                self.logger.info(f"  {current_box_size:.6f}  |  {actual_count:8d}  |  {elapsed:.2f}")
            
            if actual_count > 0:
                box_sizes.append(current_box_size)
                box_counts.append(actual_count)
            else:
                self.logger.warning(f"No boxes occupied at box size {current_box_size}. Skipping.")
            
            current_box_size /= box_size_factor
        
        if len(box_sizes) < 2:
            raise ValueError("Not enough valid box sizes for fractal dimension calculation.")
        
        total_time = time.time() - total_start_time
        self.logger.info(f"Total box counting time: {total_time:.2f} seconds")
        
        return np.array(box_sizes), np.array(box_counts), (min_x, min_y, max_x, max_y)
    
    def _count_boxes_with_grid_optimization(self, segments: List, box_size: float, 
                                          min_x: float, min_y: float, max_x: float, max_y: float,
                                          segment_grid: dict, grid_width: int, grid_height: int, 
                                          spatial_cell_size: float) -> Tuple[int, int, int]:
        """Count boxes with grid optimization - returns (min_count, max_count, grid_tests)."""
        # Adaptive grid density based on box size
        if box_size < 0.005:  # Very small boxes
            offset_increments = np.linspace(0, 0.75, 4)  # 4×4 = 16 tests
        elif box_size < 0.02:  # Medium boxes
            offset_increments = np.linspace(0, 0.5, 3)   # 3×3 = 9 tests
        else:  # Large boxes
            offset_increments = np.linspace(0, 0.5, 2)   # 2×2 = 4 tests
        
        min_count = float('inf')
        max_count = 0
        grid_tests = 0
        
        for dx_fraction in offset_increments:
            for dy_fraction in offset_increments:
                grid_tests += 1
                
                offset_x = min_x + dx_fraction * box_size
                offset_y = min_y + dy_fraction * box_size
                
                count = self._count_boxes_with_offset(
                    segments, box_size, offset_x, offset_y, max_x, max_y,
                    segment_grid, grid_width, grid_height, spatial_cell_size, min_x, min_y)
                
                min_count = min(min_count, count)
                max_count = max(max_count, count)
        
        return min_count, max_count, grid_tests
    
    def _count_boxes_standard(self, segments: List, box_size: float,
                            min_x: float, min_y: float, max_x: float, max_y: float,
                            segment_grid: dict, grid_width: int, grid_height: int,
                            spatial_cell_size: float) -> int:
        """Standard box counting without grid optimization."""
        return self._count_boxes_with_offset(
            segments, box_size, min_x, min_y, max_x, max_y,
            segment_grid, grid_width, grid_height, spatial_cell_size, min_x, min_y)
    
    def _count_boxes_with_offset(self, segments: List, box_size: float, offset_x: float, offset_y: float,
                               max_x: float, max_y: float, segment_grid: dict, grid_width: int, 
                               grid_height: int, spatial_cell_size: float,
                               spatial_min_x: float, spatial_min_y: float) -> int:
        """Count occupied boxes with specific grid offset."""
        num_boxes_x = int(np.ceil((max_x - offset_x) / box_size))
        num_boxes_y = int(np.ceil((max_y - offset_y) / box_size))
        
        occupied_boxes = set()
        
        for i in range(num_boxes_x):
            for j in range(num_boxes_y):
                box_xmin = offset_x + i * box_size
                box_ymin = offset_y + j * box_size
                box_xmax = box_xmin + box_size
                box_ymax = box_ymin + box_size
                
                # Find relevant grid cells for this box
                min_cell_x = max(0, int((box_xmin - spatial_min_x) / spatial_cell_size))
                max_cell_x = min(grid_width - 1, int((box_xmax - spatial_min_x) / spatial_cell_size))
                min_cell_y = max(0, int((box_ymin - spatial_min_y) / spatial_cell_size))
                max_cell_y = min(grid_height - 1, int((box_ymax - spatial_min_y) / spatial_cell_size))
                
                # Collect segments to check
                segments_to_check = set()
                for cell_x in range(min_cell_x, max_cell_x + 1):
                    for cell_y in range(min_cell_y, max_cell_y + 1):
                        segments_to_check.update(segment_grid.get((cell_x, cell_y), []))
                
                # Check for intersections
                for seg_idx in segments_to_check:
                    (x1, y1), (x2, y2) = segments[seg_idx]
                    if self.liang_barsky_line_box_intersection(x1, y1, x2, y2, 
                                                             box_xmin, box_ymin, box_xmax, box_ymax):
                        occupied_boxes.add((i, j))
                        break
        
        return len(occupied_boxes)

    # Step 6
    # ================ Fractal Dimension Calculation ================
    
    def calculate_fractal_dimension(self, box_sizes, box_counts):
        """Calculate the fractal dimension using box-counting method."""
        log_sizes = np.log(box_sizes)
        log_counts = np.log(box_counts)

        if np.any(np.isnan(log_sizes)) or np.any(np.isnan(log_counts)):
            valid = ~(np.isnan(log_sizes) | np.isnan(log_counts))
            log_sizes = log_sizes[valid]
            log_counts = log_counts[valid]
            self.logger.warning(f"Removed {np.sum(~valid)} invalid ln values")

        if len(log_sizes) < 2:
            self.logger.error("Not enough valid data points for regression!")
            return float('nan'), float('nan'), float('nan')

        slope, intercept, r_value, p_value, std_err = stats.linregress(log_sizes, log_counts)
        fractal_dimension = -slope

        self.logger.info(f"R-squared value: {r_value**2:.4f}")
        return fractal_dimension, std_err, intercept

    def trim_boundary_box_counts(self, box_sizes, box_counts, trim_count):
        """Trim specified number of box counts from each end of the data."""
        if trim_count == 0 or len(box_sizes) <= 2*trim_count:
            return box_sizes, box_counts
        return box_sizes[trim_count:-trim_count], box_counts[trim_count:-trim_count]

    def enhanced_boundary_removal(self, box_sizes, box_counts, trim_boundary=0):
        """Enhanced boundary artifact detection and removal with improved diagnostics."""
        original_length = len(box_sizes)

        # Apply manual trimming first
        if trim_boundary > 0:
            box_sizes, box_counts = self.trim_boundary_box_counts(
                box_sizes, box_counts, trim_boundary)
            self.logger.info(f"Applied manual boundary trimming: {trim_boundary} points from each end")

        # Enhanced automatic boundary artifact detection
        if len(box_sizes) > 8:  # Need enough points for meaningful analysis
            log_sizes = np.log(box_sizes)
            log_counts = np.log(box_counts)

            # Check for deviations from linearity at ends
            n = len(log_sizes)
            segment_size = max(3, n // 4)  # Use quarter segments, minimum 3 points

            if n >= 3 * segment_size:  # Ensure we have enough points
                try:
                    # Calculate slopes for first, middle, and last segments
                    slope_first, _, r2_first, _, _ = stats.linregress(
                        log_sizes[:segment_size], log_counts[:segment_size])
                    slope_middle, _, r2_middle, _, _ = stats.linregress(
                        log_sizes[segment_size:2*segment_size], log_counts[segment_size:2*segment_size])
                    slope_last, _, r2_last, _, _ = stats.linregress(
                        log_sizes[-segment_size:], log_counts[-segment_size:])

                    # Boundary detection thresholds
                    additional_trim = 0

                    # Check first segment - both slope deviation AND R² quality
                    first_slope_dev = abs(slope_first - slope_middle) / abs(slope_middle) if slope_middle != 0 else 0
                    if (first_slope_dev > self.SLOPE_DEVIATION_THRESHOLD) or (r2_first < self.MIN_R_SQUARED_THRESHOLD):
                        additional_trim = max(additional_trim, 1)
                        self.logger.info(f"Detected boundary artifact at start: slope deviation {first_slope_dev:.3f}, R² {r2_first:.3f}")

                    # Check last segment - both slope deviation AND R² quality
                    last_slope_dev = abs(slope_last - slope_middle) / abs(slope_middle) if slope_middle != 0 else 0
                    if (last_slope_dev > self.SLOPE_DEVIATION_THRESHOLD) or (r2_last < self.MIN_R_SQUARED_THRESHOLD):
                        additional_trim = max(additional_trim, 1)
                        self.logger.info(f"Detected boundary artifact at end: slope deviation {last_slope_dev:.3f}, R² {r2_last:.3f}")

                    # Apply additional trimming if artifacts detected
                    if additional_trim > 0 and len(box_sizes) > 2 * additional_trim + 4:  # Keep at least 4 points
                        self.logger.info(f"Removing {additional_trim} additional boundary points from each end")
                        box_sizes = box_sizes[additional_trim:-additional_trim]
                        box_counts = box_counts[additional_trim:-additional_trim]

                        # Verify the trimming improved the linearity
                        new_log_sizes = np.log(box_sizes)
                        new_log_counts = np.log(box_counts)
                        _, _, new_r2, _, _ = stats.linregress(new_log_sizes, new_log_counts)
                        self.logger.info(f"R² after boundary trimming: {new_r2:.4f} (was {r2_middle:.4f})")

                except Exception as e:
                    self.logger.warning(f"Could not perform enhanced boundary detection: {e}")

        final_length = len(box_sizes)
        total_removed = original_length - final_length
        if total_removed > 0:
            self.logger.info(f"Total boundary points removed: {total_removed} ({total_removed/original_length*100:.1f}%)")

        return box_sizes, box_counts
    
    # ================ Fractal Generators ================
    
    def generate_fractal(self, type_: str, level: int) -> Tuple[List, List]:
        """
        Generate points/segments based on fractal type.
        
        Args:
            type_: Fractal type (koch, sierpinski, etc.)
            level: Iteration level
            
        Returns:
            Tuple of (points, segments)
        """
        generators = {
            'koch': self._generate_koch,
            'sierpinski': self._generate_sierpinski,
            'minkowski': self._generate_minkowski,
            'hilbert': self._generate_hilbert,
            'dragon': self._generate_dragon
        }
        
        if type_ not in generators:
            raise ValueError(f"Unknown fractal type: {type_}")
        
        self.logger.info(f"Generating {type_} fractal at level {level}")
        return generators[type_](level)
    
    def _generate_koch(self, level: int) -> Tuple[List, List]:
        """Generate Koch curve at specified level."""
        
        @jit(nopython=True)
        def koch_points_jit(x1, y1, x2, y2, level, points_array, idx):
            """JIT-compiled Koch curve generator."""
            if level == 0:
                points_array[idx] = (x1, y1)
                return idx + 1
            else:
                angle = math.pi / 3
                x3 = x1 + (x2 - x1) / 3
                y3 = y1 + (y2 - y1) / 3
                x4 = (x1 + x2) / 2 + (y2 - y1) * math.sin(angle) / 3
                y4 = (y1 + y2) / 2 - (x2 - x1) * math.sin(angle) / 3
                x5 = x1 + 2 * (x2 - x1) / 3
                y5 = y1 + 2 * (y2 - y1) / 3
                
                idx = koch_points_jit(x1, y1, x3, y3, level - 1, points_array, idx)
                idx = koch_points_jit(x3, y3, x4, y4, level - 1, points_array, idx)
                idx = koch_points_jit(x4, y4, x5, y5, level - 1, points_array, idx)
                idx = koch_points_jit(x5, y5, x2, y2, level - 1, points_array, idx)
                
                return idx
        
        num_points = 4**level + 1
        points_array = np.zeros((num_points, 2), dtype=np.float64)
        
        final_idx = koch_points_jit(0, 0, 1, 0, level, points_array, 0)
        points_array[final_idx] = (1, 0)
        points = points_array[:final_idx+1]
        
        segments = []
        for i in range(len(points) - 1):
            segments.append(((points[i][0], points[i][1]), 
                            (points[i+1][0], points[i+1][1])))
        
        return points, segments

    # Step 7
    def _generate_sierpinski(self, level: int) -> Tuple[List, List]:
        """Generate Sierpinski triangle at specified level."""
        
        def generate_triangles(p1, p2, p3, level):
            """Recursively generate Sierpinski triangles."""
            if level == 0:
                # Return line segments of the triangle
                return [(p1, p2), (p2, p3), (p3, p1)]
            else:
                # Calculate midpoints
                mid1 = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
                mid2 = ((p2[0] + p3[0]) / 2, (p2[1] + p3[1]) / 2)
                mid3 = ((p3[0] + p1[0]) / 2, (p3[1] + p1[1]) / 2)
                
                # Recursively generate three smaller triangles
                segments = []
                segments.extend(generate_triangles(p1, mid1, mid3, level - 1))
                segments.extend(generate_triangles(mid1, p2, mid2, level - 1))
                segments.extend(generate_triangles(mid3, mid2, p3, level - 1))
                
                return segments
        
        # Initial equilateral triangle
        p1 = (0, 0)
        p2 = (1, 0)
        p3 = (0.5, np.sqrt(3) / 2)
        
        segments = generate_triangles(p1, p2, p3, level)
        
        # Extract unique points
        points = []
        for seg in segments:
            if seg[0] not in points:
                points.append(seg[0])
            if seg[1] not in points:
                points.append(seg[1])
        
        return points, segments
    
    def _generate_minkowski(self, level: int) -> Tuple[List, List]:
        """Generate Minkowski sausage/coastline at specified level."""
        
        def minkowski_recursive(x1, y1, x2, y2, level):
            """Recursively generate Minkowski curve."""
            if level == 0:
                return [(x1, y1), (x2, y2)]
            else:
                dx = x2 - x1
                dy = y2 - y1
                length = np.sqrt(dx**2 + dy**2)
                unit_x = dx / length
                unit_y = dy / length
                
                # Define the 8 segments of the Minkowski curve
                factor = 1 / 4  # Each segment is 1/4 of the original
                
                # Calculate points
                p0 = (x1, y1)
                p1 = (x1 + unit_x * length * factor, y1 + unit_y * length * factor)
                p2 = (p1[0] - unit_y * length * factor, p1[1] + unit_x * length * factor)
                p3 = (p2[0] + unit_x * length * factor, p2[1] + unit_y * length * factor)
                p4 = (p3[0] + unit_y * length * factor, p3[1] - unit_x * length * factor)
                p5 = (p4[0] + unit_x * length * factor, p4[1] + unit_y * length * factor)
                p6 = (p5[0] - unit_y * length * factor, p5[1] + unit_x * length * factor)
                p7 = (p6[0] + unit_x * length * factor, p6[1] + unit_y * length * factor)
                p8 = (x2, y2)
                
                # Recursively generate segments
                points = []
                points.extend(minkowski_recursive(p0[0], p0[1], p1[0], p1[1], level - 1)[:-1])
                points.extend(minkowski_recursive(p1[0], p1[1], p2[0], p2[1], level - 1)[:-1])
                points.extend(minkowski_recursive(p2[0], p2[1], p3[0], p3[1], level - 1)[:-1])
                points.extend(minkowski_recursive(p3[0], p3[1], p4[0], p4[1], level - 1)[:-1])
                points.extend(minkowski_recursive(p4[0], p4[1], p5[0], p5[1], level - 1)[:-1])
                points.extend(minkowski_recursive(p5[0], p5[1], p6[0], p6[1], level - 1)[:-1])
                points.extend(minkowski_recursive(p6[0], p6[1], p7[0], p7[1], level - 1)[:-1])
                points.extend(minkowski_recursive(p7[0], p7[1], p8[0], p8[1], level - 1))
            
                return points
        
        points = minkowski_recursive(0, 0, 1, 0, level)
        
        segments = []
        for i in range(len(points) - 1):
            segments.append(((points[i][0], points[i][1]), 
                            (points[i+1][0], points[i+1][1])))
        
        return points, segments

    def _generate_hilbert(self, level: int) -> Tuple[List, List]:
        """Generate Hilbert curve at specified level."""
    
        def hilbert_recursive(x0, y0, xi, xj, yi, yj, n, points):
            """Recursively generate Hilbert curve points."""
            if n <= 0:
                x = x0 + (xi + yi) / 2
                y = y0 + (xj + yj) / 2
                points.append((x, y))
            else:
                hilbert_recursive(x0, y0, yi / 2, yj / 2, xi / 2, xj / 2, n - 1, points)
                hilbert_recursive(x0 + xi / 2, y0 + xj / 2, xi / 2, xj / 2, yi / 2, yj / 2, n - 1, points)
                hilbert_recursive(x0 + xi / 2 + yi / 2, y0 + xj / 2 + yj / 2, xi / 2, xj / 2, yi / 2, yj / 2, n - 1, points)
                hilbert_recursive(x0 + xi / 2 + yi, y0 + xj / 2 + yj, -yi / 2, -yj / 2, -xi / 2, -xj / 2, n - 1, points)
    
        if level == 0:
            points = [(0, 0), (1, 0)]
        else:
            points = []
            hilbert_recursive(0, 0, 1, 0, 0, 1, level, points)
    
        # Normalize to unit square
        if len(points) > 1:
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
        
            scale = max(max_x - min_x, max_y - min_y)
            if scale > 0:
                points = [((p[0] - min_x) / scale, (p[1] - min_y) / scale) for p in points]
    
        # Create segments from consecutive points
        segments = []
        for i in range(len(points) - 1):
            segments.append(((points[i][0], points[i][1]), 
                            (points[i+1][0], points[i+1][1])))
    
        return points, segments

    def _generate_dragon(self, level: int) -> Tuple[List, List]:
        """Generate Dragon curve at specified level."""
        
        def dragon_sequence(n):
            """Generate the sequence for the dragon curve."""
            if n == 0:
                return [1]
        
            prev_seq = dragon_sequence(n - 1)
            new_seq = prev_seq.copy()
            new_seq.append(1)
            
            for i in range(len(prev_seq) - 1, -1, -1):
                new_seq.append(1 if prev_seq[i] == 0 else 0)
            
            return new_seq
        
        # Generate sequence
        sequence = dragon_sequence(level)
        
        # Convert to points
        points = [(0, 0)]
        x, y = 0, 0
        direction = 0  # 0: right, 1: up, 2: left, 3: down
        
        for turn in sequence:
            if turn == 1:  # Right turn
                direction = (direction + 1) % 4
            else:  # Left turn
                direction = (direction - 1) % 4
            
            # Move in current direction
            if direction == 0:
                x += 1
            elif direction == 1:
                y += 1
            elif direction == 2:
                x -= 1
            else:
                y -= 1
            
            points.append((x, y))
        
        # Normalize to unit square
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        
        scale = max(max_x - min_x, max_y - min_y)
        if scale > 0:
            points = [((p[0] - min_x) / scale, (p[1] - min_y) / scale) for p in points]
        
        segments = []
        for i in range(len(points) - 1):
            segments.append(((points[i][0], points[i][1]), 
                            (points[i+1][0], points[i+1][1])))
    
        return points, segments

    # Step 8
    # ================ Streamlined Analysis Methods ================
    
    def analyze_linear_region(self, segments: List, fractal_type: Optional[str] = None, 
                             plot_results: bool = True, plot_boxes: bool = True, 
                             trim_boundary: int = 0, box_size_factor: float = None,
                             use_grid_optimization: bool = True, return_box_data: bool = False,
                             plot_separate: bool = False, min_box_size: Optional[float] = None,
                             nx: Optional[int] = None, ny: Optional[int] = None,
                             filename_context: Optional[str] = None):
        """
        Unified linear region analysis with sliding window optimization.
        Supports both traditional fractals and RT interfaces with rectangular grids.
        
        Args:
            segments: List of line segments
            fractal_type: Type of fractal (for theoretical comparison)
            plot_results: Whether to create plots
            plot_boxes: Whether to show box overlay
            trim_boundary: Manual boundary trimming
            box_size_factor: Box size reduction factor
            use_grid_optimization: Use grid optimization
            return_box_data: Return box counting data
            plot_separate: Create separate plots
            min_box_size: Minimum box size (auto-estimated if None)
            nx, ny: Grid dimensions for rectangular grids
            filename_context: Filename/context for auto-detection
        
        Returns:
            Analysis results including optimal window and dimension
        """
        box_size_factor = box_size_factor or self.DEFAULT_BOX_SIZE_FACTOR
        
        self.logger.info("\n==== ANALYZING LINEAR REGION SELECTION ====\n")

        # Use provided type or instance type
        type_used = fractal_type or self.fractal_type

        # Get theoretical dimension if available
        if type_used in self.theoretical_dimensions:
            theoretical_dimension = self.theoretical_dimensions[type_used]
            self.logger.info(f"Theoretical {type_used} dimension: {theoretical_dimension:.6f}")
        else:
            theoretical_dimension = None
            self.logger.info("No theoretical dimension available for comparison")

        # Auto-detect grid resolution if not provided
        if nx is None and ny is None and filename_context is not None:
            nx, ny = self.auto_detect_resolution_from_filename(filename_context)
            if nx is not None:
                self.logger.info(f"Auto-detected grid: {nx}×{ny}")

        # Calculate extent to determine box sizes
        extent = self._calculate_domain_extent(segments)
        max_box_size = extent / 2

        # Auto-estimate min_box_size if not provided
        user_specified_min_box_size = min_box_size is not None
        if min_box_size is None:
            min_box_size = self.estimate_min_box_size(
                segments, method='auto', nx=nx, ny=ny, safety_factor=self.DEFAULT_SAFETY_FACTOR)
            self.logger.info(f"Auto-estimated min_box_size: {min_box_size:.8f}")
        else:
            self.logger.info(f"Using provided min_box_size: {min_box_size:.8f}")

        # Validation check - only validate auto-estimated values
        warning = ""
        if not user_specified_min_box_size:
            min_box_size, max_box_size, warning = self.validate_and_adjust_box_sizes(
                min_box_size, max_box_size, segments)
        else:
            # Basic sanity check for user input
            if min_box_size >= max_box_size:
                warning = f"WARNING: User-specified min_box_size ({min_box_size:.6f}) >= max_box_size ({max_box_size:.6f})"
                self.logger.warning(warning)

        self.logger.info(f"Final box size range: {min_box_size:.8f} to {max_box_size:.8f}")
        self.logger.info(f"Scaling ratio (min/max): {min_box_size/max_box_size:.6f}")

        # Calculate expected number of box sizes
        expected_steps = int(np.log(max_box_size/min_box_size) / np.log(box_size_factor))
        self.logger.info(f"Expected number of box sizes: {expected_steps}")

        if expected_steps < 5:
            self.logger.warning("Very few box sizes will be tested. Consider adjusting parameters.")

        # Perform box counting
        box_sizes, box_counts, bounding_box = self.box_counting_unified(
            segments, min_box_size, max_box_size, box_size_factor, use_grid_optimization)

        # Enhanced boundary handling
        box_sizes, box_counts = self.enhanced_boundary_removal(box_sizes, box_counts, trim_boundary)
        self.logger.info(f"Box counts after boundary handling: {len(box_counts)}")

        # Convert to ln scale for analysis
        log_sizes = np.log(box_sizes)
        log_counts = np.log(box_counts)

        # Perform sliding window analysis
        windows, dimensions, errors, r_squared, start_indices, end_indices = self._sliding_window_analysis(
            log_sizes, log_counts)

        # Check if we have any valid results
        if not dimensions:
            self.logger.error("No valid linear regions found!")
            self.logger.error("Suggestions: use smaller min_box_size, smaller box_size_factor, or higher level fractals")
            
            if return_box_data:
                return [], [], [], [], 0, float('nan'), box_sizes, box_counts, bounding_box
            else:
                return [], [], [], [], 0, float('nan')

        # Find optimal window
        closest_idx, optimal_window, optimal_dimension, optimal_start, optimal_end = self._find_optimal_window(
            windows, dimensions, r_squared, start_indices, end_indices, theoretical_dimension)

        # Calculate optimal intercept
        optimal_log_sizes = log_sizes[optimal_start:optimal_end]
        optimal_log_counts = log_counts[optimal_start:optimal_end]
        _, optimal_intercept, _, _, _ = stats.linregress(optimal_log_sizes, optimal_log_counts)

        # Log results
        self._log_analysis_results(windows, dimensions, r_squared, theoretical_dimension,
                                  optimal_window, optimal_dimension, optimal_start, optimal_end,
                                  box_sizes, closest_idx, errors)

        # Plot results if requested
        if plot_results:
            if plot_separate:
                self._create_separate_plots(windows, dimensions, errors, r_squared, optimal_window,
                                          optimal_dimension, theoretical_dimension, log_sizes, log_counts,
                                          optimal_start, optimal_end, segments, box_sizes, box_counts,
                                          bounding_box, optimal_intercept, plot_boxes, type_used)
            else:
                self._create_combined_plots(windows, dimensions, errors, r_squared, optimal_window,
                                          optimal_dimension, theoretical_dimension, log_sizes, log_counts,
                                          optimal_start, optimal_end, segments, box_sizes, box_counts,
                                          bounding_box, optimal_intercept, plot_boxes, type_used)

        if return_box_data:
            return windows, dimensions, errors, r_squared, optimal_window, optimal_dimension, box_sizes, box_counts, bounding_box
        else:
            return windows, dimensions, errors, r_squared, optimal_window, optimal_dimension

    def _sliding_window_analysis(self, log_sizes: np.ndarray, log_counts: np.ndarray):
        """Perform sliding window analysis to find optimal linear region."""
        min_window = self.MIN_WINDOW_SIZE
        max_window = len(log_sizes)

        windows = list(range(min_window, max_window + 1))
        dimensions = []
        errors = []
        r_squared = []
        start_indices = []
        end_indices = []

        self.logger.info("Window size | Start idx | End idx | Dimension | Error | R²")
        self.logger.info("-" * 65)

        # Try all possible window sizes
        for window_size in windows:
            best_r2 = -1
            best_dimension = None
            best_error = None
            best_start = None
            best_end = None

            # Try all possible starting points for this window size
            for start_idx in range(len(log_sizes) - window_size + 1):
                end_idx = start_idx + window_size

                # Perform regression on this window
                window_log_sizes = log_sizes[start_idx:end_idx]
                window_log_counts = log_counts[start_idx:end_idx]

                slope, _, r_value, _, std_err = stats.linregress(window_log_sizes, window_log_counts)
                dimension = -slope

                # Store if this is the best fit for this window size
                if r_value**2 > best_r2:
                    best_r2 = r_value**2
                    best_dimension = dimension
                    best_error = std_err
                    best_start = start_idx
                    best_end = end_idx

            # Store results if valid
            if best_dimension is not None:
                dimensions.append(best_dimension)
                errors.append(best_error)
                r_squared.append(best_r2)
                start_indices.append(best_start)
                end_indices.append(best_end)

                self.logger.info(f"{window_size:11d} | {best_start:9d} | {best_end:7d} | {best_dimension:9.6f} | {best_error:5.6f} | {best_r2:.6f}")

        return windows, dimensions, errors, r_squared, start_indices, end_indices

    # Step 9
    def _find_optimal_window(self, windows: List, dimensions: List, r_squared: List,
                           start_indices: List, end_indices: List, theoretical_dimension: Optional[float]):
        """Find the optimal window using physical constraints and theoretical comparison."""
        
        if theoretical_dimension is not None:
            closest_idx = np.argmin(np.abs(np.array(dimensions) - theoretical_dimension))
        else:
            # Apply physical constraints for RT interfaces
            valid_indices = []
            for i, (window, dim, r2) in enumerate(zip(windows, dimensions, r_squared)):
                # Physical constraint: 1.0 ≤ D ≤ 2.0 for 2D interfaces
                if 1.0 <= dim <= 2.0:
                    # Statistical constraint: window size ≥ 4
                    if window >= 4:
                        # Quality constraint: R² ≥ threshold
                        if r2 >= self.MIN_R_SQUARED_THRESHOLD:
                            valid_indices.append(i)

            if valid_indices:
                # Among valid windows, prefer larger windows when R² is close
                valid_data = [(r_squared[i], windows[i], i) for i in valid_indices]
                valid_data.sort(key=lambda x: (x[0], x[1]), reverse=True)
                closest_idx = valid_data[0][2]

                self.logger.info(f"Applied physical constraints:")
                self.logger.info(f"  Valid windows: {[windows[i] for i in valid_indices]}")
                self.logger.info(f"  Selected window {windows[closest_idx]} (D={dimensions[closest_idx]:.6f})")
            else:
                # Fallback: use best R²
                closest_idx = np.argmax(r_squared)
                self.logger.warning(f"No windows met physical constraints. Using best R² = {r_squared[closest_idx]:.6f}")
                self.logger.warning(f"Dimension {dimensions[closest_idx]:.6f} may be unphysical")

        optimal_window = windows[closest_idx]
        optimal_dimension = dimensions[closest_idx]
        optimal_start = start_indices[closest_idx]
        optimal_end = end_indices[closest_idx]

        return closest_idx, optimal_window, optimal_dimension, optimal_start, optimal_end

    def _log_analysis_results(self, windows: List, dimensions: List, r_squared: List,
                            theoretical_dimension: Optional[float], optimal_window: int,
                            optimal_dimension: float, optimal_start: int, optimal_end: int,
                            box_sizes: np.ndarray, closest_idx: int, errors: List):
        """Log detailed analysis results."""
        
        self.logger.info("\nDetailed window analysis:")
        self.logger.info("Window | Dimension | Theoretical Error | R² | Error Magnitude")
        self.logger.info("-------|-----------|------------------|----|-----------------")
        
        for i, (w, d, r2) in enumerate(zip(windows, dimensions, r_squared)):
            if theoretical_dimension is not None:
                error_pct = abs(d - theoretical_dimension) / theoretical_dimension * 100
                self.logger.info(f"{w:6d} | {d:9.6f} | {error_pct:15.1f}% | {r2:.6f} | {abs(d - theoretical_dimension):.6f}")

        if theoretical_dimension is not None:
            self.logger.info(f"Theoretical dimension: {theoretical_dimension:.6f}")
            self.logger.info(f"Closest dimension: {optimal_dimension:.6f} (window size: {optimal_window})")
        else:
            self.logger.info(f"Best dimension (highest R²): {optimal_dimension:.6f} (window size: {optimal_window})")
        
        self.logger.info(f"Optimal scaling region: points {optimal_start} to {optimal_end}")
        self.logger.info(f"Box size range: {box_sizes[optimal_start]:.8f} to {box_sizes[optimal_end-1]:.8f}")

    def _create_separate_plots(self, windows, dimensions, errors, r_squared, optimal_window,
                             optimal_dimension, theoretical_dimension, log_sizes, log_counts,
                             optimal_start, optimal_end, segments, box_sizes, box_counts,
                             bounding_box, optimal_intercept, plot_boxes, fractal_type):
        """Create separate plots for publication."""
        
        # 1. Sliding window analysis plot
        self._plot_window_analysis_separate(
            windows, dimensions, errors, r_squared, optimal_window, optimal_dimension,
            theoretical_dimension, fractal_type=fractal_type)

        # 2. Final log-log plot with optimal scaling region
        self._plot_loglog_with_region(
            log_sizes, log_counts, optimal_start, optimal_end, optimal_dimension,
            errors[windows.index(optimal_window)], fractal_type=fractal_type)

        # 3. Curve plot
        self.plot_results_separate(segments, box_sizes, box_counts, optimal_dimension,
                                 errors[windows.index(optimal_window)], bounding_box, optimal_intercept,
                                 plot_boxes=plot_boxes)

    def _create_combined_plots(self, windows, dimensions, errors, r_squared, optimal_window,
                             optimal_dimension, theoretical_dimension, log_sizes, log_counts,
                             optimal_start, optimal_end, segments, box_sizes, box_counts,
                             bounding_box, optimal_intercept, plot_boxes, fractal_type):
        """Create combined analysis plot."""
        
        self._plot_linear_region_analysis(
            windows, dimensions, errors, r_squared, optimal_window, optimal_dimension,
            theoretical_dimension, log_sizes, log_counts, optimal_start, optimal_end,
            fractal_type=fractal_type)

        # Also create a curve plot with optional box overlay
        self.plot_results_separate(segments, box_sizes, box_counts, optimal_dimension,
                                 errors[windows.index(optimal_window)], bounding_box, optimal_intercept,
                                 plot_boxes=plot_boxes)

    # ================ RT Interface Analysis Methods ================
    
    def analyze_rt_interface(self, segments: List, nx: Optional[int] = None, ny: Optional[int] = None, 
                           filename: Optional[str] = None, **kwargs):
        """
        Convenience method specifically for Rayleigh-Taylor interface analysis.
        
        Args:
            segments: Interface segments from RT simulation
            nx, ny: Grid dimensions (will auto-detect if None)
            filename: Filename for auto-detection
            **kwargs: Additional arguments for analyze_linear_region
        
        Returns:
            Analysis results with RT-specific context
        """
        self.logger.info("🌊 RAYLEIGH-TAYLOR INTERFACE FRACTAL ANALYSIS")
        self.logger.info("=" * 60)

        # Auto-detect grid if needed
        if nx is None and ny is None and filename is not None:
            nx, ny = self.auto_detect_resolution_from_filename(filename)

        # Set RT-appropriate defaults
        rt_defaults = {
            'fractal_type': None,  # RT interfaces don't have theoretical dimensions
            'use_grid_optimization': True,  # Always use for RT
            'box_size_factor': 1.4,  # Slightly more conservative for interfaces
            'trim_boundary': 1,  # Small boundary trim for RT
        }

        # Merge with user-provided kwargs
        for key, value in rt_defaults.items():
            if key not in kwargs:
                kwargs[key] = value

        # Log grid information
        if nx is not None and ny is not None:
            is_rectangular = nx != ny
            aspect_ratio = max(nx, ny) / min(nx, ny) if min(nx, ny) > 0 else 1.0
            self.logger.info(f"Grid: {nx}×{ny} ({'rectangular' if is_rectangular else 'square'})")
            self.logger.info(f"Aspect ratio: {aspect_ratio:.2f}")
            
            if is_rectangular and aspect_ratio > self.HIGH_ASPECT_RATIO_THRESHOLD:
                self.logger.warning(f"High aspect ratio detected - monitor for directional bias")

        # Perform enhanced analysis
        return self.analyze_linear_region(
            segments=segments,
            nx=nx,
            ny=ny,
            filename_context=filename,
            **kwargs
        )

    def validate_rectangular_grid_analysis(self, segments: List, nx: Optional[int] = None, 
                                         ny: Optional[int] = None, min_box_size: Optional[float] = None) -> Dict:
        """
        Validate analysis parameters for rectangular grids.
        
        Returns:
            Dictionary with validation results and recommendations
        """
        validation = {
            'is_valid': True,
            'warnings': [],
            'recommendations': [],
            'grid_info': {}
        }

        if nx is not None and ny is not None:
            is_rectangular = (nx != ny)
            aspect_ratio = max(nx, ny) / min(nx, ny) if min(nx, ny) > 0 else 1.0
            total_cells = nx * ny
            effective_resolution = max(nx, ny)

            validation['grid_info'] = {
                'nx': nx,
                'ny': ny,
                'is_rectangular': is_rectangular,
                'aspect_ratio': aspect_ratio,
                'total_cells': total_cells,
                'effective_resolution': effective_resolution
            }

            self.logger.info(f"Grid validation for {nx}×{ny}:")
            self.logger.info(f"  Rectangular: {is_rectangular}")
            self.logger.info(f"  Aspect ratio: {aspect_ratio:.2f}")
            self.logger.info(f"  Total cells: {total_cells}")

            # Rectangular grid specific warnings
            if is_rectangular:
                if aspect_ratio > self.HIGH_ASPECT_RATIO_THRESHOLD:
                    warning = f"High aspect ratio ({aspect_ratio:.2f}) may affect fractal analysis"
                    validation['warnings'].append(warning)
                    validation['recommendations'].append(
                        "Consider effects of grid anisotropy on fractal measurements")

                if aspect_ratio > 1.5:
                    validation['recommendations'].append(
                        "Monitor for directional bias in fractal dimension calculations")

            # Resolution adequacy check
            if effective_resolution < 100:
                warning = f"Low effective resolution ({effective_resolution}) for fractal analysis"
                validation['warnings'].append(warning)
                validation['recommendations'].append(
                    "Consider using higher resolution data for more accurate results")

        # Segment-based validation
        if segments:
            segment_count = len(segments)
            validation['grid_info']['segment_count'] = segment_count

            if segment_count < 100:
                warning = f"Low segment count ({segment_count}) may limit fractal accuracy"
                validation['warnings'].append(warning)

        # Min box size validation for rectangular grids
        if min_box_size is not None and nx is not None and ny is not None:
            min_grid_spacing = self.DEFAULT_DOMAIN_SIZE / max(nx, ny)
            if min_box_size < min_grid_spacing * 2:
                warning = f"Min box size ({min_box_size:.6f}) very close to grid spacing ({min_grid_spacing:.6f})"
                validation['warnings'].append(warning)
                validation['recommendations'].append(
                    "Consider larger min_box_size to avoid grid artifacts")

        if validation['warnings']:
            validation['is_valid'] = False
            self.logger.warning(f"Validation completed with {len(validation['warnings'])} warnings")
        else:
            self.logger.info(f"Validation passed - grid configuration suitable for fractal analysis")

        return validation

    # Step 10
    # ================ Iteration Analysis Methods ================

    def analyze_iterations(self, min_level: int = 1, max_level: int = 8, 
                          fractal_type: Optional[str] = None, box_ratio: float = 0.3,
                          no_plots: bool = False, no_box_plot: bool = False, 
                          box_size_factor: float = None, use_grid_optimization: bool = True,
                          min_box_size: Optional[float] = None, base_level: Optional[int] = None,
                          base_level_results: Optional[tuple] = None):
        """
        Enhanced iteration analysis that substitutes existing results into the iteration array.
        
        Args:
            base_level: Level of the pre-computed curve 
            base_level_results: Pre-computed linear region results to substitute
        """
        box_size_factor = box_size_factor or self.DEFAULT_BOX_SIZE_FACTOR
        
        self.logger.info("\n==== ITERATION ANALYSIS WITH RESULT SUBSTITUTION ====\n")

        type_used = fractal_type or self.fractal_type
        if type_used is None:
            raise ValueError("Fractal type must be specified")

        theoretical_dimension = self.theoretical_dimensions[type_used]
        self.logger.info(f"Theoretical {type_used} dimension: {theoretical_dimension:.6f}")

        # Initialize results storage
        successful_levels = []
        dimensions = []
        errors = []
        r_squared = []

        # Extract base level results for substitution
        base_dimension = None
        base_error = None
        base_r2 = None
        
        if (base_level is not None and base_level_results is not None and 
            min_level <= base_level <= max_level):
            
            # Unpack the linear region results
            windows, dims, errs, r2s, optimal_window, optimal_dimension, box_sizes, box_counts, bounding_box = base_level_results
            
            # Extract the optimal results
            base_dimension = optimal_dimension
            base_error = errs[np.where(np.array(windows) == optimal_window)[0][0]]
            base_r2 = r2s[np.where(np.array(windows) == optimal_window)[0][0]]
            
            self.logger.info(f"Will substitute level {base_level} results: D={base_dimension:.6f} ± {base_error:.6f}")

        # Process each level in the iteration range
        for level in range(min_level, max_level + 1):
            
            # SUBSTITUTE: Use pre-computed results for base level
            if level == base_level and base_dimension is not None:
                self.logger.info(f"\n--- SUBSTITUTING results for {type_used} level {level} ---")
                
                # Directly insert the pre-computed results
                successful_levels.append(level)
                dimensions.append(base_dimension)
                errors.append(base_error)
                r_squared.append(base_r2)
                
                self.logger.info(f"Level {level} - Substituted Dimension: {base_dimension:.6f} ± {base_error:.6f}")
                self.logger.info(f"Difference from theoretical: {abs(base_dimension - theoretical_dimension):.6f}")
                self.logger.info(f"R-squared: {base_r2:.6f}")

                # Plot using the base level data if requested
                if not no_plots:
                    curve_file = f"{type_used}_level_{level}_curve_substituted"
                    dimension_file = f"{type_used}_level_{level}_dimension_substituted"
                    self.logger.info(f"Plotting substituted results for level {level}")
                    
                    # Calculate intercept for plotting
                    log_sizes = np.log(box_sizes)
                    log_counts = np.log(box_counts)
                    _, calculated_intercept, _, _, _ = stats.linregress(log_sizes, log_counts)
                    
                    # Note: We can't plot the fractal curve since we don't have segments
                    # Just plot the dimension analysis
                    self._plot_loglog(box_sizes, box_counts, base_dimension, base_error,
                                      intercept=calculated_intercept, custom_filename=dimension_file)
                
                continue

            # COMPUTE: Generate and analyze new levels normally
            self.logger.info(f"\n--- Computing {type_used} curve at level {level} ---")

            try:
                # Generate the curve
                _, segments = self.generate_fractal(type_used, level)

                # Level-specific min box size estimation
                if min_box_size is None:
                    level_min_box_size = self.estimate_min_box_size(
                        segments, method='statistical', percentile=5, multiplier=1.0)
                else:
                    level_scaling_factor = 0.5 ** (level - min_level)
                    level_min_box_size = min_box_size * level_scaling_factor

                # Calculate dimension using unified analysis
                results = self.analyze_linear_region(
                    segments, fractal_type=type_used, plot_results=False,
                    box_size_factor=box_size_factor, return_box_data=True,
                    use_grid_optimization=use_grid_optimization,
                    min_box_size=level_min_box_size)

                # Unpack and validate results
                windows, dims, errs, r2s, optimal_window, optimal_dimension, box_sizes, box_counts, bounding_box = results

                if np.isnan(optimal_dimension):
                    self.logger.warning(f"Level {level} - Analysis failed: insufficient data")
                    continue

                error = errs[np.where(np.array(windows) == optimal_window)[0][0]]
                r_value_squared = r2s[np.where(np.array(windows) == optimal_window)[0][0]]

                # Store computed results
                successful_levels.append(level)
                dimensions.append(optimal_dimension)
                errors.append(error)
                r_squared.append(r_value_squared)

                self.logger.info(f"Level {level} - Computed Dimension: {optimal_dimension:.6f} ± {error:.6f}")
                self.logger.info(f"Difference from theoretical: {abs(optimal_dimension - theoretical_dimension):.6f}")

                # Plot results if requested
                if not no_plots:
                    curve_file = f"{type_used}_level_{level}_curve"
                    dimension_file = f"{type_used}_level_{level}_dimension"

                    plot_boxes = (level <= 6) and not no_box_plot
                    
                    # Calculate intercept
                    log_sizes = np.log(box_sizes)
                    log_counts = np.log(box_counts)
                    _, optimal_intercept, _, _, _ = stats.linregress(log_sizes, log_counts)

                    self.plot_results_separate(segments, box_sizes, box_counts, optimal_dimension,
                                             error, bounding_box, optimal_intercept, plot_boxes=plot_boxes,
                                             level=level, custom_filename=curve_file)

                    self._plot_loglog(box_sizes, box_counts, optimal_dimension, error,
                                      intercept=optimal_intercept, custom_filename=dimension_file)

            except Exception as e:
                self.logger.error(f"Level {level} - Error during analysis: {str(e)}")
                continue

        # Results summary
        if not successful_levels:
            self.logger.error("No levels could be successfully analyzed!")
            return [], [], [], []

        self.logger.info(f"\nIteration analysis complete:")
        self.logger.info(f"  Total levels: {len(successful_levels)}")
        self.logger.info(f"  Computed: {len([l for l in successful_levels if l != base_level])}")
        if base_level in successful_levels:
            self.logger.info(f"  Substituted: 1 (level {base_level})")
        self.logger.info(f"  Successful levels: {successful_levels}")

        # Summary statistics
        if len(dimensions) > 1:
            mean_dim = np.mean(dimensions)
            std_dim = np.std(dimensions)
            self.logger.info(f"Dimension statistics:")
            self.logger.info(f"  Mean: {mean_dim:.6f}")
            self.logger.info(f"  Std Dev: {std_dim:.6f}")
            self.logger.info(f"  Range: {min(dimensions):.6f} to {max(dimensions):.6f}")

        # Plot dimension vs level with mixed data
        if not no_plots:
            self._plot_dimension_vs_level(successful_levels, dimensions, errors, r_squared,
                                         theoretical_dimension, type_used)

        return successful_levels, dimensions, errors, r_squared

    # ================ Memory Management and Cleanup ================
    
    def clean_memory(self):
        """Force garbage collection and clean matplotlib figures."""
        import gc
        import matplotlib.pyplot as plt
        
        # Clean internal caches
        self.clean_cache()
        
        # Close all matplotlib figures
        plt.close('all')
        
        # Force garbage collection
        gc.collect()
        
        self.logger.debug("Performed memory cleanup")

    def optimize_for_large_dataset(self, segment_count: int):
        """Optimize settings for large datasets."""
        import matplotlib
        
        if segment_count > self.VERY_LARGE_DATASET_THRESHOLD:
            matplotlib.rcParams['agg.path.chunksize'] = 100000
            self.logger.info(f"Optimized for very large dataset ({segment_count} segments)")
        elif segment_count > self.LARGE_DATASET_THRESHOLD:
            matplotlib.rcParams['agg.path.chunksize'] = 50000
            self.logger.info(f"Optimized for large dataset ({segment_count} segments)")
        else:
            matplotlib.rcParams['agg.path.chunksize'] = 25000

        # Set path simplification
        matplotlib.rcParams['path.simplify_threshold'] = 0.1

    # Step 11
    # ================ Core Plotting Functions ================
    
    def plot_results_separate(self, segments: List, box_sizes: np.ndarray, box_counts: np.ndarray, 
                             fractal_dimension: float, error: float, bounding_box: tuple, 
                             intercept: Optional[float] = None, plot_boxes: bool = False, 
                             level: Optional[int] = None, custom_filename: Optional[str] = None):
        """Creates enhanced plots with better readability for publication."""
        
        segment_count = len(segments)
        
        # Determine if this is a file-based curve
        is_file_based = self.fractal_type is None or (custom_filename and not level)
        
        # Optimize for large datasets
        self.optimize_for_large_dataset(segment_count)
        use_rasterized = segment_count > self.LARGE_DATASET_THRESHOLD
        
        plt.figure(figsize=self.DEFAULT_FIGURE_SIZE)
        start_time = time.time()
        
        self.logger.info("Plotting curve segments...")

        # Handle very large datasets with sampling
        if segment_count > self.VERY_LARGE_DATASET_THRESHOLD:
            self.logger.info(f"Very large dataset ({segment_count} segments), using sampling...")
            step = max(1, segment_count // 15000)
            sampled_segments = segments[::step]
            self.logger.info(f"Sampled down to {len(sampled_segments)} segments for visualization")
            segments_to_plot = sampled_segments
        else:
            segments_to_plot = segments

        # Plot segments efficiently
        x_points = []
        y_points = []
        for (x1, y1), (x2, y2) in segments_to_plot:
            x_points.extend([x1, x2, None])
            y_points.extend([y1, y2, None])

        # Remove final None
        if x_points:
            x_points = x_points[:-1]
            y_points = y_points[:-1]

        plt.plot(x_points, y_points, 'k-', linewidth=1.8, rasterized=use_rasterized)

        elapsed = time.time() - start_time
        self.logger.info(f"Curve plotting completed in {elapsed:.2f} seconds")

        # Enhanced axis formatting
        ax = plt.gca()
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)

        # Set plot limits with margin
        min_x, min_y, max_x, max_y = bounding_box
        view_margin = max(max_x - min_x, max_y - min_y) * 0.05
        plt.xlim(min_x - view_margin, max_x + view_margin)
        plt.ylim(min_y - view_margin, max_y + view_margin)

        # Add box overlay if requested
        if plot_boxes:
            self._plot_box_overlay_enhanced(segments, box_sizes, box_counts, bounding_box)

        # Enhanced titles and labels
        if not self.no_titles:
            title = f'{self.fractal_type.capitalize() if self.fractal_type else "Fractal"} Curve'
            if level is not None:
                title += f' (Level {level})'
            if plot_boxes:
                smallest_idx = len(box_sizes) - 1
                box_size = box_sizes[smallest_idx]
                title += f'\nwith Box Counting Overlay (Box Size: {box_size:.6f})'
            plt.title(title, fontsize=18, fontweight='bold', pad=15)

        plt.xlabel('X', fontsize=16, fontweight='bold', labelpad=10)
        plt.ylabel('Y', fontsize=16, fontweight='bold', labelpad=10)
        plt.grid(True, linestyle='--', alpha=0.6, linewidth=0.8)

        # Enhanced tick formatting
        plt.tick_params(axis='both', which='major', labelsize=14, width=1.2, length=6)
        plt.tick_params(axis='both', which='minor', width=0.8, length=4)

        # Determine filename
        if custom_filename:
            curve_filename = custom_filename
        else:
            curve_filename = f'{self.fractal_type if self.fractal_type else "fractal"}_curve'
            if plot_boxes:
                curve_filename += '_with_boxes'
            if level is not None:
                curve_filename += f'_Level_{level}'

        # Save with error handling
        try:
            self._save_plot(curve_filename)
            self.logger.info(f"Successfully saved plot to {curve_filename}{self._get_plot_extension()}")
        except Exception as e:
            self.logger.error(f"Error saving {curve_filename}: {str(e)}")

        plt.close()

    def _plot_box_overlay_enhanced(self, segments: List, box_sizes: np.ndarray, 
                                  box_counts: np.ndarray, bounding_box: tuple):
        """Enhanced box overlay with better visibility."""
        box_time = time.time()
        self.logger.info("Generating enhanced box overlay...")

        # Choose the smallest box size to visualize
        smallest_idx = len(box_sizes) - 1
        box_size = box_sizes[smallest_idx]
        expected_count = box_counts[smallest_idx]

        self.logger.info(f"Box size: {box_size}, Expected count: {expected_count}")

        min_x, min_y, max_x, max_y = bounding_box

        # Calculate box coordinates
        num_boxes_x = int(np.ceil((max_x - min_x) / box_size))
        num_boxes_y = int(np.ceil((max_y - min_y) / box_size))

        # Create spatial index for efficient intersection tests
        grid_size = box_size * self.SPATIAL_CELL_MULTIPLIER
        segment_grid, _, _ = self.create_spatial_index(
            segments, min_x, min_y, max_x, max_y, grid_size)

        # Collect occupied boxes
        rectangles = []

        for i in range(num_boxes_x):
            for j in range(num_boxes_y):
                box_xmin = min_x + i * box_size
                box_ymin = min_y + j * box_size
                box_xmax = box_xmin + box_size
                box_ymax = box_ymin + box_size

                # Find relevant grid cells
                cell_x = int((box_xmin - min_x) / grid_size)
                cell_y = int((box_ymin - min_y) / grid_size)

                # Get segments that might intersect this box
                segments_to_check = set()
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        adjacent_key = (cell_x + dx, cell_y + dy)
                        segments_to_check.update(segment_grid.get(adjacent_key, []))

                # Check for intersection
                for seg_idx in segments_to_check:
                    (x1, y1), (x2, y2) = segments[seg_idx]
                    if self.liang_barsky_line_box_intersection(x1, y1, x2, y2, 
                                                             box_xmin, box_ymin, box_xmax, box_ymax):
                        rectangles.append(Rectangle((box_xmin, box_ymin), box_size, box_size))
                        break

        elapsed = time.time() - box_time
        self.logger.info(f"Box intersection tests completed in {elapsed:.2f} seconds")

        # Enhanced visualization with thicker, more visible edges
        pc = PatchCollection(rectangles, facecolor='none', edgecolor='red', 
                           linewidth=1.2, alpha=0.9)
        plt.gca().add_collection(pc)

        self.logger.info(f"Total boxes drawn: {len(rectangles)} with enhanced visibility")

    def _plot_loglog(self, box_sizes: np.ndarray, box_counts: np.ndarray, 
                    fractal_dimension: float, error: float, intercept: Optional[float] = None,
                    custom_filename: Optional[str] = None):
        """Plot ln-ln analysis with consistent regression line and enhanced readability."""
        plt.figure(figsize=(12, 9))

        # Plot data points with enhanced visibility
        plt.loglog(box_sizes, box_counts, 'bo-', markersize=10, linewidth=2.5, 
                   label='Data points', markerfacecolor='blue', markeredgecolor='darkblue', 
                   markeredgewidth=1)

        # Calculate or use provided intercept
        log_sizes = np.log(box_sizes)
        log_counts = np.log(box_counts)

        if intercept is None:
            slope, intercept, r_value, _, std_err = stats.linregress(log_sizes, log_counts)
            calculated_dimension = -slope
            if abs(calculated_dimension - fractal_dimension) > 0.001:
                self.logger.warning(f"Dimension mismatch in plot: {calculated_dimension:.6f} vs {fractal_dimension:.6f}")
                fractal_dimension = calculated_dimension
                error = std_err
        else:
            slope = -fractal_dimension

        # Plot consistent regression line
        fit_counts = np.exp(intercept + slope * log_sizes)
        plt.loglog(box_sizes, fit_counts, 'r-', linewidth=3.5,
                   label=f'Fit: D = {fractal_dimension:.4f} ± {error:.4f}')

        # Enhanced scientific formatter
        def scientific_formatter(x, pos):
            if x == 0:
                return '0'
            exponent = int(np.log10(x))
            coef = x / 10**exponent
            if abs(coef - 1.0) < 0.01:
                return r'$10^{%d}$' % exponent
            elif abs(coef - 3.0) < 0.01:
                return r'$3{\times}10^{%d}$' % exponent
            else:
                return r'${%.1f}{\times}10^{%d}$' % (coef, exponent)

        # Enhanced axis formatting
        ax = plt.gca()
        ax.xaxis.set_major_formatter(FuncFormatter(scientific_formatter))
        ax.yaxis.set_major_formatter(FuncFormatter(scientific_formatter))
        
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)

        if not self.no_titles:
            plt.title('Box Counting: ln(N) vs ln(1/r)', fontsize=20, fontweight='bold', pad=20)
        
        plt.xlabel('Box Size (r)', fontsize=18, fontweight='bold', labelpad=10)
        plt.ylabel('Number of Boxes (N)', fontsize=18, fontweight='bold', labelpad=10)
        
        plt.legend(fontsize=14, loc='best', frameon=True, fancybox=True, shadow=True)
        plt.grid(True, which='major', linestyle='-', linewidth=1.0, alpha=0.7)
        plt.grid(True, which='minor', linestyle=':', linewidth=0.5, alpha=0.5)

        plt.tick_params(axis='both', which='major', labelsize=14, width=1.2, length=6)
        plt.tick_params(axis='both', which='minor', width=0.8, length=4)

        # Determine filename
        if custom_filename:
            filename = custom_filename
        else:
            filename = 'box_counting_loglog'
            if self.fractal_type:
                filename = f'{self.fractal_type}_box_counting_loglog'

        self._save_plot(filename)
        plt.close()

    # Step 12
    def _plot_linear_region_analysis(self, windows: List, dimensions: List, errors: List, 
                                    r_squared: List, optimal_window: int, optimal_dimension: float,
                                    theoretical_dimension: Optional[float], log_sizes: np.ndarray, 
                                    log_counts: np.ndarray, optimal_start: int, optimal_end: int,
                                    fractal_type: Optional[str] = None):
        """Plot linear region analysis results."""
        plt.figure(figsize=(15, 12))

        # Plot 1: Dimension vs window size
        plt.subplot(3, 1, 1)
        plt.errorbar(windows, dimensions, yerr=errors, fmt='o-', capsize=5, markersize=4)
        
        if theoretical_dimension is not None:
            plt.axhline(y=theoretical_dimension, color='r', linestyle='--',
                        label=f'Theoretical Dimension ({theoretical_dimension:.6f})')
        
        plt.scatter([optimal_window], [optimal_dimension], color='red', s=100, zorder=5,
                   label=f'Optimal Window (size={optimal_window})')

        plt.xlabel('Window Size (number of points)')
        plt.ylabel('Calculated Fractal Dimension')
        if not self.no_titles:
            plt.title('Fractal Dimension vs. Window Size')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()

        # Plot 2: R² vs window size
        plt.subplot(3, 1, 2)
        plt.plot(windows, r_squared, 'go-', markersize=6)
        optimal_r2 = r_squared[windows.index(optimal_window)]
        plt.scatter([optimal_window], [optimal_r2],
                   color='red', s=100, zorder=5, label=f'Optimal Window (R²={optimal_r2:.6f})')
        plt.axhline(y=self.MIN_R_SQUARED_THRESHOLD, color='orange', linestyle='--', 
                    label=f'R² = {self.MIN_R_SQUARED_THRESHOLD}')

        plt.xlabel('Window Size (number of points)')
        plt.ylabel('R² Value')
        if not self.no_titles:
            plt.title('R² vs. Window Size')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.ylim(0.95, 1.001)
        plt.legend()

        # Plot 3: ln-ln plot with optimal window highlighted
        plt.subplot(3, 1, 3)
        plt.scatter(log_sizes, log_counts, label='All Data Points')
        plt.scatter(log_sizes[optimal_start:optimal_end], log_counts[optimal_start:optimal_end],
                   color='red', label=f'Optimal Window (D={optimal_dimension:.6f})')

        # Add regression line for optimal window
        opt_slope, opt_intercept, _, _, _ = stats.linregress(
            log_sizes[optimal_start:optimal_end], log_counts[optimal_start:optimal_end])
        opt_line = opt_intercept + opt_slope * log_sizes
        plt.plot(log_sizes, opt_line, 'r--', label=f'Optimal Fit (slope={-opt_slope:.6f})')

        plt.xlabel('ln(Box Size)')
        plt.ylabel('ln(Box Count)')
        if not self.no_titles:
            plt.title('ln-ln Plot with Optimal Scaling Region')

        # Adjust axis limits
        x_margin = (log_sizes.max() - log_sizes.min()) * 0.05
        y_margin = (log_counts.max() - log_counts.min()) * 0.05
        plt.xlim(log_sizes.min() - x_margin, log_sizes.max() + x_margin)
        plt.ylim(log_counts.min() - y_margin, log_counts.max() + y_margin)

        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()

        plt.tight_layout()

        # Generate filename
        filename = 'linear_region_analysis'
        if fractal_type:
            filename = f'{fractal_type}_linear_region_analysis'
        
        self._save_plot(filename)
        plt.close()

    def _plot_window_analysis_separate(self, windows: List, dimensions: List, errors: List, 
                                     r_squared: List, optimal_window: int, optimal_dimension: float,
                                     theoretical_dimension: Optional[float], fractal_type: Optional[str] = None):
        """Plot sliding window analysis as separate publication-quality figure."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        # Top panel: Dimension vs window size
        ax1.errorbar(windows, dimensions, yerr=errors, fmt='o-', capsize=5, markersize=4,
                     color='blue', label='Calculated Dimension')

        if theoretical_dimension is not None:
            ax1.axhline(y=theoretical_dimension, color='red', linestyle='--', linewidth=2,
                       label=f'Theoretical D = {theoretical_dimension:.4f}')

        # Enhanced label with best dimension info
        optimal_error = errors[windows.index(optimal_window)]
        optimal_label = f'Optimal: D = {optimal_dimension:.4f} ± {optimal_error:.4f} (size={optimal_window})'
        ax1.scatter([optimal_window], [optimal_dimension], color='red', s=100, zorder=5,
                   label=optimal_label)

        ax1.set_ylabel('Fractal Dimension')
        if not self.no_titles:
            ax1.set_title('Sliding Window Optimization Analysis')
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.legend()

        # Bottom panel: R² vs window size
        optimal_r2 = r_squared[windows.index(optimal_window)]
        ax2.plot(windows, r_squared, 'go-', markersize=6, label='R² Value')
        ax2.scatter([optimal_window], [optimal_r2],
                   color='red', s=100, zorder=5,
                   label=f'Optimal R² = {optimal_r2:.4f}')
        ax2.axhline(y=self.MIN_R_SQUARED_THRESHOLD, color='orange', linestyle='--', alpha=0.7, 
                    label=f'R² = {self.MIN_R_SQUARED_THRESHOLD}')

        # Add difference from theoretical
        if theoretical_dimension is not None:
            difference = abs(optimal_dimension - theoretical_dimension)
            ax2.text(0.02, 0.95, f'Error from theoretical: {difference:.4f}',
                    transform=ax2.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))

        ax2.set_xlabel('Window Size (number of points)')
        ax2.set_ylabel('R² Value')
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.set_ylim(0.95, 1.001)
        ax2.legend()

        plt.tight_layout()

        # Generate filename
        filename = 'sliding_window_analysis'
        if fractal_type:
            filename = f'{fractal_type}_sliding_window_analysis'

        self._save_plot(filename)
        plt.close()
        self.logger.info(f"Saved sliding window analysis to {filename}{self._get_plot_extension()}")

    def _plot_loglog_with_region(self, log_sizes: np.ndarray, log_counts: np.ndarray, 
                               optimal_start: int, optimal_end: int, optimal_dimension: float,
                               error: float, fractal_type: Optional[str] = None):
        """Plot log-log analysis with optimal scaling region highlighted."""
        plt.figure(figsize=(10, 8))

        # Convert back to regular sizes for plotting
        box_sizes = np.exp(log_sizes)
        box_counts = np.exp(log_counts)

        # Plot all data points
        plt.loglog(box_sizes, box_counts, 'bo', markersize=6, alpha=0.7, label='All Data Points')

        # Highlight optimal window
        plt.loglog(box_sizes[optimal_start:optimal_end], box_counts[optimal_start:optimal_end],
                   'ro', markersize=8, label=f'Optimal Window (D={optimal_dimension:.4f})')

        # Plot regression line for optimal window
        opt_log_sizes = log_sizes[optimal_start:optimal_end]
        opt_log_counts = log_counts[optimal_start:optimal_end]
        slope, intercept, _, _, _ = stats.linregress(opt_log_sizes, opt_log_counts)

        # Create smooth line for the fit
        fit_log_sizes = np.linspace(opt_log_sizes.min(), opt_log_sizes.max(), 100)
        fit_log_counts = intercept + slope * fit_log_sizes
        fit_sizes = np.exp(fit_log_sizes)
        fit_counts = np.exp(fit_log_counts)

        plt.loglog(fit_sizes, fit_counts, 'r-', linewidth=2,
                   label=f'Optimal Fit (slope={-slope:.4f})')

        # Custom formatter for scientific notation
        def scientific_formatter(x, pos):
            if x == 0:
                return '0'
            exponent = int(np.log10(x))
            coef = x / 10**exponent
            if abs(coef - 1.0) < 0.01:
                return r'$10^{%d}$' % exponent
            else:
                return r'${%.1f}{\times}10^{%d}$' % (coef, exponent)

        # Set axis properties
        ax = plt.gca()
        ax.xaxis.set_major_formatter(FuncFormatter(scientific_formatter))
        ax.yaxis.set_major_formatter(FuncFormatter(scientific_formatter))

        plt.xlabel('Box Size (r)')
        plt.ylabel('Number of Boxes (N)')
        if not self.no_titles:
            plt.title('Box Counting: ln(N) vs ln(1/r)')
        plt.legend()
        plt.grid(True, which='major', linestyle='-', linewidth=0.5, alpha=0.7)
        plt.grid(True, which='minor', linestyle=':', linewidth=0.3, alpha=0.5)

        # Generate filename
        filename = 'box_counting_with_optimal_region'
        if fractal_type:
            filename = f'{fractal_type}_box_counting_with_optimal_region'

        self._save_plot(filename)
        plt.close()
        self.logger.info(f"Saved log-log plot with optimal region to {filename}{self._get_plot_extension()}")

    def _plot_dimension_vs_level(self, levels: List, dimensions: List, errors: List, 
                               r_squared: List, theoretical_dimension: float, fractal_type: str):
        """Plot dimension vs. iteration level."""
        plt.figure(figsize=(10, 6))
        
        plt.errorbar(levels, dimensions, yerr=errors, fmt='o-', capsize=5,
                     label='Calculated Dimension')
        plt.axhline(y=theoretical_dimension, color='r', linestyle='--',
                    label=f'Theoretical Dimension ({theoretical_dimension:.6f})')

        plt.xlabel(f'{fractal_type.capitalize()} Curve Iteration Level')
        plt.ylabel('Fractal Dimension')
        if not self.no_titles:
            plt.title(f'Fractal Dimension vs. {fractal_type.capitalize()} Curve Iteration Level')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()

        # Add a second y-axis for R-squared values
        ax2 = plt.gca().twinx()
        ax2.plot(levels, r_squared, 'g--', marker='s', label='R-squared')
        ax2.set_ylabel('R-squared', color='g')
        ax2.tick_params(axis='y', labelcolor='g')
        ax2.set_ylim([0.9, 1.01])
        ax2.legend(loc='center right')

        plt.tight_layout()
        filename = f'{fractal_type}_dimension_vs_level'
        self._save_plot(filename)
        plt.close()

    # Steps 13 and 14
    # ================ Global Utility Functions ================

def clean_memory():
    """Global function to force garbage collection and free memory."""
    import gc
    import matplotlib.pyplot as plt
    plt.close('all')
    gc.collect()


# ================ Enhanced Main Function ================

def main():
    """Enhanced main function with comprehensive argument parsing and better error handling."""
    parser = argparse.ArgumentParser(
        description='Universal Fractal Dimension Analysis Tool with Rectangular Grid Support',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Enhanced Examples:
  # Generate and analyze a Koch curve at level 5
  python fractal_analyzer.py --generate koch --level 5

  # Analyze RT interface with auto-detected rectangular grid
  python fractal_analyzer.py --file RT160x200_interface.txt --analyze_linear_region

  # Analyze with manually specified rectangular grid
  python fractal_analyzer.py --file interface.txt --nx 160 --ny 200 --analyze_linear_region

  # RT interface validation analysis with enhanced grid support
  python fractal_analyzer.py --file RT160x200-9000.txt --rt_interface --validate_grid

  # Publication-quality EPS plots for rectangular grid
  python fractal_analyzer.py --file RT320x400_interface.txt --eps_plots --no_titles --analyze_linear_region

  # Comprehensive iteration analysis with grid optimization
  python fractal_analyzer.py --generate koch --analyze_iterations --min_level 1 --max_level 6 --use_grid_optimization

Unified Box Counting Options:
  # Standard box counting (no grid optimization)
  python fractal_analyzer.py --file data.txt --disable_grid_optimization

  # Grid optimization with custom parameters
  python fractal_analyzer.py --file data.txt --box_size_factor 1.3 --min_box_size 0.001
""")

    # File and generation arguments
    parser.add_argument('--file', help='Path to file containing line segments')
    parser.add_argument('--generate', type=str, choices=['koch', 'sierpinski', 'minkowski', 'hilbert', 'dragon'],
                        help='Generate a fractal curve of specified type')
    parser.add_argument('--level', type=int, default=5, help='Level for fractal generation (default: 5)')
    
    # Box counting parameters (unified)
    parser.add_argument('--min_box_size', type=float, default=None,
                        help='Minimum box size for calculation (default: auto-estimated)')
    parser.add_argument('--max_box_size', type=float, default=None,
                        help='Maximum box size for calculation (default: auto-determined)')
    parser.add_argument('--box_size_factor', type=float, default=1.5,
                        help='Factor by which to reduce box size in each step (default: 1.5)')

    # Enhanced grid arguments
    parser.add_argument('--nx', type=int, default=None,
                       help='Grid dimension in x-direction (for rectangular grids)')
    parser.add_argument('--ny', type=int, default=None,
                       help='Grid dimension in y-direction (for rectangular grids)')
    parser.add_argument('--resolution', type=int, default=None,
                       help='Square grid resolution (alternative to nx/ny)')
    
    # Analysis mode arguments
    parser.add_argument('--rt_interface', action='store_true',
                       help='Use RT interface analysis mode with appropriate defaults')
    parser.add_argument('--validate_grid', action='store_true',
                       help='Perform grid validation analysis')
    parser.add_argument('--analyze_linear_region', action='store_true',
                       help='Analyze how linear region selection affects dimension')
    parser.add_argument('--analyze_iterations', action='store_true',
                       help='Analyze how iteration depth affects measured dimension')

    # Iteration analysis arguments
    parser.add_argument('--min_level', type=int, default=1,
                       help='Minimum curve level for iteration analysis (default: 1)')
    parser.add_argument('--max_level', type=int, default=8,
                       help='Maximum curve level for iteration analysis (default: 8)')

    # Plotting and output arguments
    parser.add_argument('--no_plot', action='store_true', help='Disable plotting')
    parser.add_argument('--no_box_plot', action='store_true', help='Disable box overlay in curve plots')
    parser.add_argument('--plot_separate', action='store_true',
                       help='Generate separate plots instead of combined format for publication')
    parser.add_argument('--no_titles', action='store_true',
                       help='Disable plot titles for journal submissions')
    parser.add_argument('--eps_plots', action='store_true',
                       help='Generate EPS format plots for publication quality (AMC journal requirements)')

    # Analysis parameters
    parser.add_argument('--fractal_type', type=str, choices=['koch', 'sierpinski', 'minkowski', 'hilbert', 'dragon'],
                       help='Specify fractal type for analysis (needed when using --file)')
    parser.add_argument('--trim_boundary', type=int, default=0,
                       help='Number of box counts to trim from each end of the data (default: 0)')
    
    # Unified optimization control
    parser.add_argument('--use_grid_optimization', action='store_true', default=True,
                       help='Use grid optimization for improved accuracy (default: True)')
    parser.add_argument('--disable_grid_optimization', action='store_true',
                       help='Disable grid optimization (use standard method)')

    # Logging and debugging
    parser.add_argument('--log_level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Set logging level (default: INFO)')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output (equivalent to --log_level DEBUG)')

    args = parser.parse_args()

    # Process arguments
    if args.verbose:
        args.log_level = 'DEBUG'

    # Handle grid optimization flags
    use_grid_optimization = not args.disable_grid_optimization
    
    # Handle resolution vs nx/ny
    if args.resolution is not None and (args.nx is not None or args.ny is not None):
        print("Warning: Both --resolution and --nx/--ny specified. Using --nx/--ny.")
    elif args.resolution is not None:
        args.nx = args.ny = args.resolution

    # Display enhanced version info
    print(f"Enhanced Fractal Analyzer with Unified Box Counting")
    print(f"=" * 60)
    print(f"Grid optimization: {'ENABLED' if use_grid_optimization else 'DISABLED'}")
    
    if args.nx is not None and args.ny is not None:
        is_rect = args.nx != args.ny
        aspect_ratio = max(args.nx, args.ny) / min(args.nx, args.ny) if min(args.nx, args.ny) > 0 else 1.0
        print(f"Grid configuration: {args.nx}×{args.ny} ({'rectangular' if is_rect else 'square'})")
        if is_rect:
            print(f"Aspect ratio: {aspect_ratio:.2f}")
    
    if args.eps_plots:
        print(f"Output format: EPS (Publication quality)")
    else:
        print(f"Output format: PNG")
    
    print(f"Log level: {args.log_level}")

    # Create analyzer instance with enhanced settings
    analyzer = FractalAnalyzer(
        fractal_type=args.fractal_type, 
        no_titles=args.no_titles, 
        eps_plots=args.eps_plots,
        log_level=args.log_level
    )

    # Clean memory before starting
    analyzer.clean_memory()

    # Initialize segments cache
    generated_segments = None

    try:
        # Generate a fractal curve if requested
        if args.generate:
            analyzer.logger.info(f"Generating {args.generate} fractal at level {args.level}")
            _, generated_segments = analyzer.generate_fractal(args.generate, args.level)
            filename = f'{args.generate}_segments_level_{args.level}.txt'
            analyzer.write_segments_to_file(generated_segments, filename)
            analyzer.logger.info(f"{args.generate.capitalize()} curve saved to {filename}")

            if args.file is None:
                args.file = filename
                analyzer.fractal_type = args.generate

        # Process file input - use cached segments if available
        if args.file:
            if generated_segments is not None:
                # Use cached segments from generation
                segments = generated_segments
                analyzer.logger.info(f"Using cached {len(segments)} segments from generation")
            else:
                # Read from file as normal
                segments = analyzer.read_line_segments(args.file)
                analyzer.logger.info(f"Read {len(segments)} line segments from {args.file}")

            if not segments:
                analyzer.logger.error("No valid line segments found. Exiting.")
                return 1

            # Auto-detect grid resolution if not provided
            if args.nx is None and args.ny is None:
                nx, ny = analyzer.auto_detect_resolution_from_filename(args.file)
                if nx is not None:
                    args.nx, args.ny = nx, ny
                    analyzer.logger.info(f"Auto-detected grid resolution: {args.nx}×{args.ny}")

            # Grid validation if requested
            if args.validate_grid and args.nx is not None and args.ny is not None:
                analyzer.logger.info("\n=== GRID VALIDATION ANALYSIS ===")
                validation = analyzer.validate_rectangular_grid_analysis(segments, args.nx, args.ny, args.min_box_size)

                print(f"Grid validation results:")
                print(f"  Valid: {'✓' if validation['is_valid'] else '⚠'}")
                print(f"  Warnings: {len(validation['warnings'])}")
                print(f"  Recommendations: {len(validation['recommendations'])}")

                if validation['warnings']:
                    print("Warnings:")
                    for warning in validation['warnings']:
                        print(f"  - {warning}")

                if validation['recommendations']:
                    print("Recommendations:")
                    for rec in validation['recommendations']:
                        print(f"  - {rec}")
                print("=== END GRID VALIDATION ===\n")

        else:
            analyzer.logger.error("No input file or generation method specified.")
            print("Error: Must specify either --file or --generate")
            return 1

    except Exception as e:
        analyzer.logger.error(f"Error during setup: {str(e)}")
        return 1

    # Execute analysis based on selected mode
    # RT interface analysis mode (enhanced)
    if args.rt_interface:
        analyzer.logger.info("\n=== RT INTERFACE ANALYSIS MODE ===")
        results = analyzer.analyze_rt_interface(
            segments=segments,
            nx=args.nx,
            ny=args.ny,
            filename=args.file,
            plot_results=not args.no_plot,
            plot_boxes=not args.no_box_plot,
            plot_separate=args.plot_separate,
            box_size_factor=args.box_size_factor,
            use_grid_optimization=use_grid_optimization,
            min_box_size=args.min_box_size,
            trim_boundary=args.trim_boundary,
            return_box_data=False
        )
        analyzer.logger.info("=== RT INTERFACE ANALYSIS COMPLETE ===\n")

    # Linear region analysis (unified method)
    if args.analyze_linear_region:
        analyzer.logger.info("\n=== UNIFIED LINEAR REGION ANALYSIS ===")
        results = analyzer.analyze_linear_region(
            segments=segments,
            fractal_type=args.fractal_type,
            plot_results=not args.no_plot,
            plot_boxes=not args.no_box_plot,
            trim_boundary=args.trim_boundary,
            box_size_factor=args.box_size_factor,
            use_grid_optimization=use_grid_optimization,
            plot_separate=args.plot_separate,
            min_box_size=args.min_box_size,
            nx=args.nx,
            ny=args.ny,
            filename_context=args.file
        )
        analyzer.logger.info("=== LINEAR REGION ANALYSIS COMPLETE ===")

    # Iteration analysis (enhanced)
    if args.analyze_iterations:
        if not args.fractal_type and not args.generate:
            analyzer.logger.error("Must specify --fractal_type or --generate for iteration analysis")
            return 1
                
        analyzer.logger.info("\n=== ITERATION ANALYSIS ===")
        fractal_type = args.generate or args.fractal_type

        successful_levels, dimensions, errors, r_squared = analyzer.analyze_iterations(
            min_level=args.min_level, 
            max_level=args.max_level, 
            fractal_type=fractal_type,
            no_plots=args.no_plot, 
            no_box_plot=args.no_box_plot,
            box_size_factor=args.box_size_factor,
            use_grid_optimization=use_grid_optimization,
            min_box_size=args.min_box_size
        )
        analyzer.logger.info("=== ITERATION ANALYSIS COMPLETE ===")

    # Standard dimension calculation (enhanced with unified box counting)
    else:
        analyzer.logger.info("\n=== STANDARD FRACTAL DIMENSION CALCULATION ===")
                
        # Auto-determine max box size if not provided
        if args.max_box_size is None:
            extent = analyzer._calculate_domain_extent(segments)
            args.max_box_size = extent / 2
            analyzer.logger.info(f"Auto-determined max box size: {args.max_box_size}")

        # Auto-estimate min box size if not provided
        if args.min_box_size is None:
            args.min_box_size = analyzer.estimate_min_box_size(
                segments, method='auto', nx=args.nx, ny=args.ny)
            analyzer.logger.info(f"Auto-estimated min_box_size: {args.min_box_size:.8f}")

        # Validate box sizes
        args.min_box_size, args.max_box_size, warning = analyzer.validate_and_adjust_box_sizes(
            args.min_box_size, args.max_box_size, segments)
                
        if warning:
            analyzer.logger.warning(warning)

        # Perform unified box counting
        box_sizes, box_counts, bounding_box = analyzer.box_counting_unified(
            segments, args.min_box_size, args.max_box_size, 
            args.box_size_factor, use_grid_optimization)

        # Enhanced boundary removal
        box_sizes, box_counts = analyzer.enhanced_boundary_removal(
            box_sizes, box_counts, args.trim_boundary)

        # Calculate fractal dimension
        fractal_dimension, error, intercept = analyzer.calculate_fractal_dimension(
            box_sizes, box_counts)

        # Display results with grid context
        analyzer.logger.info(f"Results:")
        if args.nx is not None and args.ny is not None:
            is_rect = args.nx != args.ny
            aspect_ratio = max(args.nx, args.ny) / min(args.nx, args.ny) if min(args.nx, args.ny) > 0 else 1.0
            analyzer.logger.info(f"  Grid: {args.nx}×{args.ny} ({'rectangular' if is_rect else 'square'})")
            if is_rect:
                analyzer.logger.info(f"  Aspect ratio: {aspect_ratio:.2f}")
                
        analyzer.logger.info(f"  Fractal Dimension: {fractal_dimension:.6f} ± {error:.6f}")
                
        if args.fractal_type:
            theoretical = analyzer.theoretical_dimensions[args.fractal_type]
            analyzer.logger.info(f"  Theoretical {args.fractal_type} dimension: {theoretical:.6f}")
            analyzer.logger.info(f"  Difference: {abs(fractal_dimension - theoretical):.6f}")

        # Create plots if requested
        if not args.no_plot:
            analyzer.plot_results_separate(segments, box_sizes, box_counts,
                                         fractal_dimension, error, bounding_box, intercept,
                                         plot_boxes=not args.no_box_plot)
            analyzer._plot_loglog(box_sizes, box_counts, fractal_dimension, error, intercept)

    # Clean up and final summary
    analyzer.clean_memory()
    
    if args.eps_plots:
        print("\n" + "="*60)
        print("PUBLICATION-QUALITY EPS PLOTS GENERATED")
        print("="*60)
        print("Features:")
        print("• High resolution (300 DPI) vector graphics")
        print("• Enhanced readability and visibility")
        print("• AMC journal compliant formatting")
        if args.nx is not None and args.ny is not None:
            is_rect = args.nx != args.ny
            print(f"• Grid-specific optimization for {args.nx}×{args.ny} {'rectangular' if is_rect else 'square'} grid")
        print("• Unified box counting with", "grid optimization" if use_grid_optimization else "standard method")

    print(f"\nAnalysis completed successfully!")
    if use_grid_optimization:
        print("Used enhanced grid optimization for improved accuracy.")
    print(f"Log level: {args.log_level}")

    return 0


# ================ Enhanced Usage Examples ================

"""
COMPREHENSIVE EXAMPLES FOR ENHANCED FRACTAL ANALYZER:

Basic Fractal Generation and Analysis:
# Generate Koch curve and analyze with unified method
python fractal_analyzer.py --generate koch --level 5 --analyze_linear_region

# Generate Sierpinski triangle with iteration analysis
python fractal_analyzer.py --generate sierpinski --analyze_iterations --min_level 1 --max_level 6

Rectangular Grid RT Interface Analysis:
# Auto-detection from filename with validation
python fractal_analyzer.py --file RT160x200-9000.txt --rt_interface --validate_grid --eps_plots

# Manual grid specification with enhanced analysis
python fractal_analyzer.py --file interface_segments.txt --nx 160 --ny 200 --analyze_linear_region --log_level DEBUG

# High aspect ratio grid analysis
python fractal_analyzer.py --file RT800x200-9000.txt --rt_interface --validate_grid

Publication Quality Output:
# EPS plots with no titles for journal submission
python fractal_analyzer.py --file RT320x400-9000.txt --rt_interface --eps_plots --no_titles --plot_separate

# Comprehensive analysis with all plots
python fractal_analyzer.py --file data.txt --analyze_linear_region --plot_separate --eps_plots

Performance and Debugging:
# Debug mode with detailed logging
python fractal_analyzer.py --file large_dataset.txt --analyze_linear_region --verbose

# Disable grid optimization for comparison
python fractal_analyzer.py --file data.txt --analyze_linear_region --disable_grid_optimization

# Custom box counting parameters
python fractal_analyzer.py --file data.txt --min_box_size 0.001 --box_size_factor 1.3 --trim_boundary 2

Grid Validation Workflow:
# Complete validation and analysis pipeline
python fractal_analyzer.py --file RT640x800-9000.txt --validate_grid --rt_interface --eps_plots --verbose

# Compare square vs rectangular analysis
python fractal_analyzer.py --file interface.txt --nx 400 --ny 400 --analyze_linear_region  # Square
python fractal_analyzer.py --file interface.txt --nx 400 --ny 800 --analyze_linear_region  # Rectangular

Memory Management for Large Datasets:
# Large dataset analysis with optimization
python fractal_analyzer.py --file very_large_interface.txt --rt_interface --no_box_plot --log_level WARNING
"""


if __name__ == "__main__":
    import sys
    sys.exit(main())
