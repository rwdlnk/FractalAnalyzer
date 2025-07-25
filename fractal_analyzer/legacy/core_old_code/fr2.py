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
from typing import Tuple, List, Dict, Optional


class FractalAnalyzer:
    """Universal fractal dimension analysis tool."""
    
    def __init__(self, fractal_type: Optional[str] = None):
        """
        Initialize the fractal analyzer.
        
        Args:
            fractal_type: Type of fractal if known (koch, sierpinski, etc.)
        """
        self.theoretical_dimensions = {
            'koch': np.log(4) / np.log(3),      # ≈ 1.2619
            'sierpinski': np.log(3) / np.log(2), # ≈ 1.5850
            'minkowski': 1.5,                   # Exact value
            'hilbert': 2.0,                     # Space-filling
            'dragon': 1.5236                    # Approximate
        }
        self.fractal_type = fractal_type
    
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

# ================ Core Analysis Functions ================
    
    def read_line_segments(self, filename: str) -> List:
        """Read line segments from a file."""
        start_time = time.time()
        print(f"Reading segments from {filename}...")
        
        segments = []
        with open(filename, 'r') as file:
            for line in file:
                if not line.strip() or line.strip().startswith('#'):
                    continue
                
                try:
                    coords = [float(x) for x in line.replace(',', ' ').split()]
                    if len(coords) == 4:
                        segments.append(((coords[0], coords[1]), (coords[2], coords[3])))
                except ValueError:
                    print(f"Warning: Could not parse line: {line}")
        
        print(f"Read {len(segments)} segments in {time.time() - start_time:.2f} seconds")
        return segments
    
    def write_segments_to_file(self, segments: List, filename: str):
        """Write line segments to a file."""
        start_time = time.time()
        print(f"Writing {len(segments)} segments to file {filename}...")
        
        with open(filename, 'w') as file:
            for (x1, y1), (x2, y2) in segments:
                file.write(f"{x1} {y1} {x2} {y2}\n")
        
        print(f"File writing completed in {time.time() - start_time:.2f} seconds")
    
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
        print("Creating spatial index...")
        
        # Calculate grid dimensions
        grid_width = max(1, int(np.ceil((max_x - min_x) / cell_size)))
        grid_height = max(1, int(np.ceil((max_y - min_y) / cell_size)))
        
        # Debug information
        print(f"  Grid dimensions: {grid_width} x {grid_height}")
        print(f"  Total cells: {grid_width * grid_height}")
        print(f"  Cell size: {cell_size}")
        print(f"  Bounds: ({min_x}, {min_y}) to ({max_x}, {max_y})")
    
        # Check for extremely large grids
        if grid_width * grid_height > 1000000:
            print(f"WARNING: Grid is very large ({grid_width * grid_height} cells). This may take a while...")
        
        # Create the spatial index
        segment_grid = defaultdict(list)
        
        # Add progress reporting for large datasets
        segment_count = len(segments)
        report_interval = max(1, segment_count // 10)  # Report every 10%
        
        for i, ((x1, y1), (x2, y2)) in enumerate(segments):
            if i % report_interval == 0:
                print(f"  Progress: {i}/{segment_count} segments processed ({i*100//segment_count}%)")
            
            # Determine which grid cells this segment might intersect
            min_cell_x = max(0, int((min(x1, x2) - min_x) / cell_size))
            max_cell_x = min(grid_width - 1, int((max(x1, x2) - min_x) / cell_size))
            min_cell_y = max(0, int((min(y1, y2) - min_y) / cell_size))
            max_cell_y = min(grid_height - 1, int((max(y1, y2) - min_y) / cell_size))
            
            # Add segment to all relevant grid cells
            for cell_x in range(min_cell_x, max_cell_x + 1):
                for cell_y in range(min_cell_y, max_cell_y + 1):
                    segment_grid[(cell_x, cell_y)].append(i)
        
        print(f"Spatial index created in {time.time() - start_time:.2f} seconds")
        print(f"Total grid cells with segments: {len(segment_grid)}")
        
        return segment_grid, grid_width, grid_height
    
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
            print(f"Applied manual boundary trimming: {trim_boundary} points from each end")
    
        # Enhanced automatic boundary artifact detection
        if len(box_sizes) > 8:  # Need enough points for meaningful analysis
            log_sizes = np.log(box_sizes)
            log_counts = np.log(box_counts)
        
            # Check for deviations from linearity at ends
            n = len(log_sizes)
        
            # Calculate slopes for different segments
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
                
                    # More sophisticated boundary detection
                    slope_threshold = 0.12  # Reduced from 0.15 for better sensitivity
                    r2_threshold = 0.95     # R² threshold for quality segments
                    additional_trim = 0
                
                    # Check first segment - both slope deviation AND R² quality
                    first_slope_dev = abs(slope_first - slope_middle) / abs(slope_middle) if slope_middle != 0 else 0
                    if (first_slope_dev > slope_threshold) or (r2_first < r2_threshold):
                        additional_trim = max(additional_trim, 1)
                        print(f"Detected boundary artifact at start: slope deviation {first_slope_dev:.3f}, R² {r2_first:.3f}")
                
                    # Check last segment - both slope deviation AND R² quality
                    last_slope_dev = abs(slope_last - slope_middle) / abs(slope_middle) if slope_middle != 0 else 0
                    if (last_slope_dev > slope_threshold) or (r2_last < r2_threshold):
                        additional_trim = max(additional_trim, 1)
                        print(f"Detected boundary artifact at end: slope deviation {last_slope_dev:.3f}, R² {r2_last:.3f}")
                
                    # Apply additional trimming if artifacts detected
                    if additional_trim > 0 and len(box_sizes) > 2 * additional_trim + 4:  # Keep at least 4 points
                        print(f"Removing {additional_trim} additional boundary points from each end")
                        box_sizes = box_sizes[additional_trim:-additional_trim]
                        box_counts = box_counts[additional_trim:-additional_trim]
                    
                        # Verify the trimming improved the linearity
                        new_log_sizes = np.log(box_sizes)
                        new_log_counts = np.log(box_counts)
                        _, _, new_r2, _, _ = stats.linregress(new_log_sizes, new_log_counts)
                        print(f"R² after boundary trimming: {new_r2:.4f} (was {r2_middle:.4f})")
                
                except Exception as e:
                    print(f"Warning: Could not perform enhanced boundary detection: {e}")
    
        final_length = len(box_sizes)
        total_removed = original_length - final_length
        if total_removed > 0:
            print(f"Total boundary points removed: {total_removed} ({total_removed/original_length*100:.1f}%)")
    
        return box_sizes, box_counts

    def box_counting_with_grid_optimization(self, segments, min_box_size, max_box_size, box_size_factor=1.5):
        """Optimized box counting using spatial indexing with grid optimization."""
        total_start_time = time.time()
    
        # Find the bounding box of all segments
        min_x = min(min(s[0][0], s[1][0]) for s in segments)
        max_x = max(max(s[0][0], s[1][0]) for s in segments)
        min_y = min(min(s[0][1], s[1][1]) for s in segments)
        max_y = max(max(s[0][1], s[1][1]) for s in segments)
    
        # Add a small margin
        margin = max(max_x - min_x, max_y - min_y) * 0.01
        min_x -= margin
        max_x += margin
        min_y -= margin
        max_y += margin
    
        box_sizes = []
        box_counts = []
    
        current_box_size = max_box_size
    
        print("Box counting with grid optimization debug info:")
        print("  Box size  |  Min count |  Grid tests        | Improv | Time (s)")
        print("-----------------------------------------------------")
    
        # Use the same cell size as original approach
        spatial_cell_size = min_box_size * 2
        segment_grid, grid_width, grid_height = self.create_spatial_index(
            segments, min_x, min_y, max_x, max_y, spatial_cell_size)

        while current_box_size >= min_box_size:
            box_start_time = time.time()
        
            # Adaptive grid density based on box size
            if current_box_size < min_box_size * 5:
                # Fine grid for small boxes (most critical for accuracy)
                offset_increments = np.linspace(0, 0.75, 4)  # 4×4 = 16 tests
                grid_desc = "fine (4×4)"
            elif current_box_size < min_box_size * 20:
                # Medium grid for medium boxes
                offset_increments = np.linspace(0, 0.5, 3)   # 3×3 = 9 tests
                grid_desc = "medium (3×3)"
            else:
                # Coarse grid for large boxes (less critical)
                offset_increments = np.linspace(0, 0.5, 2)   # 2×2 = 4 tests
                grid_desc = "coarse (2×2)"
            
            # Test all grid positions
            min_count = float('inf')
            max_count = 0
            grid_tests = 0
            
            for dx_fraction in offset_increments:
                for dy_fraction in offset_increments:
                    grid_tests += 1
                    
                    # Calculate actual offsets
                    offset_x = min_x + dx_fraction * current_box_size
                    offset_y = min_y + dy_fraction * current_box_size
                    
                    # Count boxes with this offset
                    count = self._count_boxes_with_offset(
                        segments, current_box_size, offset_x, offset_y, 
                        max_x, max_y, segment_grid, grid_width, grid_height, spatial_cell_size)
                    
                    min_count = min(min_count, count)
                    max_count = max(max_count, count)
            
            # Calculate improvement metric
            improvement = (max_count - min_count) / max_count * 100 if max_count > 0 else 0
            
            elapsed = time.time() - box_start_time
            print(f"  {current_box_size:.6f}  |  {min_count:8d}  |  {grid_tests:10d} ({grid_desc})  |  {improvement:5.1f}%  |  {elapsed:.2f}")

            if min_count > 0:
                box_sizes.append(current_box_size)
                box_counts.append(min_count)
            else:
                print(f"  Warning: No boxes occupied at box size {current_box_size}. Skipping this size.")
        
            current_box_size /= box_size_factor

        if len(box_sizes) < 2:
            raise ValueError("Not enough valid box sizes for fractal dimension calculation.")
    
        print(f"\nTotal box counting time: {time.time() - total_start_time:.2f} seconds")
    
        return np.array(box_sizes), np.array(box_counts), (min_x, min_y, max_x, max_y)

    def _count_boxes_with_offset(self, segments, box_size, offset_x, offset_y, 
                               max_x, max_y, segment_grid, grid_width, grid_height, spatial_cell_size):
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
                min_cell_x = max(0, int((box_xmin - (offset_x - spatial_cell_size)) / spatial_cell_size))
                max_cell_x = min(grid_width - 1, int((box_xmax - (offset_x - spatial_cell_size)) / spatial_cell_size))
                min_cell_y = max(0, int((box_ymin - (offset_y - spatial_cell_size)) / spatial_cell_size))
                max_cell_y = min(grid_height - 1, int((box_ymax - (offset_y - spatial_cell_size)) / spatial_cell_size))
            
                segments_to_check = set()
                for cell_x in range(min_cell_x, max_cell_x + 1):
                    for cell_y in range(min_cell_y, max_cell_y + 1):
                        segments_to_check.update(segment_grid.get((cell_x, cell_y), []))
            
                for seg_idx in segments_to_check:
                    (x1, y1), (x2, y2) = segments[seg_idx]
                    if self.liang_barsky_line_box_intersection(x1, y1, x2, y2, box_xmin, box_ymin, box_xmax, box_ymax):
                        occupied_boxes.add((i, j))
                        break
    
        return len(occupied_boxes)

    def box_counting_optimized(self, segments, min_box_size, max_box_size, box_size_factor=1.5):
        """Optimized box counting using spatial indexing."""
        total_start_time = time.time()
        
        # Find the bounding box of all segments
        min_x = min(min(s[0][0], s[1][0]) for s in segments)
        max_x = max(max(s[0][0], s[1][0]) for s in segments)
        min_y = min(min(s[0][1], s[1][1]) for s in segments)
        max_y = max(max(s[0][1], s[1][1]) for s in segments)
        
        # Add a small margin
        margin = max(max_x - min_x, max_y - min_y) * 0.01
        min_x -= margin
        max_x += margin
        min_y -= margin
        max_y += margin
        
        box_sizes = []
        box_counts = []
        
        current_box_size = max_box_size
        
        print("Box counting debug info:")
        print("  Box size  |  Box count  |  Time (s)")
        print("------------------------------------------")
        
        # Use the same cell size as original fd-all.py
        spatial_cell_size = min_box_size * 2  # Conservative cell size
        segment_grid, grid_width, grid_height = self.create_spatial_index(
            segments, min_x, min_y, max_x, max_y, spatial_cell_size)
        
        while current_box_size >= min_box_size:
            box_start_time = time.time()
            
            num_boxes_x = int(np.ceil((max_x - min_x) / current_box_size))
            num_boxes_y = int(np.ceil((max_y - min_y) / current_box_size))
            
            occupied_boxes = set()
            
            for i in range(num_boxes_x):
                for j in range(num_boxes_y):
                    box_xmin = min_x + i * current_box_size
                    box_ymin = min_y + j * current_box_size
                    box_xmax = box_xmin + current_box_size
                    box_ymax = box_ymin + current_box_size
                    
                    min_cell_x = max(0, int((box_xmin - min_x) / spatial_cell_size))
                    max_cell_x = min(grid_width - 1, int((box_xmax - min_x) / spatial_cell_size))
                    min_cell_y = max(0, int((box_ymin - min_y) / spatial_cell_size))
                    max_cell_y = min(grid_height - 1, int((box_ymax - min_y) / spatial_cell_size))
                    
                    segments_to_check = set()
                    for cell_x in range(min_cell_x, max_cell_x + 1):
                        for cell_y in range(min_cell_y, max_cell_y + 1):
                            segments_to_check.update(segment_grid.get((cell_x, cell_y), []))
                    
                    for seg_idx in segments_to_check:
                        (x1, y1), (x2, y2) = segments[seg_idx]
                        if self.liang_barsky_line_box_intersection(x1, y1, x2, y2, box_xmin, box_ymin, box_xmax, box_ymax):
                            occupied_boxes.add((i, j))
                            break
            
            count = len(occupied_boxes)
            elapsed = time.time() - box_start_time
            print(f"  {current_box_size:.6f}  |  {count:8d}  |  {elapsed:.2f}")
            
            if count > 0:
                box_sizes.append(current_box_size)
                box_counts.append(count)
            else:
                print(f"  Warning: No boxes occupied at box size {current_box_size}. Skipping this size.")
            
            current_box_size /= box_size_factor
    
        if len(box_sizes) < 2:
            raise ValueError("Not enough valid box sizes for fractal dimension calculation.")
        
        print(f"\nTotal box counting time: {time.time() - total_start_time:.2f} seconds")
        
        return np.array(box_sizes), np.array(box_counts), (min_x, min_y, max_x, max_y)

    def calculate_fractal_dimension(self, box_sizes, box_counts):
            """Calculate the fractal dimension using box-counting method."""
            log_sizes = np.log(box_sizes)
            log_counts = np.log(box_counts)
        
            if np.any(np.isnan(log_sizes)) or np.any(np.isnan(log_counts)):
                valid = ~(np.isnan(log_sizes) | np.isnan(log_counts))
                log_sizes = log_sizes[valid]
                log_counts = log_counts[valid]
                print(f"Warning: Removed {np.sum(~valid)} invalid ln values")
        
            if len(log_sizes) < 2:
                print("Error: Not enough valid data points for regression!")
                return float('nan'), float('nan'), float('nan')
        
            slope, intercept, r_value, p_value, std_err = stats.linregress(log_sizes, log_counts)
        
            fractal_dimension = -slope
        
            print(f"R-squared value: {r_value**2:.4f}")
        
            return fractal_dimension, std_err, intercept

    def estimate_min_box_size_from_segments(self, segments, percentile=5, multiplier=1.5):
        """
        Estimate appropriate min_box_size from segment lengths with more conservative defaults.
    
        Args:
            segments: List of line segments ((x1,y1), (x2,y2))
            percentile: Percentile of segment lengths to use (default: 5th percentile - more conservative)
            multiplier: Factor to multiply characteristic length (default: 1.5 - reduced from 3)
    
        Returns:
            float: Suggested minimum box size
        """
        if not segments:
            print("Warning: No segments provided for box size estimation")
            return 0.001  # Fallback value
    
        # Calculate all segment lengths
        lengths = []
        for (x1, y1), (x2, y2) in segments:
            length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            if length > 0:  # Only include non-zero lengths
                lengths.append(length)
    
        if not lengths:
            print("Warning: No segments with non-zero length found")
            return 0.001
    
        lengths = np.array(lengths)
    
        # Use specified percentile (short segments) as reference
        characteristic_length = np.percentile(lengths, percentile)
    
        # Min box size should be several times the characteristic segment length
        min_box_size = multiplier * characteristic_length
    
        print(f"Segment length analysis:")
        print(f"  Total segments: {len(segments)}")
        print(f"  Mean length: {np.mean(lengths):.6f}")
        print(f"  Median length: {np.median(lengths):.6f}")
        print(f"  Min length: {np.min(lengths):.6f}")
        print(f"  {percentile}th percentile: {characteristic_length:.6f}")
        print(f"  Suggested min_box_size: {min_box_size:.6f} ({multiplier}× {percentile}th percentile)")
    
        return min_box_size

    def validate_and_adjust_box_sizes(self, min_box_size, max_box_size, segments):
        """
        Validate and adjust box sizes to ensure min_box_size < max_box_size.
        
        Args:
            min_box_size: Proposed minimum box size
            max_box_size: Maximum box size (typically extent/2)
            segments: List of line segments for fallback calculations
            
        Returns:
            tuple: (adjusted_min_box_size, max_box_size, warning_message)
        """
        warning_msg = ""
        
        # Check if min_box_size >= max_box_size
        if min_box_size >= max_box_size:
            warning_msg = f"WARNING: Auto-estimated min_box_size ({min_box_size:.6f}) >= max_box_size ({max_box_size:.6f})"
            print(warning_msg)
            
            # Strategy 1: Reduce min_box_size to a fraction of max_box_size
            safety_factor = 0.01  # Use 1% of max_box_size as fallback
            fallback_min = max_box_size * safety_factor
            
            # Strategy 2: Use the smallest segment length directly (no multiplier)
            if segments:
                lengths = []
                for (x1, y1), (x2, y2) in segments:
                    length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                    lengths.append(length)
                
                lengths = np.array(lengths)
                min_segment_length = np.min(lengths[lengths > 0])  # Avoid zero-length segments
                
                # Choose the smaller of the two fallback strategies
                adjusted_min = min(fallback_min, min_segment_length)
            else:
                adjusted_min = fallback_min
            
            print(f"Adjusting min_box_size to {adjusted_min:.6f}")
            print(f"This ensures min_box_size/max_box_size ratio = {adjusted_min/max_box_size:.4f}")
            
            # Ensure we have at least a few box size steps
            ratio = adjusted_min / max_box_size
            if ratio > 0.5:  # If still too large, be more aggressive
                adjusted_min = max_box_size * 0.001  # 0.1% of max_box_size
                print(f"Further reducing min_box_size to {adjusted_min:.6f} for sufficient scaling range")
            
            return adjusted_min, max_box_size, warning_msg
        
        # Sizes are valid - also check if we have sufficient scaling range
        ratio = min_box_size / max_box_size
        if ratio > 0.1:  # Less than 1 decade of scaling
            warning_msg = f"WARNING: Limited scaling range - min/max ratio = {ratio:.4f}"
            print(warning_msg)
            print("Consider using a smaller min_box_size for better fractal dimension accuracy")
        
        return min_box_size, max_box_size, warning_msg

# ================ Advanced Analysis Functions ================
    def analyze_linear_region(self, segments, fractal_type=None, plot_results=True, 
        plot_boxes=True, trim_boundary=0, box_size_factor=1.5, use_grid_optimization=True,
        return_box_data=False, plot_separate=False, min_box_size=None):
        """
        Analyze how the choice of linear region affects the calculated dimension.
        Uses a sliding window approach to identify the optimal scaling region.
        """
        print("\n==== ANALYZING LINEAR REGION SELECTION ====\n")
        
        # Use provided type or instance type
        type_used = fractal_type or self.fractal_type
        
        if type_used in self.theoretical_dimensions:
            theoretical_dimension = self.theoretical_dimensions[type_used]
            print(f"Theoretical {type_used} dimension: {theoretical_dimension:.6f}")
        else:
            theoretical_dimension = None
            print("No theoretical dimension available for comparison")
        
        # Calculate extent to determine box sizes
        min_x = min(min(s[0][0], s[1][0]) for s in segments)
        max_x = max(max(s[0][0], s[1][0]) for s in segments)
        min_y = min(min(s[0][1], s[1][1]) for s in segments)
        max_y = max(max(s[0][1], s[1][1]) for s in segments)
        extent = max(max_x - min_x, max_y - min_y)

        # Auto-estimate min_box_size if not provided
        if min_box_size is None:
            min_box_size = self.estimate_min_box_size_from_segments(segments)
            print(f"Auto-estimated min_box_size: {min_box_size:.6f}")
        else:
            print(f"Using provided min_box_size: {min_box_size:.6f}")

        # Determine max_box_size
        max_box_size = extent / 2

        # VALIDATION CHECK - Ensure min_box_size < max_box_size
        min_box_size, max_box_size, warning = self.validate_and_adjust_box_sizes(
            min_box_size, max_box_size, segments)

        if warning:
            print(f"Box size validation: {warning}")

        print(f"Final box size range: {min_box_size:.8f} to {max_box_size:.8f}")
        print(f"Scaling ratio (min/max): {min_box_size/max_box_size:.6f}")
        print(f"Box size reduction factor: {box_size_factor}")

        # Calculate expected number of box sizes
        expected_steps = int(np.log(max_box_size/min_box_size) / np.log(box_size_factor))
        print(f"Expected number of box sizes: {expected_steps}")

        if expected_steps < 5:
            print("WARNING: Very few box sizes will be tested. Consider adjusting parameters.")

        # Calculate fractal dimension with many data points
        if use_grid_optimization:
            print("Using grid optimization method...")
            box_sizes, box_counts, bounding_box = self.box_counting_with_grid_optimization(
                segments, min_box_size, max_box_size, box_size_factor=box_size_factor)
        else:
            print("Using basic method (no grid optimization)...")
            box_sizes, box_counts, bounding_box = self.box_counting_optimized(
                segments, min_box_size, max_box_size, box_size_factor=box_size_factor)

        # Enhanced boundary handling (automatic + manual)
        box_sizes, box_counts = self.enhanced_boundary_removal(box_sizes, box_counts, trim_boundary)
        print(f"Box counts after enhanced boundary handling: {len(box_counts)}")        

        # Convert to ln scale for analysis
        log_sizes = np.log(box_sizes)
        log_counts = np.log(box_counts)
        
        # Analyze different window sizes for linear region selection
        min_window = 3  # Minimum points for regression
        max_window = len(log_sizes)
        
        windows = range(min_window, max_window + 1)
        dimensions = []
        errors = []
        r_squared = []
        start_indices = []
        end_indices = []
        
        print("Window size | Start idx | End idx | Dimension | Error | R²")
        print("-" * 65)
        
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
            
            # Only store results if we found a valid window
            if best_dimension is not None:
                dimensions.append(best_dimension)
                errors.append(best_error)
                r_squared.append(best_r2)
                start_indices.append(best_start)
                end_indices.append(best_end)
                
                print(f"{window_size:11d} | {best_start:9d} | {best_end:7d} | {best_dimension:9.6f} | {best_error:5.6f} | {best_r2:.6f}")
            else:
                print(f"{window_size:11d} | No valid regression possible")

        # Check if we have any valid results
        if not dimensions:
            print("\nERROR: No valid linear regions found for fractal dimension calculation!")
            print("This typically happens when:")
            print("  1. Too few box sizes were generated")
            print("  2. Box size range is too limited")
            print("  3. Data quality is insufficient")
            print("\nSuggestions:")
            print("  - Use smaller min_box_size")
            print("  - Use smaller box_size_factor (e.g., 1.2)")
            print("  - Generate higher level fractals")
            
            # Return fallback values to prevent crash
            if return_box_data:
                return [], [], [], [], 0, float('nan'), box_sizes, box_counts, bounding_box
            else:
                return [], [], [], [], 0, float('nan')

        # Enhancement for fractal_analyzer.py analyze_linear_region method
        # Find the window with dimension closest to theoretical or best R² WITH PHYSICAL CONSTRAINTS
        if theoretical_dimension is not None:
            closest_idx = np.argmin(np.abs(np.array(dimensions) - theoretical_dimension))
        else:
            # For RT interfaces, apply physical constraints:
            # 1. Dimension should be between 1.0 and 2.0 (for 2D interfaces)
            # 2. Window size should be at least 4 points for reliable regression
            # 3. Prefer larger windows when R² is similar
    
            valid_indices = []
            for i, (window, dim, r2) in enumerate(zip(windows, dimensions, r_squared)):
                # Physical constraint: 1.0 ≤ D ≤ 2.0
                if 1.0 <= dim <= 2.0:
                    # Statistical constraint: window size ≥ 4 
                    if window >= 4:
                        # Quality constraint: R² ≥ 0.99
                        if r2 >= 0.99:
                            valid_indices.append(i)
    
            if valid_indices:
                # Among valid windows, prefer larger windows when R² is close
                # Sort by R² (descending), then by window size (descending) as tiebreaker
                valid_data = [(r_squared[i], windows[i], i) for i in valid_indices]
                valid_data.sort(key=lambda x: (x[0], x[1]), reverse=True)
       
                # Select the best valid window
                closest_idx = valid_data[0][2]
        
                print(f"Applied physical constraints:")
                print(f"  Valid windows: {[windows[i] for i in valid_indices]}")
                print(f"  Valid dimensions: {[f'{dimensions[i]:.3f}' for i in valid_indices]}")
                print(f"  Valid R²: {[f'{r_squared[i]:.6f}' for i in valid_indices]}")
                print(f"  Selected window {windows[closest_idx]} (D={dimensions[closest_idx]:.6f})")
            else:
                # Fallback: use best R² without constraints (but warn)
                closest_idx = np.argmax(r_squared)
                print(f"WARNING: No windows met physical constraints. Using best R² = {r_squared[closest_idx]:.6f}")
                print(f"WARNING: Dimension {dimensions[closest_idx]:.6f} may be unphysical")

        optimal_window = windows[closest_idx]
        optimal_dimension = dimensions[closest_idx]
        optimal_start = start_indices[closest_idx]
        optimal_end = end_indices[closest_idx]

        print("\nResults:")
        print("\nDetailed window analysis:")
        print("Window | Dimension | Theoretical Error | R² | Error Magnitude")
        print("-------|-----------|------------------|----|-----------------")
        for i, (w, d, r2) in enumerate(zip(windows, dimensions, r_squared)):
            if theoretical_dimension is not None:
                error_pct = abs(d - theoretical_dimension) / theoretical_dimension * 100
                print(f"{w:6d} | {d:9.6f} | {error_pct:15.1f}% | {r2:.6f} | {abs(d - theoretical_dimension):.6f}")

        if theoretical_dimension is not None:
            print(f"Theoretical dimension: {theoretical_dimension:.6f}")
            print(f"Closest dimension: {optimal_dimension:.6f} (window size: {optimal_window})")
        else:
            print(f"Best dimension (highest R²): {optimal_dimension:.6f} (window size: {optimal_window})")
        print(f"Optimal scaling region: points {optimal_start} to {optimal_end}")
        print(f"Box size range: {box_sizes[optimal_start]:.8f} to {box_sizes[optimal_end-1]:.8f}")
        
        # Plot the results
        if plot_results:
            if plot_separate:
                # Generate individual plots for publication
                
                # 1. Sliding window analysis plot
                self._plot_window_analysis_separate(
                    windows, dimensions, errors, r_squared, optimal_window, optimal_dimension,
                    theoretical_dimension, fractal_type=type_used)
                
                # 2. Final log-log plot with optimal scaling region
                self._plot_loglog_with_region(
                    log_sizes, log_counts, optimal_start, optimal_end, optimal_dimension,
                    errors[closest_idx], fractal_type=type_used)
                
                # 3. Curve plot
                self.plot_results_separate(segments, box_sizes, box_counts, optimal_dimension, 
                                         errors[closest_idx], bounding_box, None, 
                                         plot_boxes=plot_boxes)
            else:
                # Original combined plot
                self._plot_linear_region_analysis(
                    windows, dimensions, errors, r_squared, optimal_window, optimal_dimension,
                    theoretical_dimension, log_sizes, log_counts, optimal_start, optimal_end, 
                    fractal_type=type_used)
                
                # Also create a curve plot with optional box overlay
                self.plot_results_separate(segments, box_sizes, box_counts, optimal_dimension, 
                                         errors[closest_idx], bounding_box, None, 
                                         plot_boxes=plot_boxes)
        if return_box_data:
            return windows, dimensions, errors, r_squared, optimal_window, optimal_dimension, box_sizes, box_counts, bounding_box
        else:
            return windows, dimensions, errors, r_squared, optimal_window, optimal_dimension
 
    def analyze_iterations(self, min_level=1, max_level=8, fractal_type=None, 
                          box_ratio=0.3, no_plots=False, no_box_plot=False, box_size_factor=1.5):
        """
        Analyze how fractal dimension varies with iteration depth.
        Generates curves at different levels and calculates their dimensions.
        """
        print("\n==== ANALYZING DIMENSION VS ITERATION LEVEL ====\n")
        
        # Use provided type or instance type
        type_used = fractal_type or self.fractal_type
    
        if type_used is None:
            raise ValueError("Fractal type must be specified either in constructor or as argument")
    
        theoretical_dimension = self.theoretical_dimensions[type_used]
        print(f"Theoretical {type_used} dimension: {theoretical_dimension:.6f}")
    
        # Initialize results storage
        levels = list(range(min_level, max_level + 1))
        dimensions = []
        errors = []
        r_squared = []
    
        # For each level, generate a curve and calculate its dimension
        for level in levels:
            print(f"\n--- Processing {type_used} curve at level {level} ---")
        
            # Generate the curve
            _, segments = self.generate_fractal(type_used, level)
        
            # Calculate dimension by using linear region analysis WITH box data return
            results = self.analyze_linear_region(
                segments, fractal_type=type_used, plot_results=False, 
                box_size_factor=box_size_factor, return_box_data=True)  # <- REQUEST BOX DATA
        
            # Unpack results including box data
            windows, dims, errs, r2s, optimal_window, optimal_dimension, box_sizes, box_counts, bounding_box = results
        
            # Check if analysis failed
            if np.isnan(optimal_dimension):
                print(f"Level {level} - Analysis failed: insufficient data for fractal dimension calculation")
                print(f"Skipping level {level} and continuing with next level...")
                continue
        
            # Get error for the optimal window
            error = errs[np.where(np.array(windows) == optimal_window)[0][0]]
        
            # Calculate R-squared for the optimal window
            r_value_squared = r2s[np.where(np.array(windows) == optimal_window)[0][0]]
        
            # Store results
            dimensions.append(optimal_dimension)
            errors.append(error)
            r_squared.append(r_value_squared)
        
            print(f"Level {level} - Fractal Dimension: {optimal_dimension:.6f} ± {error:.6f}")
            print(f"Difference from theoretical: {abs(optimal_dimension - theoretical_dimension):.6f}")
            print(f"R-squared: {r_value_squared:.6f}")
        
            # Plot results if requested - REUSE box counting data!
            if not no_plots:
                curve_file = f"{type_used}_level_{level}_curve.png"
                dimension_file = f"{type_used}_level_{level}_dimension.png"
            
                print(f"Plotting fractal curve to {curve_file}")
                print(f"Plotting dimension analysis to {dimension_file}")
            
                # Respect the no_box_plot parameter
                plot_boxes = (level <= 6) and not no_box_plot
            
                # Plot the fractal curve - REUSE the box_sizes, box_counts, bounding_box!
                self.plot_results_separate(segments, box_sizes, box_counts, optimal_dimension, 
                                         error, bounding_box, None, plot_boxes=plot_boxes, 
                                         level=level, custom_filename=curve_file)
            
                # Plot the dimension analysis (log-log plot)
                self._plot_loglog(box_sizes, box_counts, optimal_dimension, error, 
                                  custom_filename=dimension_file)
    
        # Plot the dimension vs. level results
        self._plot_dimension_vs_level(levels, dimensions, errors, r_squared, 
                                     theoretical_dimension, type_used)
    
        return levels, dimensions, errors, r_squared

# ================ Plotting Functions ================
    def plot_results_separate(self, segments, box_sizes, box_counts, fractal_dimension, 
                            error, bounding_box, intercept=None, plot_boxes=False, level=None,
                            custom_filename=None):
        """Creates two separate plots instead of a combined figure."""
        import matplotlib
        
        # Get segment count
        segment_count = len(segments)
        
        # Determine if this is a file-based curve with unknown fractal type
        is_file_based = self.fractal_type is None or (custom_filename and not level)
        
        # Set parameters based on fractal type, level, and segment count
        if is_file_based:
            # Optimization for file-based curves based on segment count
            if segment_count > 50000:
                # Very large file-based dataset
                matplotlib.rcParams['agg.path.chunksize'] = 100000
                plot_dpi = 150
                fig_size = (14, 12)
                use_rasterized = True
                print(f"Very large file-based curve detected ({segment_count} segments). Using maximum optimization.")
            elif segment_count > 20000:
                # Large file-based dataset
                matplotlib.rcParams['agg.path.chunksize'] = 50000
                plot_dpi = 200
                fig_size = (12, 10)
                use_rasterized = True
                print(f"Large file-based curve detected ({segment_count} segments). Using enhanced optimization.")
            elif segment_count > 5000:
                # Medium file-based dataset
                matplotlib.rcParams['agg.path.chunksize'] = 25000
                plot_dpi = 250
                fig_size = (11, 9)
                use_rasterized = True
                print(f"Medium file-based curve detected ({segment_count} segments). Using standard optimization.")
            else:
                # Small file-based dataset - default settings
                matplotlib.rcParams['agg.path.chunksize'] = 20000
                plot_dpi = 300
                fig_size = (10, 8)
                use_rasterized = False
        elif self.fractal_type == 'koch' and level and level > 5:
            # Special settings for high-level Koch curves
            matplotlib.rcParams['agg.path.chunksize'] = 50000  # Increased for Koch
            plot_dpi = 200  # Lower DPI for very complex curves
            fig_size = (12, 10)  # Larger figure size
            use_rasterized = True
        elif self.fractal_type in ['sierpinski', 'hilbert'] and level and level > 6:
            # Settings for other complex fractals
            matplotlib.rcParams['agg.path.chunksize'] = 30000
            plot_dpi = 250
            fig_size = (11, 9)
            use_rasterized = True
        else:
            # Default settings for less complex curves
            matplotlib.rcParams['agg.path.chunksize'] = 20000
            plot_dpi = 300
            fig_size = (10, 8)
            use_rasterized = False
        
        matplotlib.rcParams['path.simplify_threshold'] = 0.1  # Add simplification
    
        # Figure 1: The curve with optional box overlay
        plt.figure(figsize=fig_size)
    
        start_time = time.time()
        print("Plotting curve segments...")
    
        # Convert segments to a more efficient format for line plotting
        # For large datasets, use a simplified plotting method
        if segment_count > 50000:
            print(f"Very large dataset ({segment_count} segments), using aggressive sampling...")
            # More aggressive sampling for extremely large datasets
            step = max(1, segment_count // 10000)
            sampled_segments = segments[::step]
            print(f"Sampled down to {len(sampled_segments)} segments for visualization")
            
            x_points = []
            y_points = []
            for (x1, y1), (x2, y2) in sampled_segments:
                x_points.extend([x1, x2, None])  # None creates a break in the line
                y_points.extend([y1, y2, None])
            
            x_points = x_points[:-1]
            y_points = y_points[:-1]
            
            plt.plot(x_points, y_points, 'k-', linewidth=1, rasterized=use_rasterized)
        elif segment_count > 10000:
            print(f"Large dataset ({segment_count} segments), using simplified plotting...")
            # Sample the segments for visualization
            step = max(1, segment_count // 20000)
            sampled_segments = segments[::step]
            print(f"Sampled down to {len(sampled_segments)} segments for visualization")
        
            x_points = []
            y_points = []
            for (x1, y1), (x2, y2) in sampled_segments:
                x_points.extend([x1, x2, None])  # None creates a break in the line
                y_points.extend([y1, y2, None])
        
            x_points = x_points[:-1]
            y_points = y_points[:-1]
        
            plt.plot(x_points, y_points, 'k-', linewidth=1, rasterized=use_rasterized)
        else:
            # Normal plotting for smaller datasets
            x_points = []
            y_points = []
            for (x1, y1), (x2, y2) in segments:
                x_points.extend([x1, x2, None])
                y_points.extend([y1, y2, None])
    
            x_points = x_points[:-1]
            y_points = y_points[:-1]
        
            plt.plot(x_points, y_points, 'k-', linewidth=1, rasterized=use_rasterized)
    
        print(f"Curve plotting completed in {time.time() - start_time:.2f} seconds")
    
        # Unpack the bounding box used in counting
        min_x, min_y, max_x, max_y = bounding_box

        # Set a slightly larger margin for the plot view
        view_margin = max(max_x - min_x, max_y - min_y) * 0.05
        plt.xlim(min_x - view_margin, max_x + view_margin)
        plt.ylim(min_y - view_margin, max_y + view_margin)
    
        # If requested, plot boxes at a specific scale
        if plot_boxes:
            self._plot_box_overlay(segments, box_sizes, box_counts, bounding_box)

        title = f'{self.fractal_type.capitalize() if self.fractal_type else "Fractal"} Curve'
        if level is not None:
            title += f' (Level {level})'
        if plot_boxes:
            smallest_idx = len(box_sizes) - 1
            box_size = box_sizes[smallest_idx]
            title += f'\nwith Box Counting Overlay (Box Size: {box_size:.6f})'
    
        plt.title(title)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.grid(True, linestyle='--', alpha=0.7)
    
        # Use the provided custom_filename if available, otherwise create default filename
        if custom_filename:
            curve_filename = custom_filename
        else:
            curve_filename = f'{self.fractal_type if self.fractal_type else "fractal"}_curve'
            if plot_boxes:
                curve_filename += '_with_boxes'
            if level is not None:
                curve_filename += f'_Level_{level}'
            curve_filename += '.png'
    
        # Add memory cleanup before saving large plots
        if segment_count > 20000:
            # Force garbage collection before saving large plots
            import gc
            gc.collect()
    
        try:
            print(f"Saving curve plot to {curve_filename} with DPI {plot_dpi}")
            plt.savefig(curve_filename, dpi=plot_dpi)
            print(f"Successfully saved plot to {curve_filename}")
        except Exception as e:
            print(f"Error saving {curve_filename} at DPI {plot_dpi}: {str(e)}")
            # Try a more aggressive fallback with figure simplification
            try:
                print("Attempting with simplified figure...")
                # Further reduce complexity for the save operation
                plt.clf()  # Clear the figure
                plt.plot([0, 1], [0, 1], 'k-', linewidth=1)
                plt.title(f"Simplified version of {title} - See log for details")
                plt.text(0.5, 0.5, f"Full plot was too complex to save.\nSegment count: {segment_count}", 
                         horizontalalignment='center', verticalalignment='center')
                plt.savefig(curve_filename, dpi=75)
                print(f"Saved simplified placeholder to {curve_filename}")
            except Exception as e2:
                print(f"Still failed to save: {str(e2)}")
                print("Consider using --no_plot option for very large datasets")

        plt.close()
    
        # Figure 2: Log-log plot (only if we're not using a custom_filename)
        # For custom_filename (used in analyze_iterations), we now use the _plot_loglog
        # method directly instead of handling it here
    def _plot_box_overlay(self, segments, box_sizes, box_counts, bounding_box):
        """Plot box overlay for visual verification."""
        box_time = time.time()
        print("Generating box overlay...")
        
        # Get segment count
        segment_count = len(segments)
        
        # Choose the smallest box size to visualize
        smallest_idx = len(box_sizes) - 1
        box_size = box_sizes[smallest_idx]
        expected_count = box_counts[smallest_idx]
        
        print(f"Box size: {box_size}, Expected count: {expected_count}")
        
        min_x, min_y, max_x, max_y = bounding_box
        
        # Calculate box coordinates
        num_boxes_x = int(np.ceil((max_x - min_x) / box_size))
        num_boxes_y = int(np.ceil((max_y - min_y) / box_size))
        
        # Create spatial index for efficient intersection tests
        grid_size = box_size * 2
        segment_grid, _, _ = self.create_spatial_index(
            segments, min_x, min_y, max_x, max_y, grid_size)
    
        print(f"Spatial index created in {time.time() - box_time:.2f} seconds")
        box_time = time.time()
        
        # Collect all boxes in a list for batch processing
        rectangles = []
        
        for i in range(num_boxes_x):
            for j in range(num_boxes_y):
                box_xmin = min_x + i * box_size
                box_ymin = min_y + j * box_size
                box_xmax = box_xmin + box_size
                box_ymax = box_ymin + box_size
                
                # Find which grid cell this box belongs to
                cell_x = int((box_xmin - min_x) / grid_size)
                cell_y = int((box_ymin - min_y) / grid_size)
                
                # Get segments that might intersect this box
                segments_to_check = set()
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        adjacent_key = (cell_x + dx, cell_y + dy)
                        segments_to_check.update(segment_grid.get(adjacent_key, []))
                
                # Check for intersection with the candidate segments
                for seg_idx in segments_to_check:
                    (x1, y1), (x2, y2) = segments[seg_idx]
                    if self.liang_barsky_line_box_intersection(x1, y1, x2, y2, box_xmin, box_ymin, box_xmax, box_ymax):
                        rectangles.append(Rectangle((box_xmin, box_ymin), box_size, box_size))
                        break
        
        print(f"Box intersection tests completed in {time.time() - box_time:.2f} seconds")
        box_time = time.time()
        
        # Use PatchCollection for much faster rendering with box outlines only
        pc = PatchCollection(rectangles, facecolor='none', edgecolor='r', linewidth=0.5, alpha=0.8)
        plt.gca().add_collection(pc)
    
        print(f"Box rendering completed in {time.time() - box_time:.2f} seconds")
        print(f"Total boxes drawn: {len(rectangles)}")

    def _plot_loglog(self, box_sizes, box_counts, fractal_dimension, error, intercept=None, 
                    custom_filename=None):
        """Plot ln-ln analysis."""
        plt.figure(figsize=(10, 8))
        
        plt.loglog(box_sizes, box_counts, 'bo-', label='Data points')
        
        # Perform linear regression for plotting the fit line
        log_sizes = np.log(box_sizes)
        log_counts = np.log(box_counts)
        
        # If intercept wasn't provided, calculate it
        if intercept is None:
            _, intercept, _, _, _ = stats.linregress(log_sizes, log_counts)
        
        # Plot the linear regression line
        fit_counts = np.exp(intercept + (-fractal_dimension) * log_sizes)
        plt.loglog(box_sizes, fit_counts, 'r-', 
                   label=f'Fit: D = {fractal_dimension:.4f} ± {error:.4f}')
        
        # Custom formatter for scientific notation
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
        
        # Set axis properties
        ax = plt.gca()
        ax.xaxis.set_major_formatter(FuncFormatter(scientific_formatter))
        ax.yaxis.set_major_formatter(FuncFormatter(scientific_formatter))
        
        plt.title('Box Counting: ln(N) vs ln(1/r)')
        plt.xlabel('Box Size (r)')
        plt.ylabel('Number of Boxes (N)')
        plt.legend()
        plt.grid(True, which='major', linestyle='-', linewidth=0.5, alpha=0.7)
        plt.grid(True, which='minor', linestyle=':', linewidth=0.3, alpha=0.5)
        
        # Use provided filename or generate default
        if custom_filename:
            filename = custom_filename
        else:
            # Use fractal type in filename if available
            filename = 'box_counting_loglog'
            if self.fractal_type:
                filename = f'{self.fractal_type}_box_counting_loglog'
            filename += '.png'
        
        plt.savefig(filename, dpi=300)
        plt.close()

    def _plot_linear_region_analysis(self, windows, dimensions, errors, r_squared, 
                                   optimal_window, optimal_dimension, theoretical_dimension,
                                   log_sizes, log_counts, optimal_start, optimal_end, fractal_type=None):
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
        plt.title('Fractal Dimension vs. Window Size')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        # Plot 2: R² vs window size
        plt.subplot(3, 1, 2)
        plt.plot(windows, r_squared, 'go-', markersize=6)
        plt.scatter([optimal_window], [r_squared[windows.index(optimal_window)]], 
                   color='red', s=100, zorder=5, label=f'Optimal Window (R²={r_squared[windows.index(optimal_window)]:.6f})')
        plt.axhline(y=0.99, color='orange', linestyle='--', label='R² = 0.99')
        
        plt.xlabel('Window Size (number of points)')
        plt.ylabel('R² Value')
        plt.title('R² vs. Window Size')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.ylim(0.95, 1.001)  # Zoom in on the relevant range
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
        plt.title('ln-ln Plot with Optimal Scaling Region')
        
        # Adjust axis limits to ensure all data points are visible
        x_margin = (log_sizes.max() - log_sizes.min()) * 0.05
        y_margin = (log_counts.max() - log_counts.min()) * 0.05
        plt.xlim(log_sizes.min() - x_margin, log_sizes.max() + x_margin)
        plt.ylim(log_counts.min() - y_margin, log_counts.max() + y_margin)
        
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        plt.tight_layout()
        
        # Use fractal type in filename if provided
        filename = 'linear_region_analysis'
        if fractal_type:
            filename = f'{fractal_type}_linear_region_analysis'
        plt.savefig(f'{filename}.png', dpi=300)
        plt.close()

    def _plot_dimension_vs_level(self, levels, dimensions, errors, r_squared, 
                               theoretical_dimension, fractal_type):
        """Plot dimension vs. iteration level."""
        plt.figure(figsize=(10, 6))
        plt.errorbar(levels, dimensions, yerr=errors, fmt='o-', capsize=5, 
                     label='Calculated Dimension')
        plt.axhline(y=theoretical_dimension, color='r', linestyle='--', 
                    label=f'Theoretical Dimension ({theoretical_dimension:.6f})')
        
        plt.xlabel(f'{fractal_type.capitalize()} Curve Iteration Level')
        plt.ylabel('Fractal Dimension')
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
        plt.savefig(f'{fractal_type}_dimension_vs_level.png', dpi=300)
        plt.close()

    def _plot_window_analysis_separate(self, windows, dimensions, errors, r_squared, 
                                     optimal_window, optimal_dimension, theoretical_dimension,
                                     fractal_type=None):
        """Plot sliding window analysis as separate publication-quality figure."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        
        # Top panel: Dimension vs window size
        ax1.errorbar(windows, dimensions, yerr=errors, fmt='o-', capsize=5, markersize=4,
                     color='blue', label='Calculated Dimension')
        
        if theoretical_dimension is not None:
            ax1.axhline(y=theoretical_dimension, color='red', linestyle='--', linewidth=2,
                       label=f'Theoretical D = {theoretical_dimension:.4f}')
        
        ax1.scatter([optimal_window], [optimal_dimension], color='red', s=100, zorder=5,
                   label=f'Optimal (size={optimal_window})')
        
        ax1.set_ylabel('Fractal Dimension')
        ax1.set_title('Sliding Window Optimization Analysis')
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.legend()
        
        # Bottom panel: R² vs window size
        ax2.plot(windows, r_squared, 'go-', markersize=6, label='R² Value')
        ax2.scatter([optimal_window], [r_squared[windows.index(optimal_window)]], 
                   color='red', s=100, zorder=5, 
                   label=f'Optimal R² = {r_squared[windows.index(optimal_window)]:.4f}')
        ax2.axhline(y=0.99, color='orange', linestyle='--', alpha=0.7, label='R² = 0.99')
        
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
        filename += '.png'
        
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved sliding window analysis to {filename}")

    def _plot_loglog_with_region(self, log_sizes, log_counts, optimal_start, optimal_end,
                               optimal_dimension, error, fractal_type=None):
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
        plt.title('Box Counting: ln(N) vs ln(1/r)')
        plt.legend()
        plt.grid(True, which='major', linestyle='-', linewidth=0.5, alpha=0.7)
        plt.grid(True, which='minor', linestyle=':', linewidth=0.3, alpha=0.5)
        
        # Generate filename
        filename = 'box_counting_with_optimal_region'
        if fractal_type:
            filename = f'{fractal_type}_box_counting_with_optimal_region'
        filename += '.png'
        
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved log-log plot with optimal region to {filename}")

def clean_memory():
    """Force garbage collection to free memory."""
    import gc
    import matplotlib.pyplot as plt
    plt.close('all')
    gc.collect()

def main():
    parser = argparse.ArgumentParser(
        description='Universal Fractal Dimension Analysis Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate and analyze a Koch curve at level 5
  python fractal_analyzer-v26.py --generate koch --level 5
  
  # Analyze how iteration level affects dimension
  python fractal_analyzer-v26.py --generate sierpinski --analyze_iterations
  
  # Analyze linear region selection with custom box sizes
  python fractal_analyzer-v26.py --file coastline.txt --analyze_linear_region --min_box_size 0.0005
  
  # Run both analyses in sequence
  python fractal_analyzer-v26.py --generate dragon --analyze_linear_region --analyze_iterations
""")
    parser.add_argument('--file', help='Path to file containing line segments')
    parser.add_argument('--generate', type=str, choices=['koch', 'sierpinski', 'minkowski', 'hilbert', 'dragon'],
                        help='Generate a fractal curve of specified type')
    parser.add_argument('--level', type=int, default=5, help='Level for fractal generation')
    parser.add_argument('--min_box_size', type=float, default=0.001, 
                        help='Minimum box size for calculation')
    parser.add_argument('--max_box_size', type=float, default=None, 
                        help='Maximum box size for calculation (default: auto-determined)')
    parser.add_argument('--box_size_factor', type=float, default=1.5, 
                        help='Factor by which to reduce box size in each step')
    parser.add_argument('--no_plot', action='store_true', 
                        help='Disable plotting')
    parser.add_argument('--no_box_plot', action='store_true',
                        help='Disable box overlay in the curve plot')
    parser.add_argument('--analyze_iterations', action='store_true',
                       help='Analyze how iteration depth affects measured dimension')
    parser.add_argument('--min_level', type=int, default=1,
                       help='Minimum curve level for iteration analysis')
    parser.add_argument('--max_level', type=int, default=8,
                       help='Maximum curve level for iteration analysis')
    parser.add_argument('--analyze_linear_region', action='store_true',
                       help='Analyze how linear region selection affects dimension')
    parser.add_argument('--fractal_type', type=str, choices=['koch', 'sierpinski', 'minkowski', 'hilbert', 'dragon'],
                       help='Specify fractal type for analysis (needed when using --file)')
    parser.add_argument('--trim_boundary', type=int, default=0,
                       help='Number of box counts to trim from each end of the data')
    parser.add_argument('--use_grid_optimization', action='store_true', default=True,
                       help='Use grid optimization for improved accuracy (default: True)')
    parser.add_argument('--disable_grid_optimization', action='store_true',
                       help='Disable grid optimization (use original method)')    
    parser.add_argument('--plot_separate', action='store_true',
                       help='Generate separate plots instead of combined format for publication')
    args = parser.parse_args()
    
    # Display version info
    print(f"Running Fractal Analyzer v26")
    print(f"-------------------------------")
    
    # Create analyzer instance
    analyzer = FractalAnalyzer(args.fractal_type)
    
    # Clean memory before starting
    clean_memory()
    
    # Generate a fractal curve if requested
    if args.generate:
        _, segments = analyzer.generate_fractal(args.generate, args.level)
        filename = f'{args.generate}_segments_level_{args.level}.txt'
        analyzer.write_segments_to_file(segments, filename)
        print(f"{args.generate.capitalize()} curve saved to {filename}")
        
        # Use this curve for analysis if no file is specified
        if args.file is None:
            args.file = filename
            analyzer.fractal_type = args.generate
    
    # Read line segments from file or use generated segments
    if args.file:
        segments = analyzer.read_line_segments(args.file)
        print(f"Read {len(segments)} line segments from {args.file}")
        
        if not segments:
            print("No valid line segments found. Exiting.")
            return
        
        # Analyze linear region if requested
        if args.analyze_linear_region:
            print("\n=== Starting Linear Region Analysis ===\n")
            use_grid_opt = not args.disable_grid_optimization  # Convert disable flag to use flag
            analyzer.analyze_linear_region(segments, args.fractal_type, not args.no_plot, 
                                         not args.no_box_plot, trim_boundary=args.trim_boundary,
                                         box_size_factor=args.box_size_factor,
                                         use_grid_optimization=use_grid_opt,
                                         plot_separate=args.plot_separate)

            print("\n=== Linear Region Analysis Complete ===\n")
            # Note: removed the return statement here
    
        # Standard dimension calculation if neither special analysis is requested
        if not args.analyze_linear_region and not args.analyze_iterations:
            # Auto-determine max box size if not provided
            if args.max_box_size is None:
                min_x = min(min(s[0][0], s[1][0]) for s in segments)
                max_x = max(max(s[0][0], s[1][0]) for s in segments)
                min_y = min(min(s[0][1], s[1][1]) for s in segments)
                max_y = max(max(s[0][1], s[1][1]) for s in segments)
                
                extent = max(max_x - min_x, max_y - min_y)
                args.max_box_size = extent / 2
                print(f"Auto-determined max box size: {args.max_box_size}")
            
            try:
                # Perform box counting
                if args.disable_grid_optimization:
                    box_sizes, box_counts, bounding_box = analyzer.box_counting_optimized(
                        segments, args.min_box_size, args.max_box_size, args.box_size_factor)
                else:
                    box_sizes, box_counts, bounding_box = analyzer.box_counting_with_grid_optimization(
                        segments, args.min_box_size, args.max_box_size, args.box_size_factor)
                
                # Calculate fractal dimension
                fractal_dimension, error, intercept = analyzer.calculate_fractal_dimension(
                    box_sizes, box_counts)
                
                # Print results
                print(f"Results:")
                print(f"  Fractal Dimension: {fractal_dimension:.6f} ± {error:.6f}")
                if args.fractal_type:
                    theoretical = analyzer.theoretical_dimensions[args.fractal_type]
                    print(f"  Theoretical {args.fractal_type} dimension: {theoretical:.6f}")
                    print(f"  Difference: {abs(fractal_dimension - theoretical):.6f}")
                
                # Plot if requested
                if not args.no_plot:
                    analyzer.plot_results_separate(segments, box_sizes, box_counts, 
                                                 fractal_dimension, error, bounding_box, intercept,
                                                 plot_boxes=not args.no_box_plot)
                    # Also plot the log-log analysis
                    analyzer._plot_loglog(box_sizes, box_counts, fractal_dimension, error, intercept)

            except Exception as e:
                print(f"Error during calculation: {str(e)}")
                return float('nan'), float('nan')
    
    # Analyze iteration depth relationship
    if args.analyze_iterations:
        if not args.fractal_type and not args.generate:
            print("Error: Must specify --fractal_type or --generate for iteration analysis")
            return
        print("\n=== Starting Iteration Analysis ===\n")
        fractal_type = args.generate or args.fractal_type
        analyzer.analyze_iterations(args.min_level, args.max_level, fractal_type, 
                                  no_plots=args.no_plot, no_box_plot=args.no_box_plot, box_size_factor=args.box_size_factor)
        print("\n=== Iteration Analysis Complete ===\n")
        
    if not (args.analyze_linear_region or args.analyze_iterations or args.file or args.generate):
        print("No input file specified and no curve generation requested.")
        print("Use --file to specify an input file or --generate to create a fractal curve.")
        parser.print_help()


if __name__ == "__main__":
    main()
