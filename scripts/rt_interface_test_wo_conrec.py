#!/usr/bin/env python3
"""
Debug CONREC interface extraction and provide robust alternatives for RT simulations.

This script helps diagnose CONREC issues and provides multiple interface extraction methods
specifically designed for Rayleigh-Taylor VOF data.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import measure, morphology, filters
from scipy.interpolate import RegularGridInterpolator
import time

class RTInterfaceExtractor:
    """
    Robust interface extraction for Rayleigh-Taylor VOF simulations.
    Provides multiple methods to handle highly fragmented interfaces.
    """
    
    def __init__(self, debug=True):
        self.debug = debug
    
    def diagnose_vof_data(self, f_grid, x_grid, y_grid):
        """
        Comprehensive diagnosis of VOF data quality and structure.
        """
        print("=== VOF DATA DIAGNOSIS ===")
        
        # Basic statistics
        print(f"Grid shape: {f_grid.shape}")
        print(f"F-field range: [{np.min(f_grid):.6f}, {np.max(f_grid):.6f}]")
        print(f"Unique F values: {len(np.unique(f_grid))}")
        
        # Check for binary vs continuous data
        unique_vals = np.unique(f_grid)
        is_binary = len(unique_vals) <= 10
        print(f"Data type: {'Binary VOF' if is_binary else 'Continuous VOF'}")
        
        if is_binary:
            print(f"Unique values: {unique_vals}")
        
        # Interface detection
        interface_cells = np.sum((f_grid > 0.01) & (f_grid < 0.99))
        total_cells = f_grid.size
        print(f"Mixed cells (0.01 < F < 0.99): {interface_cells}/{total_cells} ({interface_cells/total_cells*100:.1f}%)")
        
        # Sharp transitions
        transitions = 0
        for i in range(f_grid.shape[0]-1):
            for j in range(f_grid.shape[1]-1):
                if abs(f_grid[i,j] - f_grid[i+1,j]) > 0.5:
                    transitions += 1
                if abs(f_grid[i,j] - f_grid[i,j+1]) > 0.5:
                    transitions += 1
        print(f"Sharp transitions (|ΔF| > 0.5): {transitions}")
        
        # Grid spacing
        dx = x_grid[1,0] - x_grid[0,0] if x_grid.shape[0] > 1 else 0
        dy = y_grid[0,1] - y_grid[0,0] if y_grid.shape[1] > 1 else 0
        print(f"Grid spacing: Δx={dx:.8f}, Δy={dy:.8f}")
        
        return {
            'is_binary': is_binary,
            'interface_cells': interface_cells,
            'sharp_transitions': transitions,
            'dx': dx, 'dy': dy
        }
    
    def method_1_smoothed_contouring(self, f_grid, x_grid, y_grid, sigma=0.8):
        """
        Method 1: Smooth binary VOF data then use scikit-image contouring.
        This is often the most robust for RT simulations.
        """
        print("\n--- Method 1: Smoothed Contouring ---")
        
        # Apply Gaussian smoothing to create transition zones
        f_smoothed = ndimage.gaussian_filter(f_grid.astype(float), sigma=sigma)
        print(f"Applied Gaussian smoothing (σ={sigma})")
        print(f"Smoothed range: [{np.min(f_smoothed):.3f}, {np.max(f_smoothed):.3f}]")
        
        # Extract contours
        try:
            contours = measure.find_contours(f_smoothed.T, 0.5)
            print(f"Found {len(contours)} contour paths")
            
            # Convert to physical coordinates
            segments = []
            for contour in contours:
                if len(contour) > 1:
                    x_physical = np.interp(contour[:, 1], np.arange(f_grid.shape[0]), x_grid[:, 0])
                    y_physical = np.interp(contour[:, 0], np.arange(f_grid.shape[1]), y_grid[0, :])
                    
                    # Convert to segments
                    for i in range(len(x_physical) - 1):
                        segments.append(((x_physical[i], y_physical[i]), 
                                       (x_physical[i+1], y_physical[i+1])))
            
            print(f"Extracted {len(segments)} line segments")
            return segments, f_smoothed
            
        except Exception as e:
            print(f"Method 1 failed: {e}")
            return [], f_smoothed
    
    def method_2_edge_detection(self, f_grid, x_grid, y_grid):
        """
        Method 2: Edge detection for binary VOF data.
        Finds sharp transitions and creates interface segments.
        """
        print("\n--- Method 2: Edge Detection ---")
        
        segments = []
        
        # Find horizontal edges (transitions between rows)
        for i in range(f_grid.shape[0] - 1):
            for j in range(f_grid.shape[1]):
                if abs(f_grid[i, j] - f_grid[i+1, j]) > 0.5:
                    # Create horizontal interface segment
                    x1 = x_grid[i, j]
                    x2 = x_grid[i+1, j] if i+1 < x_grid.shape[0] else x_grid[i, j]
                    y_interface = (y_grid[i, j] + y_grid[i+1, j]) / 2 if i+1 < y_grid.shape[0] else y_grid[i, j]
                    
                    # Small horizontal segment
                    dx = abs(x2 - x1) * 0.1
                    x_center = (x1 + x2) / 2
                    segments.append(((x_center - dx/2, y_interface), (x_center + dx/2, y_interface)))
        
        # Find vertical edges (transitions between columns)
        for i in range(f_grid.shape[0]):
            for j in range(f_grid.shape[1] - 1):
                if abs(f_grid[i, j] - f_grid[i, j+1]) > 0.5:
                    # Create vertical interface segment
                    y1 = y_grid[i, j]
                    y2 = y_grid[i, j+1] if j+1 < y_grid.shape[1] else y_grid[i, j]
                    x_interface = (x_grid[i, j] + x_grid[i, j+1]) / 2 if j+1 < x_grid.shape[1] else x_grid[i, j]
                    
                    # Small vertical segment
                    dy = abs(y2 - y1) * 0.1
                    y_center = (y1 + y2) / 2
                    segments.append(((x_interface, y_center - dy/2), (x_interface, y_center + dy/2)))
        
        print(f"Found {len(segments)} edge segments")
        return segments
    
    def method_3_plic_reconstruction(self, f_grid, x_grid, y_grid):
        """
        Method 3: PLIC (Piecewise Linear Interface Calculation) reconstruction.
        This is the theoretically correct method for VOF interfaces.
        """
        print("\n--- Method 3: PLIC Reconstruction ---")
        
        segments = []
        
        # For each mixed cell (0 < F < 1), reconstruct the interface
        for i in range(f_grid.shape[0]):
            for j in range(f_grid.shape[1]):
                f_val = f_grid[i, j]
                
                # Only process mixed cells
                if 0.01 < f_val < 0.99:
                    # Calculate interface normal using neighboring cells
                    normal = self._calculate_interface_normal(f_grid, i, j)
                    
                    # Reconstruct interface line in cell
                    cell_segments = self._reconstruct_interface_in_cell(
                        f_val, normal, x_grid[i, j], y_grid[i, j],
                        x_grid[1, 0] - x_grid[0, 0] if x_grid.shape[0] > 1 else 0.01,
                        y_grid[0, 1] - y_grid[0, 0] if y_grid.shape[1] > 1 else 0.01
                    )
                    
                    segments.extend(cell_segments)
        
        print(f"PLIC reconstruction: {len(segments)} segments")
        return segments
    
    def _calculate_interface_normal(self, f_grid, i, j):
        """Calculate interface normal using finite differences."""
        shape = f_grid.shape
        
        # Calculate gradients with boundary handling
        if i == 0:
            df_dx = f_grid[i+1, j] - f_grid[i, j]
        elif i == shape[0] - 1:
            df_dx = f_grid[i, j] - f_grid[i-1, j]
        else:
            df_dx = (f_grid[i+1, j] - f_grid[i-1, j]) / 2
        
        if j == 0:
            df_dy = f_grid[i, j+1] - f_grid[i, j]
        elif j == shape[1] - 1:
            df_dy = f_grid[i, j] - f_grid[i, j-1]
        else:
            df_dy = (f_grid[i, j+1] - f_grid[i, j-1]) / 2
        
        # Normalize
        magnitude = np.sqrt(df_dx**2 + df_dy**2)
        if magnitude > 1e-10:
            return np.array([df_dx / magnitude, df_dy / magnitude])
        else:
            return np.array([1.0, 0.0])  # Default to horizontal
    
    def _reconstruct_interface_in_cell(self, f_val, normal, x_center, y_center, dx, dy):
        """Reconstruct interface line within a cell using PLIC."""
        # This is a simplified PLIC - full implementation is more complex
        # For now, create a line segment based on the normal direction
        
        # Interface line perpendicular to normal
        tangent = np.array([-normal[1], normal[0]])
        
        # Position interface to match volume fraction (simplified)
        # In full PLIC, this requires solving for the correct position
        offset = (f_val - 0.5) * min(dx, dy) * 0.5
        
        # Create interface segment
        x_start = x_center + tangent[0] * dx * 0.4 + normal[0] * offset
        y_start = y_center + tangent[1] * dy * 0.4 + normal[1] * offset
        x_end = x_center - tangent[0] * dx * 0.4 + normal[0] * offset
        y_end = y_center - tangent[1] * dy * 0.4 + normal[1] * offset
        
        return [((x_start, y_start), (x_end, y_end))]
    
    def method_4_adaptive_contouring(self, f_grid, x_grid, y_grid):
        """
        Method 4: Adaptive multi-level contouring.
        Try multiple contour levels and morphological operations.
        """
        print("\n--- Method 4: Adaptive Contouring ---")
        
        best_segments = []
        best_count = 0
        
        # Try multiple contour levels
        levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        
        for level in levels:
            try:
                # Apply light smoothing
                f_smooth = ndimage.gaussian_filter(f_grid.astype(float), sigma=0.5)
                contours = measure.find_contours(f_smooth.T, level)
                
                segments = []
                for contour in contours:
                    if len(contour) > 1:
                        x_physical = np.interp(contour[:, 1], np.arange(f_grid.shape[0]), x_grid[:, 0])
                        y_physical = np.interp(contour[:, 0], np.arange(f_grid.shape[1]), y_grid[0, :])
                        
                        for i in range(len(x_physical) - 1):
                            segments.append(((x_physical[i], y_physical[i]), 
                                           (x_physical[i+1], y_physical[i+1])))
                
                if len(segments) > best_count:
                    best_count = len(segments)
                    best_segments = segments
                    print(f"  Level {level:.1f}: {len(segments)} segments (best so far)")
                else:
                    print(f"  Level {level:.1f}: {len(segments)} segments")
                    
            except Exception as e:
                print(f"  Level {level:.1f}: Failed - {e}")
        
        print(f"Best result: {best_count} segments")
        return best_segments
    
    def compare_all_methods(self, f_grid, x_grid, y_grid):
        """
        Compare all extraction methods and return the best result.
        """
        print("\n=== COMPARING ALL INTERFACE EXTRACTION METHODS ===")
        
        results = {}
        
        # Method 1: Smoothed contouring
        try:
            start_time = time.time()
            segments1, f_smoothed = self.method_1_smoothed_contouring(f_grid, x_grid, y_grid)
            time1 = time.time() - start_time
            results['smoothed'] = {'segments': segments1, 'time': time1, 'data': f_smoothed}
        except Exception as e:
            print(f"Method 1 failed: {e}")
            results['smoothed'] = {'segments': [], 'time': 0, 'data': None}
        
        # Method 2: Edge detection
        try:
            start_time = time.time()
            segments2 = self.method_2_edge_detection(f_grid, x_grid, y_grid)
            time2 = time.time() - start_time
            results['edge'] = {'segments': segments2, 'time': time2}
        except Exception as e:
            print(f"Method 2 failed: {e}")
            results['edge'] = {'segments': [], 'time': 0}
        
        # Method 3: PLIC reconstruction
        try:
            start_time = time.time()
            segments3 = self.method_3_plic_reconstruction(f_grid, x_grid, y_grid)
            time3 = time.time() - start_time
            results['plic'] = {'segments': segments3, 'time': time3}
        except Exception as e:
            print(f"Method 3 failed: {e}")
            results['plic'] = {'segments': [], 'time': 0}
        
        # Method 4: Adaptive contouring
        try:
            start_time = time.time()
            segments4 = self.method_4_adaptive_contouring(f_grid, x_grid, y_grid)
            time4 = time.time() - start_time
            results['adaptive'] = {'segments': segments4, 'time': time4}
        except Exception as e:
            print(f"Method 4 failed: {e}")
            results['adaptive'] = {'segments': [], 'time': 0}
        
        # Summary
        print("\n=== COMPARISON SUMMARY ===")
        for method, result in results.items():
            count = len(result['segments'])
            time_taken = result['time']
            print(f"{method:12s}: {count:6d} segments in {time_taken:.3f}s")
        
        # Find best method
        best_method = max(results.keys(), key=lambda k: len(results[k]['segments']))
        best_segments = results[best_method]['segments']
        
        print(f"\nBest method: {best_method} with {len(best_segments)} segments")
        
        return best_segments, results
    
    def debug_conrec_input(self, f_grid, x_grid, y_grid):
        """
        Debug the input data format for CONREC to identify issues.
        """
        print("\n=== CONREC INPUT DEBUG ===")
        
        # Check data ordering and indexing
        print(f"F-grid shape: {f_grid.shape}")
        print(f"X-grid shape: {x_grid.shape}")
        print(f"Y-grid shape: {y_grid.shape}")
        
        # Check if coordinates are monotonic
        x_monotonic = np.all(np.diff(x_grid[:, 0]) > 0)
        y_monotonic = np.all(np.diff(y_grid[0, :]) > 0)
        print(f"X coordinates monotonic: {x_monotonic}")
        print(f"Y coordinates monotonic: {y_monotonic}")
        
        # Check coordinate ranges
        print(f"X range: [{np.min(x_grid):.6f}, {np.max(x_grid):.6f}]")
        print(f"Y range: [{np.min(y_grid):.6f}, {np.max(y_grid):.6f}]")
        
        # Sample the F-field at key locations
        print("\nF-field sampling:")
        ny, nx = f_grid.shape
        corners = [
            (0, 0, "bottom-left"),
            (0, nx-1, "bottom-right"), 
            (ny-1, 0, "top-left"),
            (ny-1, nx-1, "top-right"),
            (ny//2, nx//2, "center")
        ]
        
        for i, j, label in corners:
            if i < ny and j < nx:
                print(f"  {label:12s}: F[{i:3d},{j:3d}] = {f_grid[i,j]:.6f}")
        
        # Check for potential CONREC issues
        issues = []
        
        if not x_monotonic or not y_monotonic:
            issues.append("Non-monotonic coordinates")
        
        if np.any(np.isnan(f_grid)) or np.any(np.isinf(f_grid)):
            issues.append("NaN or Inf values in F-field")
        
        if len(np.unique(f_grid)) <= 2:
            issues.append("Highly binary F-field (may need smoothing)")
        
        interface_points = np.sum((f_grid > 0.1) & (f_grid < 0.9))
        if interface_points < 10:
            issues.append(f"Very few mixed cells ({interface_points})")
        
        if issues:
            print(f"\nPotential CONREC issues:")
            for issue in issues:
                print(f"  - {issue}")
        else:
            print(f"\nNo obvious CONREC input issues detected")
        
        return issues
    
    def create_visualization(self, f_grid, x_grid, y_grid, results_dict, output_file="interface_comparison.png"):
        """
        Create comprehensive visualization comparing all methods.
        Uses LineCollection for fast rendering of large segment counts.
        """
        from matplotlib.collections import LineCollection
        
        print(f"Creating visualization with fast LineCollection rendering...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Original F-field
        im1 = axes[0,0].contourf(x_grid, y_grid, f_grid, levels=20, cmap='RdYlBu_r')
        axes[0,0].set_title('Original F-field')
        axes[0,0].set_xlabel('X')
        axes[0,0].set_ylabel('Y')
        plt.colorbar(im1, ax=axes[0,0])
        
        # Smoothed F-field (if available)
        if 'smoothed' in results_dict and results_dict['smoothed']['data'] is not None:
            f_smoothed = results_dict['smoothed']['data']
            im2 = axes[0,1].contourf(x_grid, y_grid, f_smoothed, levels=20, cmap='RdYlBu_r')
            axes[0,1].contour(x_grid, y_grid, f_smoothed, levels=[0.5], colors='red', linewidths=2)
            axes[0,1].set_title('Smoothed F-field with F=0.5 contour')
            axes[0,1].set_xlabel('X')
            axes[0,1].set_ylabel('Y')
            plt.colorbar(im2, ax=axes[0,1])
        else:
            axes[0,1].text(0.5, 0.5, 'Smoothed data\nnot available', 
                          ha='center', va='center', transform=axes[0,1].transAxes)
            axes[0,1].set_title('Smoothed F-field')
        
        # Method comparison plots
        method_names = ['smoothed', 'edge', 'plic', 'adaptive']
        method_titles = ['Smoothed Contouring', 'Edge Detection', 'PLIC Reconstruction', 'Adaptive Contouring']
        plot_positions = [(0,2), (1,0), (1,1), (1,2)]
        
        for (method, title, pos) in zip(method_names, method_titles, plot_positions):
            ax = axes[pos[0], pos[1]]
            
            if method in results_dict:
                segments = results_dict[method]['segments']
                count = len(segments)
                
                print(f"  Plotting {title}: {count} segments...")
                
                # Plot F-field as background
                ax.contourf(x_grid, y_grid, f_grid, levels=10, cmap='RdYlBu_r', alpha=0.3)
                
                # Plot extracted interface segments using LineCollection for speed
                if segments:
                    if count > 1000:
                        print(f"    Large dataset ({count} segments), using LineCollection for fast rendering...")
                        # Convert segments to line collection format
                        lines = [[(x1, y1), (x2, y2)] for (x1, y1), (x2, y2) in segments]
                        
                        # For very large datasets, sample for visualization
                        if count > 10000:
                            step = max(1, count // 5000)  # Limit to ~5000 lines for visualization
                            lines = lines[::step]
                            print(f"    Sampling every {step} segments for visualization ({len(lines)} displayed)")
                        
                        lc = LineCollection(lines, colors='red', linewidths=0.5, alpha=0.8)
                        ax.add_collection(lc)
                    else:
                        # For smaller datasets, use individual plot calls
                        for (x1, y1), (x2, y2) in segments:
                            ax.plot([x1, x2], [y1, y2], 'r-', linewidth=1, alpha=0.7)
                
                ax.set_title(f'{title}\n{count} segments')
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.grid(True, alpha=0.3)
                
                # Set axis limits to data bounds
                ax.set_xlim(np.min(x_grid), np.max(x_grid))
                ax.set_ylim(np.min(y_grid), np.max(y_grid))
                
            else:
                ax.text(0.5, 0.5, f'{title}\nNot available', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(title)
        
        plt.tight_layout()
        print(f"Saving visualization to {output_file}...")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Visualization saved to {output_file}")
        
        # Also create a detailed PLIC-only plot for comparison with ParaView
        self.create_detailed_plic_plot(f_grid, x_grid, y_grid, results_dict, output_file)
    
    def create_detailed_plic_plot(self, f_grid, x_grid, y_grid, results_dict, base_output_file):
        """
        Create a detailed PLIC-only plot for comparison with ParaView.
        """
        if 'plic' not in results_dict:
            return
            
        from matplotlib.collections import LineCollection
        
        segments = results_dict['plic']['segments']
        count = len(segments)
        
        print(f"\nCreating detailed PLIC plot with {count} segments for ParaView comparison...")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Left plot: F-field with PLIC interface overlay
        im1 = ax1.contourf(x_grid, y_grid, f_grid, levels=50, cmap='RdYlBu_r')
        plt.colorbar(im1, ax=ax1, label='Volume Fraction (F)')
        
        if segments:
            # Sample segments for visualization if too many
            if count > 5000:
                step = max(1, count // 5000)
                display_segments = segments[::step]
                print(f"  Sampling every {step} segments for detailed view ({len(display_segments)} displayed)")
            else:
                display_segments = segments
            
            lines = [[(x1, y1), (x2, y2)] for (x1, y1), (x2, y2) in display_segments]
            lc = LineCollection(lines, colors='black', linewidths=1.0, alpha=0.8)
            ax1.add_collection(lc)
        
        ax1.set_title(f'PLIC Interface Reconstruction\n{count} segments total')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.grid(True, alpha=0.3)
        
        # Right plot: Interface segments only (for clear comparison)
        ax2.set_facecolor('white')
        
        if segments:
            if count > 5000:
                lines = [[(x1, y1), (x2, y2)] for (x1, y1), (x2, y2) in display_segments]
                lc = LineCollection(lines, colors='red', linewidths=1.5, alpha=0.9)
                ax2.add_collection(lc)
            else:
                for (x1, y1), (x2, y2) in segments:
                    ax2.plot([x1, x2], [y1, y2], 'r-', linewidth=1.5, alpha=0.8)
        
        ax2.set_title(f'PLIC Interface Only\n(Compare with ParaView contour)')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(np.min(x_grid), np.max(x_grid))
        ax2.set_ylim(np.min(y_grid), np.max(y_grid))
        
        plt.tight_layout()
        
        # Save detailed PLIC plot
        plic_output = base_output_file.replace('.png', '_plic_detailed.png')
        plt.savefig(plic_output, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Detailed PLIC plot saved to {plic_output}")
        print(f"Use this to compare with ParaView's F=0.5 contour visualization")


def test_interface_extraction_on_sample_data():
    """
    Test the interface extraction methods on synthetic RT-like data.
    """
    print("Creating synthetic RT interface for testing...")
    
    # Create synthetic RT interface
    nx, ny = 100, 100
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y)
    
    # Create a wavy interface with some fragmentation
    interface_y = 0.5 + 0.05 * np.sin(10 * np.pi * X) + 0.02 * np.sin(30 * np.pi * X)
    
    # Create VOF field with some noise and fragmentation
    F = np.zeros_like(X)
    F[Y > interface_y] = 1.0
    
    # Add some fragmentation and noise
    noise = np.random.random(X.shape) * 0.1 - 0.05
    F = np.clip(F + noise, 0, 1)
    
    # Make it more binary (like typical VOF output)
    F = np.where(F > 0.5, 1.0, 0.0)
    
    # Add some mixed cells near the interface
    interface_mask = np.abs(Y - interface_y) < 0.02
    F[interface_mask] = 0.3 + 0.4 * np.random.random(np.sum(interface_mask))
    
    print(f"Created synthetic data: {nx}x{ny} grid")
    print(f"F range: [{np.min(F):.3f}, {np.max(F):.3f}]")
    print(f"Mixed cells: {np.sum((F > 0.01) & (F < 0.99))}")
    
    # Test all methods
    extractor = RTInterfaceExtractor(debug=True)
    
    # Diagnose the data
    diagnosis = extractor.diagnose_vof_data(F.T, X.T, Y.T)  # Note: transpose for correct orientation
    
    # Debug CONREC input format
    extractor.debug_conrec_input(F.T, X.T, Y.T)
    
    # Compare all methods
    best_segments, all_results = extractor.compare_all_methods(F.T, X.T, Y.T)
    
    # Create visualization
    extractor.create_visualization(F.T, X.T, Y.T, all_results, "test_interface_extraction.png")
    
    return best_segments, all_results


def read_vtk_file(vtk_file):
    """
    Read VTK rectilinear grid file and extract F, X, Y data.
    Based on your rt_analyzer.py implementation.
    """
    print(f"Reading VTK file: {vtk_file}")
    
    with open(vtk_file, 'r') as f:
        lines = f.readlines()
    
    # Extract dimensions
    for i, line in enumerate(lines):
        if "DIMENSIONS" in line:
            parts = line.strip().split()
            nx, ny, nz = int(parts[1]), int(parts[2]), int(parts[3])
            print(f"Grid dimensions: {nx} x {ny} x {nz}")
            break
    
    # Extract coordinates
    x_coords = []
    y_coords = []
    
    # Find X coordinates
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
            print(f"X coordinates: {len(x_coords)} points from {x_coords[0]:.6f} to {x_coords[-1]:.6f}")
    
    # Find Y coordinates
    for i, line in enumerate(lines):
        if "Y_COORDINATES" in line:
            parts = line.strip().split()
            n_coords = int(parts[1])
            coords_data = []
            j = i + 1
            while len(coords_data) < n_coords:
                coords_data.extend(list(map(float, lines[j].strip().split())))
                j += 1
            y_coords = np.array(coords_data)
            print(f"Y coordinates: {len(y_coords)} points from {y_coords[0]:.6f} to {y_coords[-1]:.6f}")
    
    # Extract F field data
    f_data = None
    for i, line in enumerate(lines):
        if "SCALARS F" in line:
            data_values = []
            j = i + 2  # Skip the LOOKUP_TABLE line
            while j < len(lines) and not lines[j].strip().startswith("SCALARS"):
                if lines[j].strip():  # Skip empty lines
                    data_values.extend(list(map(float, lines[j].strip().split())))
                j += 1
            f_data = np.array(data_values)
            print(f"F field: {len(f_data)} values, range [{np.min(f_data):.6f}, {np.max(f_data):.6f}]")
            break
    
    if f_data is None:
        raise ValueError("No F field found in VTK file")
    
    # Check if this is cell-centered data
    is_cell_data = any("CELL_DATA" in line for line in lines)
    print(f"Data type: {'Cell-centered' if is_cell_data else 'Point-centered'}")
    
    if is_cell_data:
        # Cell-centered data: dimensions are one less than coordinates
        nx_cells, ny_cells = nx-1, ny-1
        
        # Reshape the F data
        f_grid = f_data.reshape(ny_cells, nx_cells).T
        
        # Create cell-centered coordinates
        x_cell = 0.5 * (x_coords[:-1] + x_coords[1:])
        y_cell = 0.5 * (y_coords[:-1] + y_coords[1:])
        
        # Create 2D meshgrid
        x_grid, y_grid = np.meshgrid(x_cell, y_cell)
        x_grid = x_grid.T
        y_grid = y_grid.T
        
        print(f"Cell-centered grid: {f_grid.shape}")
        
    else:
        # Point-centered data
        x_grid, y_grid = np.meshgrid(x_coords, y_coords)
        x_grid = x_grid.T
        y_grid = y_grid.T
        f_grid = f_data.reshape(ny, nx).T
        
        print(f"Point-centered grid: {f_grid.shape}")
    
    # Extract simulation time from filename
    import re
    import os
    time_match = re.search(r'(\d+)\.vtk' , os.path.basename(vtk_file))
    sim_time = float(time_match.group(1))/1000.0 if time_match else 0.0
    print(f"Simulation time: {sim_time:.6f}")
    
    return f_grid, x_grid, y_grid, sim_time


def test_interface_extraction_on_vtk_file(vtk_file):
    """
    Test interface extraction methods on a real VTK file.
    """
    print("=" * 70)
    print(f"TESTING INTERFACE EXTRACTION ON REAL VTK FILE")
    print("=" * 70)
    
    try:
        # Read VTK file
        f_grid, x_grid, y_grid, sim_time = read_vtk_file(vtk_file)
        
        # Create extractor
        extractor = RTInterfaceExtractor(debug=True)
        
        # Diagnose the data
        print(f"\n--- DATA DIAGNOSIS ---")
        diagnosis = extractor.diagnose_vof_data(f_grid, x_grid, y_grid)
        
        # Debug CONREC input format
        issues = extractor.debug_conrec_input(f_grid, x_grid, y_grid)
        
        # Compare all methods
        print(f"\n--- TESTING ALL METHODS ---")
        best_segments, all_results = extractor.compare_all_methods(f_grid, x_grid, y_grid)
        
        # Create visualization
        output_name = f"vtk_interface_extraction_{os.path.basename(vtk_file).replace('.vtk', '')}.png"
        extractor.create_visualization(f_grid, x_grid, y_grid, all_results, output_name)
        
        print(f"\n" + "=" * 70)
        print(f"VTK FILE ANALYSIS COMPLETE")
        print(f"=" * 70)
        print(f"File: {vtk_file}")
        print(f"Time: t = {sim_time:.6f}")
        print(f"Grid: {f_grid.shape[0]} x {f_grid.shape[1]}")
        print(f"Best method extracted: {len(best_segments)} segments")
        print(f"Visualization saved: {output_name}")
        
        # Comparison with your current CONREC results
        print(f"\n--- COMPARISON WITH YOUR CURRENT RESULTS ---")
        print(f"Your CONREC segments: ~1-17 (from your description)")
        print(f"Best method segments: {len(best_segments)}")
        improvement = len(best_segments) / 10 if len(best_segments) > 10 else 1  # Rough estimate
        print(f"Potential improvement: ~{improvement:.0f}x more segments")
        
        return best_segments, all_results, f_grid, x_grid, y_grid
        
    except Exception as e:
        print(f"Error reading VTK file: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None, None


if __name__ == "__main__":
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description='RT Interface Extraction Debug Tool')
    parser.add_argument('--vtk-file', help='VTK file to analyze')
    parser.add_argument('--test-synthetic', action='store_true', 
                       help='Run test with synthetic data (default if no VTK file)')
    
    args = parser.parse_args()
    
    print("RT Interface Extraction Debug Tool")
    print("=" * 50)
    
    if args.vtk_file:
        if os.path.exists(args.vtk_file):
            # Test with real VTK file
            segments, results, f_grid, x_grid, y_grid = test_interface_extraction_on_vtk_file(args.vtk_file)
            
            if segments is not None:
                print(f"\nSUCCESS: Extracted {len(segments)} interface segments from your VTK file!")
                print("\nNext steps:")
                print("1. Check the generated visualization PNG file")
                print("2. Identify which method works best for your data")
                print("3. Integrate that method into your rt_analyzer.py")
                print("4. This should solve your CONREC fragmentation issues")
            else:
                print("\nFAILED: Could not process the VTK file")
        else:
            print(f"Error: VTK file not found: {args.vtk_file}")
            args.test_synthetic = True
    
    if args.test_synthetic or not args.vtk_file:
        # Test with synthetic data
        print("\nRunning synthetic data test...")
        segments, results = test_interface_extraction_on_sample_data()
        
        print(f"\nSynthetic test result: {len(segments)} interface segments extracted")
        print("Check 'test_interface_extraction.png' for visual comparison")
    
    print("\nRecommendations for your CONREC issues:")
    print("1. Try Method 1 (smoothed contouring) first - most robust for RT")
    print("2. If binary VOF data, apply Gaussian smoothing before CONREC")
    print("3. Check that CONREC gets properly formatted, monotonic coordinate arrays")
    print("4. Consider using Method 2 (edge detection) for highly fragmented interfaces")
    print("5. Validate CONREC input data ordering (row-major vs column-major)")
    
    if args.vtk_file:
        print(f"\nTo test with your VTK file:")
        print(f"python rt_interface_debug.py --vtk-file path/to/your/file.vtk")
