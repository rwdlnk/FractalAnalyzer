"""
Multifractal Analysis Module

This module provides advanced multifractal analysis capabilities for 
interface characterization, extracted and modernized from rt_analyzer.py.

Classes:
    MultifractalAnalyzer: Core multifractal spectrum analysis
    
Functions:
    compute_singularity_spectrum: Convert tau(q) to f(alpha) spectrum
    plot_multifractal_spectrum: Visualization utilities
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
from scipy import stats
import time
import os
from collections import defaultdict

class MultifractalAnalyzer:
    """
    Advanced multifractal analysis for interface characterization.
    
    Extracted from rt_analyzer.py and modernized for the FractalAnalyzer framework.
    """
    
    def __init__(self, debug: bool = False, use_spatial_index: bool = True):
        """
        Initialize MultifractalAnalyzer.
        
        Args:
            debug: Enable debug output
            use_spatial_index: Use spatial indexing for performance
        """
        self.debug = debug
        self.use_spatial_index = use_spatial_index
        
    def create_spatial_index(self, segments, min_x, min_y, max_x, max_y, grid_size):
        """Create spatial index for fast segment lookup."""
        grid_width = int(np.ceil((max_x - min_x) / grid_size))
        grid_height = int(np.ceil((max_y - min_y) / grid_size))
        
        segment_grid = defaultdict(list)
        
        for idx, ((x1, y1), (x2, y2)) in enumerate(segments):
            # Find grid cells this segment might touch
            seg_min_x, seg_max_x = min(x1, x2), max(x1, x2)
            seg_min_y, seg_max_y = min(y1, y2), max(y1, y2)
            
            min_cell_x = max(0, int((seg_min_x - min_x) / grid_size))
            max_cell_x = min(int((seg_max_x - min_x) / grid_size) + 1, grid_width)
            min_cell_y = max(0, int((seg_min_y - min_y) / grid_size))
            max_cell_y = min(int((seg_max_y - min_y) / grid_size) + 1, grid_height)
            
            for cell_x in range(min_cell_x, max_cell_x):
                for cell_y in range(min_cell_y, max_cell_y):
                    segment_grid[(cell_x, cell_y)].append(idx)
        
        return segment_grid, grid_width, grid_height

    def liang_barsky_line_box_intersection(self, x1, y1, x2, y2, xmin, ymin, xmax, ymax):
        """Test if line segment intersects with box using Liang-Barsky algorithm."""
        dx = x2 - x1
        dy = y2 - y1
        
        # Parametric line: P = P1 + t*(P2-P1) where t ∈ [0,1]
        p = [-dx, dx, -dy, dy]
        q = [x1 - xmin, xmax - x1, y1 - ymin, ymax - y1]
        
        t_min, t_max = 0.0, 1.0
        
        for i in range(4):
            if p[i] == 0:
                # Line is parallel to this boundary
                if q[i] < 0:
                    return False  # Line is outside
            else:
                t = q[i] / p[i]
                if p[i] < 0:
                    t_min = max(t_min, t)
                else:
                    t_max = min(t_max, t)
                
                if t_min > t_max:
                    return False
        
        return True
        
    def compute_multifractal_spectrum(self, segments: List[Tuple], 
                                    min_box_size: Optional[float] = None,
                                    q_values: Optional[List[float]] = None,
                                    output_dir: Optional[str] = None,
                                    time_value: Optional[float] = None) -> Dict:
        """
        Compute multifractal spectrum from interface segments.
        
        Args:
            segments: List of line segments as ((x1,y1), (x2,y2)) tuples
            min_box_size: Minimum box size for analysis (default: auto-estimate)
            q_values: List of q moments to analyze (default: -5 to 5 in 0.5 steps)
            output_dir: Directory to save results (default: None)
            time_value: Time value for labeling plots (default: None)
            
        Returns:
            dict: Multifractal spectrum results
        """
        if not segments:
            print("No interface segments provided. Skipping multifractal analysis.")
            return None
        
        # Set default q values if not provided
        if q_values is None:
            q_values = np.arange(-5, 5.1, 0.5)
        q_values = np.array(q_values)
        
        print(f"Performing multifractal analysis with {len(q_values)} q-values")
        print(f"Using {len(segments)} interface segments")
        
        # Create output directory if specified
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Calculate extent for max box size
        min_x = min(min(s[0][0], s[1][0]) for s in segments)
        max_x = max(max(s[0][0], s[1][0]) for s in segments)
        min_y = min(min(s[0][1], s[1][1]) for s in segments)
        max_y = max(max(s[0][1], s[1][1]) for s in segments)
        
        extent = max(max_x - min_x, max_y - min_y)
        max_box_size = extent / 2
        
        # Auto-estimate min_box_size if not provided
        if min_box_size is None:
            lengths = [np.sqrt((s[1][0]-s[0][0])**2 + (s[1][1]-s[0][1])**2) for s in segments]
            avg_length = np.mean(lengths)
            min_box_size = avg_length * 2
            print(f"Auto-estimated min_box_size: {min_box_size:.6f}")
        
        print(f"Box size range: {min_box_size:.6f} to {max_box_size:.6f}")
        
        # Generate box sizes
        box_sizes = []
        current_size = max_box_size
        box_size_factor = 1.5
        
        while current_size >= min_box_size:
            box_sizes.append(current_size)
            current_size /= box_size_factor
            
        box_sizes = np.array(box_sizes)
        num_box_sizes = len(box_sizes)
        
        print(f"Using {num_box_sizes} box sizes for analysis")
        
        # Add small margin to bounding box
        margin = extent * 0.01
        min_x -= margin
        max_x += margin
        min_y -= margin
        max_y += margin
        
        # Create spatial index for segments
        start_time = time.time()
        print("Creating spatial index...")
        
        grid_size = min_box_size * 2
        segment_grid, grid_width, grid_height = self.create_spatial_index(
            segments, min_x, min_y, max_x, max_y, grid_size)
        
        print(f"Spatial index created in {time.time() - start_time:.2f} seconds")
        
        # Initialize data structures for box counting
        all_box_counts = []
        all_probabilities = []
        
        # Analyze each box size
        for box_idx, box_size in enumerate(box_sizes):
            box_start_time = time.time()
            print(f"Processing box size {box_idx+1}/{num_box_sizes}: {box_size:.6f}")
            
            num_boxes_x = int(np.ceil((max_x - min_x) / box_size))
            num_boxes_y = int(np.ceil((max_y - min_y) / box_size))
            
            # Count segments in each box
            box_counts = np.zeros((num_boxes_x, num_boxes_y))
            
            for i in range(num_boxes_x):
                for j in range(num_boxes_y):
                    box_xmin = min_x + i * box_size
                    box_ymin = min_y + j * box_size
                    box_xmax = box_xmin + box_size
                    box_ymax = box_ymin + box_size
                    
                    # Find grid cells that might overlap this box
                    min_cell_x = max(0, int((box_xmin - min_x) / grid_size))
                    max_cell_x = min(int((box_xmax - min_x) / grid_size) + 1, grid_width)
                    min_cell_y = max(0, int((box_ymin - min_y) / grid_size))
                    max_cell_y = min(int((box_ymax - min_y) / grid_size) + 1, grid_height)
                    
                    # Get segments that might intersect this box
                    segments_to_check = set()
                    for cell_x in range(min_cell_x, max_cell_x):
                        for cell_y in range(min_cell_y, max_cell_y):
                            segments_to_check.update(segment_grid.get((cell_x, cell_y), []))
                    
                    # Count intersections
                    count = 0
                    for seg_idx in segments_to_check:
                        (x1, y1), (x2, y2) = segments[seg_idx]
                        if self.liang_barsky_line_box_intersection(
                                x1, y1, x2, y2, box_xmin, box_ymin, box_xmax, box_ymax):
                            count += 1
                    
                    box_counts[i, j] = count
            
            # Keep only non-zero counts and calculate probabilities
            occupied_boxes = box_counts[box_counts > 0]
            total_segments = occupied_boxes.sum()
            
            if total_segments > 0:
                probabilities = occupied_boxes / total_segments
            else:
                probabilities = np.array([])
                
            all_box_counts.append(occupied_boxes)
            all_probabilities.append(probabilities)
            
            # Report statistics
            box_count = len(occupied_boxes)
            print(f"  Box size: {box_size:.6f}, Occupied boxes: {box_count}, Time: {time.time() - box_start_time:.2f}s")
        
        # Calculate multifractal properties
        print("Calculating multifractal spectrum...")
        
        taus = np.zeros(len(q_values))
        Dqs = np.zeros(len(q_values))
        r_squared = np.zeros(len(q_values))
        
        for q_idx, q in enumerate(q_values):
            print(f"Processing q = {q:.1f}")
            
            # Skip q=1 as it requires special treatment
            if abs(q - 1.0) < 1e-6:
                continue
                
            # Calculate partition function for each box size
            Z_q = np.zeros(num_box_sizes)
            
            for i, probabilities in enumerate(all_probabilities):
                if len(probabilities) > 0:
                    Z_q[i] = np.sum(probabilities ** q)
                else:
                    Z_q[i] = np.nan
            
            # Remove NaN values
            valid = ~np.isnan(Z_q)
            if np.sum(valid) < 3:
                print(f"Warning: Not enough valid points for q={q}")
                taus[q_idx] = np.nan
                Dqs[q_idx] = np.nan
                r_squared[q_idx] = np.nan
                continue
                
            log_eps = np.log(box_sizes[valid])
            log_Z_q = np.log(Z_q[valid])
            
            # Linear regression to find tau(q)
            slope, intercept, r_value, p_value, std_err = stats.linregress(log_eps, log_Z_q)
            
            # Calculate tau(q) and D(q)
            taus[q_idx] = slope
            Dqs[q_idx] = taus[q_idx] / (q - 1) if q != 1 else np.nan
            r_squared[q_idx] = r_value ** 2
            
            print(f"  τ({q}) = {taus[q_idx]:.4f}, D({q}) = {Dqs[q_idx]:.4f}, R² = {r_squared[q_idx]:.4f}")
        
        # Handle q=1 case (information dimension) separately
        q1_idx = np.where(np.abs(q_values - 1.0) < 1e-6)[0]
        if len(q1_idx) > 0:
            q1_idx = q1_idx[0]
            print(f"Processing q = 1.0 (information dimension)")
            
            # Calculate using L'Hôpital's rule
            mu_log_mu = np.zeros(num_box_sizes)
            
            for i, probabilities in enumerate(all_probabilities):
                if len(probabilities) > 0:
                    # Use -sum(p*log(p)) for information dimension
                    mu_log_mu[i] = -np.sum(probabilities * np.log(probabilities))
                else:
                    mu_log_mu[i] = np.nan
            
            # Remove NaN values
            valid = ~np.isnan(mu_log_mu)
            if np.sum(valid) >= 3:
                log_eps = np.log(box_sizes[valid])
                log_mu = mu_log_mu[valid]
                
                # Linear regression
                slope, intercept, r_value, p_value, std_err = stats.linregress(log_eps, log_mu)
                
                # Store information dimension
                taus[q1_idx] = -slope  # Convention: τ(1) = -D₁
                Dqs[q1_idx] = -slope   # Information dimension D₁
                r_squared[q1_idx] = r_value ** 2
                
                print(f"  τ(1) = {taus[q1_idx]:.4f}, D(1) = {Dqs[q1_idx]:.4f}, R² = {r_squared[q1_idx]:.4f}")
        
        # Calculate alpha and f(alpha) for multifractal spectrum
        alpha = np.zeros(len(q_values))
        f_alpha = np.zeros(len(q_values))
        
        print("Calculating multifractal spectrum f(α)...")
        
        for i, q in enumerate(q_values):
            if np.isnan(taus[i]):
                alpha[i] = np.nan
                f_alpha[i] = np.nan
                continue
                
            # Numerical differentiation for alpha
            if i > 0 and i < len(q_values) - 1:
                alpha[i] = -(taus[i+1] - taus[i-1]) / (q_values[i+1] - q_values[i-1])
            elif i == 0:
                alpha[i] = -(taus[i+1] - taus[i]) / (q_values[i+1] - q_values[i])
            else:
                alpha[i] = -(taus[i] - taus[i-1]) / (q_values[i] - q_values[i-1])
            
            # Calculate f(alpha)
            f_alpha[i] = q * alpha[i] + taus[i]
            
            print(f"  q = {q:.1f}, α = {alpha[i]:.4f}, f(α) = {f_alpha[i]:.4f}")
        
        # Calculate multifractal parameters
        valid_idx = ~np.isnan(Dqs)
        if np.sum(valid_idx) >= 3:
            D0 = Dqs[np.searchsorted(q_values, 0)] if 0 in q_values else np.nan
            D1 = Dqs[np.searchsorted(q_values, 1)] if 1 in q_values else np.nan
            D2 = Dqs[np.searchsorted(q_values, 2)] if 2 in q_values else np.nan
            
            # Width of multifractal spectrum
            valid = ~np.isnan(alpha)
            if np.sum(valid) >= 2:
                alpha_width = np.max(alpha[valid]) - np.min(alpha[valid])
            else:
                alpha_width = np.nan

            # Calculate degree of multifractality using available q-values
            valid_idx = ~np.isnan(Dqs)
            if np.sum(valid_idx) >= 3:
                # Use extreme available q-values
                q_min_idx = np.where(q_values == np.min(q_values[valid_idx]))[0][0]
                q_max_idx = np.where(q_values == np.max(q_values[valid_idx]))[0][0]
                degree_multifractality = Dqs[q_min_idx] - Dqs[q_max_idx]
                print(f"  Degree of multifractality (D({q_values[q_min_idx]}) - D({q_values[q_max_idx]})): {degree_multifractality:.4f}")
            else:
                degree_multifractality = np.nan
                print("  Warning: Insufficient valid q-values for degree calculation")
            
            print(f"Multifractal parameters:")
            print(f"  D(0) = {D0:.4f} (capacity dimension)")
            print(f"  D(1) = {D1:.4f} (information dimension)")
            print(f"  D(2) = {D2:.4f} (correlation dimension)")
            print(f"  α width = {alpha_width:.4f}")
            print(f"  Degree of multifractality = {degree_multifractality:.4f}")
        else:
            D0 = D1 = D2 = alpha_width = degree_multifractality = np.nan
            print("Warning: Not enough valid points to calculate multifractal parameters")
        
        # Plot results if output directory provided
        if output_dir:
            self._create_multifractal_plots(q_values, Dqs, alpha, f_alpha, r_squared, 
                                          D0, output_dir, time_value)
            self._save_multifractal_results(q_values, taus, Dqs, alpha, f_alpha, r_squared,
                                          D0, D1, D2, alpha_width, degree_multifractality,
                                          output_dir, time_value)
        
        # Return results
        return {
            'q_values': q_values,
            'tau': taus,
            'Dq': Dqs,
            'alpha': alpha,
            'f_alpha': f_alpha,
            'r_squared': r_squared,
            'D0': D0,
            'D1': D1,
            'D2': D2,
            'alpha_width': alpha_width,
            'degree_multifractality': degree_multifractality,
            'time': time_value
        }

    def _create_multifractal_plots(self, q_values, Dqs, alpha, f_alpha, r_squared, 
                                 D0, output_dir, time_value):
        """Create multifractal analysis plots."""
        time_str = f" at t = {time_value:.2f}" if time_value is not None else ""
        
        # Plot D(q) vs q
        plt.figure(figsize=(10, 6))
        valid = ~np.isnan(Dqs)
        plt.plot(q_values[valid], Dqs[valid], 'bo-', markersize=4)
        
        if not np.isnan(D0):
            plt.axhline(y=D0, color='r', linestyle='--', 
                       label=f"D(0) = {D0:.4f}")
        
        plt.xlabel('q')
        plt.ylabel('D(q)')
        plt.title(f'Generalized Dimensions D(q){time_str}')
        plt.grid(True)
        if not np.isnan(D0):
            plt.legend()
        plt.savefig(os.path.join(output_dir, "multifractal_dimensions.png"), dpi=300)
        plt.close()
        
        # Plot f(alpha) vs alpha (multifractal spectrum)
        plt.figure(figsize=(10, 6))
        valid = ~np.isnan(alpha) & ~np.isnan(f_alpha)
        plt.plot(alpha[valid], f_alpha[valid], 'bo-', markersize=4)
        
        # Add selected q values as annotations
        q_to_highlight = [-5, -2, 0, 2, 5]
        for q_val in q_to_highlight:
            if q_val in q_values:
                idx = np.searchsorted(q_values, q_val)
                if idx < len(q_values) and valid[idx]:
                    plt.annotate(f"q={q_values[idx]}", 
                                (alpha[idx], f_alpha[idx]),
                                xytext=(5, 0), textcoords='offset points')
        
        plt.xlabel('α')
        plt.ylabel('f(α)')
        plt.title(f'Multifractal Spectrum f(α){time_str}')
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, "multifractal_spectrum.png"), dpi=300)
        plt.close()
        
        # Plot R² values
        plt.figure(figsize=(10, 6))
        valid = ~np.isnan(r_squared)
        plt.plot(q_values[valid], r_squared[valid], 'go-', markersize=4)
        plt.xlabel('q')
        plt.ylabel('R²')
        plt.title(f'Fit Quality for Different q Values{time_str}')
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, "multifractal_r_squared.png"), dpi=300)
        plt.close()

    def _save_multifractal_results(self, q_values, taus, Dqs, alpha, f_alpha, r_squared,
                                 D0, D1, D2, alpha_width, degree_multifractality,
                                 output_dir, time_value):
        """Save multifractal results to CSV files."""
        # Save detailed results
        results_df = pd.DataFrame({
            'q': q_values,
            'tau': taus,
            'Dq': Dqs,
            'alpha': alpha,
            'f_alpha': f_alpha,
            'r_squared': r_squared
        })
        results_df.to_csv(os.path.join(output_dir, "multifractal_results.csv"), index=False)
        
        # Save multifractal parameters
        params_df = pd.DataFrame({
            'Parameter': ['Time', 'D0', 'D1', 'D2', 'alpha_width', 'degree_multifractality'],
            'Value': [time_value if time_value is not None else np.nan, 
                     D0, D1, D2, alpha_width, degree_multifractality]
        })
        params_df.to_csv(os.path.join(output_dir, "multifractal_parameters.csv"), index=False)

    def print_multifractal_summary(self, mf_results):
        """Print a nice summary of multifractal results."""
        if not mf_results:
            print("No multifractal results to display")
            return
            
        print(f"\n📊 MULTIFRACTAL ANALYSIS SUMMARY")
        print(f"=" * 50)
        if mf_results.get('time') is not None:
            print(f"Time: {mf_results['time']:.3f}")
        print(f"")
        print(f"Generalized Dimensions:")
        print(f"  D(0) = {mf_results['D0']:.4f} (Capacity dimension)")
        print(f"  D(1) = {mf_results['D1']:.4f} (Information dimension)")  
        print(f"  D(2) = {mf_results['D2']:.4f} (Correlation dimension)")
        print(f"")
        print(f"Multifractal Properties:")
        print(f"  α width = {mf_results['alpha_width']:.4f}")
        print(f"  Degree of multifractality = {mf_results['degree_multifractality']:.4f}")
        print(f"")
        
        # Interpretation
        if mf_results['degree_multifractality'] > 0.1:
            print(f"  🔍 Interface shows multifractal behavior")
        else:
            print(f"  📏 Interface appears monofractal")
            
        if mf_results['D0'] > 1.8:
            print(f"  🌊 Highly complex, space-filling interface")
        elif mf_results['D0'] > 1.5:
            print(f"  🌀 Moderately complex interface")
        else:
            print(f"  📐 Relatively smooth interface")

    def analyze_multifractal_evolution(self, segments_data: Dict, output_dir: Optional[str] = None, 
                                     q_values: Optional[List[float]] = None) -> List[Dict]:
        """
        Analyze how multifractal properties evolve over time or across resolutions.
        
        Args:
            segments_data: Dict mapping times/resolutions to segments data
                         e.g. {0.1: segments_list, 0.2: segments_list} for time series
                         or {100: segments_list, 200: segments_list} for resolutions
            output_dir: Directory to save results
            q_values: List of q moments to analyze (default: -5 to 5 in 0.5 steps)
            
        Returns:
            List[Dict]: Multifractal evolution results
        """
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Set default q values if not provided
        if q_values is None:
            q_values = np.arange(-5, 5.1, 0.5)
        
        # Determine type of analysis (time or resolution)
        keys = list(segments_data.keys())
        is_time_series = all(isinstance(k, (int, float)) for k in keys)
        
        if is_time_series:
            print(f"Analyzing multifractal evolution over time/resolution series: {sorted(keys)}")
            x_label = 'Time/Resolution'
            series_name = "parameter"
        else:
            print(f"Analyzing multifractal evolution across parameters: {sorted(keys)}")
            x_label = 'Parameter'
            series_name = "parameter"
        
        # Initialize results storage
        results = []
        
        # Process each segments dataset
        for key, segments in sorted(segments_data.items()):
            print(f"\nProcessing {series_name} = {key}")
            
            try:
                # Create subdirectory for this point
                if output_dir:
                    point_dir = os.path.join(output_dir, f"{series_name}_{key}")
                    os.makedirs(point_dir, exist_ok=True)
                else:
                    point_dir = None
                
                # Perform multifractal analysis
                mf_results = self.compute_multifractal_spectrum(
                    segments, q_values=q_values, output_dir=point_dir, time_value=key
                )
                
                if mf_results:
                    # Store results with the key (time or resolution)
                    mf_results[series_name] = key
                    results.append(mf_results)
                
            except Exception as e:
                print(f"Error processing {series_name}={key}: {str(e)}")
                import traceback
                traceback.print_exc()
        
        # Create summary plots
        if results and output_dir:
            self._create_evolution_plots(results, series_name, x_label, output_dir, q_values)
            self._save_evolution_summary(results, series_name, output_dir)
        
        return results
    
    def _create_evolution_plots(self, results: List[Dict], series_name: str, 
                              x_label: str, output_dir: str, q_values: np.ndarray):
        """Create evolution analysis plots."""
        # Extract evolution of key parameters
        x_values = [res[series_name] for res in results]
        D0_values = [res['D0'] for res in results]
        D1_values = [res['D1'] for res in results]
        D2_values = [res['D2'] for res in results]
        alpha_width = [res['alpha_width'] for res in results]
        degree_mf = [res['degree_multifractality'] for res in results]
        
        # Plot generalized dimensions evolution
        plt.figure(figsize=(10, 6))
        plt.plot(x_values, D0_values, 'bo-', label='D(0) - Capacity dimension')
        plt.plot(x_values, D1_values, 'ro-', label='D(1) - Information dimension')
        plt.plot(x_values, D2_values, 'go-', label='D(2) - Correlation dimension')
        plt.xlabel(x_label)
        plt.ylabel('Generalized Dimensions')
        plt.title(f'Evolution of Generalized Dimensions with {x_label}')
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(output_dir, "dimensions_evolution.png"), dpi=300)
        plt.close()
        
        # Plot multifractal parameters evolution
        plt.figure(figsize=(10, 6))
        plt.plot(x_values, alpha_width, 'ms-', label='α width')
        plt.plot(x_values, degree_mf, 'cd-', label='Degree of multifractality')
        plt.xlabel(x_label)
        plt.ylabel('Parameter Value')
        plt.title(f'Evolution of Multifractal Parameters with {x_label}')
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(output_dir, "multifractal_params_evolution.png"), dpi=300)
        plt.close()
        
        # Create 3D surface plot of D(q) evolution if possible
        try:
            from mpl_toolkits.mplot3d import Axes3D
            
            # Prepare data for 3D plot
            X, Y = np.meshgrid(x_values, q_values)
            Z = np.zeros((len(q_values), len(x_values)))
            
            for i, result in enumerate(results):
                for j, q in enumerate(q_values):
                    q_idx = np.where(result['q_values'] == q)[0]
                    if len(q_idx) > 0:
                        Z[j, i] = result['Dq'][q_idx[0]]
            
            # Create 3D plot
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection='3d')
            surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.8)
            
            ax.set_xlabel(x_label)
            ax.set_ylabel('q')
            ax.set_zlabel('D(q)')
            ax.set_title(f'Evolution of D(q) Spectrum with {x_label}')
            
            fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='D(q)')
            plt.savefig(os.path.join(output_dir, "Dq_evolution_3D.png"), dpi=300)
            plt.close()
            
        except Exception as e:
            print(f"Error creating 3D plot: {str(e)}")
    
    def _save_evolution_summary(self, results: List[Dict], series_name: str, output_dir: str):
        """Save evolution analysis summary."""
        x_values = [res[series_name] for res in results]
        D0_values = [res['D0'] for res in results]
        D1_values = [res['D1'] for res in results]
        D2_values = [res['D2'] for res in results]
        alpha_width = [res['alpha_width'] for res in results]
        degree_mf = [res['degree_multifractality'] for res in results]
        
        summary_df = pd.DataFrame({
            series_name: x_values,
            'D0': D0_values,
            'D1': D1_values,
            'D2': D2_values,
            'alpha_width': alpha_width,
            'degree_multifractality': degree_mf
        })
        summary_df.to_csv(os.path.join(output_dir, "multifractal_evolution_summary.csv"), index=False)
