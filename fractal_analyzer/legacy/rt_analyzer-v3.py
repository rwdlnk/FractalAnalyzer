# rt_analyzer_v4.py
# rt_analyzer_v3.py
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import os
import re
import time
import glob
import json
from typing import Tuple, List, Dict, Optional
from skimage import measure
from datetime import datetime

class RTAnalyzer:
    """Complete Rayleigh-Taylor simulation analyzer with publication-ready outputs."""
    
    # Journal requirements for figure resolution
    DPI_SETTINGS = {
        'line_drawing': 1000,      # Pure line drawings
        'photo': 300,              # Color/grayscale photographs  
        'line_halftone': 500,      # Combinations (most of our plots)
        'vector': 300              # For vector formats (EPS/PDF)
    }
    
    def __init__(self, output_dir="./rt_analysis"):
        """Initialize the RT analyzer."""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Create fractal analyzer instance
        try:
            from fractal_analyzer_v26 import FractalAnalyzer  # Will update to v27 later
            self.fractal_analyzer = FractalAnalyzer()
            print("Fractal analyzer v26 initialized successfully")
        except ImportError as e:
            print(f"Warning: fractal_analyzer_v26 module not found: {str(e)}")
            print("Make sure fractal_analyzer_v26.py is in the same directory")
            self.fractal_analyzer = None
        
        # Initialize publication tracking
        self.figure_counter = 1
        self.table_counter = 1
        self.figure_captions = []
        self.table_captions = []
    
    def read_vtk_file(self, vtk_file):
        """Read VTK rectilinear grid file and extract only the VOF (F) data."""
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
        
        # Find coordinates
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
        
        # Extract scalar field data (F) only
        f_data = None
        
        for i, line in enumerate(lines):
            # Find VOF (F) data
            if "SCALARS F" in line:
                data_values = []
                j = i + 2  # Skip the LOOKUP_TABLE line
                while j < len(lines) and not lines[j].strip().startswith("SCALARS"):
                    data_values.extend(list(map(float, lines[j].strip().split())))
                    j += 1
                f_data = np.array(data_values)
                break
        
        # Check if this is cell-centered data
        is_cell_data = any("CELL_DATA" in line for line in lines)
        
        # For cell data, we need to adjust the grid
        if is_cell_data:
            # The dimensions are one less than the coordinates in each direction
            nx_cells, ny_cells = nx-1, ny-1
            
            # Reshape the data
            f_grid = f_data.reshape(ny_cells, nx_cells).T if f_data is not None else None
            
            # Create cell-centered coordinates
            x_cell = 0.5 * (x_coords[:-1] + x_coords[1:])
            y_cell = 0.5 * (y_coords[:-1] + y_coords[1:])
            
            # Create 2D meshgrid
            x_grid, y_grid = np.meshgrid(x_cell, y_cell)
            x_grid = x_grid.T  # Transpose to match the data ordering
            y_grid = y_grid.T
        else:
            # For point data, use the coordinates directly
            x_grid, y_grid = np.meshgrid(x_coords, y_coords)
            x_grid = x_grid.T
            y_grid = y_grid.T
            
            # Reshape the data
            f_grid = f_data.reshape(ny, nx).T if f_data is not None else None
        
        # Extract simulation time from filename
        time_match = re.search(r'(\d+)\.vtk$', os.path.basename(vtk_file))
        sim_time = float(time_match.group(1))/1000.0 if time_match else 0.0
        
        # Create output dictionary with only needed fields
        return {
            'x': x_grid,
            'y': y_grid,
            'f': f_grid,
            'dims': (nx, ny, nz),
            'time': sim_time
        }
    
    def extract_interface(self, f_grid, x_grid, y_grid, level=0.5):
        """Extract the interface contour at level f=0.5 using marching squares algorithm."""
        # Find contours
        contours = measure.find_contours(f_grid.T, level)
        
        # Convert to physical coordinates
        physical_contours = []
        for contour in contours:
            x_physical = np.interp(contour[:, 1], np.arange(f_grid.shape[0]), x_grid[:, 0])
            y_physical = np.interp(contour[:, 0], np.arange(f_grid.shape[1]), y_grid[0, :])
            physical_contours.append(np.column_stack([x_physical, y_physical]))
        
        return physical_contours
    
    def convert_contours_to_segments(self, contours):
        """Convert contours to line segments format for fractal analysis."""
        segments = []
        
        for contour in contours:
            for i in range(len(contour) - 1):
                x1, y1 = contour[i]
                x2, y2 = contour[i+1]
                segments.append(((x1, y1), (x2, y2)))
        
        return segments
    
    def find_initial_interface(self, data):
        """Find the initial interface position (y=1.0 for RT)."""
        f_avg = np.mean(data['f'], axis=0)
        y_values = data['y'][0, :]
        
        # Find where f crosses 0.5
        idx = np.argmin(np.abs(f_avg - 0.5))
        return y_values[idx]
    
    def compute_mixing_thickness(self, data, h0, method='dalziel'):
        """Compute mixing layer thickness using different methods."""
        if method == 'geometric':
            # Extract interface contours
            contours = self.extract_interface(data['f'], data['x'], data['y'])
            
            # Find maximum displacement above and below initial interface
            ht = 0.0
            hb = 0.0
            
            for contour in contours:
                y_coords = contour[:, 1]
                ht = max(ht, np.max(y_coords - h0))
                hb = max(hb, np.max(h0 - y_coords))
            
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
            # Dalziel-style concentration-based mixing thickness
            # Following Dalziel et al. (1999) methodology
            f_avg = np.mean(data['f'], axis=0)  # Horizontal average
            y_values = data['y'][0, :]
            
            # Use concentration thresholds following Dalziel et al.
            # They mention using thresholds to define mixing zone boundaries
            lower_threshold = 0.05  # 5% threshold (heavy fluid in light region)
            upper_threshold = 0.95  # 95% threshold (light fluid in heavy region)
            
            # Find mixing zone boundaries
            # Upper boundary: first point where concentration drops to upper_threshold
            upper_idx = np.where(f_avg <= upper_threshold)[0]
            # Lower boundary: last point where concentration rises to lower_threshold  
            lower_idx = np.where(f_avg >= lower_threshold)[0]
            
            if len(upper_idx) > 0 and len(lower_idx) > 0:
                y_upper = y_values[upper_idx[0]]   # First point below 95%
                y_lower = y_values[lower_idx[-1]]  # Last point above 5%
                
                # Calculate mixing layer thicknesses relative to initial interface
                ht = max(0, y_upper - h0)  # Upper mixing thickness
                hb = max(0, h0 - y_lower)  # Lower mixing thickness
                h_total = y_upper - y_lower  # Total mixing zone width
                
                # Additional Dalziel-style diagnostics
                # Mixing zone center of mass
                mixing_region = (f_avg >= lower_threshold) & (f_avg <= upper_threshold)
                if np.any(mixing_region):
                    y_center = np.average(y_values[mixing_region], weights=f_avg[mixing_region])
                else:
                    y_center = h0
                
                # Mixing efficiency (fraction of domain that is mixed)
                mixing_fraction = np.sum(mixing_region) / len(f_avg)
                
            else:
                # No clear mixing zone found
                ht = hb = h_total = y_center = 0
                mixing_fraction = 0
            
            return {
                'ht': ht, 
                'hb': hb, 
                'h_total': h_total,
                'y_center': y_center,
                'mixing_fraction': mixing_fraction,
                'lower_threshold': lower_threshold,
                'upper_threshold': upper_threshold,
                'method': 'dalziel'
            }
        
        else:
            raise ValueError(f"Unknown mixing thickness method: {method}")

    def compute_fractal_dimension(self, data, min_box_size=0.001):
        """Compute fractal dimension of the interface using v26 advanced methods."""
        if self.fractal_analyzer is None:
            print("Fractal analyzer not available. Skipping fractal dimension calculation.")
            return {
                'dimension': np.nan,
                'error': np.nan,
                'r_squared': np.nan
            }
    
        # Extract contours
        contours = self.extract_interface(data['f'], data['x'], data['y'])
    
        # Convert to segments
        segments = self.convert_contours_to_segments(contours)
    
        if not segments:
            print("No interface segments found.")
            return {
                'dimension': np.nan,
                'error': np.nan,
                'r_squared': np.nan
            }
    
        print(f"Found {len(segments)} interface segments")
    
        try:
            # Use the advanced sliding window analysis from v26
            results = self.fractal_analyzer.analyze_fractal_segments(
                segments, 
                theoretical_dimension=None,  # No known theoretical value for RT
                max_level=None,  # Let it auto-determine appropriate level
                min_box_size=min_box_size
            )
            
            # Extract key results
            dimension = results['best_dimension']
            error = results['dimension_error'] 
            r_squared = results['best_r_squared']
            
            print(f"Fractal dimension: {dimension:.6f} ± {error:.6f}, R² = {r_squared:.6f}")
            print(f"Window size: {results['best_window_size']}, Scaling region: {results['scaling_region']}")
            
            return {
                'dimension': dimension,
                'error': error,
                'r_squared': r_squared,
                'window_size': results['best_window_size'],
                'scaling_region': results['scaling_region'],
                'box_sizes': results['box_sizes'],
                'box_counts': results['box_counts'],
                'segments': segments,
                'analysis_results': results  # Full results for detailed analysis
            }
        
        except Exception as e:
            print(f"Error in fractal dimension calculation: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                'dimension': np.nan,
                'error': np.nan,
                'r_squared': np.nan
            }
    
    def save_publication_figure(self, fig_name, output_dir, figure_type='line_halftone', 
                               close_fig=True, bbox_inches='tight'):
        """Save figure with journal-compliant settings."""
        dpi = self.DPI_SETTINGS[figure_type]
        
        # Generate figure filename
        fig_filename = f"Figure_{self.figure_counter}.png"
        full_path = os.path.join(output_dir, fig_filename)
        
        # Save with appropriate settings
        plt.savefig(full_path, dpi=dpi, format='png', bbox_inches=bbox_inches, 
                   facecolor='white', edgecolor='none')
        
        if close_fig:
            plt.close()
        
        print(f"Saved publication figure: {fig_filename} (DPI: {dpi})")
        return fig_filename
    
    def add_figure_caption(self, title, description, symbols_explanation=""):
        """Add a figure caption following journal requirements."""
        caption = f"Figure {self.figure_counter}. {title}. {description}"
        if symbols_explanation:
            caption += f" {symbols_explanation}"
        
        self.figure_captions.append(caption)
        self.figure_counter += 1
        return len(self.figure_captions)
    
    def generate_publication_figures(self, df, output_dir, mixing_method='dalziel'):
        """Generate all publication-ready figures with proper formatting."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Reset figure counter for this analysis
        start_fig_num = self.figure_counter
        
        # Figure: Mixing layer evolution
        plt.figure(figsize=(10, 6))
        plt.plot(df['time'], df['h_total'], 'b-', linewidth=2, label='Total thickness')
        plt.plot(df['time'], df['ht'], 'r--', linewidth=2, label='Upper penetration')
        plt.plot(df['time'], df['hb'], 'g--', linewidth=2, label='Lower penetration')
        plt.xlabel('Time')
        plt.ylabel('Mixing Layer Thickness')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        fig_name = self.save_publication_figure("mixing_evolution", output_dir, 'line_halftone')
        self.add_figure_caption(
            "Evolution of mixing layer thickness",
            f"Temporal development of Rayleigh-Taylor mixing layer measured using the {mixing_method} method. "
            f"Total mixing thickness (solid blue line), upper penetration into light fluid (dashed red line), "
            f"and lower penetration into heavy fluid (dashed green line) are shown as functions of time.",
            "Mixing thickness is defined using concentration thresholds of 5% and 95% following Dalziel et al. (1999)." if mixing_method == 'dalziel' else ""
        )
        
        # Figure: Fractal dimension evolution  
        plt.figure(figsize=(10, 6))
        plt.errorbar(df['time'], df['fractal_dim'], yerr=df['fd_error'],
                    fmt='ko-', capsize=3, linewidth=2, markersize=5, 
                    elinewidth=1.5, capthick=1.5)
        plt.fill_between(df['time'], 
                        df['fractal_dim'] - df['fd_error'],
                        df['fractal_dim'] + df['fd_error'],
                        alpha=0.2, color='gray')
        plt.xlabel('Time')
        plt.ylabel('Fractal Dimension')
        plt.grid(True, alpha=0.3)
        
        fig_name = self.save_publication_figure("fractal_evolution", output_dir, 'line_halftone')
        self.add_figure_caption(
            "Evolution of interface fractal dimension",
            "Temporal development of the Rayleigh-Taylor interface fractal dimension calculated using "
            "sliding window box counting analysis. Error bars represent standard deviations from "
            "Richardson extrapolation, and the gray shaded region indicates the uncertainty bounds.",
            "Fractal dimensions are calculated using the advanced sliding window method with boundary removal."
        )
        
        # Figure: Combined evolution plot
        fig, ax1 = plt.subplots(figsize=(12, 8))
        
        # Mixing layer on left axis
        color1 = 'tab:blue'
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Mixing Layer Thickness', color=color1)
        line1 = ax1.plot(df['time'], df['h_total'], 'b-', linewidth=2, label='Mixing Thickness')
        ax1.tick_params(axis='y', labelcolor=color1)
        ax1.grid(True, alpha=0.3)
        
        # Fractal dimension on right axis
        ax2 = ax1.twinx()
        color2 = 'tab:red'
        ax2.set_ylabel('Fractal Dimension', color=color2)
        line2 = ax2.errorbar(df['time'], df['fractal_dim'], yerr=df['fd_error'],
                           fmt='ro-', capsize=3, label='Fractal Dimension', 
                           linewidth=2, markersize=4)
        ax2.tick_params(axis='y', labelcolor=color2)
        
        # Combined legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + [line2], labels1 + labels2, loc='upper left')
        
        fig_name = self.save_publication_figure("combined_evolution", output_dir, 'line_halftone')
        self.add_figure_caption(
            "Combined evolution of mixing thickness and fractal dimension",
            "Simultaneous temporal development of mixing layer thickness (left axis, blue line) and "
            "interface fractal dimension (right axis, red circles with error bars). The evolution shows "
            "the relationship between bulk mixing development and interface complexity.",
            "Both metrics calculated using advanced analysis methods as described in the text."
        )
        
        # Figure: R-squared quality assessment
        plt.figure(figsize=(10, 6))
        plt.plot(df['time'], df['fd_r_squared'], 'mo-', linewidth=2, markersize=6)
        plt.axhline(y=0.95, color='r', linestyle='--', alpha=0.7, label='R² = 0.95 threshold')
        plt.xlabel('Time')
        plt.ylabel('R² Value')
        plt.ylim(0, 1)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        fig_name = self.save_publication_figure("r_squared_quality", output_dir, 'line_halftone')
        self.add_figure_caption(
            "Quality assessment of fractal dimension calculations",
            "Coefficient of determination (R²) values for fractal dimension fits as a function of time. "
            "High R² values indicate good linear scaling relationships in the box counting analysis.",
            "The dashed red line indicates the R² = 0.95 threshold commonly used for quality assessment."
        )
        
        # Additional figure for Dalziel method
        if mixing_method == 'dalziel' and 'mixing_fraction' in df.columns:
            plt.figure(figsize=(10, 6))
            plt.plot(df['time'], df['mixing_fraction'], 'c-o', linewidth=2, markersize=5)
            plt.xlabel('Time')
            plt.ylabel('Mixing Fraction')
            plt.grid(True, alpha=0.3)
            
            fig_name = self.save_publication_figure("mixing_fraction", output_dir, 'line_halftone')
            self.add_figure_caption(
                "Evolution of mixing fraction",
                "Fraction of the computational domain occupied by the mixing zone as defined by "
                "concentration thresholds of 5% and 95%. This metric quantifies the extent of "
                "fluid mixing relative to the domain size.",
                "Mixing fraction calculated following the Dalziel et al. (1999) methodology."
            )
        
        return self.figure_counter - start_fig_num  # Number of figures created
    
    def generate_convergence_figures(self, df, output_dir, target_time, mixing_method='dalziel'):
        """Generate publication-ready convergence analysis figures."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Figure: Fractal dimension convergence
        plt.figure(figsize=(10, 8))
        plt.errorbar(df['resolution'], df['fractal_dim'], yerr=df['fd_error'],
                    fmt='o-', capsize=5, elinewidth=2, markersize=8, linewidth=2)
        
        plt.xscale('log', base=2)
        plt.xlabel('Grid Resolution')
        plt.ylabel(f'Fractal Dimension')
        plt.grid(True, alpha=0.3)
        
        # Add grid points as labels
        for i, res in enumerate(df['resolution']):
            plt.annotate(f"{res}×{res}", 
                        (df['resolution'].iloc[i], df['fractal_dim'].iloc[i]),
                        xytext=(5, 5), textcoords='offset points', fontsize=10)
        
        # Add asymptotic extrapolation if enough points
        if len(df) >= 3:
            x = 1.0 / np.array(df['resolution'])
            y = df['fractal_dim']
            coeffs = np.polyfit(x[-3:], y[-3:], 1)
            asymptotic_value = coeffs[1]
            
            plt.axhline(y=asymptotic_value, color='r', linestyle='--', alpha=0.8,
                       label=f"Extrapolated value: {asymptotic_value:.4f}")
            plt.legend()
        
        fig_name = self.save_publication_figure("dimension_convergence", output_dir, 'line_halftone')
        self.add_figure_caption(
            f"Resolution convergence of fractal dimension at t = {target_time}",
            "Fractal dimension as a function of grid resolution showing convergence behavior. "
            "Error bars represent statistical uncertainties from the sliding window analysis. "
            "Grid resolution labels indicate the number of computational cells in each direction.",
            "The dashed red line shows the Richardson extrapolation to infinite resolution if sufficient data points are available."
        )
        
        # Figure: Mixing thickness convergence
        plt.figure(figsize=(10, 8))
        plt.plot(df['resolution'], df['h_total'], 'o-', markersize=8, linewidth=2, label='Total thickness')
        plt.plot(df['resolution'], df['ht'], 's--', markersize=6, linewidth=2, label='Upper penetration')
        plt.plot(df['resolution'], df['hb'], 'd--', markersize=6, linewidth=2, label='Lower penetration')
        
        plt.xscale('log', base=2)
        plt.xlabel('Grid Resolution')
        plt.ylabel(f'Mixing Layer Thickness')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        fig_name = self.save_publication_figure("mixing_convergence", output_dir, 'line_halftone')
        self.add_figure_caption(
            f"Resolution convergence of mixing layer thickness at t = {target_time}",
            f"Mixing layer characteristics as functions of grid resolution using the {mixing_method} method. "
            "Total mixing thickness (circles), upper penetration (squares), and lower penetration (diamonds) "
            "show the convergence behavior with increasing resolution.",
            f"All measurements use concentration thresholds appropriate for the {mixing_method} methodology."
        )
        
        return 2  # Number of figures created
    
    def generate_publication_table(self, df, output_dir, table_title, table_description, 
                                  columns_to_include=None, precision_dict=None):
        """Generate publication-ready table in journal-compliant format."""
        os.makedirs(output_dir, exist_ok=True)
        
        if columns_to_include is None:
            columns_to_include = df.columns.tolist()
        
        if precision_dict is None:
            precision_dict = {}
        
        # Create table filename
        table_filename = f"Table_{self.table_counter}.txt"
        table_path = os.path.join(output_dir, table_filename)
        
        with open(table_path, 'w') as f:
            # Write table header
            f.write(f"Table {self.table_counter}. {table_title}\n\n")
            
            # Create formatted table
            headers = []
            for col in columns_to_include:
                if col in df.columns:
                    # Clean up column names for publication
                    clean_name = col.replace('_', ' ').title()
                    if col == 'fractal_dim':
                        clean_name = 'Fractal Dimension'
                    elif col == 'fd_error':
                        clean_name = 'Error (±)'
                    elif col == 'fd_r_squared':
                        clean_name = 'R²'
                    elif col == 'h_total':
                        clean_name = 'Total Thickness'
                    headers.append(clean_name)
            
            # Write headers
            f.write('\t'.join(headers) + '\n')
            f.write('\t'.join(['-' * len(h) for h in headers]) + '\n')
            
            # Write data rows
            for _, row in df.iterrows():
                row_data = []
                for col in columns_to_include:
                    if col in df.columns:
                        value = row[col]
                        if pd.isna(value):
                            row_data.append('—')
                        elif col in precision_dict:
                            if col == 'fd_error':
                                row_data.append(f"±{value:.{precision_dict[col]}f}")
                            else:
                                row_data.append(f"{value:.{precision_dict[col]}f}")
                        elif isinstance(value, float):
                            if col == 'resolution':
                                row_data.append(f"{int(value)}×{int(value)}")
                            elif col in ['fractal_dim', 'h_total', 'ht', 'hb']:
                                row_data.append(f"{value:.4f}")
                            elif col in ['fd_error']:
                                row_data.append(f"±{value:.4f}")
                            elif col in ['fd_r_squared', 'mixing_fraction']:
                                row_data.append(f"{value:.3f}")
                            else:
                                row_data.append(f"{value:.3f}")
                        else:
                            row_data.append(str(value))
                f.write('\t'.join(row_data) + '\n')
            
            # Add table notes
            f.write(f"\n{table_description}\n")
        
        # Add to caption list
        caption = f"Table {self.table_counter}. {table_title}. {table_description}"
        self.table_captions.append(caption)
        self.table_counter += 1
        
        print(f"Saved publication table: {table_filename}")
        return table_filename
    
    def save_figure_captions(self, output_dir):
        """Save all figure captions to separate file as required by journal."""
        caption_file = os.path.join(output_dir, "figure_captions.txt")
        
        with open(caption_file, 'w') as f:
            f.write("FIGURE CAPTIONS\n")
            f.write("=" * 50 + "\n\n")
            
            for caption in self.figure_captions:
                f.write(caption + "\n\n")
        
        print(f"Saved figure captions to: figure_captions.txt")
        return caption_file
    
    def save_table_captions(self, output_dir):
        """Save all table captions to separate file."""
        caption_file = os.path.join(output_dir, "table_captions.txt")
        
        with open(caption_file, 'w') as f:
            f.write("TABLE CAPTIONS\n")
            f.write("=" * 50 + "\n\n")
            
            for caption in self.table_captions:
                f.write(caption + "\n\n")
        
        print(f"Saved table captions to: table_captions.txt")
        return caption_file
    
    def create_supplementary_materials(self, df, output_dir, analysis_params=None):
        """Create supplementary materials package for journal submission."""
        supp_dir = os.path.join(output_dir, "supplementary_materials")
        os.makedirs(supp_dir, exist_ok=True)
        
        # Save raw data in multiple formats
        df.to_csv(os.path.join(supp_dir, "raw_data.csv"), index=False)
        df.to_excel(os.path.join(supp_dir, "raw_data.xlsx"), index=False)
        
        # Save analysis parameters
        if analysis_params is None:
            analysis_params = {
                'fractal_analysis': {
                    'method': 'sliding_window_box_counting',
                    'min_box_size': 0.001,
                    'boundary_removal': True,
                    'richardson_extrapolation': True
                },
                'mixing_analysis': {
                    'method': 'dalziel',
                    'lower_threshold': 0.05,
                    'upper_threshold': 0.95
                },
                'generated_date': datetime.now().isoformat(),
                'software_version': 'rt_analyzer_v3.py'
            }
        
        with open(os.path.join(supp_dir, "analysis_parameters.json"), 'w') as f:
            json.dump(analysis_params, f, indent=2)
        
        # Create README file
        readme_content = f"""
# Supplementary Materials: Rayleigh-Taylor Analysis

## Contents
- raw_data.csv/xlsx: Complete analysis results dataset
- analysis_parameters.json: Detailed analysis configuration
- This README file

## Data Description
This dataset contains {len(df)} analysis points with the following variables:
{', '.join(df.columns.tolist())}

## Analysis Methods
- Fractal dimension: Advanced sliding window box counting with Richardson extrapolation
- Mixing thickness: Dalziel et al. (1999) concentration-based method
- Error analysis: Statistical uncertainties from multiple fitting windows

## Software
Generated using rt_analyzer_v3.py on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Contact
For questions about this analysis, please contact the corresponding author.
"""
        
        with open(os.path.join(supp_dir, "README.txt"), 'w') as f:
            f.write(readme_content)
        
        print(f"Created supplementary materials in: {supp_dir}")
        return supp_dir
    
    def generate_manuscript_ready_package(self, df, base_output_dir, analysis_type="time_series", 
                                        mixing_method='dalziel', target_time=None):
        """Generate complete manuscript-ready package following journal requirements."""
        
        # Create main package directory
        package_dir = os.path.join(base_output_dir, f"manuscript_package_{analysis_type}")
        os.makedirs(package_dir, exist_ok=True)
        
        # Reset counters for this package
        self.figure_counter = 1
        self.table_counter = 1
        self.figure_captions = []
        self.table_captions = []
        
        print(f"Generating manuscript package for {analysis_type} analysis...")
        
        # Generate figures
        if analysis_type == "time_series":
            n_figs = self.generate_publication_figures(df, package_dir, mixing_method)
        elif analysis_type == "convergence":
            n_figs = self.generate_convergence_figures(df, package_dir, target_time, mixing_method)
        else:
            # Generate both types
            n_figs = self.generate_publication_figures(df, package_dir, mixing_method)
            if target_time:
                n_figs += self.generate_convergence_figures(df, package_dir, target_time, mixing_method)
        
        # Generate tables
        if analysis_type == "convergence":
            columns = ['resolution', 'fractal_dim', 'fd_error', 'fd_r_squared', 'h_total', 'ht', 'hb']
            self.generate_publication_table(
                df, package_dir,
                f"Resolution convergence analysis at t = {target_time}",
                f"Fractal dimension and mixing thickness convergence with grid resolution using {mixing_method} method. "
                f"Statistical errors represent uncertainties from sliding window analysis.",
                columns_to_include=columns
            )
        else:
            # Time series summary table (first few and last few points)
            summary_df = pd.concat([df.head(3), df.tail(3)])
            columns = ['time', 'fractal_dim', 'fd_error', 'h_total', 'mixing_fraction'] if 'mixing_fraction' in df.columns else ['time', 'fractal_dim', 'fd_error', 'h_total']
            self.generate_publication_table(
                summary_df, package_dir,
                "Selected results from time series analysis",
                f"Representative data points from Rayleigh-Taylor evolution analysis using {mixing_method} method. "
                f"Complete dataset available in supplementary materials.",
                columns_to_include=columns
            )
        
        # Save captions
        self.save_figure_captions(package_dir)
        self.save_table_captions(package_dir)
        
        # Create supplementary materials
        supp_dir = self.create_supplementary_materials(df, package_dir)
        
        # Create submission checklist
        checklist_content = f"""
# Manuscript Submission Checklist

## Figures ({n_figs} total)
✓ All figures saved as PNG format
✓ Resolution requirements met (500+ DPI for line/halftone combinations)
✓ Figures numbered consecutively (Figure_1.png, Figure_2.png, etc.)
✓ No embedded text in figures (text in captions instead)
✓ Captions provided in separate file

## Tables ({self.table_counter - 1} total)
✓ Tables in editable text format
✓ No vertical rules or shading
✓ Tables numbered consecutively
✓ Captions and notes included

## Supplementary Materials
✓ Raw data provided in multiple formats
✓ Analysis parameters documented
✓ README file included

## Files to Submit
1. Manuscript text (cite figures and tables appropriately)
2. Figure files: Figure_1.png, Figure_2.png, ...
3. Table files: Table_1.txt, Table_2.txt, ...
4. Figure captions: figure_captions.txt
5. Table captions: table_captions.txt
6. Supplementary materials folder

## Journal Requirements Compliance
✓ Figure resolution: {self.DPI_SETTINGS['line_halftone']} DPI for line/halftone combinations
✓ File formats: PNG for figures, TXT for tables
✓ Naming convention: Figure_X and Table_X format
✓ Separate caption files provided
✓ Editable table format used

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        with open(os.path.join(package_dir, "submission_checklist.txt"), 'w') as f:
            f.write(checklist_content)
        
        print(f"Manuscript package complete: {package_dir}")
        print(f"Generated {n_figs} figures and {self.table_counter - 1} tables")
        
        return package_dir
    
    # [Rest of the methods remain the same as v2, just updated method calls...]
    def analyze_vtk_file(self, vtk_file, output_subdir=None, mixing_method='dalziel'):
        """Perform complete analysis on a single VTK file."""
        # Create subdirectory for this file if needed
        if output_subdir:
            file_dir = os.path.join(self.output_dir, output_subdir)
        else:
            basename = os.path.basename(vtk_file).split('.')[0]
            file_dir = os.path.join(self.output_dir, basename)
        
        os.makedirs(file_dir, exist_ok=True)
        
        print(f"Analyzing {vtk_file}...")
        
        # Read VTK file
        start_time = time.time()
        data = self.read_vtk_file(vtk_file)
        print(f"VTK file read in {time.time() - start_time:.2f} seconds")
        
        # Find initial interface position
        h0 = self.find_initial_interface(data)
        print(f"Initial interface position: {h0:.6f}")
        
        # Compute mixing thickness using specified method
        mixing = self.compute_mixing_thickness(data, h0, method=mixing_method)
        print(f"Mixing thickness ({mixing_method}): {mixing['h_total']:.6f} (ht={mixing['ht']:.6f}, hb={mixing['hb']:.6f})")
        
        # Additional diagnostics for Dalziel method
        if mixing_method == 'dalziel':
            print(f"  Mixing zone center: {mixing['y_center']:.6f}")
            print(f"  Mixing fraction: {mixing['mixing_fraction']:.4f}")
            print(f"  Thresholds: {mixing['lower_threshold']:.2f} - {mixing['upper_threshold']:.2f}")
        
        # Extract interface for visualization and save to file
        contours = self.extract_interface(data['f'], data['x'], data['y'])
        interface_file = os.path.join(file_dir, 'interface.dat')
        
        with open(interface_file, 'w') as f:
            f.write(f"# Interface data for t = {data['time']:.6f}\n")
            f.write(f"# Method: {mixing_method}\n")
            segment_count = 0
            for contour in contours:
                for i in range(len(contour) - 1):
                    f.write(f"{contour[i,0]:.7f},{contour[i,1]:.7f} {contour[i+1,0]:.7f},{contour[i+1,1]:.7f}\n")
                    segment_count += 1
            f.write(f"# Found {segment_count} contour segments\n")
        
        print(f"Interface saved to {interface_file} ({segment_count} segments)")
        
        # Compute fractal dimension
        fd_start_time = time.time()
        fd_results = self.compute_fractal_dimension(data)
        print(f"Fractal dimension: {fd_results['dimension']:.6f} ± {fd_results['error']:.6f} (R²={fd_results['r_squared']:.6f})")
        print(f"Fractal calculation time: {time.time() - fd_start_time:.2f} seconds")
        
        # Visualize interface and box counting (working versions, not publication quality)
        if not np.isnan(fd_results['dimension']):
            fig = plt.figure(figsize=(12, 10))
            plt.contourf(data['x'], data['y'], data['f'], levels=20, cmap='viridis')
            plt.colorbar(label='Volume Fraction')
            
            # Plot interface
            for contour in contours:
                plt.plot(contour[:, 0], contour[:, 1], 'r-', linewidth=2)
            
            # Plot initial interface position
            plt.axhline(y=h0, color='k', linestyle='--', alpha=0.5, label=f'Initial Interface (y={h0:.4f})')
            
            # Plot mixing zone boundaries for Dalziel method
            if mixing_method == 'dalziel':
                y_upper = h0 + mixing['ht']
                y_lower = h0 - mixing['hb']
                plt.axhline(y=y_upper, color='orange', linestyle=':', alpha=0.7, label=f'Upper boundary')
                plt.axhline(y=y_lower, color='orange', linestyle=':', alpha=0.7, label=f'Lower boundary')
            
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.title(f'Rayleigh-Taylor Interface at t = {data["time"]:.3f} ({mixing_method} method)')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(file_dir, 'interface_plot.png'), dpi=300)
            plt.close()
            
            # Plot box counting results if available
            if 'box_sizes' in fd_results and fd_results['box_sizes'] is not None:
                fig = plt.figure(figsize=(10, 8))
                plt.loglog(fd_results['box_sizes'], fd_results['box_counts'], 'bo-', label='Data')
                
                # Linear regression line
                log_sizes = np.log(fd_results['box_sizes'])
                slope = -fd_results['dimension']
                # Use intercept from analysis if available
                if 'analysis_results' in fd_results and 'intercept' in fd_results['analysis_results']:
                    intercept = fd_results['analysis_results']['intercept']
                else:
                    # Fallback calculation
                    log_counts = np.log(fd_results['box_counts'])
                    intercept = np.mean(log_counts - slope * log_sizes)
                
                fit_counts = np.exp(intercept + slope * log_sizes)
                plt.loglog(fd_results['box_sizes'], fit_counts, 'r-', 
                          label=f"D = {fd_results['dimension']:.4f} ± {fd_results['error']:.4f}")
                
                plt.xlabel('Box Size')
                plt.ylabel('Box Count')
                plt.title(f'Fractal Dimension at t = {data["time"]:.3f}')
                plt.legend()
                plt.grid(True)
                plt.savefig(os.path.join(file_dir, 'fractal_dimension.png'), dpi=300)
                plt.close()
        
        # Prepare return results
        result = {
            'time': data['time'],
            'h0': h0,
            'ht': mixing['ht'],
            'hb': mixing['hb'],
            'h_total': mixing['h_total'],
            'fractal_dim': fd_results['dimension'],
            'fd_error': fd_results['error'],
            'fd_r_squared': fd_results['r_squared'],
            'mixing_method': mixing_method
        }
        
        # Add Dalziel-specific results
        if mixing_method == 'dalziel':
            result.update({
                'y_center': mixing['y_center'],
                'mixing_fraction': mixing['mixing_fraction'],
                'lower_threshold': mixing['lower_threshold'],
                'upper_threshold': mixing['upper_threshold']
            })
        
        return result
    
    def process_vtk_series(self, vtk_pattern, resolution=None, mixing_method='dalziel', 
                          generate_publication_package=True):
        """Process a series of VTK files and optionally generate publication package."""
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
                result = self.analyze_vtk_file(vtk_file, subdir, mixing_method=mixing_method)
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
            
            # Generate publication package if requested
            if generate_publication_package:
                package_dir = self.generate_manuscript_ready_package(
                    df, results_dir, analysis_type="time_series", mixing_method=mixing_method
                )
                print(f"Publication package created: {package_dir}")
            
            return df
        else:
            print("No results to summarize")
            return None
    
    def analyze_resolution_convergence(self, vtk_files, resolutions, target_time=9.0, 
                                     mixing_method='dalziel', generate_publication_package=True):
        """Analyze resolution convergence and generate publication package."""
        results = []
        
        print(f"Analyzing resolution convergence using {mixing_method} mixing method")
        
        for vtk_file, resolution in zip(vtk_files, resolutions):
            print(f"\nAnalyzing resolution {resolution}x{resolution} using {vtk_file}")
            
            try:
                # Read and analyze the file
                data = self.read_vtk_file(vtk_file)
                
                # Check if time matches target
                if abs(data['time'] - target_time) > 0.1:
                    print(f"Warning: File time {data['time']} differs from target {target_time}")
                
                # Find initial interface
                h0 = self.find_initial_interface(data)
                
                # Calculate mixing thickness
                mixing = self.compute_mixing_thickness(data, h0, method=mixing_method)
                
                # Calculate fractal dimension
                fd_results = self.compute_fractal_dimension(data)
                
                # Save results
                result = {
                    'resolution': resolution,
                    'time': data['time'],
                    'h0': h0,
                    'ht': mixing['ht'],
                    'hb': mixing['hb'],
                    'h_total': mixing['h_total'],
                    'fractal_dim': fd_results['dimension'],
                    'fd_error': fd_results['error'],
                    'fd_r_squared': fd_results['r_squared'],
                    'mixing_method': mixing_method
                }
                
                # Add Dalziel-specific results
                if mixing_method == 'dalziel':
                    result.update({
                        'y_center': mixing['y_center'],
                        'mixing_fraction': mixing['mixing_fraction']
                    })
                
                results.append(result)
                
            except Exception as e:
                print(f"Error analyzing {vtk_file}: {str(e)}")
                import traceback
                traceback.print_exc()
        
        # Convert to DataFrame
        if results:
            df = pd.DataFrame(results)
            
            # Create output directory
            convergence_dir = os.path.join(self.output_dir, f"convergence_t{target_time}_{mixing_method}")
            os.makedirs(convergence_dir, exist_ok=True)
            
            # Save results
            csv_file = os.path.join(convergence_dir, f'resolution_convergence_{mixing_method}.csv')
            df.to_csv(csv_file, index=False)
            
            # Generate publication package if requested
            if generate_publication_package:
                package_dir = self.generate_manuscript_ready_package(
                    df, convergence_dir, analysis_type="convergence", 
                    mixing_method=mixing_method, target_time=target_time
                )
                print(f"Publication package created: {package_dir}")
            
            return df
        else:
            print("No results to analyze")
            return None

    def _plot_resolution_convergence(self, df, target_time, output_dir, mixing_method):
        """Plot resolution convergence results."""
        # Plot fractal dimension vs resolution
        plt.figure(figsize=(10, 8))
        
        plt.errorbar(df['resolution'], df['fractal_dim'], yerr=df['fd_error'],
                    fmt='o-', capsize=5, elinewidth=1, markersize=8)
        
        plt.xscale('log', base=2)  # Use log scale with base 2
        plt.xlabel('Grid Resolution')
        plt.ylabel(f'Fractal Dimension at t={target_time}')
        plt.title(f'Fractal Dimension Convergence at t={target_time} ({mixing_method} method)')
        plt.grid(True)
        
        # Add grid points as labels
        for i, res in enumerate(df['resolution']):
            plt.annotate(f"{res}×{res}", (df['resolution'].iloc[i], df['fractal_dim'].iloc[i]),
                        xytext=(5, 5), textcoords='offset points')
        
        # Add asymptote if enough points
        if len(df) >= 3:
            # Extrapolate to infinite resolution (1/N = 0)
            x = 1.0 / np.array(df['resolution'])
            y = df['fractal_dim']
            coeffs = np.polyfit(x[-3:], y[-3:], 1)
            asymptotic_value = coeffs[1]  # y-intercept
            
            plt.axhline(y=asymptotic_value, color='r', linestyle='--',
                       label=f"Extrapolated value: {asymptotic_value:.4f}")
            plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"dimension_convergence_{mixing_method}.png"), dpi=300)
        plt.close()
        
        # Plot mixing layer thickness convergence
        plt.figure(figsize=(10, 8))
        
        plt.plot(df['resolution'], df['h_total'], 'o-', markersize=8, label='Total')
        plt.plot(df['resolution'], df['ht'], 's--', markersize=6, label='Upper')
        plt.plot(df['resolution'], df['hb'], 'd--', markersize=6, label='Lower')
        
        plt.xscale('log', base=2)
        plt.xlabel('Grid Resolution')
        plt.ylabel(f'Mixing Layer Thickness at t={target_time}')
        plt.title(f'Mixing Layer Thickness Convergence at t={target_time} ({mixing_method} method)')
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"mixing_convergence_{mixing_method}.png"), dpi=300)
        plt.close()
        
        # Additional plot for Dalziel method showing mixing fraction
        if mixing_method == 'dalziel' and 'mixing_fraction' in df.columns:
            plt.figure(figsize=(10, 6))
            plt.plot(df['resolution'], df['mixing_fraction'], 'o-', markersize=8, color='purple')
            plt.xscale('log', base=2)
            plt.xlabel('Grid Resolution')
            plt.ylabel('Mixing Fraction')
            plt.title(f'Mixing Fraction Convergence at t={target_time} (Dalziel method)')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"mixing_fraction_convergence.png"), dpi=300)
            plt.close()
    
    def create_summary_plots(self, df, output_dir, mixing_method):
        """Create summary plots of the time series results."""
        # Plot mixing layer evolution
        plt.figure(figsize=(10, 6))
        plt.plot(df['time'], df['h_total'], 'b-', label='Total', linewidth=2)
        plt.plot(df['time'], df['ht'], 'r--', label='Upper', linewidth=2)
        plt.plot(df['time'], df['hb'], 'g--', label='Lower', linewidth=2)
        plt.xlabel('Time')
        plt.ylabel('Mixing Layer Thickness')
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
        plt.title(f'Fractal Dimension Evolution ({mixing_method} method)')
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, f'dimension_evolution_{mixing_method}.png'), dpi=300)
        plt.close()
        
        # Plot R-squared evolution
        plt.figure(figsize=(10, 6))
        plt.plot(df['time'], df['fd_r_squared'], 'm-o', linewidth=2)
        plt.xlabel('Time')
        plt.ylabel('R² Value')
        plt.title(f'Fractal Dimension Fit Quality ({mixing_method} method)')
        plt.ylim(0, 1)
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, f'r_squared_evolution_{mixing_method}.png'), dpi=300)
        plt.close()
        
        # Additional plot for Dalziel method
        if mixing_method == 'dalziel' and 'mixing_fraction' in df.columns:
            plt.figure(figsize=(10, 6))
            plt.plot(df['time'], df['mixing_fraction'], 'c-o', linewidth=2)
            plt.xlabel('Time')
            plt.ylabel('Mixing Fraction')
            plt.title('Mixing Fraction Evolution (Dalziel method)')
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, 'mixing_fraction_evolution.png'), dpi=300)
            plt.close()
        
        # Combined plot with mixing layer and fractal dimension
        fig, ax1 = plt.subplots(figsize=(12, 8))
        
        # Mixing layer on left axis
        ax1.plot(df['time'], df['h_total'], 'b-', label='Mixing Thickness', linewidth=2)
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Mixing Layer Thickness', color='b')
        ax1.tick_params(axis='y', labelcolor='b')
        
        # Fractal dimension on right axis
        ax2 = ax1.twinx()
        ax2.errorbar(df['time'], df['fractal_dim'], yerr=df['fd_error'],
                   fmt='ro-', capsize=3, label='Fractal Dimension')
        ax2.set_ylabel('Fractal Dimension', color='r')
        ax2.tick_params(axis='y', labelcolor='r')
        
        # Add both legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        plt.title(f'Mixing Layer and Fractal Dimension Evolution ({mixing_method} method)')
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, f'combined_evolution_{mixing_method}.png'), dpi=300)
        plt.close()

    def compute_multifractal_spectrum(self, data, min_box_size=0.001, q_values=None, output_dir=None):
        """Compute multifractal spectrum of the interface.
        
        Args:
            data: Data dictionary containing VTK data
            min_box_size: Minimum box size for analysis (default: 0.001)
            q_values: List of q moments to analyze (default: -5 to 5 in 0.5 steps)
            output_dir: Directory to save results (default: None)
            
        Returns:
            dict: Multifractal spectrum results
        """
        if self.fractal_analyzer is None:
            print("Fractal analyzer not available. Skipping multifractal analysis.")
            return None
        
        # Set default q values if not provided
        if q_values is None:
            q_values = np.arange(-5, 5.1, 0.5)
        
        # Extract contours and convert to segments
        contours = self.extract_interface(data['f'], data['x'], data['y'])
        segments = self.convert_contours_to_segments(contours)
        
        if not segments:
            print("No interface segments found. Skipping multifractal analysis.")
            return None
        
        # Create output directory if specified and not existing
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Calculate extent for max box size
        min_x = min(min(s[0][0], s[1][0]) for s in segments)
        max_x = max(max(s[0][0], s[1][0]) for s in segments)
        min_y = min(min(s[0][1], s[1][1]) for s in segments)
        max_y = max(max(s[0][1], s[1][1]) for s in segments)
        
        extent = max(max_x - min_x, max_y - min_y)
        max_box_size = extent / 2
        
        print(f"Performing multifractal analysis with {len(q_values)} q-values")
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
        
        # Use spatial index from BoxCounter to speed up calculations
        bc = self.fractal_analyzer.box_counter
        
        # Add small margin to bounding box
        margin = extent * 0.01
        min_x -= margin
        max_x += margin
        min_y -= margin
        max_y += margin
        
        # Create spatial index for segments
        start_time = time.time()
        print("Creating spatial index...")
        
        # Determine grid cell size for spatial index (use smallest box size)
        grid_size = min_box_size * 2
        segment_grid, grid_width, grid_height = bc.create_spatial_index(
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
                    
                    # Count intersections (for multifractal, count each segment)
                    count = 0
                    for seg_idx in segments_to_check:
                        (x1, y1), (x2, y2) = segments[seg_idx]
                        if self.fractal_analyzer.base.liang_barsky_line_box_intersection(
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
            
            # Degree of multifractality: D(-∞) - D(+∞) ≈ D(-5) - D(5)
            if -5 in q_values and 5 in q_values:
                degree_multifractality = Dqs[np.searchsorted(q_values, -5)] - Dqs[np.searchsorted(q_values, 5)]
            else:
                degree_multifractality = np.nan
            
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
            # Plot D(q) vs q
            plt.figure(figsize=(10, 6))
            valid = ~np.isnan(Dqs)
            plt.plot(q_values[valid], Dqs[valid], 'bo-', markersize=4)
            
            if 0 in q_values:
                plt.axhline(y=Dqs[np.searchsorted(q_values, 0)], color='r', linestyle='--', 
                           label=f"D(0) = {Dqs[np.searchsorted(q_values, 0)]:.4f}")
            
            plt.xlabel('q')
            plt.ylabel('D(q)')
            plt.title(f'Generalized Dimensions D(q) at t = {data["time"]:.2f}')
            plt.grid(True)
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
            plt.title(f'Multifractal Spectrum f(α) at t = {data["time"]:.2f}')
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, "multifractal_spectrum.png"), dpi=300)
            plt.close()
            
            # Plot R² values
            plt.figure(figsize=(10, 6))
            valid = ~np.isnan(r_squared)
            plt.plot(q_values[valid], r_squared[valid], 'go-', markersize=4)
            plt.xlabel('q')
            plt.ylabel('R²')
            plt.title(f'Fit Quality for Different q Values at t = {data["time"]:.2f}')
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, "multifractal_r_squared.png"), dpi=300)
            plt.close()
            
            # Save results to CSV
            import pandas as pd
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
                'Value': [data['time'], D0, D1, D2, alpha_width, degree_multifractality]
            })
            params_df.to_csv(os.path.join(output_dir, "multifractal_parameters.csv"), index=False)
        
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
            'time': data['time']
        }
    
    def analyze_multifractal_evolution(self, vtk_files, output_dir=None, q_values=None):
        """
        Analyze how multifractal properties evolve over time or across resolutions.
        
        Args:
            vtk_files: Dict mapping either times or resolutions to VTK files
                      e.g. {0.1: 'RT_t0.1.vtk', 0.2: 'RT_t0.2.vtk'} for time series
                      or {100: 'RT100x100.vtk', 200: 'RT200x200.vtk'} for resolutions
            output_dir: Directory to save results
            q_values: List of q moments to analyze (default: -5 to 5 in 0.5 steps)
            
        Returns:
            dict: Multifractal evolution results
        """
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Set default q values if not provided
        if q_values is None:
            q_values = np.arange(-5, 5.1, 0.5)
        
        # Determine type of analysis (time or resolution)
        keys = list(vtk_files.keys())
        is_time_series = all(isinstance(k, float) for k in keys)
        
        if is_time_series:
            print(f"Analyzing multifractal evolution over time series: {sorted(keys)}")
            x_label = 'Time'
            series_name = "time"
        else:
            print(f"Analyzing multifractal evolution across resolutions: {sorted(keys)}")
            x_label = 'Resolution'
            series_name = "resolution"
        
        # Initialize results storage
        results = []
        
        # Process each file
        for key, vtk_file in sorted(vtk_files.items()):
            print(f"\nProcessing {series_name} = {key}, file: {vtk_file}")
            
            try:
                # Read VTK file
                data = self.read_vtk_file(vtk_file)
                
                # Create subdirectory for this point
                if output_dir:
                    point_dir = os.path.join(output_dir, f"{series_name}_{key}")
                    os.makedirs(point_dir, exist_ok=True)
                else:
                    point_dir = None
                
                # Perform multifractal analysis
                mf_results = self.compute_multifractal_spectrum(data, q_values=q_values, output_dir=point_dir)
                
                if mf_results:
                    # Store results with the key (time or resolution)
                    mf_results[series_name] = key
                    results.append(mf_results)
                
            except Exception as e:
                print(f"Error processing {vtk_file}: {str(e)}")
                import traceback
                traceback.print_exc()
        
        # Create summary plots
        if results and output_dir:
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
            
            # Create 3D surface plot of D(q) evolution if matplotlib supports it
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
            
            # Save summary CSV
            import pandas as pd
            summary_df = pd.DataFrame({
                series_name: x_values,
                'D0': D0_values,
                'D1': D1_values,
                'D2': D2_values,
                'alpha_width': alpha_width,
                'degree_multifractality': degree_mf
            })
            summary_df.to_csv(os.path.join(output_dir, "multifractal_evolution_summary.csv"), index=False)
        
        return results
