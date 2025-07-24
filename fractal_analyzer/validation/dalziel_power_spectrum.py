# =================================================================
# 2. UPDATED dalziel_power_spectrum.py  
# =================================================================

#!/usr/bin/env python3
"""
UPDATED Dalziel Power Spectrum Analyzer - Compatible with fractal_analyzer suite
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import fft
import sys
import os

# Updated import to work with fractal_analyzer structure
try:
    from fractal_analyzer.core.rt_analyzer import RTAnalyzer
except ImportError:
    # Fallback for different directory structures
    sys.path.append('fractal_analyzer/core')
    from rt_analyzer import RTAnalyzer

class UpdatedDalzielPowerSpectrumAnalyzer:
    """
    Updated power spectrum analyzer compatible with current RTAnalyzer.
    """
    
    def __init__(self, debug=False):
        self.debug = debug
        self.rt_analyzer = RTAnalyzer()
        
    def find_velocity_fields(self, data):
        """
        Find velocity fields in VTK data with flexible naming.
        
        Args:
            data: VTK data dictionary
            
        Returns:
            tuple: (u_field, v_field) or (None, None) if not found
        """
        # Possible field names for velocity components
        u_names = ['U', 'u', 'velocity_x', 'vel_x', 'u_velocity']
        v_names = ['V', 'v', 'velocity_y', 'vel_y', 'v_velocity']
        
        u_field = None
        v_field = None
        
        # Find U component
        for name in u_names:
            if name in data:
                u_field = data[name]
                print(f"Found U velocity as '{name}'")
                break
        
        # Find V component  
        for name in v_names:
            if name in data:
                v_field = data[name]
                print(f"Found V velocity as '{name}'")
                break
        
        if u_field is None or v_field is None:
            print(f"Available fields: {list(data.keys())}")
            print("Could not find both U and V velocity components")
            
            # Try to extract from velocity vector if available
            if 'velocity' in data:
                vel = data['velocity']
                if len(vel.shape) == 3 and vel.shape[2] >= 2:
                    u_field = vel[:, :, 0]
                    v_field = vel[:, :, 1]
                    print("Extracted U,V from velocity vector")
        
        return u_field, v_field
    
    def compute_power_spectrum_2d(self, u_field, v_field, dx, dy):
        """
        Compute 2D power spectrum from velocity fields.
        Same implementation as before but with improved error handling.
        """
        if u_field is None or v_field is None:
            raise ValueError("Both U and V velocity fields are required")
        
        ny, nx = u_field.shape
        
        if self.debug:
            print(f"Power spectrum calculation:")
            print(f"  Grid: {nx}×{ny}")
            print(f"  Velocity ranges: u ∈ [{np.min(u_field):.6f}, {np.max(u_field):.6f}]")
            print(f"  Velocity ranges: v ∈ [{np.min(v_field):.6f}, {np.max(v_field):.6f}]")
            print(f"  Grid spacing: dx={dx:.6f}, dy={dy:.6f}")
        
        # Remove mean velocities (focus on fluctuations)
        u_fluct = u_field - np.mean(u_field)
        v_fluct = v_field - np.mean(v_field)
        
        # Compute 2D FFT of velocity components
        u_hat = fft.fft2(u_fluct)
        v_hat = fft.fft2(v_fluct)
        
        # Kinetic energy spectrum in Fourier space
        kinetic_energy_2d = 0.5 * (np.abs(u_hat)**2 + np.abs(v_hat)**2)
        
        # Physical wavenumber grids
        kx_phys = 2 * np.pi * fft.fftfreq(nx, d=dx)
        ky_phys = 2 * np.pi * fft.fftfreq(ny, d=dy)
        
        # Create 2D wavenumber magnitude grid
        KX, KY = np.meshgrid(kx_phys, ky_phys)
        K_mag = np.sqrt(KX**2 + KY**2)
        
        # Define wavenumber bins for radial averaging
        k_max = np.sqrt((np.pi/dx)**2 + (np.pi/dy)**2)
        k_min = 2 * np.pi / max(nx * dx, ny * dy)
        
        # Use logarithmic spacing
        n_bins = min(nx//2, ny//2, 50)
        k_bins = np.logspace(np.log10(k_min), np.log10(k_max), n_bins)
        
        # Perform radial averaging
        P_total = np.zeros(len(k_bins)-1)
        k_centers = np.zeros(len(k_bins)-1)
        
        for i in range(len(k_bins)-1):
            k_low, k_high = k_bins[i], k_bins[i+1]
            
            # Find points in this wavenumber shell
            mask = (K_mag >= k_low) & (K_mag < k_high)
            
            if np.any(mask):
                P_total[i] = np.sum(kinetic_energy_2d[mask])
                k_centers[i] = np.sqrt(k_low * k_high)
                
                # Normalize by number of points in shell
                shell_points = np.sum(mask)
                P_total[i] /= shell_points
            else:
                k_centers[i] = np.sqrt(k_low * k_high)
                P_total[i] = 0
        
        # Remove bins with zero power
        valid = P_total > 0
        k_centers = k_centers[valid]
        P_total = P_total[valid]
        
        # Dalziel normalization
        k_0 = k_centers[np.argmax(P_total)]
        P_0 = np.max(P_total)
        
        k_normalized = k_centers / k_0
        P_normalized = P_total / P_0
        
        if self.debug:
            print(f"  Characteristic scales:")
            print(f"    k_0 = {k_0:.4f}")
            print(f"    P_0 = {P_0:.4e}")
        
        return k_normalized, P_normalized, k_centers, P_total, k_0, P_0
    
    def analyze_rt_power_spectrum(self, vtk_file, output_dir=None):
        """
        Analyze RT simulation VTK file with updated compatibility.
        """
        print(f"🌊 UPDATED DALZIEL POWER SPECTRUM ANALYSIS")
        print(f"=" * 50)
        print(f"Analyzing: {os.path.basename(vtk_file)}")
        
        # Read VTK file
        data = self.rt_analyzer.read_vtk_file(vtk_file)
        
        # Find velocity fields with flexible naming
        u_field, v_field = self.find_velocity_fields(data)
        
        if u_field is None or v_field is None:
            raise ValueError(f"Could not find velocity fields in VTK file")
        
        # Calculate grid spacing
        x_grid = data['x']
        y_grid = data['y']
        
        # Robust grid spacing calculation
        if x_grid.shape[1] > 1:
            dx = (np.max(x_grid) - np.min(x_grid)) / (x_grid.shape[1] - 1)
        else:
            dx = 1.0 / x_grid.shape[1]
            
        if y_grid.shape[0] > 1:
            dy = (np.max(y_grid) - np.min(y_grid)) / (y_grid.shape[0] - 1)
        else:
            dy = 1.0 / y_grid.shape[0]
        
        print(f"Grid spacing: dx={dx:.6f}, dy={dy:.6f}")
        
        # Compute power spectrum
        k_norm, P_norm, k_phys, P_phys, k_0, P_0 = self.compute_power_spectrum_2d(
            u_field, v_field, dx, dy)
        
        # Create results
        results = {
            'file': vtk_file,
            'time': data.get('time', 0.0),
            'grid_shape': u_field.shape,
            'dx': dx,
            'dy': dy,
            'k_normalized': k_norm,
            'P_normalized': P_norm,
            'k_physical': k_phys, 
            'P_physical': P_phys,
            'k_0': k_0,
            'P_0': P_0,
            'total_energy': np.sum(P_phys)
        }
        
        print(f"Analysis complete:")
        print(f"  Time: {results['time']:.3f}")
        print(f"  Grid: {u_field.shape[1]}×{u_field.shape[0]}")
        print(f"  k_0: {k_0:.4f}")
        print(f"  P_0: {P_0:.4e}")
        
        # Save results if requested
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            self.plot_power_spectrum(results, output_dir)
            self.save_results(results, output_dir)
        
        return results
    
    def plot_power_spectrum(self, results, output_dir):
        """Create Dalziel-style power spectrum plot."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Normalized spectrum
        ax1.loglog(results['k_normalized'], results['P_normalized'], 'bo-', 
                   markersize=4, linewidth=1.5, label='RT Simulation')
        
        # Add reference slopes
        k_ref = results['k_normalized']
        if len(k_ref) > 5:
            k_mid = k_ref[len(k_ref)//3:2*len(k_ref)//3]
            if len(k_mid) > 0:
                P_k53 = k_mid**(-5/3) * results['P_normalized'][len(k_ref)//3]
                ax1.loglog(k_mid, P_k53, 'r--', linewidth=2, 
                          label='k^(-5/3)', alpha=0.7)
        
        ax1.set_xlabel('k/k₀', fontsize=12)
        ax1.set_ylabel('P/P₀', fontsize=12)
        ax1.set_title(f'Dalziel Power Spectrum\nt = {results["time"]:.3f}', fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Physical spectrum
        ax2.loglog(results['k_physical'], results['P_physical'], 'mo-', 
                   markersize=4, linewidth=1.5, label='Physical Spectrum')
        
        ax2.set_xlabel('k [1/length]', fontsize=12)
        ax2.set_ylabel('E(k) [energy]', fontsize=12)
        ax2.set_title(f'Physical Spectrum\nt = {results["time"]:.3f}', fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        
        plot_file = os.path.join(output_dir, f'power_spectrum_t{results["time"]:.3f}.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved: {plot_file}")
        
        plt.show()
    
    def save_results(self, results, output_dir):
        """Save results to CSV."""
        import pandas as pd
        
        # Dalziel format
        df = pd.DataFrame({
            'k_over_k0': results['k_normalized'],
            'P_over_P0': results['P_normalized']
        })
        csv_file = os.path.join(output_dir, f'dalziel_spectrum_t{results["time"]:.3f}.csv')
        df.to_csv(csv_file, index=False)
        print(f"Results saved: {csv_file}")

def main():
    """Updated main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Updated Dalziel Power Spectrum Analysis')
    parser.add_argument('--file', help='VTK file to analyze')
    parser.add_argument('--output', default='./power_analysis', help='Output directory')
    parser.add_argument('--debug', action='store_true', help='Debug mode')
    
    args = parser.parse_args()
    
    if not args.file:
        # Default test file
        args.file = "../../RT640x800-4000.vtk"  # Adjust as needed
        print(f"Using default file: {args.file}")
    
    if not os.path.exists(args.file):
        print(f"File not found: {args.file}")
        print("Please specify a valid VTK file with --file")
        return
    
    # Create analyzer and run
    analyzer = UpdatedDalzielPowerSpectrumAnalyzer(debug=args.debug)
    
    try:
        results = analyzer.analyze_rt_power_spectrum(args.file, args.output)
        print(f"✅ Analysis complete! Results in: {args.output}")
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
