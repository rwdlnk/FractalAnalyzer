# dalziel_power_spectrum_analyzer.py - Recreate Dalziel et al. (1999) power spectrum analysis

import numpy as np
import matplotlib.pyplot as plt
from scipy import fft
import sys
import os
sys.path.append('fractal_analyzer/core')
from rt_analyzer import RTAnalyzer

class DalzielPowerSpectrumAnalyzer:
    """
    Analyze RT velocity fields to create power spectrum plots matching Dalziel et al. (1999).
    
    Recreates the P/P_0 vs k/k_0 plots from their JFM paper using velocity data
    from VTK files containing U,V velocity components.
    """
    
    def __init__(self, debug=False):
        self.debug = debug
        self.rt_analyzer = RTAnalyzer()
        
    def compute_power_spectrum_2d(self, u_field, v_field, dx, dy):
        """
        Compute 2D power spectrum from velocity fields following Dalziel methodology.
        
        Args:
            u_field: 2D array of u-velocity component
            v_field: 2D array of v-velocity component  
            dx, dy: Grid spacing in x and y directions
            
        Returns:
            k_normalized: Normalized wavenumber k/k_0
            P_normalized: Normalized power P/P_0
            k_physical: Physical wavenumber
            P_total: Total power spectrum
        """
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
        # E(k) = 0.5 * (|u_hat|² + |v_hat|²)
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
        
        # Use logarithmic spacing similar to Dalziel
        n_bins = min(nx//2, ny//2, 50)
        k_bins = np.logspace(np.log10(k_min), np.log10(k_max), n_bins)
        
        # Perform radial averaging (azimuthal integration)
        P_total = np.zeros(len(k_bins)-1)
        k_centers = np.zeros(len(k_bins)-1)
        
        for i in range(len(k_bins)-1):
            k_low, k_high = k_bins[i], k_bins[i+1]
            
            # Find points in this wavenumber shell
            mask = (K_mag >= k_low) & (K_mag < k_high)
            
            if np.any(mask):
                # Sum energy in this shell and normalize by shell area
                P_total[i] = np.sum(kinetic_energy_2d[mask])
                k_centers[i] = np.sqrt(k_low * k_high)  # Geometric mean
                
                # Normalize by number of points in shell for proper averaging
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
        # k_0: characteristic wavenumber (could be most energetic mode or domain-based)
        # P_0: characteristic power (total power or peak power)
        
        # Method 1: Use peak values for normalization (most common)
        k_0 = k_centers[np.argmax(P_total)]  # Wavenumber of peak energy
        P_0 = np.max(P_total)  # Peak power
        
        # Method 2: Alternative - use domain-based characteristic scales
        # k_0_alt = 2 * np.pi / (max(nx * dx, ny * dy))  # Fundamental mode
        # P_0_alt = np.sum(P_total)  # Total energy
        
        # Create normalized quantities
        k_normalized = k_centers / k_0
        P_normalized = P_total / P_0
        
        # Fit your k/k₀ vs P/P₀ data in log space
        # Try different fitting ranges
        log_k = np.log10(k_normalized)
        log_P = np.log10(P_normalized)

        # Inertial range fitting (typically middle decade)
        valid = (k_normalized >= 1.0) & (k_normalized <= 10.0)
        if np.sum(valid) > 3:
            slope_inertial = np.polyfit(log_k[valid], log_P[valid], 1)[0]
            print(f"Inertial range slope (1 ≤ k/k₀ ≤ 10): {slope_inertial:.2f}")

        # High-k range fitting  
        valid_high = (k_normalized >= 3.0) & (k_normalized <= 30.0)
        if np.sum(valid_high) > 3:
            slope_high = np.polyfit(log_k[valid_high], log_P[valid_high], 1)[0]
            print(f"High-k range slope (3 ≤ k/k₀ ≤ 30): {slope_high:.2f}")

        if self.debug:
            print(f"  Characteristic scales:")
            print(f"    k_0 = {k_0:.4f} (peak wavenumber)")
            print(f"    P_0 = {P_0:.4e} (peak power)")
            print(f"  Spectrum range: k/k_0 ∈ [{np.min(k_normalized):.3f}, {np.max(k_normalized):.3f}]")
            print(f"  Power range: P/P_0 ∈ [{np.min(P_normalized):.3f}, {np.max(P_normalized):.3f}]")
        
        return k_normalized, P_normalized, k_centers, P_total, k_0, P_0
    
    def analyze_rt_power_spectrum(self, vtk_file, output_dir=None):
        """
        Analyze RT simulation VTK file to extract power spectrum.
        
        Args:
            vtk_file: Path to VTK file containing U,V velocity fields
            output_dir: Directory to save results (optional)
            
        Returns:
            dict: Analysis results including normalized and physical spectra
        """
        print(f"🌊 DALZIEL POWER SPECTRUM ANALYSIS")
        print(f"=" * 50)
        print(f"Analyzing: {os.path.basename(vtk_file)}")
        
        # Read VTK file with velocity data
        data = self.rt_analyzer.read_vtk_file(vtk_file)
        
        # Check for required velocity fields
        if 'U' not in data or 'V' not in data:
            raise ValueError(f"VTK file must contain U and V velocity fields")
        
        u_field = data['U']
        v_field = data['V']
        
        # Calculate grid spacing more robustly
        x_grid = data['x']
        y_grid = data['y']

        if self.debug:
            print(f"Grid shapes: x={x_grid.shape}, y={y_grid.shape}")
            print(f"X range: [{np.min(x_grid):.6f}, {np.max(x_grid):.6f}]")
            print(f"Y range: [{np.min(y_grid):.6f}, {np.max(y_grid):.6f}]")

        # Calculate dx
        if x_grid.shape[1] > 1:
            x_range = np.max(x_grid) - np.min(x_grid)
            dx = x_range / (x_grid.shape[1] - 1)
            if dx <= 0:
                dx = 1.0 / x_grid.shape[1]  # Fallback for unit domain
        else:
            dx = 1.0 / x_grid.shape[1]

        # Calculate dy  
        if y_grid.shape[0] > 1:
            y_range = np.max(y_grid) - np.min(y_grid)
            dy = y_range / (y_grid.shape[0] - 1)
            if dy <= 0:
                dy = 1.0 / y_grid.shape[0]  # Fallback for unit domain
        else:
            dy = 1.0 / y_grid.shape[0]

        # Safety check
        if dx <= 0:
            print("Warning: dx was zero or negative, using unit domain assumption")
            dx = 1.0 / x_grid.shape[1]
        if dy <= 0:
            print("Warning: dy was zero or negative, using unit domain assumption") 
            dy = 1.0 / y_grid.shape[0]
        print(f"Grid spacing: dx={dx:.6f}, dy={dy:.6f}")
        
        # Compute power spectrum
        k_norm, P_norm, k_phys, P_phys, k_0, P_0 = self.compute_power_spectrum_2d(
            u_field, v_field, dx, dy)
        
        # Create results dictionary
        results = {
            'file': vtk_file,
            'time': data['time'],
            'grid_shape': u_field.shape,
            'dx': dx,
            'dy': dy,
            'k_normalized': k_norm,
            'P_normalized': P_norm,
            'k_physical': k_phys, 
            'P_physical': P_phys,
            'k_0': k_0,
            'P_0': P_0,
            'total_energy': np.sum(P_phys),
            'peak_wavenumber': k_0,
            'peak_power': P_0
        }
        
        print(f"Analysis complete:")
        print(f"  Time: {data['time']:.3f}")
        print(f"  Grid: {u_field.shape[1]}×{u_field.shape[0]}")
        print(f"  Characteristic wavenumber k_0: {k_0:.4f}")
        print(f"  Peak power P_0: {P_0:.4e}")
        print(f"  Total kinetic energy: {results['total_energy']:.4e}")
        
        # Save results if output directory specified
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            self.save_power_spectrum_results(results, output_dir)
            self.plot_dalziel_power_spectrum(results, output_dir)
        
        return results
    
    def plot_dalziel_power_spectrum(self, results, output_dir, save_plots=True):
        """
        Create Dalziel-style power spectrum plot: P/P_0 vs k/k_0.
        
        Args:
            results: Analysis results dictionary
            output_dir: Directory to save plots
            save_plots: Whether to save plot files
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Dalziel-style normalized spectrum
        ax1.loglog(results['k_normalized'], results['P_normalized'], 'bo-', 
                   markersize=4, linewidth=1.5, label='RT Simulation')
        
        # Add reference slopes for comparison
        k_ref = results['k_normalized']
        
        # Kolmogorov -5/3 slope reference
        if len(k_ref) > 5:
            k_mid_start = len(k_ref) // 3
            k_mid_end = 2 * len(k_ref) // 3
            k_mid = k_ref[k_mid_start:k_mid_end]
            P_kolmogorov = k_mid**(-5/3) * results['P_normalized'][k_mid_start]
            ax1.loglog(k_mid, P_kolmogorov, 'r--', linewidth=2, 
                      label='Kolmogorov k^(-5/3)', alpha=0.7)
        
        # -3 slope reference (steeper cascade)
        if len(k_ref) > 5:
            P_steep = k_ref**(-3) * results['P_normalized'][len(k_ref)//4]
            ax1.loglog(k_ref[len(k_ref)//4:], P_steep[len(k_ref)//4:], 'g--', 
                      linewidth=2, label='Steep k^(-3)', alpha=0.7)
        
        ax1.set_xlabel('k/k₀', fontsize=12)
        ax1.set_ylabel('P/P₀', fontsize=12)
        ax1.set_title(f'Dalziel-Style Power Spectrum\nt = {results["time"]:.3f}', fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Add text box with key parameters
        textstr = f'''Grid: {results["grid_shape"][1]}×{results["grid_shape"][0]}
k₀ = {results["k_0"]:.4f}
P₀ = {results["P_0"]:.2e}
Total E = {results["total_energy"]:.2e}'''
        ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Plot 2: Physical spectrum
        ax2.loglog(results['k_physical'], results['P_physical'], 'mo-', 
                   markersize=4, linewidth=1.5, label='Physical Spectrum')
        
        ax2.set_xlabel('k [1/length]', fontsize=12)
        ax2.set_ylabel('E(k) [energy]', fontsize=12)
        ax2.set_title(f'Physical Power Spectrum\nt = {results["time"]:.3f}', fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        
        if save_plots:
            plot_file = os.path.join(output_dir, f'dalziel_power_spectrum_t{results["time"]:.3f}.png')
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            print(f"Plot saved: {plot_file}")
        
        plt.show()
        
        return fig
    
    def save_power_spectrum_results(self, results, output_dir):
        """Save power spectrum results to CSV files."""
        import pandas as pd
        
        # Save normalized spectrum (Dalziel format)
        dalziel_df = pd.DataFrame({
            'k_over_k0': results['k_normalized'],
            'P_over_P0': results['P_normalized']
        })
        dalziel_file = os.path.join(output_dir, f'dalziel_spectrum_t{results["time"]:.3f}.csv')
        dalziel_df.to_csv(dalziel_file, index=False)
        
        # Save physical spectrum
        physical_df = pd.DataFrame({
            'k_physical': results['k_physical'],
            'E_k': results['P_physical']
        })
        physical_file = os.path.join(output_dir, f'physical_spectrum_t{results["time"]:.3f}.csv')
        physical_df.to_csv(physical_file, index=False)
        
        # Save summary parameters
        summary_df = pd.DataFrame({
            'Parameter': ['time', 'grid_nx', 'grid_ny', 'dx', 'dy', 'k_0', 'P_0', 'total_energy'],
            'Value': [results['time'], results['grid_shape'][1], results['grid_shape'][0],
                     results['dx'], results['dy'], results['k_0'], results['P_0'], results['total_energy']]
        })
        summary_file = os.path.join(output_dir, f'spectrum_summary_t{results["time"]:.3f}.csv')
        summary_df.to_csv(summary_file, index=False)
        
        print(f"Results saved:")
        print(f"  Dalziel format: {dalziel_file}")
        print(f"  Physical spectrum: {physical_file}")
        print(f"  Summary: {summary_file}")

def main():
    """Command line interface for Dalziel power spectrum analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Dalziel Power Spectrum Analysis for RT Velocity Fields',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze single VTK file
  python dalziel_power_spectrum_analyzer.py --file RT800x800-7999.vtk

  # Analyze with output directory
  python dalziel_power_spectrum_analyzer.py --file RT800x800-7999.vtk --output dalziel_analysis

  # Analyze multiple files
  python dalziel_power_spectrum_analyzer.py --pattern "RT800x800-*.vtk" --output dalziel_series

  # Debug mode
  python dalziel_power_spectrum_analyzer.py --file RT800x800-7999.vtk --debug
""")
    
    parser.add_argument('--file', help='Single VTK file to analyze')
    parser.add_argument('--pattern', help='Pattern for multiple VTK files')
    parser.add_argument('--output', default='./dalziel_analysis', 
                       help='Output directory (default: ./dalziel_analysis)')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    
    args = parser.parse_args()
    
    if not args.file and not args.pattern:
        parser.print_help()
        return
    
    # Create analyzer
    analyzer = DalzielPowerSpectrumAnalyzer(debug=args.debug)
    
    if args.file:
        # Single file analysis
        print(f"Analyzing single file: {args.file}")
        results = analyzer.analyze_rt_power_spectrum(args.file, args.output)
        
        print(f"\n📊 ANALYSIS SUMMARY")
        print(f"File: {os.path.basename(args.file)}")
        print(f"Time: {results['time']:.3f}")
        print(f"Peak wavenumber k₀: {results['k_0']:.4f}")
        print(f"Peak power P₀: {results['P_0']:.2e}")
        
    elif args.pattern:
        # Multiple files analysis
        import glob
        vtk_files = sorted(glob.glob(args.pattern))
        
        if not vtk_files:
            print(f"No files found matching pattern: {args.pattern}")
            return
        
        print(f"Analyzing {len(vtk_files)} files matching: {args.pattern}")
        
        all_results = []
        for i, vtk_file in enumerate(vtk_files):
            print(f"\nProcessing {i+1}/{len(vtk_files)}: {os.path.basename(vtk_file)}")
            
            # Create subdirectory for each time
            file_output_dir = os.path.join(args.output, f"t_{i:04d}")
            
            try:
                results = analyzer.analyze_rt_power_spectrum(vtk_file, file_output_dir)
                all_results.append(results)
            except Exception as e:
                print(f"Error processing {vtk_file}: {e}")
        
        if all_results:
            # Create summary plot of evolution
            analyzer.plot_power_spectrum_evolution(all_results, args.output)
            print(f"\n🎉 Analysis complete! Results saved to: {args.output}")
    
if __name__ == "__main__":
    main()
