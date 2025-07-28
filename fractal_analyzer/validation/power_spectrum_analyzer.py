#!/usr/bin/env python3
"""
Enhanced Dalziel Power Spectrum Analyzer - Both Velocity and Volume Fraction Spectra
Compatible with fractal_analyzer suite
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

class EnhancedDalzielPowerSpectrumAnalyzer:
    """
    Enhanced power spectrum analyzer for both velocity and volume fraction fields.
    Implements both kinetic energy and mixing spectra following Dalziel et al. approach.
    """
    
    def __init__(self, debug=False):
        self.debug = debug
        self.rt_analyzer = RTAnalyzer()
        
    def find_fields(self, data):
        """
        Find velocity and volume fraction fields in VTK data with flexible naming.
        
        Args:
            data: VTK data dictionary
            
        Returns:
            tuple: (u_field, v_field, f_field) where f_field is volume fraction
        """
        # Possible field names for velocity components
        u_names = ['U', 'u', 'velocity_x', 'vel_x', 'u_velocity']
        v_names = ['V', 'v', 'velocity_y', 'vel_y', 'v_velocity']
        
        # Possible field names for volume fraction
        f_names = ['F', 'f', 'volume_fraction', 'vof', 'concentration', 'C']
        
        u_field = None
        v_field = None
        f_field = None
        
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
        
        # Find volume fraction
        for name in f_names:
            if name in data:
                f_field = data[name]
                print(f"Found volume fraction as '{name}'")
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
        
        if f_field is None:
            print("Warning: Could not find volume fraction field")
            print("Available fields:", list(data.keys()))
        
        return u_field, v_field, f_field
    
    def extract_domain_info(self, data):
        """
        Extract physical domain information from VTK data.
        
        Args:
            data: VTK data dictionary containing coordinate arrays
            
        Returns:
            dict: Domain information including physical dimensions and grid spacing
        """
        x_grid = data['x']
        y_grid = data['y']
        
        # Extract physical domain dimensions [m]
        x_min, x_max = np.min(x_grid), np.max(x_grid)
        y_min, y_max = np.min(y_grid), np.max(y_grid)
        
        L_x = x_max - x_min  # Domain width [m]
        L_y = y_max - y_min  # Domain height [m]
        
        # Grid dimensions
        ny, nx = x_grid.shape
        
        # Physical grid spacing [m]
        if nx > 1:
            dx = L_x / (nx - 1)
        else:
            dx = L_x / nx
            
        if ny > 1:
            dy = L_y / (ny - 1)
        else:
            dy = L_y / ny
        
        domain_info = {
            'L_x': L_x,           # Domain width [m]
            'L_y': L_y,           # Domain height [m]
            'dx': dx,             # Grid spacing x [m]
            'dy': dy,             # Grid spacing y [m]
            'nx': nx,             # Grid points x
            'ny': ny,             # Grid points y
            'x_range': (x_min, x_max),
            'y_range': (y_min, y_max)
        }
        
        if self.debug:
            print(f"Domain information:")
            print(f"  Physical domain: {L_x:.3f} × {L_y:.3f} m")
            print(f"  Grid: {nx} × {ny} points")
            print(f"  Grid spacing: dx={dx:.6f} m, dy={dy:.6f} m")
            print(f"  Expected L = 0.4 m: {'✓' if abs(L_x - 0.4) < 0.01 else '✗'}")
        
        return domain_info
    
    def compute_radial_spectrum(self, field_2d_fft, dx, dy, field_name="field"):
        """
        Generic radial averaging for any 2D FFT field.
        
        Args:
            field_2d_fft: 2D FFT of the field
            dx, dy: Grid spacing
            field_name: Name for debugging output
            
        Returns:
            tuple: (k_normalized, P_normalized, k_physical, P_physical, k_0, P_0)
        """
        ny, nx = field_2d_fft.shape
        
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
                P_total[i] = np.sum(field_2d_fft[mask])
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
            print(f"  {field_name} spectrum characteristics:")
            print(f"    k_0 = {k_0:.4f}")
            print(f"    P_0 = {P_0:.4e}")
        
        return k_normalized, P_normalized, k_centers, P_total, k_0, P_0
    
    def compute_velocity_spectrum(self, u_field, v_field, dx, dy):
        """
        Compute velocity power spectrum (kinetic energy spectrum).
        """
        if u_field is None or v_field is None:
            return None
        
        if self.debug:
            print(f"Velocity spectrum calculation:")
            print(f"  Velocity ranges: u ∈ [{np.min(u_field):.6f}, {np.max(u_field):.6f}]")
            print(f"  Velocity ranges: v ∈ [{np.min(v_field):.6f}, {np.max(v_field):.6f}]")
        
        # Remove mean velocities (focus on fluctuations)
        u_fluct = u_field - np.mean(u_field)
        v_fluct = v_field - np.mean(v_field)
        
        # Compute 2D FFT of velocity components
        u_hat = fft.fft2(u_fluct)
        v_hat = fft.fft2(v_fluct)
        
        # Kinetic energy spectrum in Fourier space
        kinetic_energy_2d = 0.5 * (np.abs(u_hat)**2 + np.abs(v_hat)**2)
        
        return self.compute_radial_spectrum(kinetic_energy_2d, dx, dy, "Velocity")
    
    def compute_volume_fraction_spectrum(self, f_field, dx, dy):
        """
        Compute volume fraction power spectrum following Dalziel et al. mixing analysis.
        
        This analyzes mixing structure and interface dynamics using volume fraction F
        as the scalar field (equivalent to Dalziel's concentration field for mixing studies).
        
        Args:
            f_field: Volume fraction field (0 ≤ F ≤ 1)
            dx, dy: Physical grid spacing in meters
            
        Returns:
            Spectral analysis results (k_norm, P_norm, k_phys, P_phys, k_0, P_0)
        """
        if f_field is None:
            return None
        
        if self.debug:
            print(f"Volume fraction spectrum calculation:")
            print(f"  Volume fraction range: F ∈ [{np.min(f_field):.6f}, {np.max(f_field):.6f}]")
            print(f"  Mean volume fraction: {np.mean(f_field):.6f}")
        
        # Remove mean volume fraction (focus on fluctuations) - key for mixing analysis
        f_fluct = f_field - np.mean(f_field)
        
        # Compute 2D FFT of volume fraction fluctuations
        f_hat = fft.fft2(f_fluct)
        
        # Volume fraction power spectrum (mixing spectrum)
        vof_power_2d = np.abs(f_hat)**2
        
        return self.compute_radial_spectrum(vof_power_2d, dx, dy, "Volume Fraction")
    
    def analyze_rt_dual_spectrum(self, vtk_file, output_dir=None):
        """
        Analyze RT simulation VTK file for both velocity and volume fraction spectra.
        Handles MKS units and physical domain scaling.
        """
        print(f"🌊 ENHANCED DALZIEL DUAL SPECTRUM ANALYSIS")
        print(f"=" * 60)
        print(f"Analyzing: {os.path.basename(vtk_file)}")
        
        # Read VTK file
        data = self.rt_analyzer.read_vtk_file(vtk_file)
        
        # Find all fields
        u_field, v_field, f_field = self.find_fields(data)
        
        # Extract domain information with proper MKS units
        domain_info = self.extract_domain_info(data)
        dx, dy = domain_info['dx'], domain_info['dy']
        
        # Compute velocity power spectrum
        velocity_results = self.compute_velocity_spectrum(u_field, v_field, dx, dy)
        
        # Compute volume fraction power spectrum (mixing analysis)
        vof_results = self.compute_volume_fraction_spectrum(f_field, dx, dy)
        
        # Create comprehensive results with physical units
        results = {
            'file': vtk_file,
            'time': data.get('time', 0.0),
            'domain_info': domain_info,
            'velocity_spectrum': velocity_results,
            'volume_fraction_spectrum': vof_results
        }
        
        # Print summary
        print(f"Analysis complete:")
        print(f"  Time: {results['time']:.3f} s")
        print(f"  Domain: {domain_info['L_x']:.3f} × {domain_info['L_y']:.3f} m")
        print(f"  Grid: {domain_info['nx']}×{domain_info['ny']}")
        
        if velocity_results:
            k0_vel = velocity_results[4]  # [1/m]
            P0_vel = velocity_results[5]  # [energy units]
            print(f"  Velocity spectrum: k₀ = {k0_vel:.4f} m⁻¹, P₀ = {P0_vel:.4e}")
        else:
            print("  Velocity spectrum: Not computed (missing fields)")
            
        if vof_results:
            k0_vof = vof_results[4]  # [1/m]  
            P0_vof = vof_results[5]  # [dimensionless^2 * m^2]
            print(f"  Volume fraction spectrum: k₀ = {k0_vof:.4f} m⁻¹, P₀ = {P0_vof:.4e}")
        else:
            print("  Volume fraction spectrum: Not computed (missing field)")
        
        # Save results if requested
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            self.plot_dual_spectrum(results, output_dir)
            self.save_dual_results(results, output_dir)
        
        return results
    
    def plot_dual_spectrum(self, results, output_dir):
        """Create comprehensive plot with both velocity and volume fraction spectra."""
        # Determine subplot layout
        has_velocity = results['velocity_spectrum'] is not None
        has_vof = results['volume_fraction_spectrum'] is not None
        
        if has_velocity and has_vof:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        elif has_velocity or has_vof:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            ax3 = ax4 = None
        else:
            print("No spectra to plot")
            return
        
        time = results['time']
        
        # Plot velocity spectrum if available
        if has_velocity:
            vel_results = results['velocity_spectrum']
            k_norm, P_norm, k_phys, P_phys = vel_results[:4]
            
            # Normalized velocity spectrum
            ax1.loglog(k_norm, P_norm, 'bo-', markersize=4, linewidth=1.5, 
                      label='Velocity (Kinetic Energy)')
            
            # Add reference slopes
            if len(k_norm) > 5:
                k_mid = k_norm[len(k_norm)//3:2*len(k_norm)//3]
                if len(k_mid) > 0:
                    P_k53 = k_mid**(-5/3) * P_norm[len(k_norm)//3]
                    ax1.loglog(k_mid, P_k53, 'r--', linewidth=2, 
                              label='k^(-5/3)', alpha=0.7)
            
            ax1.set_xlabel('k/k₀', fontsize=12)
            ax1.set_ylabel('E(k)/E₀', fontsize=12)
            ax1.set_title(f'Velocity Power Spectrum\nt = {time:.3f} s', fontsize=14)
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            
            # Physical velocity spectrum with MKS units
            L_x = results['domain_info']['L_x']
            ax2.loglog(k_phys, P_phys, 'mo-', markersize=4, linewidth=1.5, 
                      label=f'Physical Spectrum (L={L_x:.2f}m)')
            ax2.set_xlabel('k [m⁻¹]', fontsize=12)
            ax2.set_ylabel('E(k) [m³/s²]', fontsize=12)
            ax2.set_title(f'Physical Velocity Spectrum\nt = {time:.3f} s', fontsize=14)
            ax2.grid(True, alpha=0.3)
            ax2.legend()
        
        # Plot volume fraction spectrum if available
        if has_vof:
            vof_results = results['volume_fraction_spectrum']
            k_norm_c, P_norm_c, k_phys_c, P_phys_c = vof_results[:4]
            
            # Choose axes based on layout
            if has_velocity and has_vof:
                ax_norm = ax3
                ax_phys = ax4
            else:
                ax_norm = ax1
                ax_phys = ax2
            
            # Normalized volume fraction spectrum
            ax_norm.loglog(k_norm_c, P_norm_c, 'go-', markersize=4, linewidth=1.5, 
                          label='Volume Fraction F (Dalziel-style)')
            
            # Add reference slopes for volume fraction mixing
            if len(k_norm_c) > 5:
                k_mid_c = k_norm_c[len(k_norm_c)//3:2*len(k_norm_c)//3]
                if len(k_mid_c) > 0:
                    # Dalziel often shows k^(-5/3) in inertial range, k^(-3) at higher k
                    P_k53_c = k_mid_c**(-5/3) * P_norm_c[len(k_norm_c)//3]
                    ax_norm.loglog(k_mid_c, P_k53_c, 'r--', linewidth=2, 
                                  label='k^(-5/3)', alpha=0.7)
                    
                    if len(k_norm_c) > 10:
                        k_high = k_norm_c[2*len(k_norm_c)//3:]
                        if len(k_high) > 0:
                            P_k3 = k_high**(-3) * P_norm_c[2*len(k_norm_c)//3]
                            ax_norm.loglog(k_high, P_k3, 'b--', linewidth=2, 
                                          label='k^(-3)', alpha=0.7)
            
            ax_norm.set_xlabel('k/k₀', fontsize=12)
            ax_norm.set_ylabel('F(k)/F₀', fontsize=12)
            ax_norm.set_title(f'Volume Fraction Power Spectrum\nt = {time:.3f} s', fontsize=14)
            ax_norm.grid(True, alpha=0.3)
            ax_norm.legend()
            
            # Physical volume fraction spectrum with MKS units
            L_x = results['domain_info']['L_x']
            ax_phys.loglog(k_phys_c, P_phys_c, 'co-', markersize=4, linewidth=1.5, 
                          label=f'Physical Spectrum (L={L_x:.2f}m)')
            ax_phys.set_xlabel('k [m⁻¹]', fontsize=12)
            ax_phys.set_ylabel('F(k) [dimensionless²·m²]', fontsize=12)
            ax_phys.set_title(f'Physical Volume Fraction Spectrum\nt = {time:.3f} s', fontsize=14)
            ax_phys.grid(True, alpha=0.3)
            ax_phys.legend()
        
        plt.tight_layout()
        
        plot_file = os.path.join(output_dir, f'dual_spectrum_t{time:.3f}.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved: {plot_file}")
        
        plt.show()
    
    def save_dual_results(self, results, output_dir):
        """Save both velocity and volume fraction results to CSV files."""
        import pandas as pd
        
        time = results['time']
        
        # Save velocity spectrum if available
        if results['velocity_spectrum']:
            vel_results = results['velocity_spectrum']
            df_vel = pd.DataFrame({
                'k_over_k0': vel_results[0],
                'E_over_E0': vel_results[1],
                'k_physical': vel_results[2],
                'E_physical': vel_results[3]
            })
            csv_file = os.path.join(output_dir, f'velocity_spectrum_t{time:.3f}.csv')
            df_vel.to_csv(csv_file, index=False)
            print(f"Velocity results saved: {csv_file}")
        
        # Save volume fraction spectrum if available
        if results['volume_fraction_spectrum']:
            vof_results = results['volume_fraction_spectrum']
            df_vof = pd.DataFrame({
                'k_over_k0': vof_results[0],
                'F_over_F0': vof_results[1],  # Updated column name
                'k_physical': vof_results[2],
                'F_physical': vof_results[3]  # Updated column name
            })
            csv_file = os.path.join(output_dir, f'volume_fraction_spectrum_t{time:.3f}.csv')
            df_vof.to_csv(csv_file, index=False)
            print(f"Volume fraction results saved: {csv_file}")

def main():
    """Enhanced main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Enhanced Dalziel Dual Spectrum Analysis')
    parser.add_argument('--file', help='VTK file to analyze')
    parser.add_argument('--output', default='./dual_spectrum_analysis', help='Output directory')
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
    analyzer = EnhancedDalzielPowerSpectrumAnalyzer(debug=args.debug)
    
    try:
        results = analyzer.analyze_rt_dual_spectrum(args.file, args.output)
        print(f"✅ Dual spectrum analysis complete! Results in: {args.output}")
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
