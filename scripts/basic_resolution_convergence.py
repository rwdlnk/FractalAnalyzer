#!/usr/bin/env python3
"""
Simple Resolution Convergence Analysis for RT Fractal Dimension
Uses working basic fractal analysis (not multifractal)
"""

import os
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from fractal_analyzer import RTAnalyzer

def find_closest_time_file(base_dir, resolution, target_time):
    """Find VTK file closest to target time for given resolution."""
    pattern = os.path.join(base_dir, f"{resolution}x{resolution}", f"RT{resolution}x{resolution}-*.vtk")
    files = glob.glob(pattern)
    
    if not files:
        return None
    
    # Extract times from filenames and find closest
    best_file = None
    best_diff = float('inf')
    
    for file in files:
        try:
            # Extract time from filename like RT800x800-5999.vtk
            time_str = os.path.basename(file).split('-')[1].split('.')[0]
            file_time = float(time_str) / 1000.0
            diff = abs(file_time - target_time)
            
            if diff < best_diff:
                best_diff = diff
                best_file = file
        except:
            continue
    
    return best_file

def richardson_extrapolation(h, f_inf, C, p):
    """Richardson extrapolation model: f(h) = f_inf + C*h^p"""
    return f_inf + C * h**p

def analyze_resolution_convergence(base_dir, target_time=5.0, output_dir="./resolution_convergence", mixing_method='dalziel', h0=0.5):
    """Analyze fractal dimension convergence across resolutions."""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Define resolutions to test
    resolutions = [100, 200, 400, 800, 1600]
    
    print(f"Analyzing resolution convergence at target time: {target_time}")
    print("=" * 60)
    
    results = []
    
    # Process each resolution
    for resolution in resolutions:
        print(f"\nProcessing {resolution}×{resolution} resolution...")
        
        # Find closest time file
        vtk_file = find_closest_time_file(base_dir, resolution, target_time)
        
        if vtk_file is None:
            print(f"  Warning: No files found for {resolution}×{resolution}")
            continue
        
        # Extract actual time from filename
        time_str = os.path.basename(vtk_file).split('-')[1].split('.')[0]
        actual_time = float(time_str) / 1000.0
        
        print(f"  Using file: {os.path.basename(vtk_file)} (t = {actual_time:.3f})")
        
        # Analyze this file
        try:
            analyzer = RTAnalyzer(os.path.join(output_dir, f"res_{resolution}"))
            result = analyzer.analyze_vtk_file(vtk_file, mixing_method=mixing_method,h0=h0)
            
            results.append({
                'resolution': resolution,
                'actual_time': actual_time,
                'fractal_dim': result['fractal_dim'],
                'fd_error': result['fd_error'],
                'fd_r_squared': result['fd_r_squared'],
                'h_total': result['h_total'],
                'ht': result['ht'],
                'hb': result['hb'],
                'vtk_file': vtk_file
            })
            
            print(f"  Fractal dimension: {result['fractal_dim']:.6f} ± {result['fd_error']:.6f}")
            print(f"  R²: {result['fd_r_squared']:.6f}")
            print(f"  Mixing thickness: {result['h_total']:.6f}")
            
        except Exception as e:
            print(f"  Error analyzing {vtk_file}: {str(e)}")
            continue
    
    if len(results) < 2:
        print("Not enough successful analyses for convergence study")
        return None
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    df = df.sort_values('resolution')
    
    # Save results
    df.to_csv(os.path.join(output_dir, 'resolution_convergence.csv'), index=False)
    
    print(f"\nResults Summary:")
    print("=" * 60)
    for _, row in df.iterrows():
        print(f"{row['resolution']:4d}×{row['resolution']:<4d}: D = {row['fractal_dim']:.6f} ± {row['fd_error']:.6f}, "
              f"h = {row['h_total']:.6f}")
    
    # Create convergence plots
    create_convergence_plots(df, target_time, output_dir)
    
    # Perform Richardson extrapolation if we have enough points
    if len(df) >= 3:
        perform_richardson_extrapolation(df, target_time, output_dir)
    
    return df

def create_convergence_plots(df, target_time, output_dir):
    """Create convergence plots."""
    
    # Plot 1: Fractal dimension vs resolution
    plt.figure(figsize=(12, 8))
    
    plt.errorbar(df['resolution'], df['fractal_dim'], yerr=df['fd_error'],
                fmt='bo-', capsize=5, linewidth=2, markersize=8, elinewidth=2)
    
    # Add resolution labels
    for _, row in df.iterrows():
        plt.annotate(f"{row['resolution']}×{row['resolution']}", 
                    (row['resolution'], row['fractal_dim']),
                    xytext=(5, 5), textcoords='offset points', fontsize=10)
    
    plt.xscale('log', base=2)
    plt.xlabel('Grid Resolution (N)', fontsize=14)
    plt.ylabel('Fractal Dimension', fontsize=14)
    plt.title(f'Fractal Dimension Resolution Convergence (t ≈ {target_time})', fontsize=16)
    plt.grid(True, alpha=0.7)
    
    # Add asymptote line if enough points
    if len(df) >= 3:
        plt.axhline(y=df['fractal_dim'].iloc[-1], color='r', linestyle='--', alpha=0.7,
                   label=f'Highest resolution: D = {df["fractal_dim"].iloc[-1]:.4f}')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fractal_dimension_convergence.png'), dpi=300)
    plt.close()
    
    # Plot 2: Mixing layer convergence  
    plt.figure(figsize=(12, 8))
    
    plt.plot(df['resolution'], df['h_total'], 'ro-', linewidth=2, markersize=8, label='Total')
    plt.plot(df['resolution'], df['ht'], 'bs--', linewidth=2, markersize=6, label='Upper')
    plt.plot(df['resolution'], df['hb'], 'gd--', linewidth=2, markersize=6, label='Lower')
    
    plt.xscale('log', base=2)
    plt.xlabel('Grid Resolution (N)', fontsize=14)
    plt.ylabel('Mixing Layer Thickness', fontsize=14)
    plt.title(f'Mixing Layer Thickness Resolution Convergence (t ≈ {target_time})', fontsize=16)
    plt.grid(True, alpha=0.7)
    plt.legend(fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'mixing_thickness_convergence.png'), dpi=300)
    plt.close()
    
    # Plot 3: R² quality vs resolution
    plt.figure(figsize=(12, 8))
    
    plt.plot(df['resolution'], df['fd_r_squared'], 'go-', linewidth=2, markersize=8)
    
    plt.xscale('log', base=2)
    plt.xlabel('Grid Resolution (N)', fontsize=14)
    plt.ylabel('R² Value', fontsize=14)
    plt.title(f'Fractal Dimension Fit Quality vs Resolution (t ≈ {target_time})', fontsize=16)
    plt.grid(True, alpha=0.7)
    plt.ylim(0.9, 1.0)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fit_quality_convergence.png'), dpi=300)
    plt.close()

def perform_richardson_extrapolation(df, target_time, output_dir):
    """Perform Richardson extrapolation to infinite resolution."""
    
    print(f"\nPerforming Richardson extrapolation...")
    print("=" * 40)
    
    # Prepare data for extrapolation
    h_values = 1.0 / df['resolution'].values  # h = 1/N
    dim_values = df['fractal_dim'].values
    
    try:
        # Fit Richardson extrapolation model
        # Initial guess: f_inf near last value, small correction
        p0 = [dim_values[-1] + 0.01, -0.1, 1.0]
        
        params, pcov = curve_fit(richardson_extrapolation, h_values, dim_values, p0=p0)
        D_inf, C, p = params
        
        # Calculate uncertainties
        param_errors = np.sqrt(np.diag(pcov))
        D_inf_err, C_err, p_err = param_errors
        
        print(f"Richardson extrapolation results:")
        print(f"  D(∞) = {D_inf:.6f} ± {D_inf_err:.6f}")
        print(f"  Correction: C = {C:.6f} ± {C_err:.6f}")
        print(f"  Power: p = {p:.6f} ± {p_err:.6f}")
        print(f"  Model: D(N) = {D_inf:.6f} + ({C:.6f}) × N^(-{p:.6f})")
        
        # Create extrapolation plot
        plt.figure(figsize=(12, 8))
        
        # Plot measured data
        plt.errorbar(df['resolution'], df['fractal_dim'], yerr=df['fd_error'],
                    fmt='bo', capsize=5, markersize=10, elinewidth=2, 
                    label='Measured values')
        
        # Plot extrapolation curve
        N_curve = np.logspace(np.log10(df['resolution'].min()), np.log10(10000), 100)
        h_curve = 1.0 / N_curve
        D_curve = richardson_extrapolation(h_curve, D_inf, C, p)
        
        plt.plot(N_curve, D_curve, 'r--', linewidth=2, 
                label=f'Richardson extrapolation')
        
        # Add infinite resolution line
        plt.axhline(y=D_inf, color='k', linestyle=':', linewidth=2,
                   label=f'D(∞) = {D_inf:.4f} ± {D_inf_err:.4f}')
        
        # Add resolution labels
        for _, row in df.iterrows():
            plt.annotate(f"{row['resolution']}×{row['resolution']}", 
                        (row['resolution'], row['fractal_dim']),
                        xytext=(5, 5), textcoords='offset points', fontsize=10)
        
        plt.xscale('log', base=2)
        plt.xlabel('Grid Resolution (N)', fontsize=14)
        plt.ylabel('Fractal Dimension', fontsize=14)
        plt.title(f'Richardson Extrapolation to Infinite Resolution (t ≈ {target_time})', fontsize=16)
        plt.grid(True, alpha=0.7)
        plt.legend(fontsize=12)
        
        # Add model equation as text
        plt.figtext(0.5, 0.02, 
                   f"Model: D(N) = {D_inf:.4f} + ({C:.4f}) × N^(-{p:.2f})",
                   ha='center', fontsize=12,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
        
        plt.tight_layout(rect=[0, 0.05, 1, 1])
        plt.savefig(os.path.join(output_dir, 'richardson_extrapolation.png'), dpi=300)
        plt.close()
        
        # Save extrapolation results
        extrap_results = pd.DataFrame({
            'Parameter': ['D_infinity', 'Correction_C', 'Power_p'],
            'Value': [D_inf, C, p],
            'Error': [D_inf_err, C_err, p_err]
        })
        extrap_results.to_csv(os.path.join(output_dir, 'richardson_extrapolation.csv'), index=False)
        
        return D_inf, D_inf_err
        
    except Exception as e:
        print(f"Error in Richardson extrapolation: {str(e)}")
        return None, None

def main():
    """Main function for console script entry point."""
    parser = argparse.ArgumentParser(description='Resolution convergence analysis for RT fractal dimension')
    parser.add_argument('--base-dir', '-d', default='..', 
                       help='Base directory containing resolution subdirectories')
    parser.add_argument('--time', '-t', type=float, default=5.0,
                       help='Target time for analysis (default: 5.0)')
    parser.add_argument('--output', '-o', default='./resolution_convergence',
                       help='Output directory')
    parser.add_argument('--mixing-method', default='dalziel',
                      choices=['geometric', 'statistical', 'dalziel'],
                      help='Method for computing mixing layer thickness (default: dalziel)')
    parser.add_argument('--h0', type=float, default=0.5,
                      help='Initial interface position in physical coordinates (default: 0.5)')
    args = parser.parse_args()
    
    # Run the analysis
    results = analyze_resolution_convergence(args.base_dir, args.time, args.output, args.mixing_method, args.h0)
    
    if results is not None:
        print(f"\nResolution convergence analysis complete!")
        print(f"Results saved to: {args.output}")
    else:
        print("Analysis failed - check file paths and data availability")

if __name__ == "__main__":
    main()
