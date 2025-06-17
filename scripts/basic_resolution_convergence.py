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
from scipy import stats

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

def analyze_resolution_convergence(base_dir, target_time=5.0, 
    output_dir="./resolution_convergence", 
	mixing_method='dalziel', h0=0.5,
	use_grid_optimization=False, min_box_size=None, no_titles=False):
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
            analyzer = RTAnalyzer(os.path.join(output_dir, f"res_{resolution}"),
			                     use_grid_optimization=use_grid_optimization, no_titles=no_titles)
            result = analyzer.analyze_vtk_file(vtk_file, mixing_method=mixing_method,h0=h0,min_box_size=min_box_size)
            
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
    create_convergence_plots(df, target_time, output_dir, no_titles)
    
    # Perform Richardson extrapolation if we have enough points

    if len(df) >= 3:
        results_summary = alternative_convergence_analysis(df, target_time, output_dir, no_titles)
        
        if results_summary['richardson_successful']:
            print(f"Richardson extrapolation successful!")
            print(f"D(∞) = {results_summary['D_infinity']:.6f} ± {results_summary['D_infinity_error']:.6f}")
        else:
            print(f"Using alternative convergence analysis")
            print(f"Finest grid: D = {results_summary['finest_grid_dimension']:.6f}")
            print(f"Trend estimate: D ≈ {results_summary['predicted_asymptote_simple']:.6f}")
    else:
        print(f"Need at least 3 data points for convergence analysis, got {len(df)}")

    return df

def create_convergence_plots(df, target_time, output_dir, no_titles=False):
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
    if not no_titles:
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
    if not no_titles:
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
    if not no_titles:
        plt.title(f'Fractal Dimension Fit Quality vs Resolution (t ≈ {target_time})', fontsize=16)
    plt.grid(True, alpha=0.7)
    plt.ylim(0.9, 1.0)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fit_quality_convergence.png'), dpi=300)
    plt.close()

def find_monotonic_subset(df, min_points=3):
    """
    Find the longest monotonic subsequence in the data.
    Prioritizes the finest grids since they're most reliable.
    """
    print(f"\nFinding monotonic subsequence from {len(df)} points...")
    
    # Sort by resolution to ensure proper order
    df_sorted = df.sort_values('resolution').reset_index(drop=True)
    
    resolutions = df_sorted['resolution'].values
    dimensions = df_sorted['fractal_dim'].values
    
    print(f"Original data:")
    for i, (res, dim) in enumerate(zip(resolutions, dimensions)):
        print(f"  {res:4d}×{res:<4d}: D = {dim:.6f}")
    
    best_subset = None
    best_length = 0
    best_start = 0
    
    # Try all possible subsequences, prioritizing those that include finest grids
    for start in range(len(df_sorted)):
        for end in range(start + min_points, len(df_sorted) + 1):
            subset_dims = dimensions[start:end]
            
            # Check if this subsequence is monotonic
            is_increasing = all(subset_dims[i] <= subset_dims[i+1] for i in range(len(subset_dims)-1))
            is_decreasing = all(subset_dims[i] >= subset_dims[i+1] for i in range(len(subset_dims)-1))
            
            if is_increasing or is_decreasing:
                # Prefer longer subsequences, and among equal lengths, prefer those with finer grids
                length = end - start
                includes_finest = (end == len(df_sorted))  # Includes finest grid
                
                is_better = (length > best_length) or \
                           (length == best_length and includes_finest and not (best_start + best_length == len(df_sorted)))
                
                if is_better:
                    best_subset = df_sorted.iloc[start:end].copy()
                    best_length = length
                    best_start = start
                    trend = "increasing" if is_increasing else "decreasing"
                    print(f"  Found {trend} subsequence: length {length}, resolutions {resolutions[start]} to {resolutions[end-1]}")
    
    if best_subset is not None:
        print(f"\nSelected monotonic subset ({len(best_subset)} points):")
        for _, row in best_subset.iterrows():
            print(f"  {row['resolution']:4d}×{row['resolution']:<4d}: D = {row['fractal_dim']:.6f}")
        return best_subset
    else:
        print(f"No monotonic subsequence of length >= {min_points} found!")
        return None

def analyze_convergence_trend(df):
    """Analyze convergence behavior even if not perfectly monotonic."""
    
    # Sort by resolution
    df_sorted = df.sort_values('resolution').reset_index(drop=True)
    resolutions = df_sorted['resolution'].values
    dimensions = df_sorted['fractal_dim'].values
    
    print(f"\nConvergence trend analysis:")
    
    # Calculate differences between consecutive points
    res_ratios = resolutions[1:] / resolutions[:-1]
    dim_changes = dimensions[1:] - dimensions[:-1]
    
    print(f"Resolution ratios: {res_ratios}")
    print(f"Dimension changes: {dim_changes}")
    
    # Overall trend using linear regression on log(resolution) vs dimension
    log_res = np.log(resolutions)
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_res, dimensions)
    
    print(f"Overall trend (log-linear fit):")
    print(f"  Slope: {slope:.6f} (positive = increasing with resolution)")
    print(f"  R²: {r_value**2:.4f}")
    print(f"  p-value: {p_value:.4f}")
    
    # Predict asymptotic value using simple extrapolation
    finest_res = resolutions[-1]
    predicted_asymptote = intercept + slope * np.log(10000)  # Extrapolate to very fine grid
    
    print(f"Simple extrapolation to N=10000: D ≈ {predicted_asymptote:.6f}")
    
    return slope, r_value**2, predicted_asymptote

def alternative_convergence_analysis(df, target_time, output_dir, no_titles=False):
    """
    Alternative analysis for non-monotonic data that still provides useful insights.
    """
    print(f"\n{'='*60}")
    print("ALTERNATIVE CONVERGENCE ANALYSIS")
    print("(For non-monotonic Richardson extrapolation)")
    print(f"{'='*60}")
    
    # Sort by resolution
    df_sorted = df.sort_values('resolution').reset_index(drop=True)
    
    # 1. Try to find monotonic subset for Richardson extrapolation
    monotonic_subset = find_monotonic_subset(df_sorted, min_points=3)
    
    richardson_success = False
    D_inf = None
    D_inf_err = None
    
    if monotonic_subset is not None:
        print(f"\nAttempting Richardson extrapolation on monotonic subset...")
        try:
            h_values = 1.0 / monotonic_subset['resolution'].values
            dim_values = monotonic_subset['fractal_dim'].values
            
            # Try fitting with reasonable initial guess
            p0 = [dim_values[-1], -0.01, 1.0]
            params, pcov = curve_fit(richardson_extrapolation, h_values, dim_values, p0=p0)
            D_inf, C, p = params
            param_errors = np.sqrt(np.diag(pcov))
            D_inf_err = param_errors[0]
            
            # Check if result is reasonable
            if 1.0 <= D_inf <= 2.0 and 0.1 <= p <= 5.0:
                richardson_success = True
                print(f"SUCCESS: D(∞) = {D_inf:.6f} ± {D_inf_err:.6f}")
                print(f"Model: D(N) = {D_inf:.6f} + ({C:.6f}) × N^(-{p:.6f})")
            else:
                print(f"FAILED: Unreasonable parameters (D_inf={D_inf:.6f}, p={p:.6f})")
        
        except Exception as e:
            print(f"FAILED: {str(e)}")
    
    # 2. Trend analysis on all data
    slope, r_squared, predicted_asymptote = analyze_convergence_trend(df_sorted)
    
    # 3. Create comprehensive plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: All data with trend
    ax1.errorbar(df_sorted['resolution'], df_sorted['fractal_dim'], 
                yerr=df_sorted['fd_error'], fmt='bo-', capsize=5, 
                linewidth=2, markersize=8, label='All data')
    
    if monotonic_subset is not None:
        ax1.plot(monotonic_subset['resolution'], monotonic_subset['fractal_dim'], 
                'ro-', linewidth=3, markersize=10, alpha=0.7, label='Monotonic subset')
    
    # Add trend line
    res_trend = np.logspace(np.log10(df_sorted['resolution'].min()), 
                           np.log10(df_sorted['resolution'].max() * 2), 100)
    dim_trend = np.interp(np.log(res_trend), np.log(df_sorted['resolution']), 
                         df_sorted['fractal_dim'])
    ax1.plot(res_trend, dim_trend, 'g--', alpha=0.7, label=f'Trend (R²={r_squared:.3f})')
    
    ax1.set_xscale('log', base=2)
    ax1.set_xlabel('Grid Resolution (N)')
    ax1.set_ylabel('Fractal Dimension')
    if not no_titles:
        ax1.set_title(f'Fractal Dimension vs Resolution (t ≈ {target_time})')
    ax1.grid(True, alpha=0.7)
    ax1.legend()
    
    # Add resolution labels
    for _, row in df_sorted.iterrows():
        ax1.annotate(f"{row['resolution']}", 
                    (row['resolution'], row['fractal_dim']),
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    # Plot 2: Richardson extrapolation if successful
    if richardson_success and monotonic_subset is not None:
        ax2.errorbar(monotonic_subset['resolution'], monotonic_subset['fractal_dim'],
                    yerr=monotonic_subset['fd_error'], fmt='ro', capsize=5,
                    markersize=10, label='Monotonic data')
        
        # Extrapolation curve
        N_curve = np.logspace(np.log10(monotonic_subset['resolution'].min()), 
                             np.log10(10000), 100)
        h_curve = 1.0 / N_curve
        D_curve = richardson_extrapolation(h_curve, D_inf, C, p)
        
        ax2.plot(N_curve, D_curve, 'r--', linewidth=2, label='Richardson extrapolation')
        ax2.axhline(y=D_inf, color='k', linestyle=':', linewidth=2,
                   label=f'D(∞) = {D_inf:.4f}')
        
        ax2.set_xscale('log', base=2)
        ax2.set_xlabel('Grid Resolution (N)')
        ax2.set_ylabel('Fractal Dimension')
        if not no_titles:
            ax2.set_title('Richardson Extrapolation (Monotonic Subset)')
        ax2.grid(True, alpha=0.7)
        ax2.legend()
    else:
        ax2.text(0.5, 0.5, 'Richardson Extrapolation\nNot Possible\n(No monotonic subset)', 
                ha='center', va='center', transform=ax2.transAxes, fontsize=14,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.7))
        if not no_titles:
            ax2.set_title('Richardson Extrapolation Failed')
    
    # Plot 3: Convergence metrics
    if len(df_sorted) > 1:
        resolution_ratios = df_sorted['resolution'].values[1:] / df_sorted['resolution'].values[:-1]
        dimension_changes = np.diff(df_sorted['fractal_dim'].values)
        
        ax3.plot(df_sorted['resolution'].values[1:], dimension_changes, 'go-', 
                linewidth=2, markersize=8)
        ax3.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax3.set_xscale('log', base=2)
        ax3.set_xlabel('Grid Resolution (N)')
        ax3.set_ylabel('Δ Fractal Dimension')
        if not no_titles:
            ax3.set_title('Dimension Change Between Consecutive Grids')
        ax3.grid(True, alpha=0.7)
    
    # Plot 4: R² vs resolution for fit quality
    ax4.plot(df_sorted['resolution'], df_sorted['fd_r_squared'], 'mo-', 
            linewidth=2, markersize=8)
    ax4.set_xscale('log', base=2)
    ax4.set_xlabel('Grid Resolution (N)')
    ax4.set_ylabel('Fractal Analysis R²')
    if not no_titles:
        ax4.set_title('Fit Quality vs Resolution')
    ax4.grid(True, alpha=0.7)
    ax4.set_ylim(0.9, 1.0)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/alternative_convergence_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Save detailed results
    results_summary = {
        'total_resolutions': len(df_sorted),
        'monotonic_subset_length': len(monotonic_subset) if monotonic_subset is not None else 0,
        'richardson_successful': richardson_success,
        'D_infinity': D_inf,
        'D_infinity_error': D_inf_err,
        'overall_trend_slope': slope,
        'overall_trend_r_squared': r_squared,
        'predicted_asymptote_simple': predicted_asymptote,
        'finest_grid_dimension': df_sorted['fractal_dim'].iloc[-1],
        'finest_grid_error': df_sorted['fd_error'].iloc[-1]
    }
    
    # Create summary DataFrame
    summary_df = pd.DataFrame([results_summary])
    summary_df.to_csv(f'{output_dir}/convergence_summary.csv', index=False)
    
    # Print final recommendations
    print(f"\n{'='*60}")
    print("CONVERGENCE ANALYSIS SUMMARY")
    print(f"{'='*60}")
    
    print(f"Total resolutions analyzed: {len(df_sorted)}")
    print(f"Monotonic subset available: {'Yes' if monotonic_subset is not None else 'No'}")
    print(f"Richardson extrapolation: {'Successful' if richardson_success else 'Failed'}")
    
    if richardson_success:
        print(f"Best estimate: D(∞) = {D_inf:.6f} ± {D_inf_err:.6f}")
    else:
        finest_dim = df_sorted['fractal_dim'].iloc[-1]
        finest_err = df_sorted['fd_error'].iloc[-1]
        print(f"Finest grid result: D = {finest_dim:.6f} ± {finest_err:.6f}")
        print(f"Simple trend extrapolation: D ≈ {predicted_asymptote:.6f}")
    
    print(f"\nRecommendations:")
    if not richardson_success:
        print("- Consider more resolution levels to establish monotonic convergence")
        print("- Try different time points that may show better convergence")
        print("- Use finest grid result as best available estimate")
        print("- Check if interface is sufficiently resolved at coarse grids")
    else:
        print("- Richardson extrapolation successful on monotonic subset")
        print("- Results should be reliable for the resolved scales")
    
    return results_summary

def perform_richardson_extrapolation(df, target_time, output_dir, no_titles=False):
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
        if not no_titles:
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
    parser.add_argument('--disable-grid-optimization', action='store_true',
                      help='Disable grid optimization (use basic method)')
    parser.add_argument('--min-box-size', type=float, default=None,
                      help='Minimum box size for fractal analysis. If not specified, '
                        'will be auto-estimated from interface segment lengths (recommended). '
                        'Use smaller values (e.g., 0.001) for higher accuracy on fine interfaces.')
    parser.add_argument('--no-titles', action='store_true',
                       help='Disable plot titles for journal submissions')
    args = parser.parse_args()
    
	# Convert disable flag to use flag
    use_grid_optimization = not args.disable_grid_optimization
    
	# Run the analysis
    results = analyze_resolution_convergence(args.base_dir, args.time, args.output, args.mixing_method, args.h0,
	           use_grid_optimization, args.min_box_size, args.no_titles)
    
    if results is not None:
        print(f"\nResolution convergence analysis complete!")
        print(f"Results saved to: {args.output}")
    else:
        print("Analysis failed - check file paths and data availability")

if __name__ == "__main__":
    main()
