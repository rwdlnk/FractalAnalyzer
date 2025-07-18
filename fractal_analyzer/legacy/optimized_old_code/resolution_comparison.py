#!/usr/bin/env python3
"""
Simple Resolution Comparison: Analyze fractal dimension convergence across resolutions.

This script compares fractal dimensions across different grid resolutions without
parallel processing complexity, making it more reliable and easier to debug.
"""

import os
import time
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fractal_analyzer.core.rt_analyzer import RTAnalyzer

def find_timestep_file(data_dir, resolution, target_time=0.0, time_tolerance=0.5):
    """
    Find VTK file closest to target time for a given resolution.
    
    Args:
        data_dir: Directory containing VTK files
        resolution: Grid resolution (e.g., 200)
        target_time: Target simulation time
        time_tolerance: Maximum time difference allowed
        
    Returns:
        str or None: Path to best matching VTK file
    """
    pattern = os.path.join(data_dir, f"RT{resolution}x{resolution}-*.vtk")
    vtk_files = glob.glob(pattern)
    
    if not vtk_files:
        return None
    
    best_file = None
    best_diff = float('inf')
    best_time = None
    
    for vtk_file in vtk_files:
        try:
            # Extract time from filename (e.g., RT200x200-1999.vtk -> 1.999)
            basename = os.path.basename(vtk_file)
            time_str = basename.split('-')[1].split('.')[0]
            file_time = float(time_str) / 1000.0
            
            diff = abs(file_time - target_time)
            if diff < best_diff:
                best_diff = diff
                best_file = vtk_file
                best_time = file_time
        except:
            continue
    
    if best_file and best_diff <= time_tolerance:
        print(f"  {resolution}×{resolution}: {os.path.basename(best_file)} (t={best_time:.3f}, target={target_time:.3f})")
        return best_file, best_time
    else:
        print(f"  {resolution}×{resolution}: No suitable file found (best diff: {best_diff:.3f})")
        return None, None

def analyze_single_resolution(data_dir, resolution, target_time, output_dir, analysis_params):
    """
    Analyze fractal dimension for a single resolution.
    
    Args:
        data_dir: Directory containing VTK files
        resolution: Grid resolution
        target_time: Target simulation time
        output_dir: Output directory
        analysis_params: Analysis parameters
        
    Returns:
        dict: Analysis results
    """
    print(f"\n🔍 Analyzing {resolution}×{resolution} resolution")
    print(f"=" * 50)
    
    # Find appropriate VTK file
    file_result = find_timestep_file(data_dir, resolution, target_time)
    if file_result[0] is None:
        return {
            'resolution': resolution,
            'target_time': target_time,
            'actual_time': np.nan,
            'fractal_dim': np.nan,
            'fd_error': np.nan,
            'fd_r_squared': np.nan,
            'h_total': np.nan,
            'ht': np.nan,
            'hb': np.nan,
            'segments': np.nan,
            'processing_time': np.nan,
            'status': 'no_file_found',
            'data_dir': data_dir
        }
    
    vtk_file, actual_time = file_result
    
    # Create analyzer
    res_output_dir = os.path.join(output_dir, f"resolution_{resolution}x{resolution}")
    analyzer = RTAnalyzer(res_output_dir, use_grid_optimization=False, no_titles=False)
    
    try:
        start_time = time.time()
        
        # Perform analysis
        result = analyzer.analyze_vtk_file(
            vtk_file,
            mixing_method=analysis_params.get('mixing_method', 'dalziel'),
            h0=analysis_params.get('h0', 0.5),
            min_box_size=analysis_params.get('min_box_size', None)
        )
        
        processing_time = time.time() - start_time
        
        # Extract interface for segment count
        data = analyzer.read_vtk_file(vtk_file)
        contours = analyzer.extract_interface(data['f'], data['x'], data['y'])
        segments = analyzer.convert_contours_to_segments(contours)
        
        # Compile results
        analysis_result = {
            'resolution': resolution,
            'target_time': target_time,
            'actual_time': actual_time,
            'fractal_dim': result.get('fractal_dim', np.nan),
            'fd_error': result.get('fd_error', np.nan),
            'fd_r_squared': result.get('fd_r_squared', np.nan),
            'h_total': result.get('h_total', np.nan),
            'ht': result.get('ht', np.nan),
            'hb': result.get('hb', np.nan),
            'segments': len(segments),
            'processing_time': processing_time,
            'status': 'success',
            'data_dir': data_dir,
            'vtk_file': os.path.basename(vtk_file),
            'analysis_quality': result.get('analysis_quality', 'unknown')
        }
        
        print(f"✅ Analysis complete:")
        print(f"   Fractal dimension: {analysis_result['fractal_dim']:.4f} ± {analysis_result['fd_error']:.4f}")
        print(f"   R²: {analysis_result['fd_r_squared']:.4f}")
        print(f"   Mixing thickness: {analysis_result['h_total']:.4f}")
        print(f"   Interface segments: {analysis_result['segments']}")
        print(f"   Processing time: {processing_time:.2f}s")
        print(f"   Analysis quality: {analysis_result['analysis_quality']}")
        
        return analysis_result
        
    except Exception as e:
        print(f"❌ Analysis failed: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return {
            'resolution': resolution,
            'target_time': target_time,
            'actual_time': actual_time,
            'fractal_dim': np.nan,
            'fd_error': np.nan,
            'fd_r_squared': np.nan,
            'h_total': np.nan,
            'ht': np.nan,
            'hb': np.nan,
            'segments': np.nan,
            'processing_time': np.nan,
            'status': 'analysis_failed',
            'error': str(e),
            'data_dir': data_dir,
            'vtk_file': os.path.basename(vtk_file) if vtk_file else 'unknown'
        }

def run_resolution_comparison(data_dirs, resolutions, target_time, output_dir, analysis_params):
    """
    Run resolution comparison across multiple resolutions.
    
    Args:
        data_dirs: List of data directories (one per resolution)
        resolutions: List of resolutions to analyze
        target_time: Target simulation time
        output_dir: Output directory
        analysis_params: Analysis parameters
        
    Returns:
        DataFrame with comparison results
    """
    print(f"🔬 RESOLUTION COMPARISON ANALYSIS")
    print(f"=" * 60)
    print(f"Target time: {target_time}")
    print(f"Resolutions: {resolutions}")
    print(f"Output directory: {output_dir}")
    print(f"Mixing method: {analysis_params.get('mixing_method', 'dalziel')}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    results = []
    total_start = time.time()
    
    # Process each resolution sequentially
    for i, (data_dir, resolution) in enumerate(zip(data_dirs, resolutions)):
        print(f"\n🎯 Processing resolution {i+1}/{len(resolutions)}")
        
        if not os.path.exists(data_dir):
            print(f"⚠️  Directory not found: {data_dir}")
            results.append({
                'resolution': resolution,
                'target_time': target_time,
                'status': 'directory_not_found',
                'data_dir': data_dir
            })
            continue
        
        result = analyze_single_resolution(data_dir, resolution, target_time, output_dir, analysis_params)
        results.append(result)
    
    total_time = time.time() - total_start
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Save results
    results_file = os.path.join(output_dir, f'resolution_comparison_t{target_time:.1f}.csv')
    df.to_csv(results_file, index=False)
    
    print(f"\n📊 COMPARISON SUMMARY")
    print(f"=" * 60)
    print(f"Total processing time: {total_time:.1f}s")
    print(f"Results saved to: {results_file}")
    
    # Show summary table
    successful_results = df[df['status'] == 'success']
    if len(successful_results) > 0:
        print(f"\nSuccessful analyses ({len(successful_results)}/{len(df)}):")
        print(f"{'Resolution':<12} {'D':<10} {'Error':<8} {'R²':<8} {'h_total':<10} {'Segments':<10} {'Time':<8}")
        print("-" * 75)
        
        for _, row in successful_results.iterrows():
            print(f"{row['resolution']}×{row['resolution']:<8} "
                  f"{row['fractal_dim']:<10.4f} "
                  f"{row['fd_error']:<8.4f} "
                  f"{row['fd_r_squared']:<8.4f} "
                  f"{row['h_total']:<10.4f} "
                  f"{row['segments']:<10.0f} "
                  f"{row['processing_time']:<8.1f}s")
    
    # Show failed analyses
    failed_results = df[df['status'] != 'success']
    if len(failed_results) > 0:
        print(f"\nFailed analyses ({len(failed_results)}/{len(df)}):")
        for _, row in failed_results.iterrows():
            print(f"  {row['resolution']}×{row['resolution']}: {row['status']}")
    
    return df

def create_comparison_plots(df, output_dir, target_time):
    """Create resolution comparison plots."""
    
    # Filter successful results
    successful_df = df[df['status'] == 'success'].copy()
    
    if len(successful_df) == 0:
        print("⚠️  No successful results for plotting")
        return
    
    # Sort by resolution
    successful_df = successful_df.sort_values('resolution')
    
    print(f"\n📈 Creating comparison plots...")
    
    # Create comprehensive comparison plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Fractal dimension convergence
    ax1.errorbar(successful_df['resolution'], successful_df['fractal_dim'], 
                yerr=successful_df['fd_error'], fmt='bo-', capsize=5, 
                linewidth=2, markersize=8, label='Fractal Dimension')
    ax1.set_xscale('log', base=2)
    ax1.set_xlabel('Grid Resolution (N)')
    ax1.set_ylabel('Fractal Dimension')
    ax1.set_title(f'Fractal Dimension Convergence (t ≈ {target_time})')
    ax1.grid(True, alpha=0.7)
    ax1.legend()
    
    # Add resolution labels
    for _, row in successful_df.iterrows():
        ax1.annotate(f"{row['resolution']}", 
                    (row['resolution'], row['fractal_dim']),
                    xytext=(5, 5), textcoords='offset points')
    
    # Plot 2: Mixing thickness convergence
    ax2.plot(successful_df['resolution'], successful_df['h_total'], 
            'go-', linewidth=2, markersize=8, label='Total')
    ax2.plot(successful_df['resolution'], successful_df['ht'], 
            'rs--', linewidth=2, markersize=6, label='Upper')
    ax2.plot(successful_df['resolution'], successful_df['hb'], 
            'bd--', linewidth=2, markersize=6, label='Lower')
    ax2.set_xscale('log', base=2)
    ax2.set_xlabel('Grid Resolution (N)')
    ax2.set_ylabel('Mixing Layer Thickness')
    ax2.set_title(f'Mixing Layer Convergence (t ≈ {target_time})')
    ax2.grid(True, alpha=0.7)
    ax2.legend()
    
    # Plot 3: Interface complexity (segment count)
    ax3.plot(successful_df['resolution'], successful_df['segments'], 
            'mo-', linewidth=2, markersize=8, label='Interface Segments')
    ax3.set_xscale('log', base=2)
    ax3.set_yscale('log')
    ax3.set_xlabel('Grid Resolution (N)')
    ax3.set_ylabel('Number of Interface Segments')
    ax3.set_title(f'Interface Complexity vs Resolution (t ≈ {target_time})')
    ax3.grid(True, alpha=0.7)
    ax3.legend()
    
    # Plot 4: Analysis quality (R²)
    ax4.plot(successful_df['resolution'], successful_df['fd_r_squared'], 
            'co-', linewidth=2, markersize=8, label='R² Value')
    ax4.axhline(y=0.99, color='orange', linestyle='--', alpha=0.7, label='R² = 0.99')
    ax4.set_xscale('log', base=2)
    ax4.set_xlabel('Grid Resolution (N)')
    ax4.set_ylabel('R² Value')
    ax4.set_title(f'Analysis Quality vs Resolution (t ≈ {target_time})')
    ax4.set_ylim(0.95, 1.001)
    ax4.grid(True, alpha=0.7)
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'resolution_comparison_t{target_time:.1f}.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create processing time plot
    plt.figure(figsize=(10, 6))
    plt.plot(successful_df['resolution'], successful_df['processing_time'], 
            'ro-', linewidth=2, markersize=8, label='Processing Time')
    plt.xscale('log', base=2)
    plt.yscale('log')
    plt.xlabel('Grid Resolution (N)')
    plt.ylabel('Processing Time (seconds)')
    plt.title(f'Computational Cost vs Resolution (t ≈ {target_time})')
    plt.grid(True, alpha=0.7)
    plt.legend()
    
    # Add scaling reference line (N² scaling)
    if len(successful_df) >= 2:
        res_min = successful_df['resolution'].min()
        res_max = successful_df['resolution'].max()
        time_min = successful_df['processing_time'].min()
        
        # N² scaling reference
        x_ref = np.array([res_min, res_max])
        y_ref = time_min * (x_ref / res_min) ** 2
        plt.plot(x_ref, y_ref, 'k--', alpha=0.5, label='N² scaling reference')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'processing_time_scaling_t{target_time:.1f}.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   ✅ Saved comparison plots to {output_dir}")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Simple resolution comparison for fractal dimension analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare 200x200 and 400x400 at t=2.0
  python simple_resolution_comparison.py \\
    --data-dirs ~/Research/svofRuns/Dalziel/200x200 ~/Research/svofRuns/Dalziel/400x400 \\
    --resolutions 200 400 --target-time 2.0

  # Compare multiple resolutions at early time
  python simple_resolution_comparison.py \\
    --data-dirs ~/Research/svofRuns/Dalziel/100x100 ~/Research/svofRuns/Dalziel/200x200 ~/Research/svofRuns/Dalziel/400x400 \\
    --resolutions 100 200 400 --target-time 0.0

  # Custom analysis parameters
  python simple_resolution_comparison.py \\
    --data-dirs ~/Research/svofRuns/Dalziel/200x200 ~/Research/svofRuns/Dalziel/400x400 \\
    --resolutions 200 400 --target-time 1.0 --mixing-method geometric --min-box-size 0.001
""")
    
    parser.add_argument('--data-dirs', nargs='+', required=True,
                       help='Data directories (one per resolution)')
    parser.add_argument('--resolutions', nargs='+', type=int, required=True,
                       help='Grid resolutions corresponding to directories')
    parser.add_argument('--target-time', type=float, default=0.0,
                       help='Target simulation time (default: 0.0)')
    parser.add_argument('--output-dir', default=None,
                       help='Output directory (default: ./resolution_comparison_tX.X)')
    parser.add_argument('--mixing-method', default='dalziel',
                       choices=['geometric', 'statistical', 'dalziel'],
                       help='Mixing thickness calculation method')
    parser.add_argument('--h0', type=float, default=0.5,
                       help='Initial interface position')
    parser.add_argument('--min-box-size', type=float, default=None,
                       help='Minimum box size for fractal analysis')
    parser.add_argument('--time-tolerance', type=float, default=0.5,
                       help='Maximum time difference allowed (default: 0.5)')
    parser.add_argument('--no-plots', action='store_true',
                       help='Skip plot generation')
    
    args = parser.parse_args()
    
    # Validate arguments
    if len(args.data_dirs) != len(args.resolutions):
        print("❌ Number of data directories must match number of resolutions")
        return
    
    # Set output directory
    if args.output_dir is None:
        args.output_dir = f"./resolution_comparison_t{args.target_time:.1f}"
    
    # Set analysis parameters
    analysis_params = {
        'mixing_method': args.mixing_method,
        'h0': args.h0,
        'min_box_size': args.min_box_size
    }
    
    # Run comparison
    df = run_resolution_comparison(
        args.data_dirs, 
        args.resolutions, 
        args.target_time, 
        args.output_dir, 
        analysis_params
    )
    
    # Create plots
    if not args.no_plots:
        create_comparison_plots(df, args.output_dir, args.target_time)
    
    print(f"\n🎉 Resolution comparison complete!")
    print(f"Results and plots saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
