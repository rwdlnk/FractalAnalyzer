#!/usr/bin/env python3
"""
Parallel Resolution Comparison: Analyze fractal dimension convergence across resolutions in parallel.

This script compares fractal dimensions across different grid resolutions using parallel processing
for improved performance when analyzing multiple resolutions simultaneously.

UPDATED: Now supports CONREC precision interface extraction.
"""

import os
import time
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
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
        tuple: (vtk_file_path, actual_time) or (None, None)
    """
    pattern = os.path.join(data_dir, f"RT{resolution}x{resolution}-*.vtk")
    vtk_files = glob.glob(pattern)
    
    if not vtk_files:
        return None, None
    
    best_file = None
    best_diff = float('inf')
    best_time = None
    
    for vtk_file in vtk_files:
        try:
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
        return best_file, best_time
    else:
        return None, None

def analyze_single_resolution_parallel(args):
    """
    Analyze fractal dimension for a single resolution - designed for parallel execution.
    
    Args:
        args: Tuple of (data_dir, resolution, target_time, base_output_dir, analysis_params)
        
    Returns:
        dict: Analysis results
    """
    data_dir, resolution, target_time, base_output_dir, analysis_params = args
    
    # UPDATED: Support PLIC, CONREC, and scikit-image
    if analysis_params.get('use_plic', False):
        extraction_method = 'PLIC'
    elif analysis_params.get('use_conrec', False):
        extraction_method = 'CONREC'
    else:
        extraction_method = 'scikit-image'

    print(f"🔍 Worker {os.getpid()}: Analyzing {resolution}×{resolution} resolution ({extraction_method})")
    
    # Find appropriate VTK file
    vtk_file, actual_time = find_timestep_file(data_dir, resolution, target_time)
    if vtk_file is None:
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
            'extraction_method': extraction_method,  # NEW
            'data_dir': data_dir,
            'worker_pid': os.getpid()
        }
    
    # Create unique analyzer for this resolution with CONREC support
    res_output_dir = os.path.join(base_output_dir, f"parallel_resolution_{resolution}x{resolution}")
    analyzer = RTAnalyzer(
        res_output_dir, 
        use_grid_optimization=True, 
        no_titles=True,
        use_conrec=analysis_params.get('use_conrec', False),  # NEW: CONREC support
        use_plic=analysis_params.get('use_plic', False),    # NEW: PLIC support
        debug=analysis_params.get('debug', False)          # NEW: Debug support
    )
    
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
            'extraction_method': extraction_method,  # NEW
            'data_dir': data_dir,
            'vtk_file': os.path.basename(vtk_file),
            'analysis_quality': result.get('analysis_quality', 'unknown'),
            'worker_pid': os.getpid()
        }
        
        print(f"✅ Worker {os.getpid()}: {resolution}×{resolution} complete ({extraction_method}) - "
              f"D={analysis_result['fractal_dim']:.4f}±{analysis_result['fd_error']:.4f}, "
              f"Segments={analysis_result['segments']}, Time={processing_time:.2f}s")  # NEW: Include segments
        
        return analysis_result
        
    except Exception as e:
        print(f"❌ Worker {os.getpid()}: {resolution}×{resolution} failed ({extraction_method}) - {str(e)}")
        
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
            'extraction_method': extraction_method,  # NEW
            'error': str(e),
            'data_dir': data_dir,
            'vtk_file': os.path.basename(vtk_file) if vtk_file else 'unknown',
            'worker_pid': os.getpid()
        }

def run_parallel_resolution_comparison(data_dirs, resolutions, target_time, output_dir, 
                                     analysis_params, num_processes=None):
    """
    Run parallel resolution comparison across multiple resolutions.
    
    Args:
        data_dirs: List of data directories (one per resolution)
        resolutions: List of resolutions to analyze
        target_time: Target simulation time
        output_dir: Output directory
        analysis_params: Analysis parameters (including use_conrec)
        num_processes: Number of parallel processes
        
    Returns:
        DataFrame with comparison results
    """
    if num_processes is None:
        num_processes = min(len(resolutions), cpu_count())
    # UPDATED: Support all three extraction methods
    use_plic = analysis_params.get('use_plic', False)
    use_conrec = analysis_params.get('use_conrec', False)

    if use_plic:
        extraction_method = 'PLIC'
    elif use_conrec:
        extraction_method = 'CONREC'
    else:
        extraction_method = 'scikit-image'
    
    print(f"🚀 PARALLEL RESOLUTION COMPARISON ANALYSIS")
    print(f"=" * 60)
    print(f"Target time: {target_time}")
    print(f"Resolutions: {resolutions}")
    print(f"Interface extraction: {extraction_method}")  # NEW
    print(f"Output directory: {output_dir}")
    print(f"Mixing method: {analysis_params.get('mixing_method', 'dalziel')}")
    print(f"Parallel processes: {num_processes}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare arguments for parallel processing
    process_args = [(data_dir, resolution, target_time, output_dir, analysis_params)
                   for data_dir, resolution in zip(data_dirs, resolutions)]
    
    print(f"\n⚡ Starting parallel resolution analysis...")
    total_start = time.time()
    
    try:
        with Pool(processes=num_processes) as pool:
            results = pool.map(analyze_single_resolution_parallel, process_args)
    except KeyboardInterrupt:
        print("\n⚠️  Analysis interrupted by user")
        return None
    except Exception as e:
        print(f"\n❌ Parallel execution failed: {str(e)}")
        return None
    
    total_time = time.time() - total_start
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Save results
    # UPDATED: Handle all three methods
    if use_plic:
        method_suffix = "_plic"
    elif use_conrec:
        method_suffix = "_conrec"
    else:
        method_suffix = "_skimage"

    results_file = os.path.join(output_dir, f'parallel_resolution_comparison_t{target_time:.1f}{method_suffix}.csv')
    df.to_csv(results_file, index=False)
    
    print(f"\n📊 PARALLEL COMPARISON SUMMARY")
    print(f"=" * 60)
    print(f"Interface extraction method: {extraction_method}")  # NEW
    print(f"Total processing time: {total_time:.1f}s")
    print(f"Results saved to: {results_file}")
    
    # Show summary table
    successful_results = df[df['status'] == 'success']
    if len(successful_results) > 0:
        # Calculate parallel efficiency
        total_sequential_time = successful_results['processing_time'].sum()
        parallel_efficiency = total_sequential_time / (total_time * num_processes) * 100
        theoretical_speedup = total_sequential_time / total_time
        
        print(f"\n🚀 PARALLEL PERFORMANCE ANALYSIS")
        print(f"  Sequential time estimate: {total_sequential_time:.1f}s")
        print(f"  Actual parallel time: {total_time:.1f}s")
        print(f"  Theoretical speedup: {theoretical_speedup:.1f}×")
        print(f"  Parallel efficiency: {parallel_efficiency:.1f}%")
        
        print(f"\nSuccessful analyses ({len(successful_results)}/{len(df)}) using {extraction_method}:")  # NEW
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

def create_parallel_comparison_plots(df, output_dir, target_time, use_conrec=False, use_plic=False):  # UPDATED: Add PLIC parameter
    """Create resolution comparison plots optimized for parallel results."""
    
    # Filter successful results
    successful_df = df[df['status'] == 'success'].copy()
    
    if len(successful_df) == 0:
        print("⚠️  No successful results for plotting")
        return
    
    # Sort by resolution
    successful_df = successful_df.sort_values('resolution')

    # UPDATED: Support all three methods
    if use_plic:
        method_name = "PLIC"
    elif use_conrec:
        method_name = "CONREC"
    else:
        method_name = "scikit-image"

    print(f"\n📈 Creating parallel comparison plots ({method_name})...")
    
    # Create comprehensive comparison plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Fractal dimension convergence
    ax1.errorbar(successful_df['resolution'], successful_df['fractal_dim'], 
                yerr=successful_df['fd_error'], fmt='bo-', capsize=5, 
                linewidth=2, markersize=8, label='Fractal Dimension')
    ax1.set_xscale('log', base=2)
    ax1.set_xlabel('Grid Resolution (N)')
    ax1.set_ylabel('Fractal Dimension')
    ax1.set_title(f'Fractal Dimension Convergence (t ≈ {target_time}) - {method_name}')  # NEW: Include method
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
    ax2.set_title(f'Mixing Layer Convergence (t ≈ {target_time}) - {method_name}')  # NEW: Include method
    ax2.grid(True, alpha=0.7)
    ax2.legend()
    
    # Plot 3: Interface complexity (segment count) - NEW: Highlight CONREC advantage
    ax3.plot(successful_df['resolution'], successful_df['segments'], 
            'mo-', linewidth=2, markersize=8, label=f'Interface Segments ({method_name})')
    ax3.set_xscale('log', base=2)
    ax3.set_yscale('log')
    ax3.set_xlabel('Grid Resolution (N)')
    ax3.set_ylabel('Number of Interface Segments')
    ax3.set_title(f'Interface Complexity vs Resolution (t ≈ {target_time}) - {method_name}')  # NEW: Include method
    ax3.grid(True, alpha=0.7)
    ax3.legend()
    
    # Add segment count annotations to show CONREC improvement
    for _, row in successful_df.iterrows():
        ax3.annotate(f"{int(row['segments'])}", 
                    (row['resolution'], row['segments']),
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    # Plot 4: Analysis quality (R²) and parallel performance
    ax4_twin = ax4.twinx()
    
    # R² on left axis
    line1 = ax4.plot(successful_df['resolution'], successful_df['fd_r_squared'], 
                    'co-', linewidth=2, markersize=8, label='R² Value')
    ax4.axhline(y=0.99, color='orange', linestyle='--', alpha=0.7, label='R² = 0.99')
    ax4.set_xscale('log', base=2)
    ax4.set_xlabel('Grid Resolution (N)')
    ax4.set_ylabel('R² Value', color='c')
    ax4.tick_params(axis='y', labelcolor='c')
    ax4.set_ylim(0.95, 1.001)
    
    # Processing time on right axis
    line2 = ax4_twin.plot(successful_df['resolution'], successful_df['processing_time'], 
                         'ro--', linewidth=2, markersize=6, label='Processing Time')
    ax4_twin.set_ylabel('Processing Time (s)', color='r')
    ax4_twin.tick_params(axis='y', labelcolor='r')
    
    ax4.set_title(f'Analysis Quality & Performance (t ≈ {target_time}) - {method_name}')  # NEW: Include method
    ax4.grid(True, alpha=0.7)
    
    # Combine legends
    lines1, labels1 = ax4.get_legend_handles_labels()
    lines2, labels2 = ax4_twin.get_legend_handles_labels()
    ax4.legend(lines1 + lines2, labels1 + labels2, loc='center right')
    
    plt.tight_layout()

    # UPDATED: Handle all three methods
    if use_plic:
        method_suffix = "_plic"
    elif use_conrec:
        method_suffix = "_conrec"
    else:
        method_suffix = "_skimage"	

    plt.savefig(os.path.join(output_dir, f'parallel_resolution_comparison_t{target_time:.1f}{method_suffix}.png'), 

                dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   ✅ Saved parallel comparison plots to {output_dir}")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Parallel resolution comparison for fractal dimension analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare 200x200 and 400x400 at t=2.0 using CONREC in parallel
  python parallel_resolution_comparison.py \\
    --data-dirs ~/Research/svofRuns/Dalziel/200x200 ~/Research/svofRuns/Dalziel/400x400 \\
    --resolutions 200 400 --target-time 2.0 --use-conrec

  # Compare multiple resolutions with standard method
  python parallel_resolution_comparison.py \\
    --data-dirs ~/Research/svofRuns/Dalziel/100x100 ~/Research/svofRuns/Dalziel/200x200 ~/Research/svofRuns/Dalziel/400x400 \\
    --resolutions 100 200 400 --target-time 0.0 --processes 3
""")
    
    parser.add_argument('--data-dirs', nargs='+', required=True,
                       help='Data directories (one per resolution)')
    parser.add_argument('--resolutions', nargs='+', type=int, required=True,
                       help='Grid resolutions corresponding to directories')
    parser.add_argument('--target-time', type=float, default=0.0,
                       help='Target simulation time (default: 0.0)')
    parser.add_argument('--output-dir', default=None,
                       help='Output directory (default: ./parallel_resolution_comparison_tX.X_method)')
    parser.add_argument('--mixing-method', default='dalziel',
                       choices=['geometric', 'statistical', 'dalziel'],
                       help='Mixing thickness calculation method')
    parser.add_argument('--h0', type=float, default=0.5,
                       help='Initial interface position')
    parser.add_argument('--min-box-size', type=float, default=None,
                       help='Minimum box size for fractal analysis')
    parser.add_argument('--time-tolerance', type=float, default=0.5,
                       help='Maximum time difference allowed (default: 0.5)')
    parser.add_argument('--processes', type=int, default=None,
                       help='Number of parallel processes (default: auto-detect)')
    parser.add_argument('--no-plots', action='store_true',
                       help='Skip plot generation')
    parser.add_argument('--use-conrec', action='store_true',  # NEW: CONREC flag
                       help='Use CONREC precision interface extraction')
    parser.add_argument('--use-plic', action='store_true',
                       help='Use PLIC theoretical interface reconstruction')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug output for extraction methods')
    args = parser.parse_args()
    
    # Validate arguments
    if len(args.data_dirs) != len(args.resolutions):
        print("❌ Number of data directories must match number of resolutions")
        return
    
    # Validate number of processes
    if args.processes is not None:
        if args.processes < 1:
            print(f"❌ Number of processes must be at least 1")
            return
        if args.processes > cpu_count():
            print(f"⚠️  Requested {args.processes} processes, but only {cpu_count()} CPUs available")
    
    # Set output directory
    # Validate extraction method conflicts
    if args.use_conrec and args.use_plic:
        print("⚠️  Both CONREC and PLIC specified. PLIC will take precedence.")
        args.use_conrec = False

    # Set output directory with proper method suffix
    if args.output_dir is None:
        if args.use_plic:
            method_suffix = "_plic"
        elif args.use_conrec:
            method_suffix = "_conrec"
        else:
            method_suffix = "_skimage"
        args.output_dir = f"./parallel_resolution_comparison_t{args.target_time:.1f}{method_suffix}"

    # Set analysis parameters
    analysis_params = {
        'mixing_method': args.mixing_method,
        'h0': args.h0,
        'min_box_size': args.min_box_size,
        'use_conrec': args.use_conrec,
        'use_plic': args.use_plic,      # NEW: PLIC support
        'debug': args.debug             # NEW: Debug support
    }

    # Run parallel comparison
    df = run_parallel_resolution_comparison(
        args.data_dirs, 
        args.resolutions, 
        args.target_time, 
        args.output_dir, 
        analysis_params,
        args.processes
    )
    
    # Create plots
    if df is not None and not args.no_plots:
        create_parallel_comparison_plots(df, args.output_dir, args.target_time, args.use_conrec, args.use_plic)  # UPDATED: Pass both flags

    if df is not None:
        print(f"\n🎉 Parallel resolution comparison complete!")
        print(f"Results and plots saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
