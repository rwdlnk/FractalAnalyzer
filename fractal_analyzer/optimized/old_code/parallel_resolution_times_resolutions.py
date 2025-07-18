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

def analyze_resolution_batch(args):
    """
    Process multiple times for one resolution in a single worker (most efficient).
    
    Args:
        args: Tuple of (data_dir, resolution, target_times, base_output_dir, analysis_params)
        
    Returns:
        list: List of analysis results for all times
    """
    data_dir, resolution, target_times, base_output_dir, analysis_params = args
    
    # Determine extraction method
    if analysis_params.get('use_plic', False):
        extraction_method = 'PLIC'
    elif analysis_params.get('use_conrec', False):
        extraction_method = 'CONREC'
    else:
        extraction_method = 'scikit-image'
    
    print(f"🔍 Worker {os.getpid()}: Processing {resolution}×{resolution} for {len(target_times)} times ({extraction_method})")
    
    # Create one analyzer per worker (reuse for efficiency)
    res_output_dir = os.path.join(base_output_dir, f"parallel_resolution_{resolution}x{resolution}")
    analyzer = RTAnalyzer(
        res_output_dir, 
        use_grid_optimization=True,
        no_titles=True,
        use_conrec=analysis_params.get('use_conrec', False),
        use_plic=analysis_params.get('use_plic', False),
        debug=analysis_params.get('debug', False)
    )
    
    batch_results = []
    worker_start = time.time()
    
    for i, target_time in enumerate(target_times):
        print(f"   [{i+1}/{len(target_times)}] {resolution}×{resolution} at t={target_time}")
        
        # Find appropriate VTK file
        vtk_file, actual_time = find_timestep_file(data_dir, resolution, target_time)
        
        if vtk_file is None:
            print(f"   No file found for t={target_time}")
            batch_results.append({
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
                'extraction_method': extraction_method,
                'data_dir': data_dir,
                'vtk_file': 'not_found',
                'analysis_quality': 'failed',
                'worker_pid': os.getpid()
            })
            continue
        
        try:
            time_start = time.time()
            
            # Perform analysis using the reused analyzer
            result = analyzer.analyze_vtk_file(
                vtk_file,
                mixing_method=analysis_params.get('mixing_method', 'dalziel'),
                h0=analysis_params.get('h0', 0.5),
                min_box_size=analysis_params.get('min_box_size', None)
            )
            
            processing_time = time.time() - time_start
            
            # Extract interface for segment count (reuse analyzer's methods)
            data = analyzer.read_vtk_file(vtk_file)
            contours = analyzer.extract_interface(data['f'], data['x'], data['y'])
            segments = analyzer.convert_contours_to_segments(contours)
            
            # Package successful result
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
                'extraction_method': extraction_method,
                'data_dir': data_dir,
                'vtk_file': os.path.basename(vtk_file),
                'analysis_quality': result.get('analysis_quality', 'unknown'),
                'worker_pid': os.getpid()
            }
            
            print(f"   ✅ t={target_time:.1f}: D={analysis_result['fractal_dim']:.4f}±{analysis_result['fd_error']:.4f}, "
                  f"Segments={analysis_result['segments']}, Time={processing_time:.1f}s")
            
            batch_results.append(analysis_result)
            
        except Exception as e:
            print(f"   ❌ t={target_time}: Analysis failed - {str(e)}")
            
            batch_results.append({
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
                'extraction_method': extraction_method,
                'error': str(e),
                'data_dir': data_dir,
                'vtk_file': os.path.basename(vtk_file) if vtk_file else 'unknown',
                'analysis_quality': 'failed',
                'worker_pid': os.getpid()
            })
    
    worker_time = time.time() - worker_start
    successful_count = sum(1 for r in batch_results if r['status'] == 'success')
    
    print(f"✅ Worker {os.getpid()}: {resolution}×{resolution} batch complete - "
          f"{successful_count}/{len(target_times)} successful in {worker_time:.1f}s ({extraction_method})")
    
    return batch_results

def run_parallel_resolution_comparison(data_dirs, resolutions, target_times, output_dir, 
                                     analysis_params, num_processes=None):
    """
    Run parallel resolution comparison across multiple resolutions and times using smart batching.
    
    Args:
        data_dirs: List of data directories (one per resolution)
        resolutions: List of resolutions to analyze
        target_times: List of target simulation times
        output_dir: Output directory
        analysis_params: Analysis parameters
        num_processes: Number of parallel processes
        
    Returns:
        DataFrame with comparison results
    """
    if num_processes is None:
        num_processes = min(len(resolutions), cpu_count())
    
    # Determine extraction method
    use_plic = analysis_params.get('use_plic', False)
    use_conrec = analysis_params.get('use_conrec', False)
    
    if use_plic:
        extraction_method = 'PLIC'
    elif use_conrec:
        extraction_method = 'CONREC'
    else:
        extraction_method = 'scikit-image'
    
    # Handle single target_time for backward compatibility
    if not isinstance(target_times, list):
        target_times = [target_times]
    
    total_analyses = len(resolutions) * len(target_times)
    
    print(f"🚀 PARALLEL SMART BATCHING COMPARISON")
    print(f"=" * 60)
    print(f"Target times: {target_times}")
    print(f"Resolutions: {resolutions}")
    print(f"Interface extraction: {extraction_method}")
    print(f"Output directory: {output_dir}")
    print(f"Mixing method: {analysis_params.get('mixing_method', 'dalziel')}")
    print(f"Total analyses: {total_analyses} ({len(resolutions)} resolutions × {len(target_times)} times)")
    print(f"Parallel processes: {num_processes} (one per resolution)")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare arguments for parallel processing (one batch per resolution)
    process_args = [(data_dir, resolution, target_times, output_dir, analysis_params)
                   for data_dir, resolution in zip(data_dirs, resolutions)]
    
    print(f"\n⚡ Starting smart batching analysis...")
    print(f"Each worker will process {len(target_times)} times for one resolution")
    total_start = time.time()
    
    try:
        with Pool(processes=num_processes) as pool:
            # Each worker returns a list of results for all times
            batch_results = pool.map(analyze_resolution_batch, process_args)
    except KeyboardInterrupt:
        print("\n⚠️  Analysis interrupted by user")
        return None
    except Exception as e:
        print(f"\n❌ Parallel execution failed: {str(e)}")
        return None
    
    total_time = time.time() - total_start
    
    # Flatten results (each batch_result is a list)
    all_results = []
    for batch in batch_results:
        all_results.extend(batch)
    
    # Convert to DataFrame
    df = pd.DataFrame(all_results)
    
    # Save results with comprehensive filename
    if use_plic:
        method_suffix = "_plic"
    elif use_conrec:
        method_suffix = "_conrec"
    else:
        method_suffix = "_skimage"
    
    time_range_str = f"t{min(target_times):.1f}-{max(target_times):.1f}" if len(target_times) > 1 else f"t{target_times[0]:.1f}"
    results_file = os.path.join(output_dir, f'parallel_matrix_comparison_{time_range_str}{method_suffix}.csv')
    df.to_csv(results_file, index=False)
    
    print(f"\n📊 SMART BATCHING SUMMARY")
    print(f"=" * 60)
    print(f"Interface extraction method: {extraction_method}")
    print(f"Total processing time: {total_time:.1f}s")
    print(f"Results saved to: {results_file}")
    
    # Analyze results
    successful_results = df[df['status'] == 'success']
    failed_results = df[df['status'] != 'success']
    
    if len(successful_results) > 0:
        # Calculate efficiency
        total_sequential_time = successful_results['processing_time'].sum()
        parallel_efficiency = total_sequential_time / (total_time * num_processes) * 100
        theoretical_speedup = total_sequential_time / total_time
        
        print(f"\n🚀 PARALLEL PERFORMANCE ANALYSIS")
        print(f"  Sequential time estimate: {total_sequential_time:.1f}s")
        print(f"  Actual parallel time: {total_time:.1f}s")
        print(f"  Theoretical speedup: {theoretical_speedup:.1f}×")
        print(f"  Parallel efficiency: {parallel_efficiency:.1f}%")
        print(f"  Average time per analysis: {successful_results['processing_time'].mean():.1f}s")
        
        print(f"\nSuccessful analyses ({len(successful_results)}/{len(df)}) using {extraction_method}:")
        
        # Group by resolution for summary
        for resolution in sorted(resolutions):
            res_data = successful_results[successful_results['resolution'] == resolution]
            if len(res_data) > 0:
                print(f"\n  {resolution}×{resolution} Resolution ({len(res_data)}/{len(target_times)} times):")
                print(f"    Time Range | D Range | Segment Range")
                print(f"    {res_data['actual_time'].min():.1f}-{res_data['actual_time'].max():.1f}     | "
                      f"{res_data['fractal_dim'].min():.3f}-{res_data['fractal_dim'].max():.3f} | "
                      f"{int(res_data['segments'].min())}-{int(res_data['segments'].max())}")
    
    # Show failed analyses
    if len(failed_results) > 0:
        print(f"\nFailed analyses ({len(failed_results)}/{len(df)}):")
        failure_summary = failed_results.groupby(['resolution', 'status']).size()
        for (resolution, status), count in failure_summary.items():
            print(f"  {resolution}×{resolution}: {count} {status}")
    
    return df

def create_parallel_comparison_plots(df, output_dir, target_times, use_conrec=False, use_plic=False):
    """Create enhanced comparison plots for multiple times and resolutions."""
    
    # Filter successful results
    successful_df = df[df['status'] == 'success'].copy()
    
    if len(successful_df) == 0:
        print("⚠️  No successful results for plotting")
        return
    
    # Determine method name
    if use_plic:
        method_name = "PLIC"
        method_suffix = "_plic"
    elif use_conrec:
        method_name = "CONREC" 
        method_suffix = "_conrec"
    else:
        method_name = "scikit-image"
        method_suffix = "_skimage"
    
    # Handle single or multiple times
    if not isinstance(target_times, list):
        target_times = [target_times]
    
    time_range_str = f"t{min(target_times):.1f}-{max(target_times):.1f}" if len(target_times) > 1 else f"t{target_times[0]:.1f}"
    
    print(f"\n📈 Creating enhanced comparison plots ({method_name})...")
    
    # Sort by resolution and time
    successful_df = successful_df.sort_values(['resolution', 'actual_time'])
    
    # Create comprehensive comparison plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Fractal dimension evolution across time and resolution
    resolutions = sorted(successful_df['resolution'].unique())
    colors = plt.cm.viridis(np.linspace(0, 1, len(resolutions)))
    
    for i, resolution in enumerate(resolutions):
        res_data = successful_df[successful_df['resolution'] == resolution]
        if len(res_data) > 0:
            ax1.errorbar(res_data['actual_time'], res_data['fractal_dim'], 
                        yerr=res_data['fd_error'], fmt='o-', capsize=3, 
                        color=colors[i], linewidth=2, markersize=6, 
                        label=f'{resolution}×{resolution}')
    
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Fractal Dimension')
    ax1.set_title(f'Fractal Dimension Evolution - {method_name}')
    ax1.grid(True, alpha=0.7)
    ax1.legend()
    
    # Plot 2: Resolution convergence at each time
    if len(target_times) > 1:
        # Show convergence at different times
        time_colors = plt.cm.plasma(np.linspace(0, 1, len(target_times)))
        for i, target_time in enumerate(target_times):
            # Find closest actual times
            time_data = []
            for resolution in resolutions:
                res_time_data = successful_df[
                    (successful_df['resolution'] == resolution) & 
                    (np.abs(successful_df['actual_time'] - target_time) < 0.5)
                ]
                if len(res_time_data) > 0:
                    closest_match = res_time_data.iloc[
                        np.argmin(np.abs(res_time_data['actual_time'] - target_time))
                    ]
                    time_data.append(closest_match)
            
            if time_data:
                time_df = pd.DataFrame(time_data)
                ax2.errorbar(time_df['resolution'], time_df['fractal_dim'], 
                           yerr=time_df['fd_error'], fmt='s-', capsize=3,
                           color=time_colors[i], linewidth=2, markersize=6,
                           label=f't ≈ {target_time:.1f}')
        
        ax2.set_xscale('log', base=2)
        ax2.set_xlabel('Grid Resolution (N)')
        ax2.set_ylabel('Fractal Dimension')
        ax2.set_title(f'Resolution Convergence - {method_name}')
        ax2.grid(True, alpha=0.7)
        ax2.legend()
    else:
        # Single time - show segment scaling
        ax2.loglog(successful_df['resolution'], successful_df['segments'], 
                  'mo-', linewidth=2, markersize=8, label=f'Interface Segments')
        ax2.set_xlabel('Grid Resolution (N)')
        ax2.set_ylabel('Number of Interface Segments')
        ax2.set_title(f'Interface Complexity - {method_name}')
        ax2.grid(True, alpha=0.7)
        ax2.legend()
    
    # Plot 3: 3D surface plot (if multiple times and resolutions)
    if len(target_times) > 1 and len(resolutions) > 1:
        from mpl_toolkits.mplot3d import Axes3D
        ax3.remove()
        ax3 = fig.add_subplot(223, projection='3d')
        
        # Create meshgrid
        res_grid, time_grid = np.meshgrid(resolutions, target_times)
        dim_grid = np.full_like(res_grid, np.nan, dtype=float)
        
        for i, target_time in enumerate(target_times):
            for j, resolution in enumerate(resolutions):
                matching_data = successful_df[
                    (successful_df['resolution'] == resolution) & 
                    (np.abs(successful_df['actual_time'] - target_time) < 0.5)
                ]
                if len(matching_data) > 0:
                    dim_grid[i, j] = matching_data.iloc[0]['fractal_dim']
        
        # Plot surface
        surf = ax3.plot_surface(res_grid, time_grid, dim_grid, cmap='viridis', alpha=0.8)
        ax3.set_xlabel('Resolution')
        ax3.set_ylabel('Time')
        ax3.set_zlabel('Fractal Dimension')
        ax3.set_title(f'D(Resolution, Time) - {method_name}')
        fig.colorbar(surf, ax=ax3, shrink=0.5)
    else:
        # Fallback: Mixing thickness evolution
        for i, resolution in enumerate(resolutions):
            res_data = successful_df[successful_df['resolution'] == resolution]
            if len(res_data) > 0:
                ax3.plot(res_data['actual_time'], res_data['h_total'], 
                        'o-', color=colors[i], linewidth=2, markersize=6,
                        label=f'{resolution}×{resolution}')
        
        ax3.set_xlabel('Time')
        ax3.set_ylabel('Mixing Layer Thickness')
        ax3.set_title(f'Mixing Evolution - {method_name}')
        ax3.grid(True, alpha=0.7)
        ax3.legend()
    
    # Plot 4: Performance analysis
    ax4_twin = ax4.twinx()
    
    # R² on left axis
    for i, resolution in enumerate(resolutions):
        res_data = successful_df[successful_df['resolution'] == resolution]
        if len(res_data) > 0:
            ax4.plot(res_data['actual_time'], res_data['fd_r_squared'], 
                    'o-', color=colors[i], linewidth=2, markersize=4,
                    label=f'{resolution}×{resolution}')
    
    ax4.axhline(y=0.99, color='red', linestyle='--', alpha=0.7, label='R² = 0.99')
    ax4.set_xlabel('Time')
    ax4.set_ylabel('R² Value', color='black')
    ax4.set_ylim(0.95, 1.001)
    ax4.set_title(f'Analysis Quality - {method_name}')
    ax4.grid(True, alpha=0.7)
    ax4.legend(loc='lower left')
    
    # Processing time on right axis
    for i, resolution in enumerate(resolutions):
        res_data = successful_df[successful_df['resolution'] == resolution]
        if len(res_data) > 0:
            ax4_twin.plot(res_data['actual_time'], res_data['processing_time'], 
                         's--', color=colors[i], linewidth=1, markersize=3, alpha=0.7)
    
    ax4_twin.set_ylabel('Processing Time (s)', color='gray')
    ax4_twin.tick_params(axis='y', labelcolor='gray')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'parallel_matrix_comparison_{time_range_str}{method_suffix}.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   ✅ Saved enhanced matrix plots to {output_dir}")

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
    parser.add_argument('--target-time', type=float, default=None,
                       help='Single target simulation time (for backward compatibility)')
    parser.add_argument('--target-times', nargs='+', type=float, default=None,
                       help='Multiple target simulation times (e.g., --target-times 1.0 2.0 3.0)')
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

    # Handle single vs multiple times
    if args.target_times is not None:
        target_times = args.target_times
        print(f"Using multiple target times: {target_times}")
    elif args.target_time is not None:
        target_times = [args.target_time]
        print(f"Using single target time: {args.target_time}")
    else:
        target_times = [0.0]  # Default
        print(f"Using default target time: 0.0")

    # Run parallel comparison
    df = run_parallel_resolution_comparison(
        args.data_dirs, 
        args.resolutions, 
        target_times, 
        args.output_dir, 
        analysis_params,
        args.processes
    )

    # Create plots
    if df is not None and not args.no_plots:
        create_parallel_comparison_plots(df, args.output_dir, target_times, args.use_conrec, args.use_plic)

    if df is not None:
        print(f"\n🎉 Parallel resolution comparison complete!")
        print(f"Results and plots saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
