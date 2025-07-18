#!/usr/bin/env python3
"""
Parallel Temporal Evolution Analyzer: Process timesteps in parallel for fractal dimension evolution.

This script processes multiple VTK files simultaneously using multiprocessing to study how fractal
dimension evolves over time in Rayleigh-Taylor simulations with significant speedup.

UPDATED: Now supports both CONREC precision interface extraction and PLIC theoretical reconstruction.
"""

import os
import time
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
from functools import partial
from fractal_analyzer.core.rt_analyzer import RTAnalyzer

def analyze_single_timestep(args):
    """
    Analyze a single timestep - designed for parallel execution.
    
    Args:
        args: Tuple of (vtk_file, timestep_index, base_output_dir, analysis_params)
        
    Returns:
        Dictionary with analysis results
    """
    vtk_file, timestep_index, base_output_dir, analysis_params = args
    
    basename = os.path.basename(vtk_file)
    start_time = time.time()
    
    try:
        # Create unique output directory for this timestep
        timestep_output_dir = os.path.join(base_output_dir, f"timestep_{timestep_index+1:04d}")
        
        # Create analyzer for this specific timestep with PLIC/CONREC support
        analyzer = RTAnalyzer(
            timestep_output_dir, 
            use_grid_optimization=True, 
            no_titles=True,
            use_conrec=analysis_params.get('use_conrec', False),
            use_plic=analysis_params.get('use_plic', False),
			debug=analysis_params.get('debug', False)
        )
        
        # Perform analysis
        result = analyzer.analyze_vtk_file(
            vtk_file,
            mixing_method=analysis_params.get('mixing_method', 'dalziel'),
            h0=analysis_params.get('h0', 0.5),
            min_box_size=analysis_params.get('min_box_size', None)
        )
        
        processing_time = time.time() - start_time
        
        # Add metadata
        result['timestep'] = timestep_index + 1
        result['vtk_file'] = basename
        result['processing_time'] = processing_time
        result['worker_pid'] = os.getpid()  # For debugging parallel execution

        # FIXED: Proper method detection
        if analysis_params.get('use_plic', False):
            result['extraction_method'] = 'PLIC'
        elif analysis_params.get('use_conrec', False):
            result['extraction_method'] = 'CONREC'
        else:
            result['extraction_method'] = 'scikit-image'

        return result
        
    except Exception as e:
        processing_time = time.time() - start_time
        
        # FIXED: Proper method detection in error case
        if analysis_params.get('use_plic', False):
            extraction_method = 'PLIC'
        elif analysis_params.get('use_conrec', False):
            extraction_method = 'CONREC'
        else:
            extraction_method = 'scikit-image'
            
        # Return error result
        return {
            'timestep': timestep_index + 1,
            'vtk_file': basename,
            'time': np.nan,
            'fractal_dim': np.nan,
            'fd_error': np.nan,
            'fd_r_squared': np.nan,
            'h_total': np.nan,
            'ht': np.nan,
            'hb': np.nan,
            'processing_time': processing_time,
            'worker_pid': os.getpid(),
            'extraction_method': extraction_method, 
            'error': str(e),
            'status': 'failed'
        }

def analyze_parallel_temporal_evolution(data_dir, resolution, output_dir=None, 
                                       mixing_method='dalziel', max_timesteps=None,
                                       start_time=None, end_time=None, h0=0.5, 
                                       min_box_size=None, num_processes=None,
                                       use_conrec=False, use_plic=False,debug=False):
    """
    Analyze fractal dimension evolution across all timesteps using parallel processing.
    
    Args:
        data_dir: Directory containing VTK files
        resolution: Grid resolution
        output_dir: Output directory
        mixing_method: Method for mixing thickness calculation
        max_timesteps: Maximum number of timesteps to process
        start_time: Start analysis from this simulation time
        end_time: End analysis at this simulation time
        h0: Initial interface position
        min_box_size: Minimum box size for fractal analysis
        num_processes: Number of parallel processes (default: auto-detect)
        use_conrec: Use CONREC precision interface extraction
        use_plic: Use PLIC theoretical interface reconstruction
		debug: Turn debug mode on or off
        
    Returns:
        DataFrame with temporal evolution results
    """
    
    # Validate extraction method conflicts
    if use_conrec and use_plic:
        print("WARNING: Both CONREC and PLIC requested. PLIC will take precedence.")
        use_conrec = False
    
    # FIXED: Determine method suffix for output naming
    if use_plic:
        method_suffix = "_plic"
        extraction_method_name = "PLIC (theoretical reconstruction)"
    elif use_conrec:
        method_suffix = "_conrec"
        extraction_method_name = "CONREC (precision)"
    else:
        method_suffix = "_skimage"
        extraction_method_name = "scikit-image (standard)"
    
    if output_dir is None:
        output_dir = f"./parallel_temporal_evolution_{resolution}x{resolution}{method_suffix}"
    
    if num_processes is None:
        num_processes = min(cpu_count(), 8)  # Limit to 8 to avoid I/O saturation
    
    print(f"🚀 PARALLEL TEMPORAL EVOLUTION ANALYSIS")
    print(f"=" * 60)
    print(f"Data directory: {data_dir}")
    print(f"Resolution: {resolution}×{resolution}")
    print(f"Output directory: {output_dir}")
    print(f"Mixing method: {mixing_method}")
    print(f"Interface extraction: {extraction_method_name}")  # FIXED: Use correct variable
    print(f"Initial interface: h0 = {h0}")
    print(f"Parallel processes: {num_processes}")
    if min_box_size:
        print(f"Min box size: {min_box_size}")
    
    # Find all VTK files
    pattern = os.path.join(data_dir, f"RT{resolution}x{resolution}-*.vtk")
    all_vtk_files = glob.glob(pattern)
    
    # Sort by actual simulation time, not filename alphabetically
    def extract_time(filename):
        try:
            basename = os.path.basename(filename)
            time_str = basename.split('-')[1].split('.')[0]
            return float(time_str)
        except:
            return 0.0

    all_vtk_files = sorted(all_vtk_files, key=extract_time)
    
    if not all_vtk_files:
        print(f"❌ No VTK files found matching: {pattern}")
        return None
    
    # Filter files by time range if specified
    vtk_files = []
    if start_time is not None or end_time is not None:
        print(f"Filtering files by time range:")
        if start_time is not None:
            print(f"  Start time: {start_time}")
        if end_time is not None:
            print(f"  End time: {end_time}")
        
        for vtk_file in all_vtk_files:
            try:
                basename = os.path.basename(vtk_file)
                time_str = basename.split('-')[1].split('.')[0]
                file_time = float(time_str) / 1000.0
                
                if start_time is not None and file_time < start_time:
                    continue
                if end_time is not None and file_time > end_time:
                    continue
                    
                vtk_files.append(vtk_file)
                
            except:
                vtk_files.append(vtk_file)
                
        print(f"  Filtered from {len(all_vtk_files)} to {len(vtk_files)} files")
    else:
        vtk_files = all_vtk_files
    
    if max_timesteps:
        vtk_files = vtk_files[:max_timesteps]
    
    print(f"Processing {len(vtk_files)} timestep files in parallel")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # FIXED: Prepare analysis parameters with correct flags
    analysis_params = {
        'mixing_method': mixing_method,
        'h0': h0,
        'min_box_size': min_box_size,
        'use_conrec': use_conrec,
        'use_plic': use_plic,
		'debug': debug
    }
    
    # Prepare arguments for parallel processing
    process_args = [(vtk_file, i, output_dir, analysis_params) 
                   for i, vtk_file in enumerate(vtk_files)]
    
    # Execute parallel analysis
    print(f"\n⚡ Starting parallel analysis with {num_processes} processes...")
    total_start = time.time()
    
    try:
        with Pool(processes=num_processes) as pool:
            # Use map with progress tracking
            results = []
            
            # Process in chunks to show progress
            chunk_size = max(1, len(process_args) // 10)  # 10% chunks for progress
            
            for i in range(0, len(process_args), chunk_size):
                chunk = process_args[i:i+chunk_size]
                chunk_results = pool.map(analyze_single_timestep, chunk)
                results.extend(chunk_results)
                
                progress = min(100, (i + len(chunk)) * 100 // len(process_args))
                print(f"   Progress: {progress}% ({i + len(chunk)}/{len(process_args)} files)")
    
    except KeyboardInterrupt:
        print("\n⚠️  Analysis interrupted by user")
        return None
    except Exception as e:
        print(f"\n❌ Parallel execution failed: {str(e)}")
        return None
    
    total_time = time.time() - total_start
    
    # Sort results by timestep to maintain order
    results.sort(key=lambda x: x.get('timestep', 0))
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # FIXED: Save results with correct method suffix
    results_file = os.path.join(output_dir, f'parallel_temporal_evolution_{resolution}x{resolution}{method_suffix}.csv')
    df.to_csv(results_file, index=False)
    
    print(f"\n📈 PARALLEL TEMPORAL EVOLUTION SUMMARY")
    print(f"=" * 60)
    print(f"Total timesteps processed: {len(results)}")
    print(f"Interface extraction method: {extraction_method_name}")
    print(f"Total processing time: {total_time:.1f}s")
    
    # Filter valid results
    valid_results = df[~pd.isna(df['fractal_dim'])]
    failed_results = df[pd.isna(df['fractal_dim'])]
    
    print(f"Successful analyses: {len(valid_results)}/{len(results)}")
    if len(failed_results) > 0:
        print(f"Failed analyses: {len(failed_results)}")
        print("Failed files:", failed_results['vtk_file'].tolist()[:5])  # Show first 5
    
    if len(valid_results) > 0:
        print(f"Time range: t = {valid_results['time'].min():.3f} to {valid_results['time'].max():.3f}")
        print(f"Fractal dimension range: D = {valid_results['fractal_dim'].min():.4f} to {valid_results['fractal_dim'].max():.4f}")
        print(f"Final mixing thickness: h_total = {valid_results['h_total'].iloc[-1]:.4f}")
        
        # Calculate parallel efficiency
        total_sequential_time = valid_results['processing_time'].sum()
        parallel_efficiency = total_sequential_time / (total_time * num_processes) * 100
        theoretical_speedup = total_sequential_time / total_time
        
        print(f"\n🚀 PARALLEL PERFORMANCE ANALYSIS")
        print(f"  Sequential time estimate: {total_sequential_time:.1f}s")
        print(f"  Actual parallel time: {total_time:.1f}s")
        print(f"  Theoretical speedup: {theoretical_speedup:.1f}×")
        print(f"  Parallel efficiency: {parallel_efficiency:.1f}%")
        print(f"  Average time per timestep: {valid_results['processing_time'].mean():.2f}s")
    
    print(f"Results saved to: {results_file}")
    
    # FIXED: Create temporal evolution plots with correct flags
    create_parallel_evolution_plots(df, output_dir, resolution, use_conrec, use_plic)
    
    return df

def create_parallel_evolution_plots(df, output_dir, resolution, use_conrec=False, use_plic=False):
    """Create comprehensive temporal evolution plots for parallel results."""
    
    # Filter valid data
    valid_df = df[~pd.isna(df['fractal_dim'])].copy()
    
    if len(valid_df) == 0:
        print("⚠️  No valid data for plotting")
        return

    # FIXED: Determine method name and suffix for plots
    if use_plic:
        method_name = "PLIC"
        method_suffix = "_plic"
    elif use_conrec:
        method_name = "CONREC"
        method_suffix = "_conrec"
    else:
        method_name = "scikit-image"
        method_suffix = "_skimage"
    
    print(f"\n📊 Creating parallel temporal evolution plots ({method_name})...")

    # Create comprehensive evolution plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Fractal dimension evolution
    ax1.errorbar(valid_df['time'], valid_df['fractal_dim'], 
                yerr=valid_df['fd_error'], fmt='bo-', capsize=3, 
                linewidth=2, markersize=5, label='Fractal Dimension')
    ax1.fill_between(valid_df['time'], 
                     valid_df['fractal_dim'] - valid_df['fd_error'],
                     valid_df['fractal_dim'] + valid_df['fd_error'],
                     alpha=0.3, color='blue')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Fractal Dimension')
    ax1.set_title(f'Fractal Dimension Evolution ({resolution}×{resolution}) - {method_name}')
    ax1.grid(True, alpha=0.7)
    ax1.legend()
    
    # Plot 2: Mixing layer evolution
    ax2.plot(valid_df['time'], valid_df['h_total'], 'g-', linewidth=2, label='Total')
    ax2.plot(valid_df['time'], valid_df['ht'], 'r--', linewidth=2, label='Upper')
    ax2.plot(valid_df['time'], valid_df['hb'], 'b--', linewidth=2, label='Lower')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Mixing Layer Thickness')
    ax2.set_title(f'Mixing Layer Evolution ({resolution}×{resolution}) - {method_name}')
    ax2.grid(True, alpha=0.7)
    ax2.legend()
    
    # Plot 3: R² evolution (fit quality)
    ax3.plot(valid_df['time'], valid_df['fd_r_squared'], 'mo-', 
            linewidth=2, markersize=4, label='R²')
    ax3.axhline(y=0.99, color='orange', linestyle='--', alpha=0.7, label='R² = 0.99')
    ax3.set_xlabel('Time')
    ax3.set_ylabel('R² Value')
    ax3.set_title(f'Fractal Analysis Quality ({resolution}×{resolution}) - {method_name}')
    ax3.set_ylim(0.95, 1.01)
    ax3.grid(True, alpha=0.7)
    ax3.legend()
    
    # Plot 4: Processing time distribution (parallel-specific)
    ax4.hist(valid_df['processing_time'], bins=20, alpha=0.7, color='cyan', edgecolor='black')
    ax4.axvline(valid_df['processing_time'].mean(), color='red', linestyle='--', 
               label=f'Mean: {valid_df["processing_time"].mean():.2f}s')
    ax4.set_xlabel('Processing Time per Timestep (s)')
    ax4.set_ylabel('Frequency')
    ax4.set_title(f'Processing Time Distribution - {method_name}')
    ax4.grid(True, alpha=0.7)
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'parallel_temporal_evolution_{resolution}x{resolution}{method_suffix}.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create combined D vs h_total plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Main evolution path
    ax.plot(valid_df['h_total'], valid_df['fractal_dim'], 'bo-', 
           linewidth=2, markersize=6, label='Evolution Path')
    
    # Color-code by time
    scatter = ax.scatter(valid_df['h_total'], valid_df['fractal_dim'], 
                        c=valid_df['time'], cmap='viridis', s=50, alpha=0.7)
    plt.colorbar(scatter, ax=ax, label='Time')
    
    # Mark start and end
    if len(valid_df) > 1:
        ax.plot(valid_df['h_total'].iloc[0], valid_df['fractal_dim'].iloc[0], 
               'go', markersize=12, label='Start')
        ax.plot(valid_df['h_total'].iloc[-1], valid_df['fractal_dim'].iloc[-1], 
               'ro', markersize=12, label='End')
    
    ax.set_xlabel('Mixing Layer Thickness (h_total)')
    ax.set_ylabel('Fractal Dimension')
    ax.set_title(f'Fractal Dimension vs Mixing Thickness ({resolution}×{resolution}) - {method_name}')
    ax.grid(True, alpha=0.7)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'parallel_D_vs_mixing_{resolution}x{resolution}{method_suffix}.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   ✅ Saved parallel temporal evolution plots to {output_dir}")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Parallel temporal evolution analysis of fractal dimension',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all timesteps for 200x200 resolution in parallel with PLIC
  python parallel_temporal_evolution_analyzer.py --data-dir ~/Research/svofRuns/Dalziel/200x200 --resolution 200 --use-plic

  # Process all timesteps for 200x200 resolution in parallel with CONREC
  python parallel_temporal_evolution_analyzer.py --data-dir ~/Research/svofRuns/Dalziel/200x200 --resolution 200 --use-conrec

  # Process first 50 timesteps with 6 parallel processes using standard method
  python parallel_temporal_evolution_analyzer.py --data-dir ~/Research/svofRuns/Dalziel/400x400 --resolution 400 --max-timesteps 50 --processes 6

  # Analyze specific time range with CONREC and custom parameters
  python parallel_temporal_evolution_analyzer.py --data-dir ~/Research/svofRuns/Dalziel/200x200 --resolution 200 --start-time 1.0 --end-time 2.0 --processes 4 --use-conrec
""")
    
    parser.add_argument('--data-dir', required=True,
                       help='Directory containing VTK files')
    parser.add_argument('--resolution', type=int, required=True,
                       help='Grid resolution')
    parser.add_argument('--output-dir', default=None,
                       help='Output directory (default: ./parallel_temporal_evolution_RESxRES_method)')
    parser.add_argument('--mixing-method', default='dalziel',
                       choices=['geometric', 'statistical', 'dalziel'],
                       help='Mixing thickness calculation method')
    parser.add_argument('--max-timesteps', type=int, default=None,
                       help='Maximum timesteps to process (default: all)')
    parser.add_argument('--start-time', type=float, default=None,
                       help='Start analysis from this simulation time')
    parser.add_argument('--end-time', type=float, default=None,
                       help='End analysis at this simulation time')
    parser.add_argument('--h0', type=float, default=0.5,
                       help='Initial interface position')
    parser.add_argument('--min-box-size', type=float, default=None,
                       help='Minimum box size for fractal analysis')
    parser.add_argument('--processes', type=int, default=None,
                       help='Number of parallel processes (default: auto-detect, max 8)')
    parser.add_argument('--use-conrec', action='store_true',
                       help='Use CONREC precision interface extraction')
    parser.add_argument('--use-plic', action='store_true',
                       help='Use PLIC theoretical interface reconstruction')  
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug output for PLIC/CONREC extraction')    

    args = parser.parse_args()
    
    if not os.path.exists(args.data_dir):
        print(f"❌ Data directory not found: {args.data_dir}")
        return
    
    # Validate number of processes
    if args.processes is not None:
        if args.processes < 1:
            print(f"❌ Number of processes must be at least 1")
            return
        if args.processes > cpu_count():
            print(f"⚠️  Requested {args.processes} processes, but only {cpu_count()} CPUs available")
   
    # Validate extraction method conflicts
    if args.use_conrec and args.use_plic:
        print(f"⚠️  Both CONREC and PLIC specified. PLIC will take precedence.")
 
    # FIXED: Run parallel temporal evolution analysis with correct argument order
    df = analyze_parallel_temporal_evolution(
        args.data_dir,
        args.resolution,
        args.output_dir,
        args.mixing_method,
        args.max_timesteps,
        args.start_time,
        args.end_time,
        args.h0,
        args.min_box_size,
        args.processes,
        args.use_conrec,  # FIXED: Correct order
        args.use_plic,    # FIXED: Correct order
		args.debug
    )

    if df is not None:
        # Determine method name for final summary
        if args.use_plic:
            method_name = "PLIC"
        elif args.use_conrec:
            method_name = "CONREC"
        else:
            method_name = "scikit-image"
        print(f"\n🎉 Parallel temporal evolution analysis complete using {method_name}!")
        print(f"Check the output directory for detailed results and plots.")

if __name__ == "__main__":
    main()
