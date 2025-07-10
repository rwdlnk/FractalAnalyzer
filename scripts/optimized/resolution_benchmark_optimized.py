#!/usr/bin/env python3
"""
FIXED Optimized Resolution Benchmark: Enhanced convergence analysis with grid caching.

This script analyzes fractal dimension convergence across resolutions using
the optimized grid caching framework for significant performance improvements.
"""

import os
import time
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed
from fractal_analyzer.optimized import GridCacheManager, FastVTKReader
from fractal_analyzer.core.rt_analyzer import RTAnalyzer
from fractal_analyzer.core.fractal_analyzer import FractalAnalyzer

def find_resolution_files(base_dirs, target_time=0.0, time_tolerance=0.1):
    """
    Find VTK files for each resolution at target time.
    
    Args:
        base_dirs: List of directories to search (e.g., ['100x100', '200x200', ...])
        target_time: Target simulation time (default: 0.0)
        time_tolerance: Maximum time difference (default: 0.1)
        
    Returns:
        Dictionary: {resolution: vtk_file_path}
    """
    resolution_files = {}
    
    for base_dir in base_dirs:
        # Extract resolution from directory name
        try:
            if 'x' in base_dir:
                resolution = int(base_dir.split('x')[0].split('/')[-1])
            else:
                continue
        except ValueError:
            continue
            
        # Look for VTK files in this directory
        if os.path.exists(base_dir):
            import glob
            pattern = os.path.join(base_dir, f"RT{resolution}x{resolution}-*.vtk")
            vtk_files = glob.glob(pattern)
            
            # Find file closest to target time
            best_file = None
            best_diff = float('inf')
            
            for vtk_file in vtk_files:
                try:
                    # Extract time from filename
                    basename = os.path.basename(vtk_file)
                    time_str = basename.split('-')[1].split('.')[0]
                    file_time = float(time_str) / 1000.0
                    
                    diff = abs(file_time - target_time)
                    if diff < best_diff and diff <= time_tolerance:
                        best_diff = diff
                        best_file = vtk_file
                except:
                    continue
            
            if best_file:
                resolution_files[resolution] = best_file
                print(f"Found {resolution}×{resolution}: {os.path.basename(best_file)} (t={file_time:.3f})")
            else:
                print(f"Warning: No suitable file found for {resolution}×{resolution}")
    
    return resolution_files

def analyze_single_resolution_optimized(args):
    """
    FIXED: Analyze single resolution using optimized workflow.
    
    Args:
        args: Tuple of (resolution, vtk_file, output_dir, use_fast_reader, analysis_params)
        
    Returns:
        Dictionary with analysis results
    """
    resolution, vtk_file, output_dir, use_fast_reader, analysis_params = args
    
    method_str = 'optimized' if use_fast_reader else 'baseline'
    print(f"\n🔧 Processing {resolution}×{resolution} ({method_str})")
    start_time = time.time()
    
    try:
        # Create RTAnalyzer for fractal analysis (simplified approach)
        subdir = f"res_{resolution}_{method_str}"
        analyzer = RTAnalyzer(os.path.join(output_dir, subdir), use_grid_optimization=False, no_titles=True)
        
        # Perform fractal analysis 
        result = analyzer.analyze_vtk_file(
            vtk_file, 
            mixing_method=analysis_params.get('mixing_method', 'dalziel'),
            h0=analysis_params.get('h0', 0.5),
            min_box_size=analysis_params.get('min_box_size', None)
        )
        
        total_time = time.time() - start_time
        
        # CRITICAL FIX: Ensure result is a dictionary with safe key access
        if not isinstance(result, dict):
            raise ValueError(f"analyze_vtk_file returned {type(result)}, expected dict")
        
        # Build return dictionary with safe key access
        return_dict = {
            'resolution': resolution,
            'fractal_dim': result.get('fractal_dim', np.nan),
            'fd_error': result.get('fd_error', np.nan),
            'fd_r_squared': result.get('fd_r_squared', np.nan),
            'h_total': result.get('h_total', np.nan),
            'ht': result.get('ht', np.nan),
            'hb': result.get('hb', np.nan),
            'processing_time': total_time,
            'method': method_str,
            'vtk_file': os.path.basename(vtk_file),
            'time': result.get('time', np.nan)
        }
        
        print(f"✅ {resolution}×{resolution} ({method_str}): "
              f"D={return_dict['fractal_dim']:.4f}±{return_dict['fd_error']:.4f}, "
              f"time={total_time:.2f}s")
        
        return return_dict
                
    except Exception as e:
        total_time = time.time() - start_time
        print(f"❌ Error processing {resolution}×{resolution} ({method_str}): {str(e)}")
        
        return {
            'resolution': resolution,
            'fractal_dim': np.nan,
            'fd_error': np.nan,
            'fd_r_squared': np.nan,
            'h_total': np.nan,
            'ht': np.nan,
            'hb': np.nan,
            'processing_time': total_time,
            'method': method_str,
            'error': str(e),
            'vtk_file': os.path.basename(vtk_file) if vtk_file else 'unknown',
            'time': np.nan
        }

def run_resolution_benchmark(resolution_files, output_dir, 
                           compare_baseline=True, use_parallel=True,
                           analysis_params=None):
    """
    Run optimized resolution benchmark across multiple resolutions.
    
    Args:
        resolution_files: Dictionary {resolution: vtk_file}
        output_dir: Output directory for results
        compare_baseline: If True, also run baseline for comparison
        use_parallel: If True, use parallel processing
        analysis_params: Analysis parameters (mixing_method, h0, etc.)
        
    Returns:
        DataFrame with results
    """
    if analysis_params is None:
        analysis_params = {'mixing_method': 'dalziel', 'h0': 0.5}
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"🚀 FIXED OPTIMIZED RESOLUTION BENCHMARK")
    print(f"=" * 60)
    print(f"Resolutions: {sorted(resolution_files.keys())}")
    print(f"Output directory: {output_dir}")
    print(f"Parallel processing: {use_parallel}")
    print(f"Baseline comparison: {compare_baseline}")
    
    # Prepare analysis tasks
    tasks = []
    
    # Add optimized analysis tasks
    for resolution, vtk_file in resolution_files.items():
        tasks.append((resolution, vtk_file, output_dir, True, analysis_params))
    
    # Add baseline comparison tasks if requested
    if compare_baseline:
        for resolution, vtk_file in resolution_files.items():
            tasks.append((resolution, vtk_file, output_dir, False, analysis_params))
    
    # Execute analyses
    results = []
    
    if use_parallel and len(tasks) > 1:
        print(f"\n🔄 Running {len(tasks)} analyses in parallel...")
        
        with ProcessPoolExecutor(max_workers=min(4, len(tasks))) as executor:
            future_to_task = {executor.submit(analyze_single_resolution_optimized, task): task 
                            for task in tasks}
            
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    result = future.result()
                    results.append(result)
                    resolution = task[0]
                    method = "optimized" if task[2] else "baseline"
                    time_taken = result.get('processing_time', 0)
                    print(f"✅ {resolution}×{resolution} ({method}): {time_taken:.2f}s")
                except Exception as e:
                    print(f"❌ Task failed: {e}")
                    # Add failed result
                    resolution = task[0]
                    method = "optimized" if task[2] else "baseline"
                    results.append({
                        'resolution': resolution,
                        'fractal_dim': np.nan,
                        'fd_error': np.nan,
                        'fd_r_squared': np.nan,
                        'h_total': np.nan,
                        'ht': np.nan,
                        'hb': np.nan,
                        'processing_time': np.nan,
                        'method': method,
                        'error': str(e)
                    })
    else:
        print(f"\n🔄 Running {len(tasks)} analyses sequentially...")
        
        for i, task in enumerate(tasks):
            print(f"\nTask {i+1}/{len(tasks)}")
            result = analyze_single_resolution_optimized(task)
            results.append(result)
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Save results
    df.to_csv(os.path.join(output_dir, 'fixed_resolution_benchmark_results.csv'), index=False)
    
    return df

def create_benchmark_plots(df, output_dir, target_time=0.0):
    """Create performance and convergence plots."""
    
    # Separate optimized and baseline results
    df_opt = df[df['method'] == 'optimized'].copy()
    df_base = df[df['method'] == 'baseline'].copy() if 'baseline' in df['method'].values else pd.DataFrame()
    
    # Sort by resolution
    df_opt = df_opt.sort_values('resolution')
    if not df_base.empty:
        df_base = df_base.sort_values('resolution')
    
    # Filter out NaN values for plotting
    df_opt = df_opt[~pd.isna(df_opt['fractal_dim'])]
    if not df_base.empty:
        df_base = df_base[~pd.isna(df_base['fractal_dim'])]
    
    if df_opt.empty:
        print("⚠️  No valid optimized results for plotting")
        return
    
    # Create plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Fractal dimension convergence
    ax1.errorbar(df_opt['resolution'], df_opt['fractal_dim'], 
                yerr=df_opt['fd_error'], fmt='bo-', capsize=5, 
                linewidth=2, markersize=8, label='Optimized')
    
    if not df_base.empty:
        ax1.errorbar(df_base['resolution'], df_base['fractal_dim'],
                    yerr=df_base['fd_error'], fmt='ro--', capsize=5,
                    linewidth=2, markersize=8, label='Baseline')
    
    ax1.set_xscale('log', base=2)
    ax1.set_xlabel('Grid Resolution (N)')
    ax1.set_ylabel('Fractal Dimension')
    ax1.set_title(f'Fractal Dimension Convergence (t ≈ {target_time})')
    ax1.grid(True, alpha=0.7)
    ax1.legend()
    
    # Plot 2: Processing time comparison
    ax2.plot(df_opt['resolution'], df_opt['processing_time'], 
            'bo-', linewidth=2, markersize=8, label='Optimized')
    
    if not df_base.empty:
        ax2.plot(df_base['resolution'], df_base['processing_time'],
                'ro--', linewidth=2, markersize=8, label='Baseline')
    
    ax2.set_xscale('log', base=2)
    ax2.set_yscale('log')
    ax2.set_xlabel('Grid Resolution (N)')
    ax2.set_ylabel('Processing Time (seconds)')
    ax2.set_title('Processing Time vs Resolution')
    ax2.grid(True, alpha=0.7)
    ax2.legend()
    
    # Plot 3: Mixing thickness convergence
    ax3.plot(df_opt['resolution'], df_opt['h_total'], 
            'go-', linewidth=2, markersize=8, label='Total (Optimized)')
    ax3.plot(df_opt['resolution'], df_opt['ht'], 
            'bs--', linewidth=2, markersize=6, label='Upper (Optimized)')
    ax3.plot(df_opt['resolution'], df_opt['hb'], 
            'rd--', linewidth=2, markersize=6, label='Lower (Optimized)')
    
    ax3.set_xscale('log', base=2)
    ax3.set_xlabel('Grid Resolution (N)')
    ax3.set_ylabel('Mixing Layer Thickness')
    ax3.set_title(f'Mixing Layer Convergence (t ≈ {target_time})')
    ax3.grid(True, alpha=0.7)
    ax3.legend()
    
    # Plot 4: R² quality vs resolution
    ax4.plot(df_opt['resolution'], df_opt['fd_r_squared'], 
            'mo-', linewidth=2, markersize=8, label='Optimized')
    
    if not df_base.empty:
        ax4.plot(df_base['resolution'], df_base['fd_r_squared'],
                'co--', linewidth=2, markersize=8, label='Baseline')
    
    ax4.set_xscale('log', base=2)
    ax4.set_xlabel('Grid Resolution (N)')
    ax4.set_ylabel('R² Value')
    ax4.set_title('Fit Quality vs Resolution')
    ax4.grid(True, alpha=0.7)
    ax4.set_ylim(0.9, 1.0)
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fixed_resolution_benchmark_analysis.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Performance summary plot
    if not df_base.empty:
        plt.figure(figsize=(10, 6))
        
        # Calculate speedup
        speedup_data = []
        for res in df_opt['resolution']:
            opt_time = df_opt[df_opt['resolution'] == res]['processing_time'].values
            base_time = df_base[df_base['resolution'] == res]['processing_time'].values
            
            if len(opt_time) > 0 and len(base_time) > 0:
                speedup = base_time[0] / opt_time[0]
                speedup_data.append((res, speedup))
        
        if speedup_data:
            resolutions, speedups = zip(*speedup_data)
            plt.plot(resolutions, speedups, 'go-', linewidth=2, markersize=8)
            plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='No speedup')
            
            plt.xscale('log', base=2)
            plt.xlabel('Grid Resolution (N)')
            plt.ylabel('Speedup Factor')
            plt.title('Optimization Speedup vs Resolution')
            plt.grid(True, alpha=0.7)
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'fixed_speedup_analysis.png'), 
                        dpi=300, bbox_inches='tight')
            plt.close()

def main():
    """Main benchmark function."""
    parser = argparse.ArgumentParser(description='FIXED Optimized resolution convergence benchmark')
    parser.add_argument('--base-dirs', nargs='+', required=True,
                       help='Base directories containing resolution data')
    parser.add_argument('--target-time', type=float, default=0.0,
                       help='Target simulation time (default: 0.0)')
    parser.add_argument('--output', default='./fixed_resolution_benchmark_optimized',
                       help='Output directory')
    parser.add_argument('--no-baseline', action='store_true',
                       help='Skip baseline comparison (faster)')
    parser.add_argument('--no-parallel', action='store_true',
                       help='Disable parallel processing')
    parser.add_argument('--mixing-method', default='dalziel',
                       choices=['geometric', 'statistical', 'dalziel'],
                       help='Mixing thickness method')
    parser.add_argument('--h0', type=float, default=0.5,
                       help='Initial interface position')
    parser.add_argument('--min-box-size', type=float, default=None,
                       help='Minimum box size for fractal analysis')
    
    args = parser.parse_args()
    
    print(f"🔬 FIXED OPTIMIZED RESOLUTION BENCHMARK")
    print(f"=" * 60)
    
    # Find resolution files
    resolution_files = find_resolution_files(args.base_dirs, args.target_time)
    
    if not resolution_files:
        print("❌ No suitable VTK files found!")
        return
    
    # Set analysis parameters
    analysis_params = {
        'mixing_method': args.mixing_method,
        'h0': args.h0,
        'min_box_size': args.min_box_size
    }
    
    # Run benchmark
    df = run_resolution_benchmark(
        resolution_files,
        args.output,
        compare_baseline=not args.no_baseline,
        use_parallel=not args.no_parallel,
        analysis_params=analysis_params
    )
    
    # Create plots
    create_benchmark_plots(df, args.output, args.target_time)
    
    # Print summary
    print(f"\n📊 FIXED BENCHMARK SUMMARY")
    print(f"=" * 60)
    
    df_opt = df[df['method'] == 'optimized']
    valid_opt = df_opt[~pd.isna(df_opt['fractal_dim'])]
    
    if not valid_opt.empty:
        print(f"Optimized results:")
        for _, row in valid_opt.iterrows():
            print(f"  {row['resolution']:4d}×{row['resolution']:<4d}: "
                  f"D = {row['fractal_dim']:.4f} ± {row['fd_error']:.4f}, "
                  f"Time = {row['processing_time']:.2f}s")
    
    if not args.no_baseline:
        df_base = df[df['method'] == 'baseline']
        valid_base = df_base[~pd.isna(df_base['fractal_dim'])]
        
        if not valid_base.empty:
            print(f"\nBaseline comparison:")
            total_opt_time = valid_opt['processing_time'].sum()
            total_base_time = valid_base['processing_time'].sum()
            if total_opt_time > 0:
                overall_speedup = total_base_time / total_opt_time
                print(f"  Total optimized time: {total_opt_time:.2f}s")
                print(f"  Total baseline time: {total_base_time:.2f}s")
                print(f"  Overall speedup: {overall_speedup:.2f}×")
    
    print(f"\n✅ Fixed results saved to: {args.output}")

if __name__ == "__main__":
    main()
