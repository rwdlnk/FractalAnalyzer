#!/usr/bin/env python3
"""
Full Temporal Evolution Analyzer: Process all timesteps to analyze fractal dimension evolution.

This script processes all available VTK files in a directory to study how fractal
dimension evolves over time in Rayleigh-Taylor simulations.
"""

import os
import time
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fractal_analyzer.core.rt_analyzer import RTAnalyzer

def analyze_temporal_evolution(data_dir, resolution, output_dir=None, 
                              mixing_method='dalziel', max_timesteps=None,
                              h0=0.5, min_box_size=None):
    """
    Analyze fractal dimension evolution across all timesteps.
    
    Args:
        data_dir: Directory containing VTK files
        resolution: Grid resolution
        output_dir: Output directory (default: ./temporal_evolution_{resolution})
        mixing_method: Method for mixing thickness calculation
        max_timesteps: Maximum number of timesteps to process (None = all)
        h0: Initial interface position
        min_box_size: Minimum box size for fractal analysis
        
    Returns:
        DataFrame with temporal evolution results
    """
    
    if output_dir is None:
        output_dir = f"./temporal_evolution_{resolution}x{resolution}"
    
    print(f"🕰️  TEMPORAL EVOLUTION ANALYSIS")
    print(f"=" * 60)
    print(f"Data directory: {data_dir}")
    print(f"Resolution: {resolution}×{resolution}")
    print(f"Output directory: {output_dir}")
    print(f"Mixing method: {mixing_method}")
    print(f"Initial interface: h0 = {h0}")
    if min_box_size:
        print(f"Min box size: {min_box_size}")
    
    # Find all VTK files
    pattern = os.path.join(data_dir, f"RT{resolution}x{resolution}-*.vtk")
    vtk_files = sorted(glob.glob(pattern))
    
    if max_timesteps:
        vtk_files = vtk_files[:max_timesteps]
    
    if not vtk_files:
        print(f"❌ No VTK files found matching: {pattern}")
        return None
    
    print(f"Found {len(vtk_files)} timestep files")
    
    # Create analyzer
    analyzer = RTAnalyzer(output_dir, use_grid_optimization=False, no_titles=False)
    
    # Process all files
    results = []
    start_time = time.time()
    
    for i, vtk_file in enumerate(vtk_files):
        basename = os.path.basename(vtk_file)
        print(f"\n📊 Processing {i+1}/{len(vtk_files)}: {basename}")
        
        try:
            # Analyze this timestep
            file_start = time.time()
            
            result = analyzer.analyze_vtk_file(
                vtk_file,
                output_subdir=f"timestep_{i+1:04d}",
                mixing_method=mixing_method,
                h0=h0,
                min_box_size=min_box_size
            )
            
            file_time = time.time() - file_start
            
            # Add processing info
            result['timestep'] = i + 1
            result['vtk_file'] = basename
            result['processing_time'] = file_time
            
            results.append(result)
            
            # Progress report
            print(f"   ✅ t={result['time']:.3f}: D={result['fractal_dim']:.4f}±{result['fd_error']:.4f}, "
                  f"h_total={result['h_total']:.4f}, time={file_time:.2f}s")
            
        except Exception as e:
            print(f"   ❌ Failed: {str(e)}")
            # Add failed result to maintain timestep indexing
            results.append({
                'timestep': i + 1,
                'vtk_file': basename,
                'time': np.nan,
                'fractal_dim': np.nan,
                'fd_error': np.nan,
                'fd_r_squared': np.nan,
                'h_total': np.nan,
                'ht': np.nan,
                'hb': np.nan,
                'processing_time': np.nan,
                'error': str(e)
            })
    
    total_time = time.time() - start_time
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Save results
    results_file = os.path.join(output_dir, f'temporal_evolution_{resolution}x{resolution}.csv')
    df.to_csv(results_file, index=False)
    
    print(f"\n📈 TEMPORAL EVOLUTION SUMMARY")
    print(f"=" * 60)
    print(f"Total timesteps processed: {len(results)}")
    
    # Filter valid results
    valid_results = df[~pd.isna(df['fractal_dim'])]
    print(f"Successful analyses: {len(valid_results)}/{len(results)}")
    
    if len(valid_results) > 0:
        print(f"Time range: t = {valid_results['time'].min():.3f} to {valid_results['time'].max():.3f}")
        print(f"Fractal dimension range: D = {valid_results['fractal_dim'].min():.4f} to {valid_results['fractal_dim'].max():.4f}")
        print(f"Final mixing thickness: h_total = {valid_results['h_total'].iloc[-1]:.4f}")
        print(f"Average processing time: {valid_results['processing_time'].mean():.2f}s per timestep")
    
    print(f"Total processing time: {total_time:.1f}s")
    print(f"Results saved to: {results_file}")
    
    # Create temporal evolution plots
    create_evolution_plots(df, output_dir, resolution)
    
    return df

def create_evolution_plots(df, output_dir, resolution):
    """Create comprehensive temporal evolution plots."""
    
    # Filter valid data
    valid_df = df[~pd.isna(df['fractal_dim'])].copy()
    
    if len(valid_df) == 0:
        print("⚠️  No valid data for plotting")
        return
    
    print(f"\n📊 Creating temporal evolution plots...")
    
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
    ax1.set_title(f'Fractal Dimension Evolution ({resolution}×{resolution})')
    ax1.grid(True, alpha=0.7)
    ax1.legend()
    
    # Plot 2: Mixing layer evolution
    ax2.plot(valid_df['time'], valid_df['h_total'], 'g-', linewidth=2, label='Total')
    ax2.plot(valid_df['time'], valid_df['ht'], 'r--', linewidth=2, label='Upper')
    ax2.plot(valid_df['time'], valid_df['hb'], 'b--', linewidth=2, label='Lower')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Mixing Layer Thickness')
    ax2.set_title(f'Mixing Layer Evolution ({resolution}×{resolution})')
    ax2.grid(True, alpha=0.7)
    ax2.legend()
    
    # Plot 3: R² evolution (fit quality)
    ax3.plot(valid_df['time'], valid_df['fd_r_squared'], 'mo-', 
            linewidth=2, markersize=4, label='R²')
    ax3.axhline(y=0.99, color='orange', linestyle='--', alpha=0.7, label='R² = 0.99')
    ax3.set_xlabel('Time')
    ax3.set_ylabel('R² Value')
    ax3.set_title(f'Fractal Analysis Quality ({resolution}×{resolution})')
    ax3.set_ylim(0.95, 1.01)
    ax3.grid(True, alpha=0.7)
    ax3.legend()
    
    # Plot 4: Processing time evolution
    ax4.plot(valid_df['time'], valid_df['processing_time'], 'co-', 
            linewidth=2, markersize=4, label='Processing Time')
    ax4.set_xlabel('Time')
    ax4.set_ylabel('Processing Time (s)')
    ax4.set_title(f'Computational Performance ({resolution}×{resolution})')
    ax4.grid(True, alpha=0.7)
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'temporal_evolution_{resolution}x{resolution}.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create combined D vs h_total plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Main plot
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
    ax.set_title(f'Fractal Dimension vs Mixing Thickness ({resolution}×{resolution})')
    ax.grid(True, alpha=0.7)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'D_vs_mixing_{resolution}x{resolution}.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   ✅ Saved temporal evolution plots to {output_dir}")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Full temporal evolution analysis of fractal dimension',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all timesteps for 200x200 resolution
  python temporal_evolution_analyzer.py --data-dir ~/Research/svofRuns/Dalziel/200x200 --resolution 200

  # Process first 50 timesteps with custom parameters
  python temporal_evolution_analyzer.py --data-dir ~/Research/svofRuns/Dalziel/400x400 --resolution 400 --max-timesteps 50 --min-box-size 0.001

  # Use geometric mixing method
  python temporal_evolution_analyzer.py --data-dir ~/Research/svofRuns/Dalziel/200x200 --resolution 200 --mixing-method geometric
""")
    
    parser.add_argument('--data-dir', required=True,
                       help='Directory containing VTK files')
    parser.add_argument('--resolution', type=int, required=True,
                       help='Grid resolution')
    parser.add_argument('--output-dir', default=None,
                       help='Output directory (default: ./temporal_evolution_RESxRES)')
    parser.add_argument('--mixing-method', default='dalziel',
                       choices=['geometric', 'statistical', 'dalziel'],
                       help='Mixing thickness calculation method')
    parser.add_argument('--max-timesteps', type=int, default=None,
                       help='Maximum timesteps to process (default: all)')
    parser.add_argument('--h0', type=float, default=0.5,
                       help='Initial interface position')
    parser.add_argument('--min-box-size', type=float, default=None,
                       help='Minimum box size for fractal analysis')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.data_dir):
        print(f"❌ Data directory not found: {args.data_dir}")
        return
    
    # Run temporal evolution analysis
    df = analyze_temporal_evolution(
        args.data_dir,
        args.resolution,
        args.output_dir,
        args.mixing_method,
        args.max_timesteps,
        args.h0,
        args.min_box_size
    )
    
    if df is not None:
        print(f"\n🎉 Temporal evolution analysis complete!")
        print(f"Check the output directory for detailed results and plots.")

if __name__ == "__main__":
    main()
