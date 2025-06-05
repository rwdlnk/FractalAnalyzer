#!/usr/bin/env python3
import os
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fractal_analyzer import RTAnalyzer

def find_vtk_files_with_times(base_pattern, target_times=None):
    """
    Find VTK files and extract their times, optionally filtering by target times.
    
    Args:
        base_pattern: Pattern like "../800x800/RT800x800-*.vtk"
        target_times: List of target times to find closest matches for (optional)
    
    Returns:
        List of (time, filename) tuples, sorted by time
    """
    # Find all VTK files matching the pattern
    vtk_files = glob.glob(base_pattern)
    
    if not vtk_files:
        print(f"Warning: No VTK files found matching pattern: {base_pattern}")
        return []
    
    # Extract times from filenames
    file_times = []
    for vtk_file in vtk_files:
        try:
            # Extract time from filename like RT800x800-5999.vtk
            basename = os.path.basename(vtk_file)
            time_str = basename.split('-')[1].split('.')[0]
            file_time = float(time_str) / 1000.0
            file_times.append((file_time, vtk_file))
        except (IndexError, ValueError) as e:
            print(f"Warning: Could not extract time from {vtk_file}: {e}")
            continue
    
    # Sort by time
    file_times.sort(key=lambda x: x[0])
    
    if target_times is None:
        # Return all files
        return file_times
    else:
        # Find closest matches to target times
        closest_matches = []
        for target_time in target_times:
            best_file = None
            best_diff = float('inf')
            
            for file_time, vtk_file in file_times:
                diff = abs(file_time - target_time)
                if diff < best_diff:
                    best_diff = diff
                    best_file = (file_time, vtk_file)
            
            if best_file and best_diff < 0.5:  # Within 0.5 time units
                closest_matches.append(best_file)
                print(f"Target time {target_time:.1f} → Found {os.path.basename(best_file[1])} (t={best_file[0]:.3f})")
            else:
                print(f"Warning: No file found within 0.5 time units of target {target_time:.1f}")
        
        return sorted(closest_matches, key=lambda x: x[0])

def analyze_temporal_evolution(output_dir, resolutions, base_pattern=None, specific_times=None, 
                              time_tolerance=0.5, auto_detect_times=False, mixing_method='dalziel',h0=0.5):
    """Analyze fractal dimension evolution over time for different resolutions."""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Dictionary to store results for each resolution
    all_results = {}
    
    # Process each resolution
    for resolution in resolutions:
        print(f"\n=== Analyzing temporal evolution for {resolution}x{resolution} resolution ===\n")
        
        # Create analyzer instance for this resolution
        res_dir = os.path.join(output_dir, f'res_{resolution}')
        analyzer = RTAnalyzer(res_dir)
        
        # Create pattern for this resolution
        if base_pattern:
           pattern = base_pattern.format(resolution=resolution)
        else:
            pattern = f"./{resolution}x{resolution}/RT{resolution}x{resolution}-*.vtk"  # Changed ../ to ./
        print(f"Using pattern: {pattern}")

        # Check if resolution directory exists
        resolution_dir = f"./{resolution}x{resolution}"
        if not os.path.exists(resolution_dir):
            print(f"Warning: Directory {resolution_dir} does not exist. Skipping {resolution}x{resolution}")
            continue

        # Find VTK files and their times
        if specific_times:
            # Find closest matches to specific times
            file_time_pairs = find_vtk_files_with_times(pattern, specific_times)
        else:
            # Find all files or auto-detect reasonable time range
            all_files = find_vtk_files_with_times(pattern)
            
            if auto_detect_times and all_files:
                # Auto-select reasonable time range (e.g., every 1-2 time units)
                min_time = all_files[0][0]
                max_time = all_files[-1][0]
                
                # Create target times with reasonable spacing
                if max_time - min_time > 10:
                    # For long simulations, sample every 2 time units
                    spacing = 2.0
                else:
                    # For shorter simulations, sample every 1 time unit
                    spacing = 1.0
                
                target_times = np.arange(
                    np.ceil(min_time), 
                    np.floor(max_time) + 0.1, 
                    spacing
                )
                
                print(f"Auto-detected time range: {min_time:.1f} to {max_time:.1f}")
                print(f"Using target times: {list(target_times)}")
                
                file_time_pairs = find_vtk_files_with_times(pattern, target_times)
            else:
                # Use all available files
                file_time_pairs = all_files
        
        if not file_time_pairs:
            print(f"Warning: No suitable VTK files found for {resolution}x{resolution}")
            continue
        
        print(f"Found {len(file_time_pairs)} files for {resolution}x{resolution} resolution")
        
        # Results for this resolution
        results = []
        
        # Process each file
        for i, (actual_time, vtk_file) in enumerate(file_time_pairs):
            print(f"Processing file {i+1}/{len(file_time_pairs)}: {os.path.basename(vtk_file)} (t={actual_time:.3f})")

            try:
                # Use the complete analysis method instead of separate calls
                result = analyzer.analyze_vtk_file(vtk_file, mixing_method=mixing_method, h0=h0)
    
                # Store results directly from the complete analysis
                results.append({
                    'time': actual_time,
                    'h0': h0,
                    'ht': result['ht'],
                    'hb': result['hb'],
                    'h_total': result['h_total'],
                    'fractal_dim': result['fractal_dim'],
                    'fd_error': result['fd_error'],
                    'fd_r_squared': result['fd_r_squared'],
                    'resolution': resolution,
                    'vtk_file': vtk_file
                })
    
                print(f"  Time: {actual_time:.3f}, Dimension: {result['fractal_dim']:.4f}, "
                      f"Mixing: {result['h_total']:.4f}, R²: {result['fd_r_squared']:.4f}")
    
                # Validate results
                if not (1.0 <= result['fractal_dim'] <= 2.0):
                    print(f"  Warning: Fractal dimension {result['fractal_dim']:.3f} outside physical range [1.0, 2.0]")
    
                if result['h_total'] <= 0:
                    print(f"  Warning: Non-positive mixing thickness {result['h_total']:.6f}")

            except Exception as e:
                print(f"Error processing {vtk_file}: {str(e)}")
                import traceback
                traceback.print_exc()
        
        # Convert results to DataFrame and sort by time
        if results:
            df = pd.DataFrame(results)
            df = df.sort_values('time')
            all_results[resolution] = df
            
            # Save results for this resolution
            os.makedirs(res_dir, exist_ok=True)
            df.to_csv(os.path.join(res_dir, 'temporal_evolution.csv'), index=False)
            
            # Create individual plots for this resolution
            plot_single_resolution_evolution(df, resolution, res_dir)
            
            print(f"Completed {resolution}x{resolution}: {len(results)} time points analyzed")
        else:
            print(f"No results for {resolution}x{resolution} resolution")
    
    # Create combined plots across resolutions
    if all_results:
        plot_multi_resolution_evolution(all_results, resolutions, output_dir)
        
        # Create summary CSV with all results
        create_combined_summary(all_results, output_dir)
    
    return all_results

def create_combined_summary(all_results, output_dir):
    """Create a combined CSV with all temporal evolution results."""
    combined_data = []
    
    for resolution, df in all_results.items():
        for _, row in df.iterrows():
            combined_data.append(row.to_dict())
    
    combined_df = pd.DataFrame(combined_data)
    combined_df = combined_df.sort_values(['resolution', 'time'])
    
    # Save combined results
    combined_df.to_csv(os.path.join(output_dir, 'all_temporal_results.csv'), index=False)
    
    # Create summary statistics
    summary_stats = combined_df.groupby('resolution').agg({
        'time': ['min', 'max', 'count'],
        'fractal_dim': ['mean', 'std', 'min', 'max'],
        'h_total': ['mean', 'std', 'min', 'max'],
        'fd_r_squared': ['mean', 'min']
    }).round(4)
    
    # Flatten column names
    summary_stats.columns = ['_'.join(col).strip() for col in summary_stats.columns.values]
    summary_stats.to_csv(os.path.join(output_dir, 'temporal_summary_statistics.csv'))
    
    print(f"Combined summary saved with {len(combined_data)} total analyses")

def plot_single_resolution_evolution(df, resolution, output_dir):
    """Create plots for a single resolution's temporal evolution."""
    # Plot fractal dimension vs time
    plt.figure(figsize=(10, 6))
    plt.errorbar(df['time'], df['fractal_dim'], yerr=df['fd_error'],
                fmt='o-', capsize=3, linewidth=2, markersize=5)
    plt.xlabel('Time')
    plt.ylabel('Fractal Dimension')
    plt.title(f'Fractal Dimension Evolution ({resolution}x{resolution})')
    plt.grid(True)
    
    # Add trend line if enough points
    if len(df) >= 3:
        # Fit polynomial trend
        z = np.polyfit(df['time'], df['fractal_dim'], 2)
        p = np.poly1d(z)
        time_smooth = np.linspace(df['time'].min(), df['time'].max(), 100)
        plt.plot(time_smooth, p(time_smooth), 'r--', alpha=0.7, label='Trend')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'dimension_evolution.png'), dpi=300)
    plt.close()
    
    # Plot mixing layer thickness vs time
    plt.figure(figsize=(10, 6))
    plt.plot(df['time'], df['h_total'], 'b-', label='Total', linewidth=2)
    plt.plot(df['time'], df['ht'], 'r--', label='Upper', linewidth=2)
    plt.plot(df['time'], df['hb'], 'g--', label='Lower', linewidth=2)
    plt.xlabel('Time')
    plt.ylabel('Mixing Layer Thickness')
    plt.title(f'Mixing Layer Evolution ({resolution}x{resolution})')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'mixing_evolution.png'), dpi=300)
    plt.close()
    
    # Combined plot of fractal dimension and mixing thickness
    fig, ax1 = plt.subplots(figsize=(12, 8))
    
    # Fractal dimension on left axis
    ax1.errorbar(df['time'], df['fractal_dim'], yerr=df['fd_error'],
               fmt='bo-', capsize=3, label='Fractal Dimension')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Fractal Dimension', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    
    # Mixing layer on right axis
    ax2 = ax1.twinx()
    ax2.plot(df['time'], df['h_total'], 'r-', label='Mixing Thickness', linewidth=2)
    ax2.set_ylabel('Mixing Layer Thickness', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    
    # Add both legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.title(f'Fractal Dimension and Mixing Layer Evolution ({resolution}x{resolution})')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'combined_evolution.png'), dpi=300)
    plt.close()

def plot_multi_resolution_evolution(all_results, resolutions, output_dir):
    """Create plots comparing temporal evolution across multiple resolutions."""
    # Create output directory
    multi_res_dir = os.path.join(output_dir, 'multi_resolution')
    os.makedirs(multi_res_dir, exist_ok=True)
    
    # Plot fractal dimension evolution
    plt.figure(figsize=(12, 8))
    
    colors = ['b', 'g', 'r', 'm', 'c', 'y', 'k']
    markers = ['o', 's', '^', 'D', 'v', '<', '>']
    
    for i, resolution in enumerate(resolutions):
        if resolution not in all_results:
            continue
        
        df = all_results[resolution]
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]
        
        plt.errorbar(df['time'], df['fractal_dim'], yerr=df['fd_error'],
                   fmt=f'{color}{marker}-', capsize=3, linewidth=1.5, 
                   label=f'{resolution}x{resolution}')
    
    plt.xlabel('Time')
    plt.ylabel('Fractal Dimension')
    plt.title('Fractal Dimension Evolution Across Resolutions')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(multi_res_dir, 'dimension_evolution_comparison.png'), dpi=300)
    plt.close()
    
    # Plot mixing layer evolution
    plt.figure(figsize=(12, 8))
    
    for i, resolution in enumerate(resolutions):
        if resolution not in all_results:
            continue
        
        df = all_results[resolution]
        color = colors[i % len(colors)]
        
        plt.plot(df['time'], df['h_total'], f'{color}-', linewidth=2, 
                label=f'{resolution}x{resolution}')
    
    plt.xlabel('Time')
    plt.ylabel('Mixing Layer Thickness')
    plt.title('Mixing Layer Evolution Across Resolutions')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(multi_res_dir, 'mixing_evolution_comparison.png'), dpi=300)
    plt.close()
    
    # Phase plot: Fractal dimension vs. Mixing layer thickness
    plt.figure(figsize=(12, 8))
    
    for i, resolution in enumerate(resolutions):
        if resolution not in all_results:
            continue
        
        df = all_results[resolution]
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]
        
        plt.plot(df['h_total'], df['fractal_dim'], f'{color}{marker}-', linewidth=1.5, 
               label=f'{resolution}x{resolution}')
        
        # Add time labels to selected points
        if len(df) > 4:
            # Add labels every nth point
            n = max(1, len(df) // 5)
            for j in range(0, len(df), n):
                time = df['time'].iloc[j]
                plt.annotate(f't={time:.1f}', 
                            (df['h_total'].iloc[j], df['fractal_dim'].iloc[j]),
                            textcoords="offset points", 
                            xytext=(0,10), 
                            ha='center', fontsize=8)
    
    plt.xlabel('Mixing Layer Thickness')
    plt.ylabel('Fractal Dimension')
    plt.title('Phase Portrait: Fractal Dimension vs. Mixing Layer Thickness')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(multi_res_dir, 'phase_portrait.png'), dpi=300)
    plt.close()

def main():
    """Main function for console script entry point."""
    parser = argparse.ArgumentParser(description='Analyze fractal dimension temporal evolution with improved file finding')
    parser.add_argument('--resolutions', '-r', type=int, nargs='+', required=True,
                      help='Resolutions to analyze (e.g., 100 200 400 800)')
    parser.add_argument('--output', '-o', default='./temporal_analysis',
                      help='Output directory')
    parser.add_argument('--pattern', default=None,
                      help='Pattern for VTK files with {resolution} placeholder')
    parser.add_argument('--times', type=float, nargs='*',
                      help='Specific time points to analyze (optional)')
    parser.add_argument('--auto-times', action='store_true',
                      help='Auto-detect reasonable time range for analysis')
    parser.add_argument('--time-tolerance', type=float, default=0.5,
                      help='Maximum time difference for matching files (default: 0.5)')
    parser.add_argument('--mixing-method', default='dalziel', 
                  choices=['geometric', 'statistical', 'dalziel'],
                  help='Method for computing mixing layer thickness (default: dalziel)')
    parser.add_argument('--h0', type=float, default=0.5,
                      help='Initial interface position in physical coordinates (default: 0.5)')
    args = parser.parse_args()
    
    # Run analysis
    results = analyze_temporal_evolution(
        args.output, 
        args.resolutions, 
        args.pattern, 
        args.times,
        args.time_tolerance,
        args.auto_times,
        args.mixing_method,
        args.h0
    )
    
    if results:
        print(f"\n{'='*60}")
        print("Temporal evolution analysis complete!")
        print(f"Results saved to: {args.output}")
        
        # Print summary
        total_analyses = sum(len(df) for df in results.values())
        print(f"Total analyses completed: {total_analyses}")
        
        for resolution, df in results.items():
            time_range = f"{df['time'].min():.1f} to {df['time'].max():.1f}"
            avg_dim = df['fractal_dim'].mean()
            print(f"  {resolution}x{resolution}: {len(df)} points, t={time_range}, avg D={avg_dim:.3f}")
    else:
        print("Analysis failed - check file paths and parameters")

if __name__ == "__main__":
    main()
