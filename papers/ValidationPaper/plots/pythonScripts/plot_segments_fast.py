#!/usr/bin/env python3
"""
Fast Curve Segment Plotter - Optimized for large datasets

Plots a curve defined by line segments from a data file.
Each line in the input file should contain 4 space-separated coordinates: x1 y1 x2 y2

Usage:
    python plot_curve_fast.py input_file.txt
    python plot_curve_fast.py input_file.txt --outfile curve.pdf
"""

import argparse
import matplotlib.pyplot as plt
import matplotlib.collections as mc
import numpy as np
from pathlib import Path
import sys
import time

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Plot curve segments from a data file (optimized for large files)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  %(prog)s segments.txt
  %(prog)s segments.txt --outfile curve.pdf
  %(prog)s segments.txt --outfile curve.svg --title "Dragon Curve"
  %(prog)s segments.txt --subsample 10000

Supported output formats:
  PNG, PDF, SVG, EPS, PS, TIFF, JPEG

Performance tips:
  - Use --subsample for very large files (>100k segments)
  - Use vector formats (PDF, SVG) for best quality
  - Use PNG with lower DPI for faster rendering
        '''
    )
    
    parser.add_argument('input_file', 
                       help='Input file containing segment coordinates (x1 y1 x2 y2 per line)')
    
    parser.add_argument('--outfile', '-o',
                       help='Output file name (default: input_name.png)')
    
    parser.add_argument('--title', '-t',
                       help='Plot title (default: derived from filename)')
    
    parser.add_argument('--figsize', nargs=2, type=float, default=[10, 10],
                       metavar=('WIDTH', 'HEIGHT'),
                       help='Figure size in inches (default: 10 10)')
    
    parser.add_argument('--grid', action='store_true',
                       help='Show grid on plot')
    
    parser.add_argument('--dpi', type=int, default=150,
                       help='DPI for raster formats (default: 150, use 300+ for high quality)')
    
    parser.add_argument('--linewidth', '-lw', type=float, default=0.5,
                       help='Line width for plotting (default: 0.5 for large datasets)')
    
    parser.add_argument('--color', '-c', default='black',
                       help='Line color (default: black)')
    
    parser.add_argument('--subsample', type=int,
                       help='Subsample to N segments for faster rendering (e.g., --subsample 50000)')
    
    parser.add_argument('--alpha', type=float, default=1.0,
                       help='Line transparency (0.0-1.0, default: 1.0)')
    
    parser.add_argument('--no-title', action='store_true',
                       help='Suppress plot title (recommended for journal figures)')
    
    return parser.parse_args()

def read_segments_fast(filename, subsample=None):
    """
    Read line segments from file using numpy for speed.
    
    Args:
        filename: Path to input file
        subsample: If specified, randomly subsample to this many segments
        
    Returns:
        numpy.ndarray: Array of shape (N, 4) containing segments
    """
    print(f"Reading segments from '{filename}'...")
    start_time = time.time()
    
    try:
        # Use numpy to read the entire file at once - much faster!
        data = np.loadtxt(filename)
        
        if data.ndim == 1:
            data = data.reshape(1, -1)
        
        if data.shape[1] != 4:
            print(f"Error: Expected 4 columns, got {data.shape[1]}")
            sys.exit(1)
        
        read_time = time.time() - start_time
        print(f"Read {len(data)} segments in {read_time:.2f} seconds")
        
        # Subsample if requested
        if subsample and len(data) > subsample:
            print(f"Subsampling to {subsample} segments...")
            indices = np.random.choice(len(data), subsample, replace=False)
            data = data[indices]
            print(f"Using {len(data)} segments after subsampling")
        
        return data
        
    except Exception as e:
        print(f"Error reading file '{filename}': {e}")
        sys.exit(1)

def plot_curve_fast(segments, args):
    """
    Plot the curve segments using LineCollection for speed.
    
    Args:
        segments: numpy array of shape (N, 4) with segments
        args: Parsed command line arguments
    """
    print("Creating plot...")
    start_time = time.time()
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=args.figsize)
    
    # Convert segments to line collection format
    # segments has shape (N, 4) where each row is [x1, y1, x2, y2]
    # LineCollection needs list of [(x1,y1), (x2,y2)] pairs
    lines = [[(x1, y1), (x2, y2)] for x1, y1, x2, y2 in segments]
    
    # Create LineCollection - MUCH faster than individual plot calls
    lc = mc.LineCollection(lines, 
                          linewidths=args.linewidth,
                          colors=args.color,
                          alpha=args.alpha,
                          capstyle='round')
    
    ax.add_collection(lc)
    
    # Set axis limits based on data
    all_x = np.concatenate([segments[:, 0], segments[:, 2]])
    all_y = np.concatenate([segments[:, 1], segments[:, 3]])
    
    margin = 0.02 * max(all_x.max() - all_x.min(), all_y.max() - all_y.min())
    ax.set_xlim(all_x.min() - margin, all_x.max() + margin)
    ax.set_ylim(all_y.min() - margin, all_y.max() + margin)
    
    # Set title
    if not args.no_title:
        if args.title:
            title = args.title
        else:
            input_path = Path(args.input_file)
            title = input_path.stem.replace('_', ' ').title()
            if len(segments) > 10000:
                title += f" ({len(segments):,} segments)"
        
        ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Labels and formatting
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    
    # Grid
    if args.grid:
        ax.grid(True, alpha=0.3, zorder=0)
    
    # Equal aspect ratio for proper geometry
    ax.set_aspect('equal', adjustable='box')
    
    # Tight layout
    plt.tight_layout()
    
    plot_time = time.time() - start_time
    print(f"Created plot in {plot_time:.2f} seconds")
    
    return fig

def determine_output_filename(input_file, output_file):
    """Determine the output filename and validate format."""
    supported_formats = {'.png', '.pdf', '.svg', '.eps', '.ps', '.tiff', '.tif', '.jpg', '.jpeg'}
    
    if output_file is None:
        input_path = Path(input_file)
        output_file = input_path.with_suffix('.png')
    else:
        output_path = Path(output_file)
        if output_path.suffix.lower() not in supported_formats:
            print(f"Warning: Unsupported format '{output_path.suffix}'. Using .png")
            output_file = output_path.with_suffix('.png')
    
    return str(output_file)

def save_plot(fig, output_file, dpi):
    """Save the plot to file."""
    print(f"Saving plot to '{output_file}'...")
    start_time = time.time()
    
    output_path = Path(output_file)
    
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if output_path.suffix.lower() in {'.png', '.jpg', '.jpeg', '.tiff', '.tif'}:
            fig.savefig(output_file, dpi=dpi, bbox_inches='tight')
        else:
            fig.savefig(output_file, bbox_inches='tight')
        
        save_time = time.time() - start_time
        print(f"Plot saved in {save_time:.2f} seconds")
        
    except Exception as e:
        print(f"Error saving plot to '{output_file}': {e}")
        sys.exit(1)

def main():
    """Main function."""
    total_start = time.time()
    
    args = parse_arguments()
    
    # Read segments from input file
    segments = read_segments_fast(args.input_file, args.subsample)
    
    # Performance recommendations
    if len(segments) > 100000:
        print(f"\nPerformance note: {len(segments):,} segments is quite large.")
        print("Consider using:")
        print("  --subsample 50000     (for faster rendering)")
        print("  --linewidth 0.3       (thinner lines)")
        print("  --dpi 150             (lower resolution)")
        print("  --outfile plot.pdf    (vector format)\n")
    
    # Determine output filename
    output_file = determine_output_filename(args.input_file, args.outfile)
    
    # Create plot
    fig = plot_curve_fast(segments, args)
    
    # Save plot
    save_plot(fig, output_file, args.dpi)
    
    total_time = time.time() - total_start
    print(f"\nTotal time: {total_time:.2f} seconds")
    print(f"Plotted {len(segments):,} line segments")
    
    plt.close(fig)

if __name__ == '__main__':
    main()
