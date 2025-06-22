#!/usr/bin/env python3
"""
Curve Segment Plotter

Plots a curve defined by line segments from a data file.
Each line in the input file should contain 4 space-separated coordinates: x1 y1 x2 y2

Usage:
    python plot_curve.py input_file.txt
    python plot_curve.py input_file.txt --outfile curve.pdf
    python plot_curve.py input_file.txt --outfile curve.svg --title "My Curve"
"""

import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import sys
import os

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Plot curve segments from a data file',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  %(prog)s segments.txt
  %(prog)s segments.txt --outfile curve.pdf
  %(prog)s segments.txt --outfile curve.svg --title "Dragon Curve"
  %(prog)s segments.txt --grid --figsize 10 8

Supported output formats:
  PNG, PDF, SVG, EPS, PS, TIFF, JPEG
        '''
    )
    
    parser.add_argument('input_file', 
                       help='Input file containing segment coordinates (x1 y1 x2 y2 per line)')
    
    parser.add_argument('--outfile', '-o',
                       help='Output file name (default: input_name.png). '
                            'File extension determines format.')
    
    parser.add_argument('--title', '-t',
                       help='Plot title (default: derived from filename)')
    
    parser.add_argument('--figsize', nargs=2, type=float, default=[8, 6],
                       metavar=('WIDTH', 'HEIGHT'),
                       help='Figure size in inches (default: 8 6)')
    
    parser.add_argument('--grid', action='store_true',
                       help='Show grid on plot')
    
    parser.add_argument('--dpi', type=int, default=300,
                       help='DPI for raster formats (default: 300)')
    
    parser.add_argument('--linewidth', '-lw', type=float, default=1.0,
                       help='Line width for plotting (default: 1.0)')
    
    parser.add_argument('--color', '-c', default='black',
                       help='Line color (default: black)')
    
    return parser.parse_args()

def read_segments(filename):
    """
    Read line segments from file.
    
    Args:
        filename: Path to input file
        
    Returns:
        list: List of tuples (x1, y1, x2, y2) for each segment
    """
    segments = []
    
    try:
        with open(filename, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue  # Skip empty lines and comments
                
                try:
                    coords = list(map(float, line.split()))
                    if len(coords) != 4:
                        print(f"Warning: Line {line_num} has {len(coords)} coordinates, expected 4. Skipping.")
                        continue
                    
                    x1, y1, x2, y2 = coords
                    segments.append((x1, y1, x2, y2))
                    
                except ValueError:
                    print(f"Warning: Could not parse line {line_num}: '{line}'. Skipping.")
                    continue
                    
    except FileNotFoundError:
        print(f"Error: Input file '{filename}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file '{filename}': {e}")
        sys.exit(1)
    
    if not segments:
        print(f"Error: No valid segments found in '{filename}'")
        sys.exit(1)
    
    print(f"Read {len(segments)} segments from '{filename}'")
    return segments

def plot_curve(segments, args):
    """
    Plot the curve segments.
    
    Args:
        segments: List of segment tuples (x1, y1, x2, y2)
        args: Parsed command line arguments
    """
    # Create figure and axis
    fig, ax = plt.subplots(figsize=args.figsize)
    
    # Plot each segment
    for x1, y1, x2, y2 in segments:
        ax.plot([x1, x2], [y1, y2], 
               color=args.color, 
               linewidth=args.linewidth,
               solid_capstyle='round')
    
    # Set title
    if args.title:
        title = args.title
    else:
        # Derive title from filename
        input_path = Path(args.input_file)
        title = input_path.stem.replace('_', ' ').title()
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Labels and formatting
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    
    # Grid
    if args.grid:
        ax.grid(True, alpha=0.3)
    
    # Equal aspect ratio for proper geometry
    ax.set_aspect('equal', adjustable='box')
    
    # Tight layout
    plt.tight_layout()
    
    return fig

def determine_output_filename(input_file, output_file):
    """
    Determine the output filename and validate format.
    
    Args:
        input_file: Input filename
        output_file: User-specified output filename or None
        
    Returns:
        str: Output filename
    """
    supported_formats = {'.png', '.pdf', '.svg', '.eps', '.ps', '.tiff', '.tif', '.jpg', '.jpeg'}
    
    if output_file is None:
        # Default: input_name.png
        input_path = Path(input_file)
        output_file = input_path.with_suffix('.png')
    else:
        output_path = Path(output_file)
        if output_path.suffix.lower() not in supported_formats:
            print(f"Warning: Unsupported format '{output_path.suffix}'. Using .png")
            output_file = output_path.with_suffix('.png')
    
    return str(output_file)

def save_plot(fig, output_file, dpi):
    """
    Save the plot to file.
    
    Args:
        fig: matplotlib figure
        output_file: Output filename
        dpi: DPI for raster formats
    """
    output_path = Path(output_file)
    
    try:
        # Create output directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save with appropriate settings
        if output_path.suffix.lower() in {'.png', '.jpg', '.jpeg', '.tiff', '.tif'}:
            fig.savefig(output_file, dpi=dpi, bbox_inches='tight')
        else:
            fig.savefig(output_file, bbox_inches='tight')
        
        print(f"Plot saved to '{output_file}'")
        
    except Exception as e:
        print(f"Error saving plot to '{output_file}': {e}")
        sys.exit(1)

def main():
    """Main function."""
    args = parse_arguments()
    
    # Read segments from input file
    segments = read_segments(args.input_file)
    
    # Determine output filename
    output_file = determine_output_filename(args.input_file, args.outfile)
    
    # Create plot
    fig = plot_curve(segments, args)
    
    # Save plot
    save_plot(fig, output_file, args.dpi)
    
    # Show statistics
    print(f"Plotted {len(segments)} line segments")
    
    # Close the figure to free memory
    plt.close(fig)

if __name__ == '__main__':
    main()
