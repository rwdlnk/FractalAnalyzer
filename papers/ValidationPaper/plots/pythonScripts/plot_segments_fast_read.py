#!/usr/bin/env python3
"""
Fast Curve Segment Plotter - Optimized for large datasets with format auto-detection

Plots a curve defined by line segments from a data file.
Supports both formats:
1. Space-separated: x1 y1 x2 y2
2. Comma decimals: x1,y1 x2,y2

Usage:
    python plot_segments_fast.py input_file.txt
    python plot_segments_fast.py input_file.txt --outfile curve.pdf
    python plot_segments_fast.py input_file.txt --eps_plots --no_title
"""

import argparse
import matplotlib.pyplot as plt
import matplotlib.collections as mc
import numpy as np
from pathlib import Path
import sys
import time

def configure_matplotlib_for_eps():
    """Configure matplotlib for high-quality EPS output meeting AMC requirements."""
    import matplotlib
    
    # Set backend that supports EPS
    matplotlib.use('Agg')
    
    # Enhanced configuration for publication-quality EPS with better readability
    plt.rcParams.update({
        # FONT SETTINGS - Increased sizes for better readability
        'font.size': 14,              # Increased from 12
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif', 'Computer Modern Roman'],
        'text.usetex': False,
        
        # AXES AND LABELS - Larger for better visibility
        'axes.linewidth': 1.2,        # Slightly thicker
        'axes.labelsize': 16,         # Increased from 12
        'axes.titlesize': 18,         # Increased from 14
        'axes.labelweight': 'bold',   # Bold labels for better visibility
        
        # TICK LABELS - Larger and bolder
        'xtick.labelsize': 14,        # Increased from 10
        'ytick.labelsize': 14,        # Increased from 10
        'xtick.major.width': 1.2,
        'ytick.major.width': 1.2,
        'xtick.minor.width': 0.8,
        'ytick.minor.width': 0.8,
        
        # LEGEND - More readable
        'legend.fontsize': 12,        # Increased from 10
        'legend.frameon': True,
        'legend.fancybox': True,
        'legend.shadow': True,
        'legend.framealpha': 0.9,
        
        # FIGURE SETTINGS - Optimized for AMC
        'figure.figsize': [10, 8],    # Larger default size
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.format': 'eps',
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.2,    # Slightly more padding
        
        # LINES AND MARKERS - More visible
        'lines.linewidth': 2.0,       # Increased from 1.5
        'lines.markersize': 8,        # Increased from 6
        'lines.markeredgewidth': 1.0,
        
        # GRID - Subtle but visible
        'grid.linewidth': 0.8,        # Slightly thicker
        'grid.alpha': 0.6,            # Less transparent
        
        # MATH TEXT - Better rendering
        'mathtext.fontset': 'stix',
        'mathtext.default': 'regular'
    })
    
    print("Configured matplotlib for publication-quality EPS output with enhanced readability")


def detect_file_format(filename):
    """
    Detect the format of the segment file by examining the first few lines.
    
    Returns:
        str: 'comma' for comma decimal separators, 'space' for space-separated format
    """
    with open(filename, 'r') as f:
        for _ in range(min(10, sum(1 for _ in f))):  # Check first 10 lines or all lines if fewer
            f.seek(0)  # Reset to start
            line = f.readline().strip()
            if line:
                if ',' in line and len(line.split()) == 2:
                    # Format: x1,y1 x2,y2
                    return 'comma'
                elif len(line.split()) == 4:
                    # Format: x1 y1 x2 y2
                    return 'space'
    
    # Default assumption
    return 'space'

def read_segments_numpy_space_format(filename, subsample=None):
    """
    Original numpy method for space-separated format: x1 y1 x2 y2
    """
    start_time = time.time()
    
    # Use numpy to read the entire file at once - much faster!
    data = np.loadtxt(filename)
    
    if data.ndim == 1:
        data = data.reshape(1, -1)
    
    if data.shape[1] != 4:
        raise ValueError(f"Expected 4 columns, got {data.shape[1]}")
    
    read_time = time.time() - start_time
    print(f"Read {len(data)} segments in {read_time:.2f} seconds (numpy space format)")
    
    # Subsample if requested
    if subsample and len(data) > subsample:
        print(f"Subsampling to {subsample} segments...")
        indices = np.random.choice(len(data), subsample, replace=False)
        data = data[indices]
        print(f"Using {len(data)} segments after subsampling")
    
    return data

def read_segments_numpy_comma_format(filename, subsample=None):
    """
    Numpy-based approach for comma decimal format: x1,y1 x2,y2
    """
    start_time = time.time()
    
    # Read entire file as text
    with open(filename, 'r') as f:
        content = f.read()
    
    # Replace commas with spaces and split into tokens
    content = content.replace(',', ' ')
    tokens = content.split()
    
    # Convert to numpy array and reshape
    # Each line has 4 numbers: x1 y1 x2 y2
    data = np.array(tokens, dtype=float)
    
    if len(data) % 4 != 0:
        print(f"Warning: Data length {len(data)} is not divisible by 4")
        # Truncate to nearest multiple of 4
        data = data[:len(data) - (len(data) % 4)]
    
    # Reshape to (N, 4) where each row is [x1, y1, x2, y2]
    data = data.reshape(-1, 4)
    
    read_time = time.time() - start_time
    print(f"Read {len(data)} segments in {read_time:.2f} seconds (numpy comma format)")
    
    # Subsample if requested
    if subsample and len(data) > subsample:
        print(f"Subsampling to {subsample} segments...")
        indices = np.random.choice(len(data), subsample, replace=False)
        data = data[indices]
        print(f"Using {len(data)} segments after subsampling")
    
    return data

def read_segments_robust_fallback(filename, subsample=None):
    """
    Robust line-by-line parsing that handles both formats.
    """
    segments = []
    
    with open(filename, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:  # Skip empty lines
                continue
            
            try:
                parts = line.split()
                
                if len(parts) == 4:
                    # Format: x1 y1 x2 y2
                    x1, y1, x2, y2 = map(float, parts)
                    
                elif len(parts) == 2:
                    # Format: x1,y1 x2,y2
                    coord1 = parts[0].split(',')
                    coord2 = parts[1].split(',')
                    
                    if len(coord1) != 2 or len(coord2) != 2:
                        print(f"Warning: Line {line_num}, malformed coordinates: {line}")
                        continue
                        
                    x1, y1 = float(coord1[0]), float(coord1[1])
                    x2, y2 = float(coord2[0]), float(coord2[1])
                    
                else:
                    print(f"Warning: Line {line_num} has {len(parts)} parts, expected 2 or 4")
                    continue
                
                segments.append([x1, y1, x2, y2])
                
            except ValueError as ve:
                print(f"Warning: Line {line_num} parse error: {ve}")
                print(f"  Line content: '{line}'")
                continue
    
    if not segments:
        print("Error: No valid segments found in file")
        sys.exit(1)
    
    data = np.array(segments)
    
    # Subsample if requested
    if subsample and len(data) > subsample:
        print(f"Subsampling to {subsample} segments...")
        indices = np.random.choice(len(data), subsample, replace=False)
        data = data[indices]
        print(f"Using {len(data)} segments after subsampling")
    
    return data

def read_segments_fast(filename, subsample=None):
    """
    Read line segments from file - automatically detects format.
    
    Supports two formats:
    1. Space-separated: x1 y1 x2 y2
    2. Comma decimals: x1,y1 x2,y2
    
    Args:
        filename: Path to input file
        subsample: If specified, randomly subsample to this many segments
        
    Returns:
        numpy.ndarray: Array of shape (N, 4) containing segments as [x1, y1, x2, y2]
    """
    print(f"Reading segments from '{filename}'...")
    start_time = time.time()
    
    # Detect file format
    file_format = detect_file_format(filename)
    print(f"Detected format: {file_format}")
    
    try:
        if file_format == 'space':
            # Try original numpy method for space-separated format
            return read_segments_numpy_space_format(filename, subsample)
        else:
            # Use comma format method
            return read_segments_numpy_comma_format(filename, subsample)
            
    except Exception as e:
        print(f"Fast method failed ({e}), using robust line-by-line parsing...")
        return read_segments_robust_fallback(filename, subsample)

def parse_arguments():
    """Parse command line arguments with enhanced defaults for readability."""
    parser = argparse.ArgumentParser(
        description='Plot curve segments from a data file (optimized for large files with enhanced readability)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  %(prog)s segments.txt
  %(prog)s segments.txt --outfile curve.pdf
  %(prog)s segments.txt --outfile curve.svg --title "Dragon Curve"
  %(prog)s segments.txt --subsample 10000
  %(prog)s segments.txt --eps_plots --no_title

Supported formats:
  Space-separated: x1 y1 x2 y2
  Comma decimals:  x1,y1 x2,y2

Supported output formats:
  PNG, PDF, SVG, EPS, PS, TIFF, JPEG

Performance tips:
  - Use --subsample for very large files (>100k segments)
  - Use vector formats (PDF, SVG, EPS) for best quality
  - Use PNG with lower DPI for faster rendering
  - Use --eps_plots for publication-quality AMC-compliant figures

Readability enhancements:
  - Larger fonts and thicker lines for better visibility
  - Enhanced axis formatting and grid display
  - Optimized figure sizes for publication quality
        '''
    )
    
    parser.add_argument('input_file', 
                       help='Input file containing segment coordinates')
    
    parser.add_argument('--outfile', '-o',
                       help='Output file name (default: input_name.png, or .eps if --eps_plots)')
    
    parser.add_argument('--title', '-t',
                       help='Plot title (default: derived from filename)')
    
    parser.add_argument('--figsize', nargs=2, type=float, default=[12, 10],  # Enhanced default
                       metavar=('WIDTH', 'HEIGHT'),
                       help='Figure size in inches (default: 12 10 for better readability, or 10 8 for EPS)')
    
    parser.add_argument('--grid', action='store_true',
                       help='Show enhanced grid on plot')
    
    parser.add_argument('--dpi', type=int, default=300,  # Higher default DPI
                       help='DPI for raster formats (default: 300 for better quality)')
    
    parser.add_argument('--linewidth', '-lw', type=float, default=1.0,  # Thicker default
                       help='Line width for plotting (default: 1.0 for better visibility, 2.0 for EPS)')
    
    parser.add_argument('--color', '-c', default='black',
                       help='Line color (default: black)')
    
    parser.add_argument('--subsample', type=int,
                       help='Subsample to N segments for faster rendering (e.g., --subsample 50000)')
    
    parser.add_argument('--alpha', type=float, default=1.0,
                       help='Line transparency (0.0-1.0, default: 1.0)')
    
    parser.add_argument('--no_title', action='store_true',
                       help='Suppress plot title (recommended for journal figures)')
    
    parser.add_argument('--eps_plots', action='store_true',
                       help='Generate EPS format plots for publication quality (AMC journal requirements)')
    
    return parser.parse_args()

def get_plot_extension(eps_plots):
    """Get the appropriate file extension for plots."""
    return '.eps' if eps_plots else '.png'

def get_plot_dpi(eps_plots, default_dpi):
    """Get the appropriate DPI for plots."""
    return 300 if eps_plots else default_dpi

def save_plot_with_format(fig, filename_base, eps_plots, dpi):
    """
    Save plot with appropriate format and settings.
    
    Args:
        fig: matplotlib figure
        filename_base: Base filename without extension
        eps_plots: Whether to use EPS format
        dpi: DPI setting
    """
    extension = get_plot_extension(eps_plots)
    filename = filename_base + extension
    actual_dpi = get_plot_dpi(eps_plots, dpi)
    
    try:
        if eps_plots:
            # EPS-specific save settings for AMC compliance
            fig.savefig(filename, format='eps', dpi=actual_dpi, bbox_inches='tight', 
                       pad_inches=0.1, facecolor='white', edgecolor='none')
            print(f"Saved EPS plot: {filename}")
        else:
            # Standard save settings
            fig.savefig(filename, dpi=actual_dpi, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            print(f"Saved plot: {filename}")
            
    except Exception as e:
        print(f"Error saving plot {filename}: {str(e)}")
        # Try simplified save
        try:
            fig.savefig(filename, format='eps' if eps_plots else 'png', dpi=150)
            print(f"Saved simplified plot: {filename}")
        except Exception as e2:
            print(f"Failed to save plot completely: {str(e2)}")

def plot_curve_fast(segments, args):
    """
    Plot the curve segments using LineCollection for speed with enhanced readability.
    
    Args:
        segments: numpy array of shape (N, 4) with segments
        args: Parsed command line arguments
    """
    print("Creating enhanced plot...")
    start_time = time.time()
    
    # Enhanced parameters for better readability
    if args.eps_plots:
        figsize = [10, 8]  # Larger AMC size for better readability
        linewidth = 2.0 if args.linewidth == 0.5 else max(1.5, args.linewidth)  # Thicker default
    else:
        figsize = [12, 10] if args.figsize == [10, 10] else args.figsize  # Larger default
        linewidth = max(1.0, args.linewidth)  # Ensure minimum readable width
    
    # Create figure and axis with enhanced styling
    fig, ax = plt.subplots(figsize=figsize)
    
    # Enhance axis spines for better visibility
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    
    # Convert segments to line collection format
    lines = [[(x1, y1), (x2, y2)] for x1, y1, x2, y2 in segments]
    
    # Create LineCollection with enhanced visibility
    lc = mc.LineCollection(lines, 
                          linewidths=linewidth,
                          colors=args.color,
                          alpha=args.alpha,
                          capstyle='round',
                          joinstyle='round')  # Better line joins
    
    ax.add_collection(lc)
    
    # Set axis limits based on data with better margins
    all_x = np.concatenate([segments[:, 0], segments[:, 2]])
    all_y = np.concatenate([segments[:, 1], segments[:, 3]])
    
    margin = 0.03 * max(all_x.max() - all_x.min(), all_y.max() - all_y.min())  # Slightly larger margin
    ax.set_xlim(all_x.min() - margin, all_x.max() + margin)
    ax.set_ylim(all_y.min() - margin, all_y.max() + margin)
    
    # Enhanced title formatting
    if not args.no_title:
        if args.title:
            title = args.title
        else:
            input_path = Path(args.input_file)
            title = input_path.stem.replace('_', ' ').title()
            if len(segments) > 10000:
                title += f" ({len(segments):,} segments)"
        
        ax.set_title(title, fontsize=18, fontweight='bold', pad=15)
    
    # Enhanced labels and formatting
    ax.set_xlabel('X', fontsize=16, fontweight='bold', labelpad=10)
    ax.set_ylabel('Y', fontsize=16, fontweight='bold', labelpad=10)
    
    # Enhanced tick formatting
    ax.tick_params(axis='both', which='major', labelsize=14, width=1.2, length=6)
    ax.tick_params(axis='both', which='minor', width=0.8, length=4)
    
    # Enhanced grid
    if args.grid:
        ax.grid(True, alpha=0.6, zorder=0, linewidth=0.8)
    
    # Equal aspect ratio for proper geometry
    ax.set_aspect('equal', adjustable='box')
    
    # Tight layout with better spacing
    plt.tight_layout(pad=1.5)
    
    plot_time = time.time() - start_time
    print(f"Created enhanced plot in {plot_time:.2f} seconds")
    
    return fig

def determine_output_filename(input_file, output_file, eps_plots):
    """Determine the output filename and validate format."""
    supported_formats = {'.png', '.pdf', '.svg', '.eps', '.ps', '.tiff', '.tif', '.jpg', '.jpeg'}
    
    if output_file is None:
        input_path = Path(input_file)
        extension = get_plot_extension(eps_plots)
        output_file = input_path.with_suffix(extension)
    else:
        output_path = Path(output_file)
        if output_path.suffix.lower() not in supported_formats:
            extension = get_plot_extension(eps_plots)
            print(f"Warning: Unsupported format '{output_path.suffix}'. Using {extension}")
            output_file = output_path.with_suffix(extension)
    
    return str(output_file)

def save_plot(fig, output_file, dpi, eps_plots):
    """Save the plot to file."""
    print(f"Saving plot to '{output_file}'...")
    start_time = time.time()
    
    output_path = Path(output_file)
    actual_dpi = get_plot_dpi(eps_plots, dpi)
    
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if eps_plots or output_path.suffix.lower() == '.eps':
            # EPS-specific save settings
            fig.savefig(output_file, format='eps', dpi=actual_dpi, bbox_inches='tight',
                       pad_inches=0.1, facecolor='white', edgecolor='none')
        elif output_path.suffix.lower() in {'.png', '.jpg', '.jpeg', '.tiff', '.tif'}:
            fig.savefig(output_file, dpi=actual_dpi, bbox_inches='tight')
        else:
            fig.savefig(output_file, bbox_inches='tight')
        
        save_time = time.time() - start_time
        format_info = f" (EPS format, {actual_dpi} DPI)" if eps_plots else f" ({actual_dpi} DPI)"
        print(f"Plot saved in {save_time:.2f} seconds{format_info}")
        
    except Exception as e:
        print(f"Error saving plot to '{output_file}': {e}")
        sys.exit(1)

def main():
    """Main function with enhanced readability features."""
    total_start = time.time()
    
    args = parse_arguments()
    
    # Configure matplotlib for EPS if requested
    if args.eps_plots:
        configure_matplotlib_for_eps()
        print("EPS plots: ENABLED (Publication quality for AMC journal)")
        print("Enhanced readability features active for publication")
    else:
        print("EPS plots: DISABLED (Using enhanced PNG format)")
        print("Enhanced readability features active for better visualization")
    
    # Read segments from input file
    segments = read_segments_fast(args.input_file, args.subsample)
    
    # Enhanced performance recommendations
    if len(segments) > 100000:
        print(f"\nPerformance note: {len(segments):,} segments is quite large.")
        if not args.eps_plots:
            print("Consider using:")
            print("  --subsample 50000     (for faster rendering)")
            print("  --linewidth 0.8       (thinner lines, still readable)")
            print("  --dpi 200             (balanced quality/speed)")
            print("  --outfile plot.pdf    (vector format)")
            print("  --eps_plots           (publication quality)")
        else:
            print("EPS mode active - optimized for publication quality with enhanced readability")
        print()
    
    # Determine output filename
    output_file = determine_output_filename(args.input_file, args.outfile, args.eps_plots)
    
    # Create enhanced plot
    fig = plot_curve_fast(segments, args)
    
    # Save plot
    save_plot(fig, output_file, args.dpi, args.eps_plots)
    
    total_time = time.time() - total_start
    print(f"\nTotal time: {total_time:.2f} seconds")
    print(f"Plotted {len(segments):,} line segments with enhanced readability")
    
    if args.eps_plots:
        print("\n" + "="*60)
        print("PUBLICATION-QUALITY EPS PLOT GENERATED")
        print("="*60)
        print("• High resolution (300 DPI)")
        print("• Vector format for scalability")
        print("• Enhanced fonts and line weights for readability")
        print("• AMC journal compliant formatting")
        print("• Ready for academic publication")
    else:
        print("\n" + "="*50)
        print("ENHANCED READABILITY PLOT GENERATED")
        print("="*50)
        print("• Larger fonts and thicker lines")
        print("• Better axis formatting")
        print("• Optimized for clear visualization")
    
    plt.close(fig)

# Test function to verify the reading works
def test_reader():
    """Test the segment reader with sample data."""
    # Create test data for both formats
    test_content_space = """0.7862022 0.4724715 0.7867778 0.4721812
0.7867778 0.4721812 0.7868809 0.4714294
0.7868809 0.4714294 0.7868272 0.4706743"""
    
    test_content_comma = """0.7862022,0.4724715 0.7867778,0.4721812
0.7867778,0.4721812 0.7868809,0.4714294
0.7868809,0.4714294 0.7868272,0.4706743"""
    
    # Test space format
    with open('test_segments_space.dat', 'w') as f:
        f.write(test_content_space)
    
    # Test comma format
    with open('test_segments_comma.dat', 'w') as f:
        f.write(test_content_comma)
    
    # Test reading both
    print("Testing space format:")
    segments_space = read_segments_fast('test_segments_space.dat')
    print(f"Space format shape: {segments_space.shape}")
    print("First segment:", segments_space[0])
    
    print("\nTesting comma format:")
    segments_comma = read_segments_fast('test_segments_comma.dat')
    print(f"Comma format shape: {segments_comma.shape}")
    print("First segment:", segments_comma[0])
    
    # Verify they're the same
    if np.allclose(segments_space, segments_comma):
        print("\n✓ Both formats produce identical results!")
    else:
        print("\n✗ Formats produce different results!")
    
    # Clean up
    import os
    os.remove('test_segments_space.dat')
    os.remove('test_segments_comma.dat')

if __name__ == '__main__':
    if len(sys.argv) == 2 and sys.argv[1] == '--test':
        test_reader()
    else:
        main()
