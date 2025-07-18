import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.collections import LineCollection

# Set up the figure with publication-quality settings
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'axes.linewidth': 1.2,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

def load_fractal_segments(data_string):
    """
    Load fractal segments from string data in format: x1 y1 x2 y2
    Returns array of line segments for plotting
    """
    lines = data_string.strip().split('\n')
    segments = []
    
    for line in lines:
        if line.strip():  # Skip empty lines
            try:
                coords = [float(x) for x in line.split()]
                if len(coords) == 4:
                    x1, y1, x2, y2 = coords
                    segments.append([(x1, y1), (x2, y2)])
            except ValueError:
                continue  # Skip malformed lines
    
    return segments

def normalize_segments(segments):
    """Normalize segments to fit in [0,1] x [0,1] square"""
    if not segments:
        return segments
    
    # Extract all coordinates
    all_x = []
    all_y = []
    for segment in segments:
        (x1, y1), (x2, y2) = segment
        all_x.extend([x1, x2])
        all_y.extend([y1, y2])
    
    # Find bounds
    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)
    
    # Normalize
    x_range = x_max - x_min if x_max != x_min else 1
    y_range = y_max - y_min if y_max != y_min else 1
    
    # Use the larger range to maintain aspect ratio
    max_range = max(x_range, y_range)
    
    normalized_segments = []
    for segment in segments:
        (x1, y1), (x2, y2) = segment
        norm_x1 = (x1 - x_min) / max_range
        norm_y1 = (y1 - y_min) / max_range
        norm_x2 = (x2 - x_min) / max_range
        norm_y2 = (y2 - y_min) / max_range
        normalized_segments.append([(norm_x1, norm_y1), (norm_x2, norm_y2)])
    
    return normalized_segments

def plot_fractal_segments_optimized(ax, segments, color='blue', linewidth=1.0, strategy='auto'):
    """
    Optimized plotting for large fractal datasets using LineCollection and smart subsampling
    
    Strategies:
    - 'auto': Automatically choose best strategy based on segment count
    - 'linecollection': Fast batch rendering using matplotlib LineCollection
    - 'subsample': Random subsampling for huge datasets (>500K segments)
    """
    if not segments:
        return
    
    num_segments = len(segments)
    
    # Auto-select strategy based on dataset size
    if strategy == 'auto':
        if num_segments < 100000:
            strategy = 'linecollection'
        else:
            strategy = 'subsample'
    
    print(f"  Plotting {num_segments} segments using '{strategy}' strategy...")
    
    if strategy == 'subsample':
        # For huge datasets: subsample to manageable size
        max_segments = 75000  # Good balance of visual quality vs speed
        if num_segments > max_segments:
            import random
            segments = random.sample(segments, max_segments)
            print(f"  Subsampled to {max_segments} segments for performance ({num_segments//max_segments}x reduction)")
    
    # Use LineCollection for fast batch rendering
    lines = []
    for segment in segments:
        (x1, y1), (x2, y2) = segment
        lines.append([(x1, y1), (x2, y2)])
    
    # Create LineCollection with rasterization for large datasets
    rasterized = len(lines) > 50000
    lc = plt.matplotlib.collections.LineCollection(lines, 
                                                  colors=color, 
                                                  linewidths=linewidth,
                                                  rasterized=rasterized)
    ax.add_collection(lc)
    
    if rasterized:
        print(f"  Applied rasterization for smooth rendering")

def load_fractal_from_file(filename):
    """Load fractal segments from file using your naming convention"""
    try:
        print(f"Loading {filename}...")
        with open(filename, 'r') as f:
            data = f.read()
        segments = load_fractal_segments(data)
        normalized = normalize_segments(segments)
        print(f"Loaded {len(segments)} segments from {filename}")
        return normalized
    except FileNotFoundError:
        print(f"Warning: File {filename} not found")
        return []
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return []

# Create figure with 5 subplots
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Theoretical Fractal Validation Suite\nFive Diverse Fractal Types for Algorithm Testing', 
             fontsize=16, fontweight='bold', y=0.95)

# Remove the last subplot (we only need 5)
axes[1, 2].remove()

# Flatten axes for easier indexing
ax_flat = [axes[0, 0], axes[0, 1], axes[0, 2], axes[1, 0], axes[1, 1]]

# Define fractal properties with optimized plotting strategies
fractals = [
    {
        'name': 'Koch Curve',
        'theoretical_D': 1.2619,
        'optimized_D': 1.289,
        'error': 2.2,
        'color': 'darkred',
        'description': 'Self-similar\ncoastline fractal',
        'max_iteration': 9,
        'segments_at_max': '4^9 = 262,144',
        'filename': 'koch_level9_segments.dat',
        'linewidth': 0.8,
        'strategy': 'linecollection'  # Medium dataset
    },
    {
        'name': 'Sierpinski Triangle', 
        'theoretical_D': 1.5850,
        'optimized_D': 1.642,
        'error': 3.6,
        'color': 'darkblue',
        'description': 'Self-similar\ntriangular fractal',
        'max_iteration': 9,
        'segments_at_max': '3^10 = 59,049',
        'filename': 'sierpinski_level9_segments.dat',
        'linewidth': 0.6,
        'strategy': 'linecollection'  # Medium dataset
    },
    {
        'name': 'Dragon Curve',
        'theoretical_D': 1.5236,
        'optimized_D': 1.533,
        'error': 0.6,
        'color': 'darkgreen',
        'description': 'Complex folded\nspace-filling curve',
        'max_iteration': 11,
        'segments_at_max': '2^11 = 2,048',
        'filename': 'dragon_level11_segments.dat',
        'linewidth': 1.2,
        'strategy': 'linecollection'  # Small dataset
    },
    {
        'name': 'Minkowski Sausage',
        'theoretical_D': 1.5000,
        'optimized_D': 1.548,
        'error': 3.2,
        'color': 'purple',
        'description': 'Exact D=1.5\nrectangular pattern',
        'max_iteration': 8,
        'segments_at_max': '8^8 = 16.7M',
        'filename': 'minkowski_level8_segments.dat',
        'linewidth': 0.4,
        'strategy': 'subsample'  # Huge dataset - MUST subsample!
    },
    {
        'name': 'Hilbert Curve',
        'theoretical_D': 2.0000,
        'optimized_D': 2.002,
        'error': 0.08,
        'color': 'red',
        'description': 'Space-filling\ncurve (D→2)',
        'max_iteration': 9,
        'segments_at_max': '4^9 = 262,144',
        'filename': 'hilbert_level9_segments.dat',
        'linewidth': 0.5,
        'strategy': 'linecollection'  # Medium dataset
    }
]

print("Loading fractal data files with performance optimizations...")
print("Expected files and strategies:")
for fractal in fractals:
    print(f"  - {fractal['filename']:<35} → {fractal['strategy']}")
print()

# Performance expectations
print("Performance optimizations applied:")
print("  • LineCollection: 10-100x faster than individual plot() calls")
print("  • Subsampling: Reduces 16.7M → 75K segments (220x reduction)")
print("  • Rasterization: Vector→bitmap for large datasets")
print("  • Expected total time: <2 minutes instead of 30+ minutes")
print()

# Load and plot each fractal
for i, (ax, fractal) in enumerate(zip(ax_flat, fractals)):
    # Load fractal data from your files
    segments = load_fractal_from_file(fractal['filename'])
    
    if segments:
        # Plot using optimized strategy
        plot_fractal_segments_optimized(ax, segments, 
                                       color=fractal['color'], 
                                       linewidth=fractal['linewidth'],
                                       strategy=fractal['strategy'])
        status_text = f"OK: Loaded {len(segments)} segments"
    else:
        # Show placeholder if file not found
        ax.text(0.5, 0.5, f'File not found:\n{fractal["filename"]}\n\nExpected format:\nx1 y1 x2 y2 per line', 
                transform=ax.transAxes, ha='center', va='center', fontsize=10,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcoral', alpha=0.7))
        status_text = "WARN: File not found"
    
    # Customize subplot
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_title(f'{fractal["name"]}\n{fractal["description"]}', 
                fontweight='bold', pad=15)
    
    # Add dimension and file information
    info_text = f'Theoretical D = {fractal["theoretical_D"]:.4f}\n'
    info_text += f'Measured D = {fractal["optimized_D"]:.3f}\n'
    info_text += f'Error = {fractal["error"]:.1f}%\n'
    info_text += f'Max Level = {fractal["max_iteration"]}\n'
    info_text += f'Segments: {fractal["segments_at_max"]}\n'
    info_text += status_text
    
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
            verticalalignment='top', fontsize=9,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))

# Add summary text
fig.text(0.68, 0.35, 
         'Validation Results Summary:\n\n'
         '• Five diverse fractal types\n'
         '• Iteration levels: 8-11 (convergence validated)\n'
         '• Segment counts: 2K - 16.7M elements\n'
         '• Theoretical dimensions: 1.26 - 2.00\n'
         '• Sliding window optimization\n'
         '• Errors: 0.08% - 3.6%\n'
         '• Excellent statistical fits (R² > 0.98)\n\n'
         'Algorithm Performance:\n'
         '• Space-filling curves: <0.1% error\n'
         '• Complex geometries: <1% error\n'
         '• Simple fractals: 2-4% error\n'
         '• Automatic window selection\n'
         '• Computational scaling well-characterized\n\n'
         'Data Source: Actual fractal segments\n'
         'from validation calculations',
         fontsize=11, verticalalignment='top',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.3))

plt.tight_layout()
plt.subplots_adjust(top=0.88, right=0.65)

# Save the figure
plt.savefig('five_fractals_validation_suite.png', dpi=300, bbox_inches='tight')
plt.savefig('five_fractals_validation_suite.pdf', bbox_inches='tight')

plt.show()

print("\n" + "="*60)
print("FRACTAL VALIDATION SUITE COMPLETE")
print("="*60)
print(f"\nOptimization strategies used:")
for fractal in fractals:
    exists = "OK" if os.path.exists(fractal['filename']) else "MISSING"
    print(f"{exists} {fractal['filename']:<35} -> {fractal['strategy']}")

print(f"\nPerformance improvements:")
print(f"  * Dragon (2K segments): LineCollection -> ~1 second")
print(f"  * Sierpinski (59K segments): LineCollection -> ~2 seconds") 
print(f"  * Koch/Hilbert (262K segments): LineCollection -> ~5 seconds")
print(f"  * Minkowski (16.7M segments): Subsample -> ~30 seconds")
print(f"  * Total estimated time: ~1-2 minutes (vs 30+ minutes)")

print(f"\nFigures saved:")
print(f"  - five_fractals_validation_suite.png (high-res)")
print(f"  - five_fractals_validation_suite.pdf (vector)")

print(f"\nVisual quality maintained:")
print(f"  * LineCollection: Identical visual quality to individual plots")
print(f"  * Subsampling: 75K segments still shows full fractal structure")
print(f"  * Rasterization: Smooth rendering for publication")

print(f"\nFile naming convention used:")
print(f"  Format: fractal-name_levelN_segments.dat")
print(f"  Examples:")
print(f"    koch_level9_segments.dat")
print(f"    dragon_level11_segments.dat")
print(f"    minkowski_level8_segments.dat")

print(f"\nTo create soft links from your actual files:")
print(f"  ln -s /path/to/your/koch/file.dat koch_level9_segments.dat")
print(f"  ln -s /path/to/your/dragon/file.dat dragon_level11_segments.dat")
print(f"  # etc...")
