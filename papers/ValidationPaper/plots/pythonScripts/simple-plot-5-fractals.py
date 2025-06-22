import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.collections import LineCollection

# Set up publication-quality settings (no titles in plots per journal requirements)
plt.rcParams.update({
    'font.size': 10,
    'font.family': 'serif',
    'axes.linewidth': 1.0,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

def load_fractal_segments(filename):
    """Load fractal segments from file with format: x1 y1 x2 y2 per line"""
    try:
        print(f"Loading {filename}...")
        segments = []
        
        with open(filename, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:  # Skip empty lines
                    continue
                    
                try:
                    coords = [float(x) for x in line.split()]
                    if len(coords) == 4:
                        x1, y1, x2, y2 = coords
                        segments.append([(x1, y1), (x2, y2)])
                    else:
                        print(f"  Warning: Line {line_num} has {len(coords)} values (expected 4)")
                except ValueError as e:
                    print(f"  Warning: Could not parse line {line_num}: {e}")
                    continue
        
        print(f"  Successfully loaded {len(segments)} segments")
        return segments
        
    except FileNotFoundError:
        print(f"  Error: File {filename} not found")
        return []
    except Exception as e:
        print(f"  Error loading {filename}: {e}")
        return []

def normalize_segments(segments):
    """Normalize segments to fit in [0,1] x [0,1] square maintaining aspect ratio"""
    if not segments:
        return segments
    
    # Extract all coordinates
    all_coords = []
    for segment in segments:
        (x1, y1), (x2, y2) = segment
        all_coords.extend([x1, x2, y1, y2])
    
    # Find bounds
    x_coords = [segments[i][j][0] for i in range(len(segments)) for j in range(2)]
    y_coords = [segments[i][j][1] for i in range(len(segments)) for j in range(2)]
    
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    
    # Use uniform scaling to maintain aspect ratio
    x_range = x_max - x_min if x_max != x_min else 1
    y_range = y_max - y_min if y_max != y_min else 1
    max_range = max(x_range, y_range)
    
    # Normalize with centering
    normalized_segments = []
    for segment in segments:
        (x1, y1), (x2, y2) = segment
        norm_x1 = (x1 - x_min) / max_range
        norm_y1 = (y1 - y_min) / max_range
        norm_x2 = (x2 - x_min) / max_range
        norm_y2 = (y2 - y_min) / max_range
        normalized_segments.append([(norm_x1, norm_y1), (norm_x2, norm_y2)])
    
    return normalized_segments

def plot_fractal(ax, segments, color='blue', linewidth=1.0):
    """Plot fractal using LineCollection for optimal performance"""
    if not segments:
        ax.text(0.5, 0.5, 'No data\navailable', ha='center', va='center',
                transform=ax.transAxes, fontsize=12, 
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.7))
        return
    
    # Subsample if dataset is too large (>100k segments)
    if len(segments) > 100000:
        import random
        segments = random.sample(segments, 75000)
        print(f"  Subsampled to 75k segments for performance")
    
    # Create LineCollection for fast rendering
    lc = LineCollection(segments, colors=color, linewidths=linewidth,
                       rasterized=len(segments) > 50000)
    ax.add_collection(lc)

# Define fractal properties (updated with your actual results)
fractals = [
    {
        'name': 'Koch',
        'filename': 'koch_segments_level_9.dat',
        'color': '#8B0000',  # dark red
        'linewidth': 0.8
    },
    {
        'name': 'Sierpinski',
        'filename': 'sierpinski_segments_level_9.dat',
        'color': '#00008B',  # dark blue
        'linewidth': 0.6
    },
    {
        'name': 'Dragon',
        'filename': 'dragon_segments_level_11.dat',
        'color': '#006400',  # dark green
        'linewidth': 1.0
    },
    {
        'name': 'Minkowski',
        'filename': 'minkowski_segments_level_6.dat',  # Note: you said max level 6
        'color': '#000080',  # dark blue
        'linewidth': 0.8,
        'strategy': 'subsample_dense'  # Keep 200K-300K segments instead of 75K
    },
    {
        'name': 'Hilbert',
        'filename': 'hilbert_segments_level_9.dat',
        'color': '#B22222',  # fire brick
        'linewidth': 0.6
    }
]

# Create figure with clean layout
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Remove the empty subplot
axes[1, 2].remove()

# Flatten for easier access
ax_list = [axes[0, 0], axes[0, 1], axes[0, 2], axes[1, 0], axes[1, 1]]

print("Loading and plotting fractals...")
print("=" * 50)

# Load and plot each fractal
for ax, fractal in zip(ax_list, fractals):
    # Load segments
    segments = load_fractal_segments(fractal['filename'])
    
    if segments:
        # Normalize coordinates
        segments = normalize_segments(segments)
        
        # Plot
        plot_fractal(ax, segments, 
                    color=fractal['color'], 
                    linewidth=fractal['linewidth'])
    
    # Clean subplot formatting (journal style - no titles)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2, linewidth=0.5)
    
    # Minimal axis labels
    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([0, 0.5, 1])
    
    # Add subtle fractal label in corner
    ax.text(0.02, 0.98, fractal['name'], transform=ax.transAxes,
            verticalalignment='top', fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

# Clean overall layout
plt.tight_layout(pad=1.0)

# Save figures
print("\nSaving figures...")
plt.savefig('fractal_validation_suite.png', dpi=300, bbox_inches='tight')
plt.savefig('fractal_validation_suite.pdf', bbox_inches='tight')

print("Figures saved:")
print("  - fractal_validation_suite.png")
print("  - fractal_validation_suite.pdf")

plt.show()

print("\n" + "=" * 50)
print("FRACTAL PLOTTING COMPLETE")
print("=" * 50)

# File status summary
print("\nFile loading summary:")
for fractal in fractals:
    if os.path.exists(fractal['filename']):
        print(f"✓ {fractal['filename']}")
    else:
        print(f"✗ {fractal['filename']} (not found)")

print("\nExpected file format (4 numbers per line):")
print("x1 y1 x2 y2")
print("x1 y1 x2 y2")
print("...")
print("\nExample from your data:")
print("0.0 0.0 5.0805263425290864e-05 0.0")
print("5.0805263425290864e-05 0.0 7.620789513793629e-05 -4.399864877226229e-05")
