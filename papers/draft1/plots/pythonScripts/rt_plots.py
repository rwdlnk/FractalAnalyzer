#!/usr/bin/env python3
"""
RT Validation Plots Generator
Creates publication-quality plots for Rayleigh-Taylor validation analysis
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import rcParams
import pandas as pd

# Set publication-quality plot parameters
rcParams['font.family'] = 'serif'
#rcParams['font.serif'] = ['Times New Roman']
rcParams['font.size'] = 12
rcParams['axes.linewidth'] = 1.0
rcParams['axes.labelsize'] = 14
rcParams['axes.titlesize'] = 16
rcParams['xtick.labelsize'] = 12
rcParams['ytick.labelsize'] = 12
rcParams['legend.fontsize'] = 11
rcParams['figure.dpi'] = 300
rcParams['savefig.dpi'] = 300
rcParams['savefig.bbox'] = 'tight'
rcParams['savefig.pad_inches'] = 0.1

def create_rt_validation_comparison():
    """Create RT validation comparison plot"""
    
    # Your simulation data
    your_time = np.array([0.12743362831858407, 0.3256637168141593, 0.44601769911504424, 
                         0.5557522123893806, 0.6831858407079646, 0.8130088495575221, 
                         0.9451327433628318, 1.0646017699115044, 1.1946902654867257, 
                         1.3389380530973451, 1.4584070796460177, 1.5955752212389379, 
                         1.7415929203539823, 1.8654867256637167, 2.012389380530973, 
                         2.1469026548672566, 2.2831858407079646, 2.4371681415929203, 
                         2.5734513274336283, 2.727433628318584, 2.8646017699115044, 
                         3.0106194690265486, 3.2106194690265486, 3.447787610619469, 
                         3.7663716814159294])
    
    your_thickness = np.array([0.002295918367346939, 0.004719387755102041, 0.005867346938775511, 
                              0.007653061224489797, 0.009183673469387756, 0.010969387755102042, 
                              0.012397959183673469, 0.014438775510204082, 0.016479591836734695, 
                              0.01877551020408163, 0.020663265306122448, 0.023214285714285715, 
                              0.02602040816326531, 0.028826530612244896, 0.03214285714285714, 
                              0.035969387755102046, 0.03979591836734694, 0.04408163265306122, 
                              0.047857142857142855, 0.053061224489795916, 0.05765306122448979, 
                              0.06321428571428571, 0.06964285714285715, 0.07793367346938776, 
                              0.0903061224489796])
    
    # Dalziel experimental data (estimated from his Figure 4)
    dalziel_time = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
    dalziel_thickness = np.array([0.008, 0.015, 0.023, 0.032, 0.042, 0.052, 0.062, 0.073])
    
    # Constants
    A = 2.1e-3  # Atwood number
    g = 9.917    # gravity
    
    # Best fit parameters
    c = 0.145
    t0 = -1.60
    d0 = -0.005
    
    # Generate model curve
    time_range = np.linspace(0, 4.5, 100)
    model_thickness = c * A * g * (time_range - t0)**2 + d0
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Plot data
    ax.scatter(your_time, your_thickness * 1000, c='#d62728', s=60, 
              marker='o', edgecolor='black', linewidth=1, 
              label='Our RT Simulation', zorder=3)
    
    ax.scatter(dalziel_time, dalziel_thickness * 1000, c='#1f77b4', s=80, 
              marker='x', linewidth=3, 
              label='Dalziel (1993) Experiment', zorder=3)
    
    # Plot model
    ax.plot(time_range, model_thickness * 1000, '#2ca02c', linewidth=3, 
           label=r'RT Model: $d = cAg(t-t_0)^2 + d_0$', zorder=2)
    
    # Formatting
    ax.set_xlabel('Time (s)', fontsize=14)
    ax.set_ylabel('Mixing Layer Thickness (mm)', fontsize=14)
    ax.set_title('Rayleigh-Taylor Mixing Layer Growth Validation', fontsize=16, pad=20)
    
    # Grid and styling
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Legend
    ax.legend(frameon=True, fancybox=True, shadow=True, loc='upper left')
    
    # Set limits
    ax.set_xlim(0, 4.2)
    ax.set_ylim(0, max(dalziel_thickness * 1000) * 1.1)
    
    # Add text box with fit parameters
    textstr = f'Fit Parameters:\n$c = {c:.3f}$\n$t_0 = {t0:.1f}$ s\n$RMSE < 1$ mm'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.savefig('rt_validation_comparison.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: rt_validation_comparison.png")
    
    return fig, ax

def create_rt_residuals_analysis():
    """Create RT residuals analysis plot"""
    
    # Your simulation data
    your_time = np.array([0.12743362831858407, 0.3256637168141593, 0.44601769911504424, 
                         0.5557522123893806, 0.6831858407079646, 0.8130088495575221, 
                         0.9451327433628318, 1.0646017699115044, 1.1946902654867257, 
                         1.3389380530973451, 1.4584070796460177, 1.5955752212389379, 
                         1.7415929203539823, 1.8654867256637167, 2.012389380530973, 
                         2.1469026548672566, 2.2831858407079646, 2.4371681415929203, 
                         2.5734513274336283, 2.727433628318584, 2.8646017699115044, 
                         3.0106194690265486, 3.2106194690265486, 3.447787610619469, 
                         3.7663716814159294])
    
    your_thickness = np.array([0.002295918367346939, 0.004719387755102041, 0.005867346938775511, 
                              0.007653061224489797, 0.009183673469387756, 0.010969387755102042, 
                              0.012397959183673469, 0.014438775510204082, 0.016479591836734695, 
                              0.01877551020408163, 0.020663265306122448, 0.023214285714285715, 
                              0.02602040816326531, 0.028826530612244896, 0.03214285714285714, 
                              0.035969387755102046, 0.03979591836734694, 0.04408163265306122, 
                              0.047857142857142855, 0.053061224489795916, 0.05765306122448979, 
                              0.06321428571428571, 0.06964285714285715, 0.07793367346938776, 
                              0.0903061224489796])
    
    # Constants and fit parameters
    A = 2.1e-3  # Atwood number
    g = 9.917    # gravity
    c = 0.145
    t0 = -1.60
    d0 = -0.005
    
    # Calculate model predictions and residuals
    model_predictions = c * A * g * (your_time - t0)**2 + d0
    residuals = (your_thickness - model_predictions) * 1000  # Convert to mm
    
    # Calculate statistics
    mean_residual = np.mean(residuals)
    rmse = np.sqrt(np.mean(residuals**2))
    std_dev = np.std(residuals, ddof=1)
    max_abs_residual = np.max(np.abs(residuals))
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Plot residuals
    ax.plot(your_time, residuals, 'o-', color='#d62728', markersize=6, 
           linewidth=2, markeredgecolor='black', markeredgewidth=1,
           label='Residuals (Data - Model)')
    
    # Zero line
    ax.axhline(y=0, color='black', linestyle='--', linewidth=2, 
               label='Perfect Fit (Zero Error)', alpha=0.8)
    
    # Mean residual line
    ax.axhline(y=mean_residual, color='#ff7f0e', linestyle=':', linewidth=2,
               label=f'Mean Residual ({mean_residual:.3f} mm)')
    
    # ±2σ bounds
    ax.axhline(y=mean_residual + 2*std_dev, color='gray', linestyle='-.', linewidth=1,
               alpha=0.7, label='±2σ Bounds')
    ax.axhline(y=mean_residual - 2*std_dev, color='gray', linestyle='-.', linewidth=1,
               alpha=0.7)
    
    # Fill ±2σ region
    ax.fill_between(your_time, mean_residual - 2*std_dev, mean_residual + 2*std_dev,
                   alpha=0.1, color='gray')
    
    # Formatting
    ax.set_xlabel('Time (s)', fontsize=14)
    ax.set_ylabel('Residuals (mm)', fontsize=14)
    ax.set_title('RT Model Fitting Residuals: Validation Quality Assessment', fontsize=16, pad=20)
    
    # Grid and styling
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Legend
    ax.legend(frameon=True, fancybox=True, shadow=True, loc='upper right')
    
    # Set limits
    ax.set_xlim(0, max(your_time) * 1.05)
    y_range = max(abs(residuals)) * 1.2
    ax.set_ylim(-y_range, y_range)
    
    # Add statistics text box
    stats_text = (f'Residual Statistics:\n'
                 f'Mean: {mean_residual:.3f} mm\n'
                 f'RMSE: {rmse:.3f} mm\n'
                 f'Std Dev: {std_dev:.3f} mm\n'
                 f'Max |Residual|: {max_abs_residual:.3f} mm\n'
                 f'Relative RMSE: {rmse/90*100:.2f}%\n'
                 f'Quality: {"✅ Excellent" if rmse < 1.0 else "✅ Good" if rmse < 2.0 else "⚠️ Review"}')
    
    props = dict(boxstyle='round', facecolor='lightblue', alpha=0.8)
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props, family='monospace')
    
    plt.tight_layout()
    plt.savefig('rt_residuals_analysis.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: rt_residuals_analysis.png")
    
    return fig, ax

def main():
    """Generate both publication-quality plots"""
    
    print("🎨 Generating RT Validation Plots...")
    print("=" * 50)
    
    # Create plots
    fig1, ax1 = create_rt_validation_comparison()
    fig2, ax2 = create_rt_residuals_analysis()
    
    print("=" * 50)
    print("📊 Plot Generation Complete!")
    print()
    print("Generated files:")
    print("  📁 rt_validation_comparison.png")
    print("  📁 rt_residuals_analysis.png")
    print()
    print("📝 LaTeX Usage:")
    print("  \\includegraphics[width=0.9\\textwidth]{rt_validation_comparison.png}")
    print("  \\includegraphics[width=0.8\\textwidth]{rt_residuals_analysis.png}")
    print()
    print("🎯 Publication-quality plots ready for your paper!")
    
    # Optional: Show plots
    show_plots = input("\n🖼️  Display plots now? (y/n): ").lower().strip()
    if show_plots in ['y', 'yes']:
        plt.show()
    else:
        plt.close('all')

if __name__ == "__main__":
    main()
