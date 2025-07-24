import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Load Dalziel's experimental data (mm vs seconds)
try:
    fig4_data = pd.read_csv('Fig4datamm.csv')
    dalziel_time = fig4_data.iloc[:, 0].values  # seconds
    dalziel_thickness_mm = fig4_data.iloc[:, 1].values  # mm
    print(f"✅ Loaded Dalziel data: {len(dalziel_time)} points")
    print(f"   Time range: {dalziel_time.min():.1f} - {dalziel_time.max():.1f} s")
    print(f"   Thickness range: {dalziel_thickness_mm.min():.1f} - {dalziel_thickness_mm.max():.1f} mm")
except Exception as e:
    print(f"❌ Error loading Dalziel data: {e}")
    dalziel_time = None
    dalziel_thickness_mm = None

# Load your simulation data
try:
    sim_data = pd.read_csv('hybrid_analysis_temporal_evolution_t0.010.0_conrec.csv')
    
    # Extract time and h_total, convert h_total from m to mm
    sim_time_raw = sim_data['actual_time'].values  # seconds
    sim_time = sim_time_raw - 2.0  # Subtract 2 seconds for comparison
    sim_h_total_m = sim_data['h_total'].values  # meters
    sim_h_total_mm = sim_h_total_m * 1000  # convert to mm
    
    print(f"✅ Loaded simulation data: {len(sim_time)} points")
    print(f"   Original time range: {sim_time_raw.min():.1f} - {sim_time_raw.max():.1f} s")
    print(f"   Shifted time range: {sim_time.min():.1f} - {sim_time.max():.1f} s (t - 2.0s)")
    print(f"   h_total range: {sim_h_total_m.min():.6f} - {sim_h_total_m.max():.6f} m")
    print(f"   h_total range: {sim_h_total_mm.min():.1f} - {sim_h_total_mm.max():.1f} mm")
    
except Exception as e:
    print(f"❌ Error loading simulation data: {e}")
    sim_time = None
    sim_h_total_mm = None

# Create the comparison plot
fig, ax = plt.subplots(figsize=(12, 8))

# Plot Dalziel experimental data
if dalziel_time is not None and dalziel_thickness_mm is not None:
    ax.scatter(dalziel_time, dalziel_thickness_mm, 
              marker='o', s=60, color='red', 
              label="Dalziel 1993 Experiment", zorder=3)
    ax.plot(dalziel_time, dalziel_thickness_mm, 
           'r-', linewidth=2, alpha=0.7, zorder=2)

# Plot your simulation data
if sim_time is not None and sim_h_total_mm is not None:
    # Plot all simulation data
    ax.plot(sim_time, sim_h_total_mm, 
           'b-', linewidth=2, marker='s', markersize=4,
           label="Your Simulation (t - 2.0s)", zorder=2)
    
    # Highlight the comparison range (0-4s to match Dalziel)
    comparison_mask = sim_time <= 4.0
    if np.any(comparison_mask):
        ax.plot(sim_time[comparison_mask], sim_h_total_mm[comparison_mask], 
               'b-', linewidth=3, alpha=0.8, 
               label="Simulation (0-4s comparison range)")

# Formatting
ax.set_xlabel("Time (seconds)", fontsize=12)
ax.set_ylabel("Mixing Layer Thickness (mm)", fontsize=12)
ax.set_title("Dalziel Figure 4 Comparison (Simulation Time Shifted by -2s)\nMixing Layer Thickness vs Time", fontsize=14)

# Set axis limits
ax.set_xlim(0., 10.0)  # Show full simulation range
ax.set_ylim(0., None)  # Auto-scale y-axis

# Add grid and legend
ax.grid(True, alpha=0.3)
ax.legend(fontsize=11, loc='upper left')

# Add comparison statistics in the 0-4s range
if (dalziel_time is not None and dalziel_thickness_mm is not None and 
    sim_time is not None and sim_h_total_mm is not None):
    
    # Compare at t=4s (or max Dalziel time)
    t_compare = min(4.0, dalziel_time.max())
    
    dalziel_at_t = np.interp(t_compare, dalziel_time, dalziel_thickness_mm)
    
    sim_mask = sim_time <= t_compare
    if np.any(sim_mask):
        sim_at_t = np.interp(t_compare, sim_time[sim_mask], sim_h_total_mm[sim_mask])
        ratio = sim_at_t / dalziel_at_t
        
        # Add text box with comparison
        textstr = f'At t = {t_compare:.1f}s:\n'
        textstr += f'Dalziel: {dalziel_at_t:.1f} mm\n'
        textstr += f'Simulation (t-2s): {sim_at_t:.1f} mm\n'
        textstr += f'Ratio: {ratio:.2f}\n'
        textstr += f'Time shift: -2.0s'
        
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)

# Add vertical line at t=4s to show Dalziel data limit
if dalziel_time is not None:
    ax.axvline(dalziel_time.max(), color='red', linestyle='--', alpha=0.5,
               label=f'Dalziel data limit (t={dalziel_time.max():.1f}s)')

# Save the plot
plt.tight_layout()
plt.savefig('DalzielComp_shifted.png', dpi=300, bbox_inches='tight')
plt.savefig('DalzielComp_shifted.pdf', bbox_inches='tight')
print("\n📈 Plots saved as 'DalzielComp_shifted.png' and 'DalzielComp_shifted.pdf'")

plt.show()

# Print detailed comparison
print(f"\n📊 DETAILED COMPARISON:")
print(f"="*50)

if (dalziel_time is not None and sim_time is not None):
    comparison_times = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    
    print(f"{'Time (s)':<10} {'Dalziel (mm)':<15} {'Simulation (mm)':<18} {'Ratio':<8} {'Status'}")
    print("-" * 65)
    
    for t in comparison_times:
        if t <= dalziel_time.max() and t <= sim_time.max():
            dalziel_h = np.interp(t, dalziel_time, dalziel_thickness_mm)
            sim_h = np.interp(t, sim_time, sim_h_total_mm)
            ratio = sim_h / dalziel_h
            
            if 0.7 <= ratio <= 1.43:
                status = "✅ Excellent"
            elif 0.5 <= ratio <= 2.0:
                status = "✅ Good"
            elif 0.25 <= ratio <= 4.0:
                status = "⚠️ Fair"
            else:
                status = "❌ Poor"
            
            print(f"{t:<10.1f} {dalziel_h:<15.1f} {sim_h:<18.1f} {ratio:<8.2f} {status}")

# Growth rate analysis
if sim_time is not None and sim_h_total_mm is not None:
    print(f"\n📈 GROWTH RATE ANALYSIS:")
    
    # Linear growth rate in 1-3s range (avoiding early transients)
    growth_mask = (sim_time >= 1.0) & (sim_time <= 3.0)
    if np.sum(growth_mask) > 3:
        t_growth = sim_time[growth_mask]
        h_growth = sim_h_total_mm[growth_mask]
        
        coeffs = np.polyfit(t_growth, h_growth, 1)
        growth_rate = coeffs[0]  # mm/s
        
        print(f"  Simulation growth rate (1-3s): {growth_rate:.2f} mm/s")
    
    # Compare with Dalziel if available
    if dalziel_time is not None:
        dalziel_growth_mask = (dalziel_time >= 1.0) & (dalziel_time <= 3.0)
        if np.sum(dalziel_growth_mask) > 3:
            t_dalziel_growth = dalziel_time[dalziel_growth_mask]
            h_dalziel_growth = dalziel_thickness_mm[dalziel_growth_mask]
            
            dalziel_coeffs = np.polyfit(t_dalziel_growth, h_dalziel_growth, 1)
            dalziel_growth_rate = dalziel_coeffs[0]  # mm/s
            
            print(f"  Dalziel growth rate (1-3s): {dalziel_growth_rate:.2f} mm/s")
            print(f"  Growth rate ratio: {growth_rate/dalziel_growth_rate:.2f}")
