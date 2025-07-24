import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def complete_dalziel_jfm_comparison(sim_csv_file, shift_time_by=0):
    """
    Complete comparison with Dalziel JFM 1999 using correct parameters
    
    Parameters from your analysis:
    - H = 500 mm (domain height)
    - g = 9.81 m/s²
    - A = 2e-3 (Atwood number)
    - h_total = h10 + h11 (mixing thickness)
    """
    
    print("🎯 COMPLETE DALZIEL JFM 1999 COMPARISON")
    print("="*50)
    
    # Dalziel experimental parameters
    dalziel_params = {
        'H_mm': 500.0,          # Domain height in mm
        'H_m': 0.5,             # Domain height in meters  
        'g': 9.81,              # m/s²
        'A': 2e-3,              # Atwood number
    }
    
    print(f"Dalziel parameters:")
    print(f"  Domain height H: {dalziel_params['H_mm']:.0f} mm = {dalziel_params['H_m']:.1f} m")
    print(f"  Atwood number A: {dalziel_params['A']:.1e}")
    print(f"  Gravity g: {dalziel_params['g']:.2f} m/s²")
    
    # Calculate dimensionless time scale
    time_scale = np.sqrt(dalziel_params['A'] * dalziel_params['g'] / dalziel_params['H_m'])
    print(f"  Time scale √(Ag/H): {time_scale:.6f} s⁻¹")
    
    # Load Dalziel JFM data
    try:
        # Fig 14a: h11 data
        fig14a = pd.read_csv('Fig14a.csv')
        tau_14a = fig14a.iloc[:, 0].values
        h11_norm_14a = fig14a.iloc[:, 1].values
        
        # Fig 14b: h10 data  
        fig14b = pd.read_csv('Fig14b.csv')
        tau_14b = fig14b.iloc[:, 0].values
        h10_norm_14b = fig14b.iloc[:, 1].values
        
        print(f"\n✅ Loaded Dalziel JFM data:")
        print(f"  Fig 14a (h11): {len(fig14a)} points, τ ∈ [{tau_14a.min():.2f}, {tau_14a.max():.2f}]")
        print(f"  Fig 14b (h10): {len(fig14b)} points, τ ∈ [{tau_14b.min():.2f}, {tau_14b.max():.2f}]")
        
    except Exception as e:
        print(f"❌ Error loading Dalziel data: {e}")
        return None
    
    # Create common tau grid and interpolate
    tau_common = np.linspace(0, min(tau_14a.max(), tau_14b.max()), 100)
    h10_norm_interp = np.interp(tau_common, tau_14b, h10_norm_14b)
    h11_norm_interp = np.interp(tau_common, tau_14a, h11_norm_14a)
    
    # Calculate total mixing thickness (h10 + h11)
    h_total_norm_dalziel = h10_norm_interp + h11_norm_interp
    h_total_mm_dalziel = h_total_norm_dalziel * dalziel_params['H_mm']
    
    # Convert tau back to time for comparison
    time_dalziel = tau_common / time_scale
    
    print(f"\nDalziel JFM results:")
    print(f"  τ range: {tau_common.min():.2f} - {tau_common.max():.2f}")
    print(f"  Time range: {time_dalziel.min():.1f} - {time_dalziel.max():.1f} s")
    print(f"  h_total range: {h_total_mm_dalziel.min():.1f} - {h_total_mm_dalziel.max():.1f} mm")
    
    # Load your simulation data
    try:
        sim_data = pd.read_csv(sim_csv_file)
        
        # Apply time shift if requested
        sim_time_raw = sim_data['actual_time'].values
        sim_time = sim_time_raw + shift_time_by
        sim_h_total_m = sim_data['h_total'].values
        sim_h_total_mm = sim_h_total_m * 1000  # Convert m to mm
        
        print(f"\n✅ Loaded simulation data:")
        print(f"  Points: {len(sim_time)}")
        if shift_time_by != 0:
            print(f"  Original time: {sim_time_raw.min():.1f} - {sim_time_raw.max():.1f} s")
            print(f"  Shifted time: {sim_time.min():.1f} - {sim_time.max():.1f} s (shift: {shift_time_by:+.1f}s)")
        else:
            print(f"  Time range: {sim_time.min():.1f} - {sim_time.max():.1f} s")
        print(f"  h_total range: {sim_h_total_mm.min():.1f} - {sim_h_total_mm.max():.1f} mm")
        
    except Exception as e:
        print(f"❌ Error loading simulation data: {e}")
        return None
    
    # Create comprehensive comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Direct time comparison
    ax1 = axes[0, 0]
    ax1.plot(time_dalziel, h_total_mm_dalziel, 'r-', linewidth=3, 
             label='Dalziel JFM 1999 (h₁₀ + h₁₁)')
    ax1.plot(sim_time, sim_h_total_mm, 'b-', linewidth=2, marker='s', markersize=4,
             label=f'Your Simulation{" (shifted)" if shift_time_by != 0 else ""}')
    
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Total Mixing Thickness (mm)')
    ax1.set_title('Direct Time Comparison')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_xlim(0, 20)
    
    # Plot 2: Dimensionless comparison
    ax2 = axes[0, 1]
    ax2.plot(tau_common, h_total_norm_dalziel, 'r-', linewidth=3, 
             label='Dalziel JFM 1999')
    
    # Convert your simulation to dimensionless
    sim_tau = sim_time * time_scale
    sim_h_norm = sim_h_total_mm / dalziel_params['H_mm']
    ax2.plot(sim_tau, sim_h_norm, 'b-', linewidth=2, marker='s', markersize=4,
             label='Your Simulation')
    
    ax2.set_xlabel('τ = √(Ag/H)t')
    ax2.set_ylabel('h_total/H')
    ax2.set_title('Dimensionless Comparison')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Plot 3: Component comparison (if available)
    ax3 = axes[1, 0]
    ax3.plot(tau_common, h10_norm_interp, 'r-', linewidth=2, label='Dalziel h₁₀/H')
    ax3.plot(tau_common, h11_norm_interp, 'r--', linewidth=2, label='Dalziel h₁₁/H')
    ax3.plot(tau_common, h_total_norm_dalziel, 'r-', linewidth=3, label='Dalziel Total')
    
    # Add your simulation components if available
    if 'h_10' in sim_data.columns and 'h_11' in sim_data.columns:
        sim_h10_norm = sim_data['h_10'].values / dalziel_params['H_m']
        sim_h11_norm = sim_data['h_11'].values / dalziel_params['H_m']
        ax3.plot(sim_tau, sim_h10_norm, 'b-', linewidth=2, label='Your h₁₀/H')
        ax3.plot(sim_tau, sim_h11_norm, 'b--', linewidth=2, label='Your h₁₁/H')
    
    ax3.plot(sim_tau, sim_h_norm, 'b-', linewidth=3, label='Your Total')
    ax3.set_xlabel('τ = √(Ag/H)t')
    ax3.set_ylabel('h/H')
    ax3.set_title('Component Comparison')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Plot 4: Ratio analysis
    ax4 = axes[1, 1]
    
    # Calculate ratios at common time points
    comparison_tau = np.linspace(0.5, min(tau_common.max(), sim_tau.max()), 20)
    dalziel_interp = np.interp(comparison_tau, tau_common, h_total_norm_dalziel)
    sim_interp = np.interp(comparison_tau, sim_tau, sim_h_norm)
    ratios = sim_interp / dalziel_interp
    
    ax4.plot(comparison_tau, ratios, 'go-', linewidth=2, markersize=6)
    ax4.axhline(1.0, color='black', linestyle='--', alpha=0.5, label='Perfect match')
    ax4.axhline(0.65, color='red', linestyle=':', alpha=0.7, label='Typical sim/exp ratio')
    ax4.axhline(0.8, color='orange', linestyle=':', alpha=0.7, label='Good agreement')
    
    ax4.set_xlabel('τ = √(Ag/H)t')
    ax4.set_ylabel('Your Simulation / Dalziel JFM')
    ax4.set_title('Ratio Analysis')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    ax4.set_ylim(0, 2)
    
    plt.tight_layout()
    
    # Save the comprehensive plot
    output_name = f"dalziel_jfm_comparison{'_shifted' if shift_time_by != 0 else ''}.png"
    plt.savefig(output_name, dpi=300, bbox_inches='tight')
    print(f"\n📈 Comprehensive comparison saved: {output_name}")
    
    plt.show()
    
    # Detailed quantitative comparison
    print(f"\n📊 DETAILED QUANTITATIVE COMPARISON:")
    print(f"="*60)
    
    comparison_times = [2, 4, 6, 8, 10, 12, 15, 18, 20]
    print(f"{'Time (s)':<10} {'Dalziel (mm)':<15} {'Your Sim (mm)':<15} {'Ratio':<8} {'Status'}")
    print("-" * 70)
    
    agreements = []
    for t in comparison_times:
        if t <= time_dalziel.max() and t <= sim_time.max():
            dalziel_h = np.interp(t, time_dalziel, h_total_mm_dalziel)
            sim_h = np.interp(t, sim_time, sim_h_total_mm)
            ratio = sim_h / dalziel_h
            
            if 0.8 <= ratio <= 1.25:
                status = "✅ Excellent"
                agreements.append(True)
            elif 0.6 <= ratio <= 1.5:
                status = "✅ Good"
                agreements.append(True)
            elif 0.4 <= ratio <= 2.0:
                status = "⚠️ Fair"
                agreements.append(False)
            else:
                status = "❌ Poor"
                agreements.append(False)
            
            print(f"{t:<10.0f} {dalziel_h:<15.1f} {sim_h:<15.1f} {ratio:<8.2f} {status}")
    
    # Overall assessment
    if agreements:
        agreement_rate = sum(agreements) / len(agreements)
        print(f"\nOverall agreement: {agreement_rate:.1%}")
        
        average_ratio = np.mean([sim_interp[i]/dalziel_interp[i] for i in range(len(comparison_tau)) 
                                if dalziel_interp[i] > 0])
        
        print(f"Average ratio (sim/dalziel): {average_ratio:.2f}")
        
        if agreement_rate >= 0.8:
            print("🎉 EXCELLENT agreement with Dalziel JFM 1999!")
        elif agreement_rate >= 0.6:
            print("✅ Good agreement with Dalziel JFM 1999")
        else:
            print("⚠️ Moderate agreement - check parameters")
        
        # Interpretation
        if 0.6 <= average_ratio <= 0.8:
            print("📊 Ratio suggests this matches Dalziel's NUMERICAL results (expected!)")
        elif 0.8 <= average_ratio <= 1.2:
            print("📊 Ratio suggests this matches Dalziel's EXPERIMENTAL results (excellent!)")
    
    return {
        'agreement_rate': agreement_rate if 'agreement_rate' in locals() else 0,
        'average_ratio': average_ratio if 'average_ratio' in locals() else 0,
        'dalziel_time': time_dalziel,
        'dalziel_thickness': h_total_mm_dalziel,
        'sim_time': sim_time,
        'sim_thickness': sim_h_total_mm
    }

# Example usage:
if __name__ == "__main__":
    # Run comparison with your data
    result = complete_dalziel_jfm_comparison('hybrid_analysis_temporal_evolution_t0.010.0_conrec.csv')
    
    # Also try with time shift to see if it improves
    print("\n" + "="*70)
    print("COMPARISON WITH -2s TIME SHIFT:")
    result_shifted = complete_dalziel_jfm_comparison('hybrid_analysis_temporal_evolution_t0.010.0_conrec.csv', 
                                                    shift_time_by=-2)
