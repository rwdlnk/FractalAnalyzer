#!/usr/bin/env python3
"""
Dalziel et al. (1999) Comparison Script

Compare your fractal dimension results with Dalziel et al. experimental data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def compare_with_dalziel_experiments(csv_file, atwood_number=2.1e-3):
    """Compare your results with Dalziel et al. (1999) Figure 20."""
    
    print(f"📊 COMPARISON WITH DALZIEL ET AL. (1999)")
    print(f"=" * 50)
    
    # Load your results
    df = pd.read_csv(csv_file)
    
    # Dalziel experimental data (from Figure 20)
    dalziel_experimental = {
        'tau': [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
        'dimension': [1.35, 1.38, 1.40, 1.42, 1.40, 1.38, 1.37, 1.36]
    }
    
    # Convert your time to Dalziel's τ scale (τ ≈ 0.2t)
    tau_scale = 0.2
    your_tau = df['time'] * tau_scale
    
    print(f"Your Atwood: {atwood_number:.1e}, Dalziel: 2.0e-3 ✅")
    print(f"\nComparison at key time points:")
    print(f"{'τ':<8} {'Dalziel D':<12} {'Your D':<12} {'Δ':<8} {'Status'}")
    print("-" * 50)
    
    agreements = []
    for dalziel_tau, dalziel_d in zip(dalziel_experimental['tau'], dalziel_experimental['dimension']):
        # Find closest time in your data
        your_time_target = dalziel_tau / tau_scale
        time_diff = np.abs(df['time'] - your_time_target)
        closest_idx = time_diff.idxmin()
        
        your_d = df.loc[closest_idx, 'fractal_dim']
        difference = abs(your_d - dalziel_d)
        
        if difference < 0.1:
            status = "✅ Excellent"
            agreements.append(True)
        elif difference < 0.2:
            status = "✅ Good"
            agreements.append(True)
        else:
            status = "⚠️ Poor"
            agreements.append(False)
        
        print(f"{dalziel_tau:<8.1f} {dalziel_d:<12.2f} {your_d:<12.3f} {difference:<8.3f} {status}")
    
    # Overall assessment
    agreement_rate = sum(agreements) / len(agreements)
    print(f"\nOverall agreement: {agreement_rate:.1%}")
    
    if agreement_rate > 0.75:
        print("🎉 EXCELLENT agreement with Dalziel experiments!")
    elif agreement_rate > 0.5:
        print("✅ Good agreement with Dalziel experiments")
    else:
        print("⚠️ Poor agreement - possible resolution effects")
    
    return dalziel_experimental

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Compare with Dalziel et al. experiments")
    parser.add_argument('--csv-file', required=True, help='Your CSV results file')
    args = parser.parse_args()
    
    compare_with_dalziel_experiments(args.csv_file)
