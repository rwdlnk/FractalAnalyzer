#!/usr/bin/env python3
"""
Debug Test 3 - Check what the actual residual values are
"""

from fractal_analyzer import FractalAnalyzer
import numpy as np
from scipy import stats

def debug_residuals():
    print("DEBUGGING TEST 3 RESIDUALS")
    print("="*50)
    
    analyzer = FractalAnalyzer('koch')
    _, segments = analyzer.generate_fractal('koch', 4)
    
    # Get box counting data
    min_x = min(min(s[0][0], s[1][0]) for s in segments)
    max_x = max(max(s[0][0], s[1][0]) for s in segments)
    min_y = min(min(s[0][1], s[1][1]) for s in segments)
    max_y = max(max(s[0][1], s[1][1]) for s in segments)
    extent = max(max_x - min_x, max_y - min_y)
    
    min_box_size = analyzer.estimate_min_box_size_from_segments(segments)
    max_box_size = extent / 2
    
    print(f"Box size range: {min_box_size:.8f} to {max_box_size:.8f}")
    
    box_sizes, box_counts, _ = analyzer.box_counting_with_grid_optimization(
        segments, min_box_size, max_box_size)
    
    print(f"Generated {len(box_sizes)} box sizes")
    print(f"Box counts range: {min(box_counts)} to {max(box_counts)}")
    
    # Calculate dimension manually
    log_sizes = np.log(box_sizes)
    log_counts = np.log(box_counts)
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_sizes, log_counts)
    manual_dimension = -slope
    
    print(f"\nRegression results:")
    print(f"  Slope: {slope:.8f}")
    print(f"  Intercept: {intercept:.8f}")
    print(f"  Dimension: {manual_dimension:.8f}")
    print(f"  R²: {r_value**2:.8f}")
    print(f"  Std error: {std_err:.8f}")
    
    # Check residuals in detail
    predicted_log_counts = intercept + slope * log_sizes
    residuals = log_counts - predicted_log_counts
    
    print(f"\nRESIDUAL ANALYSIS:")
    print(f"  Number of points: {len(residuals)}")
    print(f"  Max absolute residual: {np.max(np.abs(residuals)):.2e}")
    print(f"  RMS residual: {np.sqrt(np.mean(residuals**2)):.2e}")
    print(f"  Mean residual: {np.mean(residuals):.2e}")
    print(f"  Std residual: {np.std(residuals):.2e}")
    
    # Show individual residuals
    print(f"\nINDIVIDUAL RESIDUALS:")
    for i, (size, count, pred, res) in enumerate(zip(box_sizes, box_counts, np.exp(predicted_log_counts), residuals)):
        print(f"  Point {i}: size={size:.6f}, count={count:4d}, predicted={pred:.1f}, residual={res:.2e}")
    
    # Interpretation
    max_residual = np.max(np.abs(residuals))
    if max_residual < 1e-12:
        print(f"\n✅ PERFECT fit (residual < 1e-12)")
    elif max_residual < 1e-10:
        print(f"\n✅ EXCELLENT fit (residual < 1e-10)")
    elif max_residual < 1e-8:
        print(f"\n✅ VERY GOOD fit (residual < 1e-8)")
    elif max_residual < 1e-6:
        print(f"\n✅ GOOD fit (residual < 1e-6)")
    elif max_residual < 1e-4:
        print(f"\n⚠️  ACCEPTABLE fit (residual < 1e-4) - this is normal for real data")
    else:
        print(f"\n❌ POOR fit (residual > 1e-4) - may indicate a problem")
    
    # Check if R² is good (more important than tiny residuals)
    if r_value**2 > 0.999:
        print(f"✅ EXCELLENT R² = {r_value**2:.6f}")
    elif r_value**2 > 0.99:
        print(f"✅ GOOD R² = {r_value**2:.6f}")
    else:
        print(f"❌ POOR R² = {r_value**2:.6f}")
        
    return max_residual, r_value**2, manual_dimension

if __name__ == "__main__":
    debug_residuals()
