#!/usr/bin/env python3
"""
Comprehensive test script to verify all fractal analyzer fixes
"""

import numpy as np
import matplotlib.pyplot as plt
from fractal_analyzer import FractalAnalyzer
import os

def test_fit_line_consistency():
    """Test 1: Verify that fit lines pass through data points"""
    print("="*60)
    print("TEST 1: FIT LINE CONSISTENCY")
    print("="*60)
    
    analyzer = FractalAnalyzer('koch')
    
    # Test with a simple Koch curve
    _, segments = analyzer.generate_fractal('koch', 4)
    print(f"Generated Koch level 4: {len(segments)} segments")
    
    # Run analyze_linear_region
    results = analyzer.analyze_linear_region(
        segments, fractal_type='koch', plot_results=True, 
        return_box_data=True, plot_separate=True
    )
    
    if len(results) == 9:  # return_box_data=True
        windows, dims, errs, r2s, opt_window, opt_dim, box_sizes, box_counts, bbox = results
        
        print(f"Optimal dimension: {opt_dim:.6f}")
        print(f"Theoretical Koch: {analyzer.theoretical_dimensions['koch']:.6f}")
        print(f"Error: {abs(opt_dim - analyzer.theoretical_dimensions['koch']):.6f}")
        
        # Visual check: Look at the generated plots
        print("\nCHECK: Look at the generated plots:")
        print("  - koch_box_counting_with_optimal_region.png")
        print("  - Does the RED fit line pass through the BLUE data points?")
        print("  - If yes, fix is working!")
        
        return opt_dim, analyzer.theoretical_dimensions['koch']
    else:
        print("ERROR: Unexpected return format from analyze_linear_region")
        return None, None

def test_iteration_consistency():
    """Test 2: Check dimension consistency across iteration levels"""
    print("\n" + "="*60)
    print("TEST 2: ITERATION LEVEL CONSISTENCY")
    print("="*60)
    
    analyzer = FractalAnalyzer('koch')
    
    # Test multiple levels
    levels, dims, errs, r2s = analyzer.analyze_iterations(
        min_level=3, max_level=6, fractal_type='koch',
        no_plots=False  # Generate plots to check visually
    )
    
    if len(levels) > 0:
        theoretical = analyzer.theoretical_dimensions['koch']
        print(f"\nRESULTS SUMMARY:")
        print(f"Theoretical Koch dimension: {theoretical:.6f}")
        print("-" * 50)
        
        for i, (level, dim, err, r2) in enumerate(zip(levels, dims, errs, r2s)):
            error_pct = abs(dim - theoretical) / theoretical * 100
            print(f"Level {level}: {dim:.6f} ± {err:.6f} (R²={r2:.4f}) "
                  f"Error: {error_pct:.2f}%")
        
        # Check consistency
        dim_std = np.std(dims)
        dim_mean = np.mean(dims)
        
        print(f"\nCONSISTENCY METRICS:")
        print(f"Mean dimension: {dim_mean:.6f}")
        print(f"Standard deviation: {dim_std:.6f}")
        print(f"Coefficient of variation: {dim_std/dim_mean*100:.2f}%")
        
        # Good consistency check
        if dim_std < 0.01:  # Less than 1% variation
            print("✅ EXCELLENT consistency across levels")
        elif dim_std < 0.02:
            print("✅ GOOD consistency across levels") 
        else:
            print("❌ POOR consistency - may indicate remaining issues")
            
        # Check plots
        print(f"\nCHECK GENERATED PLOTS:")
        for level in levels:
            print(f"  - koch_level_{level}_dimension.png (fit line should pass through data)")
            
        return levels, dims, errs
    else:
        print("ERROR: No levels successfully analyzed")
        return [], [], []

def test_intercept_calculation():
    """Test 3: Verify intercept calculation is correct"""
    print("\n" + "="*60)
    print("TEST 3: INTERCEPT CALCULATION VERIFICATION")
    print("="*60)
    
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
    
    box_sizes, box_counts, _ = analyzer.box_counting_with_grid_optimization(
        segments, min_box_size, max_box_size)
    
    # Calculate dimension manually
    log_sizes = np.log(box_sizes)
    log_counts = np.log(box_counts)
    
    from scipy import stats
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_sizes, log_counts)
    manual_dimension = -slope
    
    print(f"Manual calculation:")
    print(f"  Slope: {slope:.6f}")
    print(f"  Intercept: {intercept:.6f}")
    print(f"  Dimension: {manual_dimension:.6f}")
    print(f"  R²: {r_value**2:.6f}")
    
    # Verify fit line passes through data
    predicted_log_counts = intercept + slope * log_sizes
    residuals = log_counts - predicted_log_counts
    max_residual = np.max(np.abs(residuals))
    
    print(f"\nFit quality check:")
    print(f"  Maximum residual: {max_residual:.8f}")
    print(f"  RMS residual: {np.sqrt(np.mean(residuals**2)):.8f}")
    
    if max_residual < 1e-10:
        print("✅ PERFECT fit - line passes exactly through data points")
    elif max_residual < 1e-6:
        print("✅ EXCELLENT fit - line passes very close to data points")
    else:
        print("❌ POOR fit - line doesn't pass close to data points")
        
    return manual_dimension, intercept, max_residual

def test_different_fractals():
    """Test 4: Quick test with different fractal types"""
    print("\n" + "="*60)
    print("TEST 4: DIFFERENT FRACTAL TYPES")
    print("="*60)
    
    fractals = ['koch', 'sierpinski', 'dragon']
    results = {}
    
    for fractal_type in fractals:
        print(f"\nTesting {fractal_type}...")
        analyzer = FractalAnalyzer(fractal_type)
        
        try:
            _, segments = analyzer.generate_fractal(fractal_type, 4)
            
            # Quick analysis
            windows, dims, errs, r2s, opt_window, opt_dim = analyzer.analyze_linear_region(
                segments, fractal_type=fractal_type, plot_results=False
            )
            
            theoretical = analyzer.theoretical_dimensions[fractal_type]
            error = abs(opt_dim - theoretical)
            error_pct = error / theoretical * 100
            
            results[fractal_type] = {
                'measured': opt_dim,
                'theoretical': theoretical,
                'error': error,
                'error_pct': error_pct
            }
            
            print(f"  Measured: {opt_dim:.6f}")
            print(f"  Theoretical: {theoretical:.6f}")
            print(f"  Error: {error_pct:.2f}%")
            
            if error_pct < 2.0:
                print(f"  ✅ GOOD accuracy")
            else:
                print(f"  ❌ POOR accuracy")
                
        except Exception as e:
            print(f"  ❌ ERROR: {str(e)}")
            results[fractal_type] = None
    
    return results

def main():
    """Run all tests"""
    print("COMPREHENSIVE FRACTAL ANALYZER TEST SUITE")
    print("This will test all the fixes you've implemented")
    print("Look for ✅ (good) and ❌ (needs work) indicators")
    
    # Test 1: Fit line consistency
    opt_dim, theoretical = test_fit_line_consistency()
    
    # Test 2: Iteration consistency  
    levels, dims, errs = test_iteration_consistency()
    
    # Test 3: Intercept calculation
    manual_dim, intercept, residual = test_intercept_calculation()
    
    # Test 4: Different fractals
    fractal_results = test_different_fractals()
    
    # Overall summary
    print("\n" + "="*60)
    print("OVERALL TEST SUMMARY")
    print("="*60)
    
    issues_found = []
    
    if opt_dim and theoretical:
        error_pct = abs(opt_dim - theoretical) / theoretical * 100
        if error_pct > 2.0:
            issues_found.append(f"High dimension error: {error_pct:.2f}%")
    
    if len(dims) > 1:
        dim_std = np.std(dims)
        if dim_std > 0.02:
            issues_found.append(f"Poor iteration consistency: std={dim_std:.4f}")
    
    if residual and residual > 1e-6:
        issues_found.append(f"Poor fit line quality: residual={residual:.2e}")
    
    if issues_found:
        print("❌ ISSUES FOUND:")
        for issue in issues_found:
            print(f"  - {issue}")
        print("\nYou may need to check your implementation again.")
    else:
        print("✅ ALL TESTS PASSED!")
        print("Your fixes appear to be working correctly.")
    
    print(f"\nGenerated plot files to check visually:")
    plot_files = [
        "koch_box_counting_with_optimal_region.png",
        "koch_sliding_window_analysis.png", 
        "koch_dimension_vs_level.png"
    ]
    
    for level in levels:
        plot_files.extend([
            f"koch_level_{level}_curve.png",
            f"koch_level_{level}_dimension.png"
        ])
    
    for filename in plot_files:
        if os.path.exists(filename):
            print(f"  ✅ {filename}")
        else:
            print(f"  ❌ {filename} (missing)")

if __name__ == "__main__":
    main()
