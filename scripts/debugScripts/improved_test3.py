#!/usr/bin/env python3
"""
Test with improved box counting parameters
"""

from fractal_analyzer import FractalAnalyzer
import numpy as np
from scipy import stats

def test_with_better_parameters():
    print("TESTING WITH IMPROVED BOX COUNTING PARAMETERS")
    print("="*60)
    
    analyzer = FractalAnalyzer('koch')
    _, segments = analyzer.generate_fractal('koch', 5)  # Use level 5 for more detail
    
    print(f"Generated Koch level 5: {len(segments)} segments")
    
    # Get extent
    min_x = min(min(s[0][0], s[1][0]) for s in segments)
    max_x = max(max(s[0][0], s[1][0]) for s in segments)
    min_y = min(min(s[0][1], s[1][1]) for s in segments)
    max_y = max(max(s[0][1], s[1][1]) for s in segments)
    extent = max(max_x - min_x, max_y - min_y)
    max_box_size = extent / 2
    
    print(f"Extent: {extent:.6f}")
    print(f"Max box size: {max_box_size:.6f}")
    
    # Try different approaches
    approaches = [
        {
            'name': 'Auto-estimated (original)',
            'min_box_size': analyzer.estimate_min_box_size_from_segments(segments),
            'box_size_factor': 1.5
        },
        {
            'name': 'Conservative (larger min_box_size)',
            'min_box_size': analyzer.estimate_min_box_size_from_segments(segments) * 3,
            'box_size_factor': 1.5
        },
        {
            'name': 'Finer steps (smaller factor)',
            'min_box_size': analyzer.estimate_min_box_size_from_segments(segments) * 2,
            'box_size_factor': 1.3
        },
        {
            'name': 'Manual (based on segment length)',
            'min_box_size': 0.01,  # Roughly segment length
            'box_size_factor': 1.4
        }
    ]
    
    best_result = None
    best_r2 = 0
    
    for approach in approaches:
        print(f"\n{approach['name']}:")
        print("-" * 40)
        
        min_box_size = approach['min_box_size']
        box_size_factor = approach['box_size_factor']
        
        print(f"min_box_size: {min_box_size:.8f}")
        print(f"box_size_factor: {box_size_factor}")
        print(f"Expected scaling range: {max_box_size/min_box_size:.1f}")
        
        try:
            # Test box counting
            box_sizes, box_counts, _ = analyzer.box_counting_with_grid_optimization(
                segments, min_box_size, max_box_size, box_size_factor=box_size_factor)
            
            if len(box_sizes) < 5:
                print("❌ Too few box sizes generated")
                continue
                
            # Calculate dimension
            log_sizes = np.log(box_sizes)
            log_counts = np.log(box_counts)
            
            slope, intercept, r_value, p_value, std_err = stats.linregress(log_sizes, log_counts)
            dimension = -slope
            r2 = r_value**2
            
            print(f"Generated {len(box_sizes)} box sizes")
            print(f"Box counts: {min(box_counts)} to {max(box_counts)}")
            print(f"Dimension: {dimension:.6f}")
            print(f"R²: {r2:.6f}")
            print(f"Error from theoretical: {abs(dimension - 1.2619):.6f}")
            
            # Check for non-monotonic behavior (bad sign)
            box_count_diffs = np.diff(box_counts)
            non_monotonic = np.sum(box_count_diffs < 0)
            if non_monotonic > 0:
                print(f"⚠️  Warning: {non_monotonic} non-monotonic steps in box counts")
            
            # Quality assessment
            if r2 > 0.99 and abs(dimension - 1.2619) < 0.05:
                print("✅ EXCELLENT result")
                if r2 > best_r2:
                    best_result = {
                        'approach': approach['name'],
                        'dimension': dimension,
                        'r2': r2,
                        'min_box_size': min_box_size,
                        'box_size_factor': box_size_factor,
                        'box_sizes': box_sizes,
                        'box_counts': box_counts
                    }
                    best_r2 = r2
            elif r2 > 0.98 and abs(dimension - 1.2619) < 0.1:
                print("✅ GOOD result")
                if r2 > best_r2:
                    best_result = {
                        'approach': approach['name'],
                        'dimension': dimension,
                        'r2': r2,
                        'min_box_size': min_box_size,
                        'box_size_factor': box_size_factor,
                        'box_sizes': box_sizes,
                        'box_counts': box_counts
                    }
                    best_r2 = r2
            elif r2 > 0.95:
                print("⚠️  ACCEPTABLE result")
                if best_r2 < 0.95:  # Only update if we don't have better
                    best_result = {
                        'approach': approach['name'],
                        'dimension': dimension,
                        'r2': r2,
                        'min_box_size': min_box_size,
                        'box_size_factor': box_size_factor,
                        'box_sizes': box_sizes,
                        'box_counts': box_counts
                    }
                    best_r2 = r2
            else:
                print("❌ POOR result")
                
        except Exception as e:
            print(f"❌ Error: {str(e)}")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    if best_result:
        print(f"Best approach: {best_result['approach']}")
        print(f"Dimension: {best_result['dimension']:.6f}")
        print(f"R²: {best_result['r2']:.6f}")
        print(f"Parameters: min_box_size={best_result['min_box_size']:.8f}, "
              f"factor={best_result['box_size_factor']}")
        
        # Test the best result with visual output
        print(f"\nTesting best approach with visual output...")
        analyzer_visual = FractalAnalyzer('koch')
        _, segments_visual = analyzer_visual.generate_fractal('koch', 5)
        
        # Use analyze_linear_region with the best parameters
        results = analyzer_visual.analyze_linear_region(
            segments_visual, fractal_type='koch', plot_results=True, plot_separate=True,
            min_box_size=best_result['min_box_size']
        )
        
        print("✅ Generated plots with optimized parameters")
        print("Check: koch_box_counting_with_optimal_region.png")
        
    else:
        print("❌ No good results found with any approach")
        print("This suggests a fundamental issue with the box counting implementation")
        
    return best_result

if __name__ == "__main__":
    test_with_better_parameters()
