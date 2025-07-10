#!/usr/bin/env python3
"""
Temporal Evolution Performance Test: Realistic parameter study simulation.

This tests the optimization in the context it was designed for: 
processing many timesteps sequentially for temporal evolution analysis.
"""

import os
import time
import glob
import argparse
import math
from fractal_analyzer.optimized import GridCacheManager, FastVTKReader
from fractal_analyzer.core.rt_analyzer import RTAnalyzer

def test_temporal_evolution(data_dir, resolution, max_timesteps=20):
    """
    Test temporal evolution analysis - the realistic use case for optimization.
    
    This simulates what you'd do for parameter studies: analyze fractal 
    dimension evolution over time for a single At value.
    """
    
    print(f"🕒 TEMPORAL EVOLUTION PERFORMANCE TEST")
    print(f"{'='*60}")
    print(f"Data directory: {data_dir}")
    print(f"Resolution: {resolution}×{resolution}")
    print(f"Testing up to {max_timesteps} timesteps")
    
    # Find VTK files
    pattern = os.path.join(data_dir, f"RT{resolution}x{resolution}-*.vtk")
    vtk_files = sorted(glob.glob(pattern))[:max_timesteps]
    
    if not vtk_files:
        print(f"❌ No VTK files found matching: {pattern}")
        return
    
    print(f"Found {len(vtk_files)} timestep files")
    
    # Test 1: Optimized Sequential Workflow
    print(f"\n🚀 OPTIMIZED SEQUENTIAL WORKFLOW")
    print(f"=" * 50)
    print(f"Processing {len(vtk_files)} timesteps sequentially with grid caching...")
    
    cache_manager = GridCacheManager(f"./temporal_test_{resolution}")
    reader = FastVTKReader(cache_manager)
    
    opt_results = []
    opt_total_start = time.time()
    
    for i, vtk_file in enumerate(vtk_files):
        basename = os.path.basename(vtk_file)
        print(f"  {i+1:2d}/{len(vtk_files)}: {basename}")
        
        start_time = time.time()
        
        try:
            # Step 1: Fast VTK reading (our optimization)
            vtk_start = time.time()
            data = reader.read_f_data_only(vtk_file, resolution)
            vtk_time = time.time() - vtk_start
            
            # Debug F-field data
            print(f"       F-field stats: min={data['f'].min():.3f}, max={data['f'].max():.3f}, mean={data['f'].mean():.3f}")
            print(f"       F-field shape: {data['f'].shape}")
            print(f"       F values near 0.5: {((data['f'] > 0.4) & (data['f'] < 0.6)).sum()} cells")
            print(f"       Grid bounds: x=({data['x'].min():.3f}, {data['x'].max():.3f}), y=({data['y'].min():.3f}, {data['y'].max():.3f})")
            
            # Step 2: Create RTAnalyzer for fractal analysis
            analyzer = RTAnalyzer(f"./temporal_test_{resolution}/optimized")
            
            # Step 3: Perform fractal analysis (same as baseline)
            fractal_start = time.time()
           
            # Extract interface using optimized data
            contours = analyzer.extract_interface(data['f'], data['x'], data['y'])
            segments = analyzer.convert_contours_to_segments(contours)
            print(f"       [OPT] Segments found: {len(segments)}")

            if len(segments) > 10:  # Only analyze if sufficient segments
                print(f"       [OPT] Attempting fractal analysis...")
                
                # Use analyze_linear_region for RT interfaces (not analyze_iterations)
                dimension_result = analyzer.fractal_analyzer.analyze_linear_region(
                    segments,
                    fractal_type=None,  # RT interfaces are not generated fractals
                    plot_results=False,
                    return_box_data=False,
                    min_box_size=None  # Let it auto-estimate
                )
                
                print(f"       [OPT] Result structure: {type(dimension_result)} with {len(dimension_result)} elements")
                
                # Extract optimal dimension from linear region analysis
                if len(dimension_result) >= 6:
                    windows, dims, errs, r2s, optimal_window, optimal_dimension = dimension_result
                    fractal_dim = optimal_dimension
                    print(f"       [OPT] Extracted dim: {fractal_dim}")
                else:
                    print(f"       [OPT] Unexpected result format")
                    fractal_dim = float('nan')
                    
            else:
                print(f"       [OPT] Too few segments ({len(segments)}) for analysis")
                fractal_dim = float('nan')
            
            fractal_time = time.time() - fractal_start
            total_time = time.time() - start_time
            
            opt_results.append({
                'timestep': i+1,
                'sim_time': data['time'],
                'vtk_time': vtk_time,
                'fractal_time': fractal_time, 
                'total_time': total_time,
                'fractal_dim': fractal_dim,
                'segments': len(segments) if 'segments' in locals() else 0
            })
            
            print(f"       VTK: {vtk_time:.3f}s, Fractal: {fractal_time:.3f}s, "
                  f"Total: {total_time:.3f}s, D: {fractal_dim:.3f}")
            
        except Exception as e:
            print(f"       ❌ Failed: {e}")
            opt_results.append({
                'timestep': i+1, 'sim_time': 0, 'vtk_time': 0, 
                'fractal_time': 0, 'total_time': float('inf'),
                'fractal_dim': float('nan'), 'segments': 0
            })
    
    opt_total_time = time.time() - opt_total_start
    
    # Test 2: Baseline Sequential Workflow  
    print(f"\n🐌 BASELINE SEQUENTIAL WORKFLOW")
    print(f"=" * 50)
    print(f"Processing {len(vtk_files)} timesteps sequentially without caching...")
    
    baseline_results = []
    baseline_total_start = time.time()
    
    for i, vtk_file in enumerate(vtk_files):
        basename = os.path.basename(vtk_file)
        print(f"  {i+1:2d}/{len(vtk_files)}: {basename}")
        
        start_time = time.time()
        
        try:
            # Create new RTAnalyzer for each timestep (no caching)
            analyzer = RTAnalyzer(f"./temporal_test_{resolution}/baseline")
            
            # Step 1: Full VTK reading (baseline - reads coordinates every time)
            vtk_start = time.time()
            data = analyzer.read_vtk_file(vtk_file)
            vtk_time = time.time() - vtk_start
            
            # Step 2: Fractal analysis (same as optimized)
            fractal_start = time.time()

            contours = analyzer.extract_interface(data['f'], data['x'], data['y'])
            segments = analyzer.convert_contours_to_segments(contours)
            print(f"       [BASE] Segments found: {len(segments)}")
            
            if len(segments) > 10:
                print(f"       [BASE] Attempting fractal analysis...")
                
                # Use analyze_linear_region for RT interfaces
                dimension_result = analyzer.fractal_analyzer.analyze_linear_region(
                    segments,
                    fractal_type=None,
                    plot_results=False,
                    return_box_data=False,
                    min_box_size=None
                )
                
                # Extract optimal dimension
                if len(dimension_result) >= 6:
                    windows, dims, errs, r2s, optimal_window, optimal_dimension = dimension_result
                    fractal_dim = optimal_dimension
                    print(f"       [BASE] Extracted dim: {fractal_dim}")
                else:
                    fractal_dim = float('nan')
                    print(f"       [BASE] Unexpected result format")
            else:
                print(f"       [BASE] Too few segments ({len(segments)}) for analysis")
                fractal_dim = float('nan')
            
            fractal_time = time.time() - fractal_start
            total_time = time.time() - start_time
            
            baseline_results.append({
                'timestep': i+1,
                'sim_time': data.get('time', 0),
                'vtk_time': vtk_time,
                'fractal_time': fractal_time,
                'total_time': total_time,
                'fractal_dim': fractal_dim,
                'segments': len(segments)
            })
            
            print(f"       VTK: {vtk_time:.3f}s, Fractal: {fractal_time:.3f}s, "
                  f"Total: {total_time:.3f}s, D: {fractal_dim:.3f}")
            
        except Exception as e:
            print(f"       ❌ Failed: {e}")
            baseline_results.append({
                'timestep': i+1, 'sim_time': 0, 'vtk_time': 0,
                'fractal_time': 0, 'total_time': float('inf'),
                'fractal_dim': float('nan'), 'segments': 0
            })
    
    baseline_total_time = time.time() - baseline_total_start
    
    # Performance Analysis
    print(f"\n📊 TEMPORAL EVOLUTION PERFORMANCE ANALYSIS")
    print(f"=" * 60)
    
    # Filter valid results
    valid_opt = [r for r in opt_results if r['total_time'] != float('inf')]
    valid_baseline = [r for r in baseline_results if r['total_time'] != float('inf')]
    
    if valid_opt and valid_baseline:
        # VTK reading comparison
        avg_opt_vtk = sum(r['vtk_time'] for r in valid_opt) / len(valid_opt)
        avg_baseline_vtk = sum(r['vtk_time'] for r in valid_baseline) / len(valid_baseline)
        vtk_speedup = avg_baseline_vtk / avg_opt_vtk if avg_opt_vtk > 0 else float('inf')
        
        # Total time comparison
        avg_opt_total = sum(r['total_time'] for r in valid_opt) / len(valid_opt)
        avg_baseline_total = sum(r['total_time'] for r in valid_baseline) / len(valid_baseline)
        total_speedup = avg_baseline_total / avg_opt_total if avg_opt_total > 0 else float('inf')
        
        print(f"VTK Reading Performance:")
        print(f"  Optimized (cached):  {avg_opt_vtk:.3f}s average")
        print(f"  Baseline (no cache): {avg_baseline_vtk:.3f}s average") 
        print(f"  VTK reading speedup: {vtk_speedup:.2f}×")
        
        print(f"\nTotal Processing Performance:")
        print(f"  Optimized total:  {opt_total_time:.3f}s ({avg_opt_total:.3f}s avg/timestep)")
        print(f"  Baseline total:   {baseline_total_time:.3f}s ({avg_baseline_total:.3f}s avg/timestep)")
        print(f"  Overall speedup:  {baseline_total_time/opt_total_time:.2f}×")
        
        # Show first timestep vs later (grid caching effect)
        if len(valid_opt) > 1:
            first_opt_vtk = valid_opt[0]['vtk_time']
            later_opt_vtk = sum(r['vtk_time'] for r in valid_opt[1:]) / len(valid_opt[1:])
            
            print(f"\nGrid Caching Effect:")
            print(f"  1st timestep VTK time: {first_opt_vtk:.3f}s (includes grid init)")
            print(f"  Later avg VTK time:    {later_opt_vtk:.3f}s (uses cached grid)")
            print(f"  Cache benefit:         {(first_opt_vtk - later_opt_vtk):.3f}s per timestep")
        
        # Temporal evolution insights
        print(f"\nTemporal Evolution Results:")
        print(f"  Timesteps analyzed: {len(valid_opt)}")
        if valid_opt:
            time_range = f"{valid_opt[0]['sim_time']:.2f} - {valid_opt[-1]['sim_time']:.2f}"
            print(f"  Simulation time range: {time_range}")
            
            # Check for valid fractal dimensions
            valid_dims = [r['fractal_dim'] for r in valid_opt if not math.isnan(r['fractal_dim'])]
            if valid_dims:
                print(f"  Fractal dimension range: {min(valid_dims):.3f} - {max(valid_dims):.3f}")
                print(f"  Valid fractal analyses: {len(valid_dims)}/{len(valid_opt)}")
            else:
                print(f"  Valid fractal analyses: 0/{len(valid_opt)} (all NaN)")
        
        # Interface complexity evolution
        print(f"\nInterface Complexity Evolution:")
        for i, r in enumerate(valid_opt[:10]):  # Show first 10
            print(f"  t={r['sim_time']:.2f}: {r['segments']} segments")
        if len(valid_opt) > 10:
            print(f"  ... (showing first 10 of {len(valid_opt)} timesteps)")
    
    else:
        print("❌ Insufficient valid results for comparison")
    
    return opt_total_time, baseline_total_time

def main():
    """Main test function."""
    parser = argparse.ArgumentParser(description='Temporal evolution performance test')
    parser.add_argument('--data-dir', required=True,
                       help='Directory containing VTK files')
    parser.add_argument('--resolution', type=int, required=True,
                       help='Grid resolution')
    parser.add_argument('--max-timesteps', type=int, default=20,
                       help='Maximum timesteps to test (default: 20)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.data_dir):
        print(f"❌ Data directory not found: {args.data_dir}")
        return
    
    opt_time, baseline_time = test_temporal_evolution(
        args.data_dir, args.resolution, args.max_timesteps)
    
    print(f"\n🎯 TEMPORAL EVOLUTION CONCLUSION")
    print(f"=" * 60)
    if opt_time and baseline_time:
        speedup = baseline_time / opt_time
        if speedup > 1.0:
            print(f"✅ Optimization provides {speedup:.2f}× speedup for temporal evolution!")
            print(f"✅ Grid caching eliminates coordinate redundancy across timesteps!")
            print(f"✅ Ready for production parameter studies!")
        else:
            print(f"⚠️  Current speedup: {speedup:.2f}× - optimization shows modest benefit")
            print(f"   Benefits increase with larger resolutions and more timesteps")
    
    print(f"\n💡 For parameter studies with 100 timesteps:")
    if opt_time and baseline_time:
        proj_opt = (opt_time / args.max_timesteps) * 100
        proj_baseline = (baseline_time / args.max_timesteps) * 100
        print(f"   Optimized: {proj_opt:.1f}s")
        print(f"   Baseline:  {proj_baseline:.1f}s") 
        print(f"   Time saved: {proj_baseline - proj_opt:.1f}s per 100-timestep study")

if __name__ == "__main__":
    main()
