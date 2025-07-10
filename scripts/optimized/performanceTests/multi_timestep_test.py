#!/usr/bin/env python3
"""
Multi-Timestep Performance Test: Show grid caching benefits across many timesteps.

This script demonstrates where the optimization really shines - when processing
multiple timesteps from the same resolution simulation.
"""

import os
import time
import glob
import argparse
from fractal_analyzer.optimized import GridCacheManager, FastVTKReader
from fractal_analyzer.core.rt_analyzer import RTAnalyzer

def test_multi_timestep_performance(data_dir, resolution, max_timesteps=10):
    """
    Test performance across multiple timesteps to show grid caching benefits.
    
    Args:
        data_dir: Directory containing VTK files (e.g., ~/Research/svofRuns/Dalziel/200x200)
        resolution: Grid resolution (e.g., 200)
        max_timesteps: Maximum number of timesteps to test
    """
    
    print(f"🔬 MULTI-TIMESTEP PERFORMANCE TEST")
    print(f"{'='*60}")
    print(f"Data directory: {data_dir}")
    print(f"Resolution: {resolution}×{resolution}")
    
    # Find VTK files
    pattern = os.path.join(data_dir, f"RT{resolution}x{resolution}-*.vtk")
    vtk_files = sorted(glob.glob(pattern))[:max_timesteps]
    
    if not vtk_files:
        print(f"❌ No VTK files found matching: {pattern}")
        return
    
    print(f"Found {len(vtk_files)} timestep files")
    
    # Test 1: Optimized workflow with grid caching
    print(f"\n🚀 OPTIMIZED WORKFLOW (Grid Caching)")
    print(f"=" * 50)
    
    cache_manager = GridCacheManager(f"./multi_timestep_test_{resolution}")
    reader = FastVTKReader(cache_manager)
    
    opt_times = []
    total_start = time.time()
    
    for i, vtk_file in enumerate(vtk_files):
        print(f"  Processing timestep {i+1}/{len(vtk_files)}: {os.path.basename(vtk_file)}")
        
        start_time = time.time()
        
        try:
            # This uses cached grid after first file
            data = reader.read_f_data_only(vtk_file, resolution)
            process_time = time.time() - start_time
            opt_times.append(process_time)
            
            print(f"    Time: {process_time:.3f}s, F-shape: {data['f'].shape}, Sim-time: {data['time']:.3f}")
            
        except Exception as e:
            print(f"    ❌ Failed: {e}")
            opt_times.append(float('inf'))
    
    opt_total_time = time.time() - total_start
    
    # Test 2: Baseline workflow (re-reads coordinates every time)
    print(f"\n🐌 BASELINE WORKFLOW (No Caching)")
    print(f"=" * 50)
    
    analyzer = RTAnalyzer(f"./baseline_multi_test_{resolution}")
    
    baseline_times = []
    total_start = time.time()
    
    for i, vtk_file in enumerate(vtk_files):
        print(f"  Processing timestep {i+1}/{len(vtk_files)}: {os.path.basename(vtk_file)}")
        
        start_time = time.time()
        
        try:
            # This re-reads coordinates every time
            data = analyzer.read_vtk_file(vtk_file)
            process_time = time.time() - start_time
            baseline_times.append(process_time)
            
            print(f"    Time: {process_time:.3f}s, F-shape: {data['f'].shape}")
            
        except Exception as e:
            print(f"    ❌ Failed: {e}")
            baseline_times.append(float('inf'))
    
    baseline_total_time = time.time() - total_start
    
    # Performance Analysis
    print(f"\n📊 MULTI-TIMESTEP PERFORMANCE ANALYSIS")
    print(f"=" * 60)
    
    valid_opt = [t for t in opt_times if t != float('inf')]
    valid_baseline = [t for t in baseline_times if t != float('inf')]
    
    if valid_opt and valid_baseline:
        avg_opt = sum(valid_opt) / len(valid_opt)
        avg_baseline = sum(valid_baseline) / len(valid_baseline)
        
        print(f"Average per-timestep times:")
        print(f"  Optimized (cached):  {avg_opt:.3f}s")
        print(f"  Baseline (no cache): {avg_baseline:.3f}s")
        print(f"  Per-timestep speedup: {avg_baseline/avg_opt:.1f}×")
        
        print(f"\nTotal processing times:")
        print(f"  Optimized total:  {opt_total_time:.3f}s")
        print(f"  Baseline total:   {baseline_total_time:.3f}s")
        print(f"  Overall speedup:  {baseline_total_time/opt_total_time:.1f}×")
        
        # Show first vs later timesteps (grid caching benefit)
        if len(valid_opt) > 1 and len(valid_baseline) > 1:
            print(f"\nGrid caching effect:")
            print(f"  Optimized 1st timestep: {valid_opt[0]:.3f}s (includes grid init)")
            print(f"  Optimized later avg:    {sum(valid_opt[1:])/len(valid_opt[1:]):.3f}s (uses cached grid)")
            print(f"  Baseline consistency:   {avg_baseline:.3f}s (re-reads grid every time)")
        
        # Parameter study projections
        print(f"\nParameter study projections (for {len(vtk_files)} timesteps):")
        at_values = [3, 5, 10]
        
        print(f"{'At Values':<10} {'Baseline':<12} {'Optimized':<12} {'Speedup':<8}")
        print(f"-" * 45)
        
        for nat in at_values:
            baseline_proj = nat * baseline_total_time
            opt_proj = nat * opt_total_time  
            speedup = baseline_proj / opt_proj
            
            print(f"{nat:<10} {baseline_proj:>8.1f}s    {opt_proj:>8.1f}s    {speedup:>6.1f}×")
    
    else:
        print("❌ Performance analysis failed - insufficient valid data")
    
    print(f"\n🎯 CONCLUSION")
    print(f"=" * 50)
    if valid_opt and valid_baseline and len(vtk_files) > 1:
        total_speedup = baseline_total_time / opt_total_time
        if total_speedup > 1.1:
            print(f"✅ Grid caching provides {total_speedup:.1f}× speedup for multi-timestep analysis!")
            print(f"✅ Optimization scales well with number of timesteps!")
        else:
            print(f"⚠️  Minimal speedup at {resolution}×{resolution} resolution")
            print(f"   Grid caching benefits increase with larger resolutions")
    else:
        print(f"ℹ️  Test completed - check results above")

def main():
    """Main test function."""
    parser = argparse.ArgumentParser(description='Multi-timestep performance test')
    parser.add_argument('--data-dir', required=True,
                       help='Directory containing VTK files')
    parser.add_argument('--resolution', type=int, required=True,
                       help='Grid resolution')
    parser.add_argument('--max-timesteps', type=int, default=10,
                       help='Maximum timesteps to test (default: 10)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.data_dir):
        print(f"❌ Data directory not found: {args.data_dir}")
        return
    
    test_multi_timestep_performance(args.data_dir, args.resolution, args.max_timesteps)

if __name__ == "__main__":
    main()
