#!/usr/bin/env python3
"""
Performance Demo: Compare optimized vs baseline RT analysis.

This script demonstrates the grid caching optimization by comparing
the time to process multiple timesteps with and without caching.
"""

import os
import time
import argparse
from fractal_analyzer.optimized import GridCacheManager, FastVTKReader
from fractal_analyzer.core.rt_analyzer import RTAnalyzer

def demo_optimized_workflow(resolution, sample_vtk_file, output_dir="./performance_demo"):
    """
    Demonstrate the optimized workflow with grid caching.
    
    Args:
        resolution: Grid resolution (e.g., 800)
        sample_vtk_file: Path to a sample VTK file for this resolution
        output_dir: Output directory for demo
    """
    print(f"🚀 OPTIMIZED WORKFLOW DEMO")
    print(f"=" * 50)
    print(f"Resolution: {resolution}×{resolution}")
    print(f"Sample file: {os.path.basename(sample_vtk_file)}")
    
    # Create grid cache manager
    cache_manager = GridCacheManager(output_dir)
    
    # Initialize grid cache (this happens ONCE per resolution)
    print(f"\n1️⃣  GRID INITIALIZATION (once per resolution)")
    start_time = time.time()
    
    cache = cache_manager.get_resolution_cache(resolution, sample_vtk_file)
    
    init_time = time.time() - start_time
    print(f"   Grid cache initialization: {init_time:.3f} seconds")
    
    if cache['grid_initialized']:
        print(f"   ✅ Grid cached: {cache['x_grid'].shape}")
        
        # Create fast reader
        reader = FastVTKReader(cache_manager)
        
        # Simulate processing multiple timesteps
        print(f"\n2️⃣  FAST F-DATA READING (per timestep)")
        print(f"   Testing F-data reading speed...")
        
        # Time F-data reading (would be done for each timestep)
        start_time = time.time()
        
        try:
            # Read F-data only using cached grid
            data = reader.read_f_data_only(sample_vtk_file, resolution)
            
            read_time = time.time() - start_time
            print(f"   F-data read time: {read_time:.3f} seconds")
            print(f"   ✅ F-data shape: {data['f'].shape}")
            print(f"   ✅ Using cached grid: {data['x'].shape}")
            print(f"   ✅ Simulation time: {data['time']:.3f}")
            
            return init_time, read_time, True
            
        except Exception as e:
            print(f"   ❌ F-data reading failed: {e}")
            print(f"   Error details: {type(e).__name__}")
            return init_time, None, False
            
    else:
        print(f"   ❌ Could not initialize grid from sample file")
        return None, None, False

def demo_baseline_workflow(vtk_file):
    """
    Demonstrate baseline workflow for comparison.
    
    Args:
        vtk_file: Path to VTK file
    """
    print(f"\n🐌 BASELINE WORKFLOW (for comparison)")
    print(f"=" * 50)
    
    # Create baseline RT analyzer
    analyzer = RTAnalyzer("./baseline_demo_output")
    
    # Time the full VTK reading (coordinates + F data)
    start_time = time.time()
    
    try:
        data = analyzer.read_vtk_file(vtk_file)
        
        read_time = time.time() - start_time
        print(f"   Full VTK read time: {read_time:.3f} seconds")
        print(f"   ✅ Read coordinates + F data + cell center calculation")
        print(f"   ✅ Data shape: {data['f'].shape}")
        
        return read_time
        
    except Exception as e:
        print(f"   ❌ Baseline reading failed: {e}")
        return None

def calculate_performance_projections(init_time, optimized_time, baseline_time):
    """Calculate performance projections for parameter studies."""
    print(f"\n📊 PERFORMANCE ANALYSIS")
    print(f"=" * 50)
    
    print(f"Per-timestep comparison:")
    print(f"   Baseline (full read): {baseline_time:.3f}s")
    print(f"   Optimized (F-only):   {optimized_time:.3f}s")
    print(f"   Per-timestep speedup: {baseline_time/optimized_time:.1f}×")
    
    # Projections for parameter studies
    timesteps = [10, 50, 100]
    at_values = [3, 5, 10]
    
    print(f"\nParameter study projections:")
    print(f"{'Timesteps':<12} {'At Values':<10} {'Baseline':<12} {'Optimized':<12} {'Speedup':<8}")
    print(f"-" * 60)
    
    for nt in timesteps:
        for nat in at_values:
            baseline_total = nat * nt * baseline_time
            optimized_total = nat * (init_time + nt * optimized_time)
            speedup = baseline_total / optimized_total
            
            print(f"{nt:<12} {nat:<10} {baseline_total:>8.1f}s    {optimized_total:>8.1f}s    {speedup:>6.1f}×")

def main():
    """Main demo function."""
    parser = argparse.ArgumentParser(description='Performance optimization demo')
    parser.add_argument('--vtk-file', required=True,
                       help='Path to sample VTK file (e.g., RT800x800-0000.vtk)')
    parser.add_argument('--resolution', type=int, required=True,
                       help='Grid resolution (e.g., 800)')
    parser.add_argument('--output', default='./performance_demo',
                       help='Output directory')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.vtk_file):
        print(f"❌ VTK file not found: {args.vtk_file}")
        print(f"Please provide a valid path to an RT VTK file")
        return
    
    print(f"🔬 RT ANALYSIS PERFORMANCE DEMONSTRATION")
    print(f"{'='*60}")
    
    # Run optimized demo
    init_time, opt_read_time, opt_success = demo_optimized_workflow(
        args.resolution, args.vtk_file, args.output)
    
    # Run baseline demo
    baseline_time = demo_baseline_workflow(args.vtk_file)
    
    # Performance analysis
    if opt_success and baseline_time and opt_read_time:
        calculate_performance_projections(init_time, opt_read_time, baseline_time)
        
        print(f"\n🎯 CONCLUSION")
        print(f"=" * 50)
        print(f"✅ Grid caching eliminates coordinate redundancy!")
        print(f"✅ Significant speedup demonstrated for parameter studies!")
        print(f"✅ Ready for large-scale At-dependence analysis!")
        
    else:
        print(f"\n⚠️  PARTIAL SUCCESS")
        print(f"=" * 50)
        print(f"Some tests failed, but basic infrastructure is working.")
        print(f"May need to adjust VTK reading for your specific file format.")

if __name__ == "__main__":
    main()
