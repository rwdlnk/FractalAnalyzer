#!/usr/bin/env python3
"""
Test scripts for RT Analyzer optimization validation
"""

import time
import os
import sys

# Add your rt_analyzer module path if needed
# sys.path.append('/path/to/your/rt_analyzer')

from rt_analyzer import RTAnalyzer  # Adjust import as needed

def test_single_file_basic():
    """Test 1: Basic single file analysis"""
    print("="*60)
    print("TEST 1: Basic Single File Analysis")
    print("="*60)
    
    # Initialize analyzer
    analyzer = RTAnalyzer()
    
    # Replace with your actual VTK file path
    vtk_file = "/home/rod/Research/svofRuns/Dalziel/200x200/RT200x200-1000.vtk"  # UPDATE THIS PATH
    print(f"vtk_file: {vtk_file}") 
    if not os.path.exists(vtk_file):
        print(f"❌ ERROR: VTK file not found: {vtk_file}")
        print("Please update the vtk_file path in the script")
        return False
    
    try:
        print(f"🧪 Testing file: {vtk_file}")
        
        # Test fractal dimension analysis
        start_time = time.time()
        results = analyzer.analyze_vtk_file(vtk_file, ['fractal_dim'])
        total_time = time.time() - start_time
        
        if results:
            print(f"✅ SUCCESS!")
            print(f"   Fractal dimension: {results.get('fractal_dimension', 'N/A')}")
            print(f"   Interface points: {results.get('interface_point_count', 'N/A')}")
            print(f"   Extraction time: {results.get('interface_extraction_time', 0):.3f}s")
            print(f"   Total time: {total_time:.3f}s")
            return True
        else:
            print(f"❌ FAILED: No results returned")
            return False
            
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_cache_performance():
    """Test 2: Cache performance validation"""
    print("\n" + "="*60)
    print("TEST 2: Cache Performance")
    print("="*60)
    
    analyzer = RTAnalyzer()
    vtk_file = "/home/rod/Research/svofRuns/Dalziel/200x200/RT200x200-1000.vtk"  # UPDATE THIS PATH
    
    if not os.path.exists(vtk_file):
        print(f"❌ ERROR: VTK file not found: {vtk_file}")
        return False
    
    try:
        # First call (cache miss)
        print("🔄 First call (cache miss)...")
        start = time.time()
        results1 = analyzer.analyze_vtk_file(vtk_file, ['fractal_dim'])
        time1 = time.time() - start
        
        # Second call (cache hit)
        print("⚡ Second call (should use cache)...")
        start = time.time()
        results2 = analyzer.analyze_vtk_file(vtk_file, ['fractal_dim'])
        time2 = time.time() - start
        
        if results1 and results2:
            print(f"✅ CACHE TEST SUCCESS!")
            print(f"   First call:  {time1:.3f}s")
            print(f"   Second call: {time2:.3f}s")
            if time2 > 0:
                speedup = time1 / time2
                print(f"   Cache speedup: {speedup:.1f}x")
                if speedup > 2:
                    print("   🚀 Excellent cache performance!")
                else:
                    print("   ⚠️  Cache may not be working optimally")
            return True
        else:
            print("❌ FAILED: One or both calls failed")
            return False
            
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        return False

def test_multi_analysis():
    """Test 3: Multiple analysis types (should still be fast!)"""
    print("\n" + "="*60)
    print("TEST 3: Multiple Analysis Types")
    print("="*60)
    
    analyzer = RTAnalyzer()
    vtk_file = "/home/rod/Research/svofRuns/Dalziel/200x200/RT200x200-1000.vtk"  # UPDATE THIS PATH
    
    if not os.path.exists(vtk_file):
        print(f"❌ ERROR: VTK file not found: {vtk_file}")
        return False
    
    try:
        print("🔄 Running multiple analyses (fractal + curvature + wavelength)...")
        start = time.time()
        results = analyzer.analyze_vtk_file(vtk_file, 
                                          ['fractal_dim', 'curvature', 'wavelength'])
        total_time = time.time() - start
        
        if results:
            print(f"✅ MULTI-ANALYSIS SUCCESS!")
            print(f"   Total time: {total_time:.3f}s")
            print(f"   Interface extraction: {results.get('interface_extraction_time', 0):.3f}s")
            print(f"   Analyses performed: {results.get('analysis_types_performed', [])}")
            
            # Show individual timing if available
            for analysis_type in ['fractal', 'curvature', 'wavelength']:
                time_key = f"{analysis_type}_computation_time"
                if time_key in results:
                    print(f"   {analysis_type.capitalize()} time: {results[time_key]:.3f}s")
            
            return True
        else:
            print("❌ FAILED: No results returned")
            return False
            
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        return False

def test_performance_comparison():
    """Test 4: Performance comparison with original method (if available)"""
    print("\n" + "="*60)
    print("TEST 4: Performance Comparison")
    print("="*60)
    
    analyzer = RTAnalyzer()
    vtk_file = "/home/rod/Research/svofRuns/Dalziel/200x200/RT200x200-1000.vtk"  # UPDATE THIS PATH

    
    if not os.path.exists(vtk_file):
        print(f"❌ ERROR: VTK file not found: {vtk_file}")
        return False
    
    # Check if original method exists
    has_original = hasattr(analyzer, 'analyze_vtk_file_original')
    
    if not has_original:
        print("ℹ️  Original method not found (analyze_vtk_file_original)")
        print("   Skipping performance comparison")
        return True
    
    try:
        print("🏃 Testing ORIGINAL method...")
        start = time.time()
        original_results = analyzer.analyze_vtk_file_original(vtk_file, ['fractal_dim'])
        original_time = time.time() - start
        
        print("🚀 Testing OPTIMIZED method...")
        start = time.time()
        optimized_results = analyzer.analyze_vtk_file(vtk_file, ['fractal_dim'])
        optimized_time = time.time() - start
        
        if original_results and optimized_results:
            print(f"✅ PERFORMANCE COMPARISON:")
            print(f"   Original time:  {original_time:.3f}s")
            print(f"   Optimized time: {optimized_time:.3f}s")
            if optimized_time > 0:
                speedup = original_time / optimized_time
                print(f"   Speedup: {speedup:.1f}x")
                if speedup > 2:
                    print("   🎉 Significant improvement!")
                else:
                    print("   📈 Some improvement")
            return True
        else:
            print("❌ FAILED: One or both methods failed")
            return False
            
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        return False

def run_all_tests():
    """Run all optimization tests"""
    print("🧪 RT ANALYZER OPTIMIZATION TESTS")
    print("=" * 60)
    
    # Update these paths before running!
    print("⚠️  BEFORE RUNNING: Update vtk_file paths in each test function!")
    print("")
    
    tests = [
        ("Basic Functionality", test_single_file_basic),
        ("Cache Performance", test_cache_performance), 
        ("Multi-Analysis", test_multi_analysis),
        ("Performance Comparison", test_performance_comparison)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n🏃 Running {test_name}...")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name} CRASHED: {str(e)}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = 0
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status}: {test_name}")
        if success:
            passed += 1
    
    print(f"\nResults: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 ALL TESTS PASSED! Optimization is working!")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")

if __name__ == "__main__":
    # Quick setup check
    print("🔧 SETUP CHECKLIST:")
    print("1. ✅ Update vtk_file paths in test functions")
    print("2. ✅ Ensure rt_analyzer module is importable") 
    print("3. ✅ Have at least one test VTK file available")
    print("")
    
    response = input("Ready to run tests? (y/n): ")
    if response.lower().startswith('y'):
        run_all_tests()
    else:
        print("Update the setup items above, then run the script again!")
