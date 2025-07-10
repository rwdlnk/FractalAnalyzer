#!/usr/bin/env python3

# Test importing from optimized package
try:
    from fractal_analyzer.optimized import GridCacheManager
    print("✅ Can import GridCacheManager from package!")
    
    # Test basic functionality
    manager = GridCacheManager("./import_test_output")
    cache = manager.get_resolution_cache(200, "test.vtk")
    print(f"✅ GridCacheManager works: {cache['output_dir']}")
    
except ImportError as e:
    print(f"❌ Import failed: {e}")

print("Optimized package test complete!")
