#!/usr/bin/env python3

# Test updated package imports
try:
    from fractal_analyzer.optimized import GridCacheManager, FastVTKReader
    print('✅ Both modules imported successfully!')

    # Test basic functionality
    manager = GridCacheManager('./import_test')
    reader = FastVTKReader(manager)
    print('✅ Both classes instantiated successfully!')
    
except ImportError as e:
    print(f'❌ Import failed: {e}')
except Exception as e:
    print(f'❌ Error: {e}')

print('Package import test complete!')
