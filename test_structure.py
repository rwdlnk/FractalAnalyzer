#!/usr/bin/env python3

# Test that we can import from the new optimized package
try:
    import fractal_analyzer.optimized
    print("✅ fractal_analyzer.optimized package created successfully!")
except ImportError as e:
    print(f"❌ Import failed: {e}")

# Test existing imports still work
try:
    from fractal_analyzer.core.fractal_analyzer import FractalAnalyzer
    from fractal_analyzer.core.rt_analyzer import RTAnalyzer
    print("✅ Existing core modules still work!")
except ImportError as e:
    print(f"❌ Core import failed: {e}")

print("Package structure test complete!")
