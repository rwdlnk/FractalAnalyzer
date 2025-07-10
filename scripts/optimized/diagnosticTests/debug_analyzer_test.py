#!/usr/bin/env python3
"""
Debug test to diagnose the "string indices must be integers" error.
"""

import os
import sys
import numpy as np
sys.path.append('/home/rod/Research/FractalAnalyzer')

from fractal_analyzer.core.rt_analyzer import RTAnalyzer

def debug_analyze_vtk_file():
    """Debug what analyze_vtk_file returns."""
    
    # Test with a known file
    vtk_file = "/home/rod/Research/svofRuns/Dalziel/200x200/RT200x200-2999.vtk"
    
    if not os.path.exists(vtk_file):
        print(f"File not found: {vtk_file}")
        return
    
    print(f"🔍 Debugging analyze_vtk_file with: {vtk_file}")
    
    try:
        # Create analyzer
        analyzer = RTAnalyzer("./debug_test", use_grid_optimization=False, no_titles=False)
        
        print(f"📖 Step 1: Reading VTK file...")
        data = analyzer.read_vtk_file(vtk_file)
        print(f"   ✅ VTK read successful, data type: {type(data)}")
        print(f"   Keys: {list(data.keys()) if isinstance(data, dict) else 'Not a dict'}")
        
        print(f"🔬 Step 2: Extract interface...")
        contours = analyzer.extract_interface(data['f'], data['x'], data['y'])
        print(f"   ✅ Interface extracted, type: {type(contours)}")
        if isinstance(contours, dict):
            print(f"   Keys: {list(contours.keys())}")
        
        print(f"🧮 Step 3: Convert to segments...")
        segments = analyzer.convert_contours_to_segments(contours)
        print(f"   ✅ Segments converted, count: {len(segments)}")
        
        print(f"📊 Step 4: Calling analyze_vtk_file...")
        result = analyzer.analyze_vtk_file(
            vtk_file,
            mixing_method='dalziel',
            h0=0.5
        )
        
        print(f"   ✅ analyze_vtk_file completed!")
        print(f"   Result type: {type(result)}")
        
        if isinstance(result, dict):
            print(f"   Result keys: {list(result.keys())}")
            print(f"   fractal_dim: {result.get('fractal_dim', 'NOT FOUND')}")
        else:
            print(f"   ❌ Result is not a dictionary! Content: {result}")
        
        return result
        
    except Exception as e:
        print(f"❌ Error during debug: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    result = debug_analyze_vtk_file()
    print(f"\n🎯 Final result: {result}")
