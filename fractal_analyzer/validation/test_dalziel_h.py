#!/usr/bin/env python3
"""
Test script for comparing Dalziel mixing methods
"""

import numpy as np
import sys
import os

# Import your RT analyzer (adjust the import path as needed)
from rt_analyzer import RTAnalyzer  # Adjust this import to match your file structure

def test_corrected_dalziel_method(vtk_filename):
    """
    Test the corrected Dalziel method on a VTK file.
    
    Args:
        vtk_filename: Path to VTK file to analyze
    """
    print(f"Testing CORRECTED Dalziel method on: {vtk_filename}")
    
    # Check if file exists
    if not os.path.exists(vtk_filename):
        print(f"ERROR: File {vtk_filename} not found!")
        return None
    
    # Initialize analyzer
    analyzer = RTAnalyzer()
    
    try:
        # Load VTK data
        print(f"Loading VTK file...")
        data = analyzer.read_vtk_file(vtk_filename)
        print(f"Successfully loaded VTK data")
        
        # Set parameters for your RT setup
        h0 = 0.25    # Initial interface position (middle of domain)
        H = 0.5      # Domain height (500 mm)
        L = 0.4      # Domain width (400 mm)
        
        print(f"Parameters: h0={h0}, H={H}, L={L}")
        
        # Run comparison of all three methods
        results = analyzer.compare_all_dalziel_methods(data, h0, H, L)
        
        return results
        
    except Exception as e:
        print(f"ERROR during analysis: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_dalziel_methods_on_file(vtk_filename):
    """
    Test both Dalziel methods on a specific VTK file.
    
    Args:
        vtk_filename: Path to VTK file to analyze
    
    Returns:
        dict: Results from both methods
    """
    print(f"Testing Dalziel methods on: {vtk_filename}")
    
    # Check if file exists
    if not os.path.exists(vtk_filename):
        print(f"ERROR: File {vtk_filename} not found!")
        return None
    
    # Initialize analyzer
    analyzer = RTAnalyzer()
    
    try:
        # Load VTK data
        print(f"Loading VTK file...")
        data = analyzer.read_vtk_file(vtk_filename)
        print(f"Successfully loaded VTK data")
        
        # Set parameters for your RT setup
        h0 = 0.25    # Initial interface position (middle of domain)
        H = 0.5      # Domain height (500 mm)
        L = 0.4      # Domain width (400 mm)
        
        print(f"Parameters: h0={h0}, H={H}, L={L}")
        
        # Run comparison of both methods
        results = analyzer.compare_dalziel_methods(data, h0, H, L)
        
        return results
        
    except Exception as e:
        print(f"ERROR during analysis: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main test function"""
    
    # Test on your specific file
    vtk_file = "/home/rod/Research/svofRuns/Dalziel_1999/640x800/RT640x800-10000.vtk"  # Adjust path as needed
    
    print("="*80)
    print("DALZIEL MIXING METHODS COMPARISON TEST")
    print("="*80)
    
    # Run the test
    #results = test_dalziel_methods_on_file(vtk_file)
    results = test_corrected_dalziel_method(vtk_file)
    if results is not None:
        print(f"\n" + "="*60)
        print(f"FINAL RESULTS SUMMARY")
        print(f"="*60)
        
        mixing = results['mixing_weighted']
        mass = results['mass_based']
        
        print(f"\n1. MIXING-WEIGHTED METHOD (Dalziel's actual method):")
        print(f"   h₁,₁/(H/2) = {mixing['h_11_normalized']:.4f}")
        print(f"   h₁,₀/(H/2) = {mixing['h_10_normalized']:.4f}")
        
        print(f"\n2. MASS-BASED METHOD (your original approach):")
        print(f"   h₁,₁/(H/2) = {mass['h_11_normalized']:.4f}")
        print(f"   h₁,₀/(H/2) = {mass['h_10_normalized']:.4f}")
        
        print(f"\n3. DALZIEL FIGURE 14 REFERENCE (τ≈2.0):")
        print(f"   h₁,₁/(H/2) ≈ 0.62 (expected)")
        print(f"   h₁,₀/(H/2) ≈ 0.37 (expected)")
        
        print(f"\n4. COMPARISON:")
        #if results['comparison']['ratio_h11'] is not None:
        #    print(f"   Mixing/Mass ratio h₁,₁: {results['comparison']['ratio_h11']:.3f}")
        #if results['comparison']['ratio_h10'] is not None:
        #    print(f"   Mixing/Mass ratio h₁,₀: {results['comparison']['ratio_h10']:.3f}")
        
        # Determine which method matches better
        mixing_error_11 = abs(mixing['h_11_normalized'] - 0.62)
        mixing_error_10 = abs(mixing['h_10_normalized'] - 0.37)
        mass_error_11 = abs(mass['h_11_normalized'] - 0.62)
        mass_error_10 = abs(mass['h_10_normalized'] - 0.37)
        
        print(f"\n5. AGREEMENT WITH DALZIEL FIGURE 14:")
        print(f"   Mixing-weighted error: h₁,₁={mixing_error_11:.3f}, h₁,₀={mixing_error_10:.3f}")
        print(f"   Mass-based error:      h₁,₁={mass_error_11:.3f}, h₁,₀={mass_error_10:.3f}")
        
        if (mixing_error_11 + mixing_error_10) < (mass_error_11 + mass_error_10):
            print(f"   → MIXING-WEIGHTED method agrees better with Dalziel!")
        else:
            print(f"   → MASS-BASED method agrees better with Dalziel!")
    
    else:
        print("Test failed - no results obtained")

# Additional utility function for batch testing
def test_multiple_files(vtk_files):
    """
    Test multiple VTK files and compare temporal evolution
    
    Args:
        vtk_files: List of VTK filenames to test
    """
    results_list = []
    
    for vtk_file in vtk_files:
        print(f"\n" + "-"*60)
        print(f"Processing: {vtk_file}")
        print(f"-"*60)
        
        result = test_dalziel_methods_on_file(vtk_file)
        if result is not None:
            # Extract time from filename (assuming format like RT160x200-10000.vtk)
            try:
                time_str = vtk_file.split('-')[-1].split('.')[0]
                time_step = int(time_str)
                result['time_step'] = time_step
                result['filename'] = vtk_file
                results_list.append(result)
            except:
                print(f"Warning: Could not extract time from filename {vtk_file}")
    
    return results_list

if __name__ == "__main__":
    # Run the main test
    main()
    
    # Optional: Test multiple time steps
    # Uncomment and modify as needed:
    
    # vtk_files = [
    #     "RT160x200-05000.vtk",
    #     "RT160x200-10000.vtk", 
    #     "RT160x200-15000.vtk"
    # ]
    # 
    # print(f"\n" + "="*80)
    # print(f"TEMPORAL EVOLUTION TEST")
    # print(f"="*80)
    # 
    # temporal_results = test_multiple_files(vtk_files)
    # 
    # if temporal_results:
    #     print(f"\nTemporal Evolution Summary:")
    #     print(f"Time Step    Mix h₁,₁    Mix h₁,₀    Mass h₁,₁   Mass h₁,₀")
    #     for result in temporal_results:
    #         mix = result['mixing_weighted']
    #         mass = result['mass_based']
    #         print(f"{result['time_step']:8d}    {mix['h_11_normalized']:8.4f}   {mix['h_10_normalized']:8.4f}    {mass['h_11_normalized']:8.4f}   {mass['h_10_normalized']:8.4f}")
