#!/usr/bin/env python3
"""
Hybrid Parallel Resolution Analyzer: Comprehensive tool for both temporal evolution and convergence analysis.

This script combines the best features of temporal evolution analysis and resolution convergence studies,
with full support for rectangular grids, multiple interface extraction methods, and smart parallel processing.

FEATURES:
- Temporal evolution analysis (multiple times, single/multiple resolutions)
- Resolution convergence analysis (single time, multiple resolutions)  
- Rectangular grid support (160x200, 320x400, etc.)
- Mixed grid types (square + rectangular)
- PLIC, CONREC, and scikit-image interface extraction
- Smart batching for optimal parallel efficiency
- Comprehensive visualization and analysis

Part 1: Core functions and utilities
"""

import os
import time
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
from functools import partial
from fractal_analyzer.core.rt_analyzer import RTAnalyzer

def parse_grid_resolution(resolution_str):
    """
    Parse grid resolution from string, supporting both square and rectangular formats.
    
    Args:
        resolution_str: String like "200", "200x200", "160x200", etc.
        
    Returns:
        tuple: (nx, ny) for the grid dimensions
    """
    if 'x' in resolution_str:
        # Rectangular format: "160x200"
        parts = resolution_str.split('x')
        if len(parts) != 2:
            raise ValueError(f"Invalid resolution format: {resolution_str}")
        nx, ny = int(parts[0]), int(parts[1])
        return nx, ny
    else:
        # Square format: "200"
        res = int(resolution_str)
        return res, res

def format_grid_resolution(nx, ny):
    """Format grid resolution for display and file naming."""
    return f"{nx}x{ny}"

def validate_grid_resolution_input(resolution_input):
    """
    Validate and standardize resolution input.
    
    Args:
        resolution_input: Can be int, string, or mixed list
        
    Returns:
        list: List of standardized resolution strings
    """
    if isinstance(resolution_input, (int, str)):
        # Single resolution
        return [str(resolution_input)]
    elif isinstance(resolution_input, list):
        # Multiple resolutions - convert all to strings
        return [str(res) for res in resolution_input]
    else:
        raise ValueError(f"Invalid resolution input type: {type(resolution_input)}")

def find_timestep_files_for_resolution(data_dir, resolution_str, target_times=None, time_tolerance=0.5):
    """
    Find VTK files for a given resolution, supporting rectangular grids.
    
    Args:
        data_dir: Directory containing VTK files
        resolution_str: Resolution string (e.g., "200", "160x200")
        target_times: List of target times (None = find all files)
        time_tolerance: Maximum time difference allowed
        
    Returns:
        dict: {target_time: (vtk_file_path, actual_time)} or {actual_time: vtk_file_path}
    """
    # Parse resolution
    nx, ny = parse_grid_resolution(resolution_str)
    grid_resolution_str = format_grid_resolution(nx, ny)
    is_rectangular = (nx != ny)
    
    # Try multiple filename patterns
    patterns = [
        os.path.join(data_dir, f"RT{grid_resolution_str}-*.vtk"),  # RT160x200-*.vtk
        os.path.join(data_dir, f"RT{nx}x{ny}-*.vtk"),              # Alternative format
        os.path.join(data_dir, f"{grid_resolution_str}-*.vtk"),    # 160x200-*.vtk
        os.path.join(data_dir, f"{nx}x{ny}-*.vtk"),                # Alternative format
    ]
    
    # For backward compatibility with square grids
    if not is_rectangular:
        patterns.extend([
            os.path.join(data_dir, f"RT{nx}-*.vtk"),  # Legacy: RT200-*.vtk
            os.path.join(data_dir, f"{nx}-*.vtk")     # Legacy: 200-*.vtk
        ])
    
    # Find files using first matching pattern
    vtk_files = []
    for pattern in patterns:
        found_files = glob.glob(pattern)
        if found_files:
            vtk_files = found_files
            break
    
    if not vtk_files:
        return {}
    
    # Extract times from filenames
    file_time_map = {}
    for vtk_file in vtk_files:
        try:
            basename = os.path.basename(vtk_file)
            if '-' in basename:
                time_str = basename.split('-')[1].split('.')[0]
            else:
                # Fallback: extract number from filename
                import re
                match = re.search(r'(\d+)\.vtk$', basename)
                time_str = match.group(1) if match else "0"
            
            file_time = float(time_str) / 1000.0
            file_time_map[file_time] = vtk_file
        except:
            continue
    
    if target_times is None:
        # Return all files found
        return file_time_map
    
    # Find best matches for target times
    result = {}
    for target_time in target_times:
        best_file = None
        best_diff = float('inf')
        best_time = None
        
        for file_time, vtk_file in file_time_map.items():
            diff = abs(file_time - target_time)
            if diff < best_diff:
                best_diff = diff
                best_file = vtk_file
                best_time = file_time
        
        if best_file and best_diff <= time_tolerance:
            result[target_time] = (best_file, best_time)
    
    return result

def determine_analysis_mode(resolutions, target_times):
    """
    Determine the optimal analysis mode based on inputs.
    
    Args:
        resolutions: List of resolution strings
        target_times: List of target times
        
    Returns:
        str: 'temporal_evolution', 'convergence_study', or 'matrix_analysis'
    """
    num_resolutions = len(resolutions)
    num_times = len(target_times)
    
    if num_resolutions == 1 and num_times > 1:
        return 'temporal_evolution'
    elif num_resolutions > 1 and num_times == 1:
        return 'convergence_study'
    elif num_resolutions > 1 and num_times > 1:
        return 'matrix_analysis'
    else:
        # Single resolution, single time - treat as temporal evolution
        return 'temporal_evolution'

def get_method_info(analysis_params):
    """Get extraction method information for naming and display."""
    if analysis_params.get('use_plic', False):
        return 'PLIC', '_plic', 'PLIC (theoretical reconstruction)'
    elif analysis_params.get('use_conrec', False):
        return 'CONREC', '_conrec', 'CONREC (precision)'
    else:
        return 'scikit-image', '_skimage', 'scikit-image (standard)'

def create_output_directory_name(mode, resolutions, target_times, method_suffix):
    """Create descriptive output directory name based on analysis mode."""
    if mode == 'temporal_evolution':
        if len(resolutions) == 1:
            res_str = resolutions[0]
            nx, ny = parse_grid_resolution(res_str)
            grid_str = format_grid_resolution(nx, ny)
            time_range = f"t{min(target_times):.1f}-{max(target_times):.1f}" if len(target_times) > 1 else f"t{target_times[0]:.1f}"
            return f"temporal_evolution_{grid_str}_{time_range}{method_suffix}"
        else:
            time_range = f"t{min(target_times):.1f}-{max(target_times):.1f}" if len(target_times) > 1 else f"t{target_times[0]:.1f}"
            return f"temporal_evolution_multi_res_{time_range}{method_suffix}"
    
    elif mode == 'convergence_study':
        time_str = f"t{target_times[0]:.1f}"
        return f"convergence_study_{time_str}{method_suffix}"
    
    elif mode == 'matrix_analysis':
        time_range = f"t{min(target_times):.1f}-{max(target_times):.1f}"
        res_range = f"{len(resolutions)}res"
        return f"matrix_analysis_{res_range}_{time_range}{method_suffix}"
    
    else:
        return f"hybrid_analysis{method_suffix}"

def validate_inputs(data_dirs, resolutions, target_times):
    """
    Validate input parameters for consistency.
    
    Args:
        data_dirs: List of data directories
        resolutions: List of resolution strings
        target_times: List of target times
        
    Returns:
        tuple: (is_valid, error_message)
    """
    # Check data directories
    if len(data_dirs) != len(resolutions):
        return False, f"Number of data directories ({len(data_dirs)}) must match number of resolutions ({len(resolutions)})"
    
    for i, data_dir in enumerate(data_dirs):
        if not os.path.exists(data_dir):
            return False, f"Data directory not found: {data_dir}"
    
    # Validate resolution formats
    try:
        for res_str in resolutions:
            nx, ny = parse_grid_resolution(res_str)
            if nx <= 0 or ny <= 0:
                return False, f"Invalid resolution: {res_str}"
    except ValueError as e:
        return False, str(e)
    
    # Validate target times
    if not target_times:
        return False, "At least one target time must be specified"
    
    for t in target_times:
        if not isinstance(t, (int, float)) or t < 0:
            return False, f"Invalid target time: {t}"
    
    return True, None

def analyze_grid_types(resolutions):
    """
    Analyze the types of grids in the resolution list.
    
    Returns:
        dict: Information about grid types
    """
    square_count = 0
    rectangular_count = 0
    aspect_ratios = []
    
    for res_str in resolutions:
        nx, ny = parse_grid_resolution(res_str)
        if nx == ny:
            square_count += 1
        else:
            rectangular_count += 1
            aspect_ratios.append(max(nx, ny) / min(nx, ny))
    
    return {
        'total': len(resolutions),
        'square_count': square_count,
        'rectangular_count': rectangular_count,
        'aspect_ratios': aspect_ratios,
        'max_aspect_ratio': max(aspect_ratios) if aspect_ratios else 1.0,
        'has_mixed_types': square_count > 0 and rectangular_count > 0
    }

def print_analysis_header(mode, resolutions, target_times, data_dirs, analysis_params, num_processes):
    """Print comprehensive analysis header with all relevant information."""
    method_name, method_suffix, method_description = get_method_info(analysis_params)
    grid_info = analyze_grid_types(resolutions)
    
    print(f"🚀 HYBRID PARALLEL RESOLUTION ANALYZER")
    print(f"=" * 70)
    print(f"Analysis mode: {mode.replace('_', ' ').title()}")
    print(f"Interface extraction: {method_description}")
    print(f"Mixing method: {analysis_params.get('mixing_method', 'dalziel')}")
    print(f"Parallel processes: {num_processes}")
    
    print(f"\n📊 ANALYSIS SCOPE:")
    print(f"Resolutions ({len(resolutions)}): {resolutions}")
    print(f"Target times ({len(target_times)}): {target_times}")
    print(f"Data directories: {len(data_dirs)}")
    
    print(f"\n📐 GRID ANALYSIS:")
    if grid_info['has_mixed_types']:
        print(f"Mixed grid types: {grid_info['square_count']} square, {grid_info['rectangular_count']} rectangular")
    elif grid_info['rectangular_count'] > 0:
        print(f"All rectangular grids (max aspect ratio: {grid_info['max_aspect_ratio']:.2f})")
    else:
        print(f"All square grids")
    
    total_analyses = len(resolutions) * len(target_times)
    print(f"\n⚡ PARALLEL EXECUTION:")
    print(f"Total analyses: {total_analyses} ({len(resolutions)} resolutions × {len(target_times)} times)")
    
    if mode == 'temporal_evolution':
        print(f"Strategy: Smart batching (each worker processes all times for one resolution)")
    elif mode == 'convergence_study':
        print(f"Strategy: Resolution parallel (each worker processes one resolution)")
    else:
        print(f"Strategy: Matrix parallel (adaptive based on workload)")

# Analysis result processing utilities
def create_base_result_dict(resolution_str, target_time, extraction_method, worker_pid):
    """Create base result dictionary with common fields."""
    nx, ny = parse_grid_resolution(resolution_str)
    grid_resolution_str = format_grid_resolution(nx, ny)
    
    return {
        'resolution_str': resolution_str,
        'nx': nx,
        'ny': ny,
        'grid_resolution': grid_resolution_str,
        'is_rectangular': (nx != ny),
        'aspect_ratio': max(nx, ny) / min(nx, ny),
        'effective_resolution': max(nx, ny),
        'total_cells': nx * ny,
        'target_time': target_time,
        'extraction_method': extraction_method,
        'worker_pid': worker_pid
    }

def create_success_result(base_result, vtk_analysis_result, vtk_file, actual_time, 
                         processing_time, segments_count):
    """Create successful analysis result dictionary."""
    result = base_result.copy()
    result.update({
        'actual_time': actual_time,
        'time_error': abs(actual_time - base_result['target_time']),
        'fractal_dim': vtk_analysis_result.get('fractal_dim', np.nan),
        'fd_error': vtk_analysis_result.get('fd_error', np.nan),
        'fd_r_squared': vtk_analysis_result.get('fd_r_squared', np.nan),
        'h_total': vtk_analysis_result.get('h_total', np.nan),
        'ht': vtk_analysis_result.get('ht', np.nan),
        'hb': vtk_analysis_result.get('hb', np.nan),
        'segments': segments_count,
        'processing_time': processing_time,
        'vtk_file': os.path.basename(vtk_file),
        'analysis_quality': vtk_analysis_result.get('analysis_quality', 'unknown'),
        'status': 'success'
    })
    return result

def create_failure_result(base_result, error_message, vtk_file=None, actual_time=None, processing_time=None):
    """Create failed analysis result dictionary."""
    result = base_result.copy()
    result.update({
        'actual_time': actual_time if actual_time is not None else np.nan,
        'time_error': np.nan,
        'fractal_dim': np.nan,
        'fd_error': np.nan,
        'fd_r_squared': np.nan,
        'h_total': np.nan,
        'ht': np.nan,
        'hb': np.nan,
        'segments': np.nan,
        'processing_time': processing_time if processing_time is not None else np.nan,
        'vtk_file': os.path.basename(vtk_file) if vtk_file else 'not_found',
        'analysis_quality': 'failed',
        'status': 'failed',
        'error': error_message
    })
    return result

# Part 2: Core analysis functions

def analyze_single_file(vtk_file, analyzer, analysis_params):
    """
    Analyze a single VTK file using the provided analyzer.
    
    Args:
        vtk_file: Path to VTK file
        analyzer: RTAnalyzer instance
        analysis_params: Analysis parameters
        
    Returns:
        tuple: (analysis_result, segments_count, processing_time)
    """
    start_time = time.time()
    
    try:
        # Perform VTK analysis
        result = analyzer.analyze_vtk_file(
            vtk_file,
            mixing_method=analysis_params.get('mixing_method', 'dalziel'),
            h0=analysis_params.get('h0', 0.5),
            min_box_size=analysis_params.get('min_box_size', None)
        )
        
        # Extract interface for segment count
        data = analyzer.read_vtk_file(vtk_file)
        contours = analyzer.extract_interface(data['f'], data['x'], data['y'])
        segments = analyzer.convert_contours_to_segments(contours)
        
        processing_time = time.time() - start_time
        return result, len(segments), processing_time
        
    except Exception as e:
        processing_time = time.time() - start_time
        raise Exception(f"Analysis failed: {str(e)}")

def analyze_temporal_evolution_batch(args):
    """
    Analyze temporal evolution for one resolution across multiple times.
    Optimized for temporal evolution studies.
    
    Args:
        args: Tuple of (data_dir, resolution_str, target_times, base_output_dir, analysis_params)
        
    Returns:
        list: List of analysis results for all times
    """
    data_dir, resolution_str, target_times, base_output_dir, analysis_params = args
    
    method_name, method_suffix, method_description = get_method_info(analysis_params)
    nx, ny = parse_grid_resolution(resolution_str)
    grid_resolution_str = format_grid_resolution(nx, ny)
    
    print(f"🔍 Worker {os.getpid()}: Temporal evolution {grid_resolution_str} for {len(target_times)} times ({method_name})")
    
    # Create analyzer for this resolution (reuse for efficiency)
    res_output_dir = os.path.join(base_output_dir, f"temporal_evolution_{grid_resolution_str}")
    analyzer = RTAnalyzer(
        res_output_dir, 
        use_grid_optimization=True,
        no_titles=True,
        use_conrec=analysis_params.get('use_conrec', False),
        use_plic=analysis_params.get('use_plic', False),
        debug=analysis_params.get('debug', False)
    )
    
    # Find all available files for this resolution
    file_time_map = find_timestep_files_for_resolution(data_dir, resolution_str, target_times)
    
    batch_results = []
    worker_start = time.time()
    
    for i, target_time in enumerate(target_times):
        print(f"   [{i+1}/{len(target_times)}] {grid_resolution_str} at t={target_time}")
        
        base_result = create_base_result_dict(resolution_str, target_time, method_name, os.getpid())
        
        if target_time not in file_time_map:
            print(f"   No file found for t={target_time}")
            failure_result = create_failure_result(base_result, "No file found for target time")
            batch_results.append(failure_result)
            continue
        
        vtk_file, actual_time = file_time_map[target_time]
        
        try:
            vtk_result, segments_count, processing_time = analyze_single_file(vtk_file, analyzer, analysis_params)
            
            success_result = create_success_result(
                base_result, vtk_result, vtk_file, actual_time, processing_time, segments_count
            )
            
            print(f"   ✅ t={target_time:.1f}: D={success_result['fractal_dim']:.4f}±{success_result['fd_error']:.4f}, "
                  f"Segments={segments_count}, Time={processing_time:.1f}s")
            
            batch_results.append(success_result)
            
        except Exception as e:
            print(f"   ❌ t={target_time}: {str(e)}")
            failure_result = create_failure_result(base_result, str(e), vtk_file, actual_time)
            batch_results.append(failure_result)
    
    worker_time = time.time() - worker_start
    successful_count = sum(1 for r in batch_results if r['status'] == 'success')
    
    print(f"✅ Worker {os.getpid()}: {grid_resolution_str} temporal evolution complete - "
          f"{successful_count}/{len(target_times)} successful in {worker_time:.1f}s ({method_name})")
    
    return batch_results

def analyze_convergence_single_resolution(args):
    """
    Analyze single resolution for convergence study.
    Optimized for resolution convergence studies.
    
    Args:
        args: Tuple of (data_dir, resolution_str, target_time, base_output_dir, analysis_params)
        
    Returns:
        dict: Analysis result for this resolution
    """
    data_dir, resolution_str, target_time, base_output_dir, analysis_params = args
    
    method_name, method_suffix, method_description = get_method_info(analysis_params)
    nx, ny = parse_grid_resolution(resolution_str)
    grid_resolution_str = format_grid_resolution(nx, ny)
    
    print(f"🔍 Worker {os.getpid()}: Convergence analysis {grid_resolution_str} at t={target_time} ({method_name})")
    
    base_result = create_base_result_dict(resolution_str, target_time, method_name, os.getpid())
    
    # Create analyzer for this resolution
    res_output_dir = os.path.join(base_output_dir, f"convergence_{grid_resolution_str}")
    analyzer = RTAnalyzer(
        res_output_dir, 
        use_grid_optimization=True,
        no_titles=True,
        use_conrec=analysis_params.get('use_conrec', False),
        use_plic=analysis_params.get('use_plic', False),
        debug=analysis_params.get('debug', False)
    )
    
    try:
        # Find file for target time
        file_time_map = find_timestep_files_for_resolution(data_dir, resolution_str, [target_time])
        
        if target_time not in file_time_map:
            print(f"   No file found for t={target_time}")
            return create_failure_result(base_result, "No file found for target time")
        
        vtk_file, actual_time = file_time_map[target_time]
        
        # Perform analysis
        vtk_result, segments_count, processing_time = analyze_single_file(vtk_file, analyzer, analysis_params)
        
        success_result = create_success_result(
            base_result, vtk_result, vtk_file, actual_time, processing_time, segments_count
        )
        
        print(f"✅ Worker {os.getpid()}: {grid_resolution_str} D={success_result['fractal_dim']:.4f}±{success_result['fd_error']:.4f}, "
              f"Segments={segments_count}, Time={processing_time:.1f}s ({method_name})")
        
        return success_result
        
    except Exception as e:
        print(f"❌ Worker {os.getpid()}: {grid_resolution_str} failed - {str(e)}")
        return create_failure_result(base_result, str(e))

def analyze_matrix_single_point(args):
    """
    Analyze single (resolution, time) point for matrix analysis.
    Optimized for comprehensive matrix studies.
    
    Args:
        args: Tuple of (data_dir, resolution_str, target_time, base_output_dir, analysis_params)
        
    Returns:
        dict: Analysis result for this point
    """
    data_dir, resolution_str, target_time, base_output_dir, analysis_params = args
    
    method_name, method_suffix, method_description = get_method_info(analysis_params)
    nx, ny = parse_grid_resolution(resolution_str)
    grid_resolution_str = format_grid_resolution(nx, ny)
    
    base_result = create_base_result_dict(resolution_str, target_time, method_name, os.getpid())
    
    # Create analyzer for this point
    point_output_dir = os.path.join(base_output_dir, f"matrix_{grid_resolution_str}_t{target_time:.1f}")
    analyzer = RTAnalyzer(
        point_output_dir, 
        use_grid_optimization=True,
        no_titles=True,
        use_conrec=analysis_params.get('use_conrec', False),
        use_plic=analysis_params.get('use_plic', False),
        debug=analysis_params.get('debug', False)
    )
    
    try:
        # Find file for target time
        file_time_map = find_timestep_files_for_resolution(data_dir, resolution_str, [target_time])
        
        if target_time not in file_time_map:
            return create_failure_result(base_result, "No file found for target time")
        
        vtk_file, actual_time = file_time_map[target_time]
        
        # Perform analysis
        vtk_result, segments_count, processing_time = analyze_single_file(vtk_file, analyzer, analysis_params)
        
        success_result = create_success_result(
            base_result, vtk_result, vtk_file, actual_time, processing_time, segments_count
        )
        
        return success_result
        
    except Exception as e:
        return create_failure_result(base_result, str(e))

def run_temporal_evolution_analysis(data_dirs, resolutions, target_times, output_dir, 
                                  analysis_params, num_processes):
    """
    Run temporal evolution analysis using smart batching.
    Each worker processes all times for one resolution.
    """
    print(f"\n⚡ TEMPORAL EVOLUTION ANALYSIS")
    print(f"Strategy: Smart batching (reuse analyzer per resolution)")
    
    # Prepare arguments for parallel processing
    process_args = [(data_dir, resolution_str, target_times, output_dir, analysis_params)
                   for data_dir, resolution_str in zip(data_dirs, resolutions)]
    
    total_start = time.time()
    
    try:
        with Pool(processes=num_processes) as pool:
            batch_results = pool.map(analyze_temporal_evolution_batch, process_args)
    except KeyboardInterrupt:
        print("\n⚠️  Analysis interrupted by user")
        return None
    except Exception as e:
        print(f"\n❌ Parallel execution failed: {str(e)}")
        return None
    
    total_time = time.time() - total_start
    
    # Flatten results
    all_results = []
    for batch in batch_results:
        all_results.extend(batch)
    
    return all_results, total_time

def run_convergence_analysis(data_dirs, resolutions, target_time, output_dir, 
                            analysis_params, num_processes):
    """
    Run convergence analysis with one worker per resolution.
    Optimized for single time, multiple resolutions.
    """
    print(f"\n⚡ CONVERGENCE ANALYSIS")
    print(f"Strategy: Resolution parallel (one worker per resolution)")
    
    # Prepare arguments for parallel processing
    process_args = [(data_dir, resolution_str, target_time, output_dir, analysis_params)
                   for data_dir, resolution_str in zip(data_dirs, resolutions)]
    
    total_start = time.time()
    
    try:
        with Pool(processes=num_processes) as pool:
            results = pool.map(analyze_convergence_single_resolution, process_args)
    except KeyboardInterrupt:
        print("\n⚠️  Analysis interrupted by user")
        return None
    except Exception as e:
        print(f"\n❌ Parallel execution failed: {str(e)}")
        return None
    
    total_time = time.time() - total_start
    
    return results, total_time

def run_matrix_analysis(data_dirs, resolutions, target_times, output_dir, 
                       analysis_params, num_processes):
    """
    Run matrix analysis with adaptive parallelization.
    Create all (resolution, time) combinations and distribute optimally.
    """
    print(f"\n⚡ MATRIX ANALYSIS")
    print(f"Strategy: Matrix parallel (distribute all combinations)")
    
    # Create all (resolution, time) combinations
    process_args = []
    for data_dir, resolution_str in zip(data_dirs, resolutions):
        for target_time in target_times:
            process_args.append((data_dir, resolution_str, target_time, output_dir, analysis_params))
    
    print(f"Total matrix points: {len(process_args)} ({len(resolutions)} × {len(target_times)})")
    
    total_start = time.time()
    
    try:
        with Pool(processes=num_processes) as pool:
            # Process in chunks to show progress
            chunk_size = max(1, len(process_args) // 10)  # 10% chunks
            results = []
            
            for i in range(0, len(process_args), chunk_size):
                chunk = process_args[i:i+chunk_size]
                chunk_results = pool.map(analyze_matrix_single_point, chunk)
                results.extend(chunk_results)
                
                progress = min(100, (i + len(chunk)) * 100 // len(process_args))
                print(f"   Progress: {progress}% ({i + len(chunk)}/{len(process_args)} points)")
                
    except KeyboardInterrupt:
        print("\n⚠️  Analysis interrupted by user")
        return None
    except Exception as e:
        print(f"\n❌ Parallel execution failed: {str(e)}")
        return None
    
    total_time = time.time() - total_start
    
    return results, total_time

def run_hybrid_analysis(data_dirs, resolutions, target_times, output_dir, 
                       analysis_params, num_processes=None):
    """
    Main hybrid analysis function that automatically selects the best strategy.
    
    Args:
        data_dirs: List of data directories
        resolutions: List of resolution strings  
        target_times: List of target times
        output_dir: Output directory
        analysis_params: Analysis parameters dictionary
        num_processes: Number of parallel processes
        
    Returns:
        DataFrame with analysis results
    """
    # Validate inputs
    is_valid, error_msg = validate_inputs(data_dirs, resolutions, target_times)
    if not is_valid:
        print(f"❌ Input validation failed: {error_msg}")
        return None
    
    # Determine analysis mode
    mode = determine_analysis_mode(resolutions, target_times)
    
    # Set optimal number of processes
    if num_processes is None:
        if mode == 'temporal_evolution':
            num_processes = min(len(resolutions), cpu_count())
        elif mode == 'convergence_study':  
            num_processes = min(len(resolutions), cpu_count())
        else:  # matrix_analysis
            num_processes = min(cpu_count(), 8)  # Limit for I/O
    
    # Create output directory
    method_name, method_suffix, method_description = get_method_info(analysis_params)
    if output_dir is None:
        output_dir = create_output_directory_name(mode, resolutions, target_times, method_suffix)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Print analysis header
    print_analysis_header(mode, resolutions, target_times, data_dirs, analysis_params, num_processes)
    
    # Run appropriate analysis
    if mode == 'temporal_evolution':
        results, total_time = run_temporal_evolution_analysis(
            data_dirs, resolutions, target_times, output_dir, analysis_params, num_processes)
    elif mode == 'convergence_study':
        results, total_time = run_convergence_analysis(
            data_dirs, resolutions, target_times[0], output_dir, analysis_params, num_processes)
    else:  # matrix_analysis
        results, total_time = run_matrix_analysis(
            data_dirs, resolutions, target_times, output_dir, analysis_params, num_processes)
    
    if results is None:
        return None
    
    # Convert to DataFrame and add metadata
    df = pd.DataFrame(results)
    df['analysis_mode'] = mode
    
    # Save results
    time_range_str = f"t{min(target_times):.1f}-{max(target_times):.1f}" if len(target_times) > 1 else f"t{target_times[0]:.1f}"
    results_file = os.path.join(output_dir, f'hybrid_analysis_{mode}_{time_range_str}{method_suffix}.csv')
    df.to_csv(results_file, index=False)
    
    # Print summary
    print_analysis_summary(df, mode, total_time, method_description, results_file)
    
    return df
# Part 3: Analysis summary and plotting functions

def print_analysis_summary(df, mode, total_time, method_description, results_file):
    """Print comprehensive analysis summary."""
    print(f"\n📊 HYBRID ANALYSIS SUMMARY")
    print(f"=" * 70)
    print(f"Analysis mode: {mode.replace('_', ' ').title()}")
    print(f"Interface extraction: {method_description}")
    print(f"Total processing time: {total_time:.1f}s")
    print(f"Results saved to: {results_file}")
    
    # Filter results
    successful_results = df[df['status'] == 'success']
    failed_results = df[df['status'] == 'failed']
    
    print(f"\nSuccessful analyses: {len(successful_results)}/{len(df)}")
    if len(failed_results) > 0:
        print(f"Failed analyses: {len(failed_results)}")
        
        # Show failure summary
        failure_summary = failed_results.groupby(['grid_resolution', 'error']).size()
        print("Failure breakdown:")
        for (resolution, error), count in failure_summary.items():
            print(f"  {resolution}: {count} - {error[:50]}...")
    
    if len(successful_results) > 0:
        # Calculate efficiency metrics
        total_sequential_time = successful_results['processing_time'].sum()
        theoretical_speedup = total_sequential_time / total_time
        
        # Determine process count based on mode
        if mode == 'temporal_evolution':
            effective_processes = len(successful_results['grid_resolution'].unique())
        elif mode == 'convergence_study':
            effective_processes = len(successful_results['grid_resolution'].unique())
        else:  # matrix_analysis
            effective_processes = min(8, len(successful_results))  # Estimate
        
        parallel_efficiency = total_sequential_time / (total_time * effective_processes) * 100
        
        print(f"\n🚀 PARALLEL PERFORMANCE:")
        print(f"  Sequential time estimate: {total_sequential_time:.1f}s")
        print(f"  Actual parallel time: {total_time:.1f}s")
        print(f"  Theoretical speedup: {theoretical_speedup:.1f}×")
        print(f"  Parallel efficiency: {parallel_efficiency:.1f}%")
        print(f"  Average time per analysis: {successful_results['processing_time'].mean():.1f}s")
        
        print(f"\n📈 ANALYSIS RESULTS:")
        print(f"  Time range: {successful_results['actual_time'].min():.3f} to {successful_results['actual_time'].max():.3f}")
        print(f"  Fractal dimension range: {successful_results['fractal_dim'].min():.4f} to {successful_results['fractal_dim'].max():.4f}")
        print(f"  Segment count range: {int(successful_results['segments'].min())} to {int(successful_results['segments'].max())}")
        
        # Grid type analysis
        grid_info = analyze_grid_types(successful_results['resolution_str'].unique())
        if grid_info['has_mixed_types']:
            print(f"  Grid types: {grid_info['square_count']} square, {grid_info['rectangular_count']} rectangular")
            if grid_info['aspect_ratios']:
                print(f"  Max aspect ratio: {grid_info['max_aspect_ratio']:.2f}")
        
        # Mode-specific summaries
        if mode == 'temporal_evolution':
            print_temporal_evolution_summary(successful_results)
        elif mode == 'convergence_study':
            print_convergence_summary(successful_results)
        else:  # matrix_analysis
            print_matrix_summary(successful_results)

def print_temporal_evolution_summary(df):
    """Print summary for temporal evolution analysis."""
    print(f"\n🌊 TEMPORAL EVOLUTION SUMMARY:")
    
    for resolution_str in sorted(df['resolution_str'].unique()):
        res_data = df[df['resolution_str'] == resolution_str]
        if len(res_data) > 0:
            nx, ny = parse_grid_resolution(resolution_str)
            grid_str = format_grid_resolution(nx, ny)
            
            print(f"\n  {grid_str} Resolution ({len(res_data)} time points):")
            print(f"    Time range: {res_data['actual_time'].min():.3f} to {res_data['actual_time'].max():.3f}")
            print(f"    D range: {res_data['fractal_dim'].min():.4f} to {res_data['fractal_dim'].max():.4f}")
            print(f"    Final mixing thickness: {res_data['h_total'].iloc[-1]:.4f}")
            print(f"    Segment range: {int(res_data['segments'].min())} to {int(res_data['segments'].max())}")

def print_convergence_summary(df):
    """Print summary for convergence analysis."""
    print(f"\n📈 CONVERGENCE SUMMARY:")
    
    df_sorted = df.sort_values('effective_resolution')
    print(f"  Resolution progression:")
    
    for _, row in df_sorted.iterrows():
        print(f"    {row['grid_resolution']}: D = {row['fractal_dim']:.4f} ± {row['fd_error']:.4f}")
    
    # Check for convergence
    if len(df_sorted) >= 2:
        fd_change = df_sorted['fractal_dim'].iloc[-1] - df_sorted['fractal_dim'].iloc[-2]
        fd_rel_change = abs(fd_change) / df_sorted['fractal_dim'].iloc[-1]
        
        print(f"\n  Convergence analysis:")
        print(f"    Latest change: {fd_change:.6f}")
        print(f"    Relative change: {fd_rel_change:.3%}")
        
        if fd_rel_change < 0.01:
            print(f"    ✅ Appears converged (< 1% change)")
        elif fd_rel_change < 0.05:
            print(f"    ⚠️  Near convergence (< 5% change)")
        else:
            print(f"    ❌ Not converged (≥ 5% change)")

def print_matrix_summary(df):
    """Print summary for matrix analysis."""
    print(f"\n🔲 MATRIX SUMMARY:")
    
    resolutions = sorted(df['resolution_str'].unique())
    times = sorted(df['actual_time'].unique())
    
    print(f"  Matrix dimensions: {len(resolutions)} resolutions × {len(times)} times")
    print(f"  Coverage: {len(df)}/{len(resolutions) * len(times)} points")
    
    # Show data availability matrix
    print(f"\n  Data availability matrix:")
    print(f"  {'Resolution':<12} | ", end="")
    for t in times[:5]:  # Show first 5 times
        print(f"{t:>6.1f}", end=" ")
    if len(times) > 5:
        print(f"... (+{len(times)-5})")
    else:
        print()
    
    print(f"  {'-'*12}-+-{'-'*(7*min(5, len(times)))}")
    
    for res_str in resolutions:
        nx, ny = parse_grid_resolution(res_str)
        grid_str = format_grid_resolution(nx, ny)
        print(f"  {grid_str:<12} | ", end="")
        
        for t in times[:5]:
            point_data = df[(df['resolution_str'] == res_str) & (abs(df['actual_time'] - t) < 0.1)]
            if len(point_data) > 0:
                print(f"{'  ✓  '}", end=" ")
            else:
                print(f"{'  ✗  '}", end=" ")
        print()

def create_temporal_evolution_plots(df, output_dir, method_suffix):
    """Create plots for temporal evolution analysis."""
    print(f"\n📊 Creating temporal evolution plots...")
    
    successful_df = df[df['status'] == 'success'].copy()
    if len(successful_df) == 0:
        print("⚠️  No successful results for plotting")
        return
    
    # Sort data
    successful_df = successful_df.sort_values(['resolution_str', 'actual_time'])
    
    # Create comprehensive evolution plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Fractal dimension evolution
    resolutions = sorted(successful_df['resolution_str'].unique())
    colors = plt.cm.viridis(np.linspace(0, 1, len(resolutions)))
    
    for i, resolution_str in enumerate(resolutions):
        res_data = successful_df[successful_df['resolution_str'] == resolution_str]
        nx, ny = parse_grid_resolution(resolution_str)
        grid_str = format_grid_resolution(nx, ny)
        
        if len(res_data) > 0:
            ax1.errorbar(res_data['actual_time'], res_data['fractal_dim'], 
                        yerr=res_data['fd_error'], fmt='o-', capsize=3, 
                        color=colors[i], linewidth=2, markersize=6, 
                        label=grid_str)
    
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Fractal Dimension')
    ax1.set_title('Fractal Dimension Temporal Evolution')
    ax1.grid(True, alpha=0.7)
    ax1.legend()
    
    # Plot 2: Mixing layer evolution
    for i, resolution_str in enumerate(resolutions):
        res_data = successful_df[successful_df['resolution_str'] == resolution_str]
        nx, ny = parse_grid_resolution(resolution_str)
        grid_str = format_grid_resolution(nx, ny)
        
        if len(res_data) > 0:
            ax2.plot(res_data['actual_time'], res_data['h_total'], 
                    'o-', color=colors[i], linewidth=2, markersize=6,
                    label=grid_str)
    
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Mixing Layer Thickness')
    ax2.set_title('Mixing Layer Temporal Evolution')
    ax2.grid(True, alpha=0.7)
    ax2.legend()
    
    # Plot 3: Interface complexity evolution
    for i, resolution_str in enumerate(resolutions):
        res_data = successful_df[successful_df['resolution_str'] == resolution_str]
        nx, ny = parse_grid_resolution(resolution_str)
        grid_str = format_grid_resolution(nx, ny)
        
        if len(res_data) > 0:
            ax3.plot(res_data['actual_time'], res_data['segments'], 
                    'o-', color=colors[i], linewidth=2, markersize=6,
                    label=grid_str)
    
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Number of Interface Segments')
    ax3.set_title('Interface Complexity Evolution')
    ax3.grid(True, alpha=0.7)
    ax3.legend()
    
    # Plot 4: Analysis quality
    for i, resolution_str in enumerate(resolutions):
        res_data = successful_df[successful_df['resolution_str'] == resolution_str]
        nx, ny = parse_grid_resolution(resolution_str)
        grid_str = format_grid_resolution(nx, ny)
        
        if len(res_data) > 0:
            ax4.plot(res_data['actual_time'], res_data['fd_r_squared'], 
                    'o-', color=colors[i], linewidth=2, markersize=4,
                    label=grid_str)
    
    ax4.axhline(y=0.99, color='red', linestyle='--', alpha=0.7, label='R² = 0.99')
    ax4.set_xlabel('Time')
    ax4.set_ylabel('R² Value')
    ax4.set_title('Fractal Analysis Quality')
    ax4.set_ylim(0.95, 1.01)
    ax4.grid(True, alpha=0.7)
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'temporal_evolution{method_suffix}.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   ✅ Saved temporal evolution plots")

def create_convergence_plots(df, output_dir, method_suffix):
    """Create plots for convergence analysis."""
    print(f"\n📊 Creating convergence plots...")
    
    successful_df = df[df['status'] == 'success'].copy()
    if len(successful_df) == 0:
        print("⚠️  No successful results for plotting")
        return
    
    # Sort by effective resolution
    successful_df = successful_df.sort_values('effective_resolution')
    
    # Create comprehensive convergence plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Fractal dimension convergence
    # Color-code by grid type
    square_mask = ~successful_df['is_rectangular']
    rect_mask = successful_df['is_rectangular']
    
    ax1.errorbar(successful_df['effective_resolution'], successful_df['fractal_dim'], 
                yerr=successful_df['fd_error'], fmt='bo-', capsize=5, 
                linewidth=2, markersize=8, label='All grids')
    
    if np.any(square_mask):
        ax1.scatter(successful_df[square_mask]['effective_resolution'], 
                   successful_df[square_mask]['fractal_dim'], 
                   c='blue', s=100, marker='s', label='Square grids', alpha=0.7)
    
    if np.any(rect_mask):
        ax1.scatter(successful_df[rect_mask]['effective_resolution'], 
                   successful_df[rect_mask]['fractal_dim'], 
                   c='red', s=100, marker='^', label='Rectangular grids', alpha=0.7)
    
    ax1.set_xscale('log', base=2)
    ax1.set_xlabel('Effective Resolution (max of nx, ny)')
    ax1.set_ylabel('Fractal Dimension')
    ax1.set_title('Fractal Dimension Convergence')
    ax1.grid(True, alpha=0.7)
    ax1.legend()
    
    # Add grid labels
    for _, row in successful_df.iterrows():
        ax1.annotate(f"{row['grid_resolution']}", 
                    (row['effective_resolution'], row['fractal_dim']),
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    # Plot 2: Mixing thickness convergence
    ax2.plot(successful_df['effective_resolution'], successful_df['h_total'], 'go-', 
            linewidth=2, markersize=8, label='Total')
    ax2.plot(successful_df['effective_resolution'], successful_df['ht'], 'r--', 
            linewidth=2, markersize=6, label='Upper')
    ax2.plot(successful_df['effective_resolution'], successful_df['hb'], 'b--', 
            linewidth=2, markersize=6, label='Lower')
    
    ax2.set_xscale('log', base=2)
    ax2.set_xlabel('Effective Resolution')
    ax2.set_ylabel('Mixing Layer Thickness')
    ax2.set_title('Mixing Layer Convergence')
    ax2.grid(True, alpha=0.7)
    ax2.legend()
    
    # Plot 3: Interface complexity scaling
    ax3.loglog(successful_df['effective_resolution'], successful_df['segments'], 
              'mo-', linewidth=2, markersize=8, label='Interface Segments')
    
    # Add power law fit if enough points
    if len(successful_df) >= 3:
        log_res = np.log(successful_df['effective_resolution'])
        log_seg = np.log(successful_df['segments'])
        coeffs = np.polyfit(log_res, log_seg, 1)
        slope = coeffs[0]
        
        fit_res = successful_df['effective_resolution']
        fit_seg = np.exp(coeffs[1]) * fit_res**slope
        ax3.loglog(fit_res, fit_seg, 'r--', alpha=0.7, 
                  label=f'Power law: slope = {slope:.2f}')
    
    ax3.set_xlabel('Effective Resolution')
    ax3.set_ylabel('Number of Interface Segments')
    ax3.set_title('Interface Complexity Scaling')
    ax3.grid(True, alpha=0.7)
    ax3.legend()
    
    # Plot 4: Processing time scaling
    ax4.scatter(successful_df['total_cells'], successful_df['processing_time'], 
               c=['blue' if not rect else 'red' for rect in successful_df['is_rectangular']], 
               s=100, alpha=0.7)
    
    for _, row in successful_df.iterrows():
        ax4.annotate(f"{row['grid_resolution']}", 
                    (row['total_cells'], row['processing_time']),
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    ax4.set_xscale('log')
    ax4.set_yscale('log')
    ax4.set_xlabel('Total Cells (nx × ny)')
    ax4.set_ylabel('Processing Time (s)')
    ax4.set_title('Processing Time Scaling')
    ax4.grid(True, alpha=0.7)
    
    # Add legend for grid types
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='blue', label='Square grids'),
                      Patch(facecolor='red', label='Rectangular grids')]
    ax4.legend(handles=legend_elements)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'convergence_analysis{method_suffix}.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create aspect ratio analysis for rectangular grids
    if np.any(successful_df['is_rectangular']):
        create_aspect_ratio_plots(successful_df, output_dir, method_suffix)
    
    print(f"   ✅ Saved convergence plots")

def create_aspect_ratio_plots(df, output_dir, method_suffix):
    """Create aspect ratio analysis plots for rectangular grids."""
    rect_data = df[df['is_rectangular']].copy()
    
    if len(rect_data) == 0:
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Fractal dimension vs aspect ratio
    ax1.scatter(rect_data['aspect_ratio'], rect_data['fractal_dim'], 
               c=rect_data['effective_resolution'], cmap='viridis', s=100)
    ax1.set_xlabel('Aspect Ratio (max/min dimension)')
    ax1.set_ylabel('Fractal Dimension')
    ax1.set_title('Fractal Dimension vs Aspect Ratio')
    ax1.grid(True, alpha=0.7)
    
    cbar1 = plt.colorbar(ax1.collections[0], ax=ax1, label='Effective Resolution')
    
    # Add grid labels
    for _, row in rect_data.iterrows():
        ax1.annotate(f"{row['grid_resolution']}", 
                    (row['aspect_ratio'], row['fractal_dim']),
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    # Mixing thickness vs aspect ratio
    ax2.scatter(rect_data['aspect_ratio'], rect_data['h_total'], 
               c=rect_data['effective_resolution'], cmap='plasma', s=100)
    ax2.set_xlabel('Aspect Ratio (max/min dimension)')
    ax2.set_ylabel('Mixing Layer Thickness')
    ax2.set_title('Mixing Thickness vs Aspect Ratio')
    ax2.grid(True, alpha=0.7)
    
    cbar2 = plt.colorbar(ax2.collections[0], ax=ax2, label='Effective Resolution')
    
    # Add grid labels
    for _, row in rect_data.iterrows():
        ax2.annotate(f"{row['grid_resolution']}", 
                    (row['aspect_ratio'], row['h_total']),
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'rectangular_grid_analysis{method_suffix}.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   ✅ Saved rectangular grid analysis plots")

# Part 4: Matrix analysis plots and main function

def create_matrix_plots(df, output_dir, method_suffix):
    """Create plots for matrix analysis."""
    print(f"\n📊 Creating matrix analysis plots...")
    
    successful_df = df[df['status'] == 'success'].copy()
    if len(successful_df) == 0:
        print("⚠️  No successful results for plotting")
        return
    
    resolutions = sorted(successful_df['resolution_str'].unique())
    times = sorted(successful_df['actual_time'].unique())
    
    # Create comprehensive matrix plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: 3D surface plot of fractal dimension
    try:
        from mpl_toolkits.mplot3d import Axes3D
        
        # Create meshgrid for 3D plotting
        res_values = [parse_grid_resolution(res)[0] * parse_grid_resolution(res)[1] for res in resolutions]  # Use total cells
        time_grid, res_grid = np.meshgrid(times, res_values)
        dim_grid = np.full_like(time_grid, np.nan, dtype=float)
        
        for i, res_str in enumerate(resolutions):
            for j, target_time in enumerate(times):
                matching_data = successful_df[
                    (successful_df['resolution_str'] == res_str) & 
                    (abs(successful_df['actual_time'] - target_time) < 0.1)
                ]
                if len(matching_data) > 0:
                    dim_grid[i, j] = matching_data.iloc[0]['fractal_dim']
        
        ax1.remove()
        ax1 = fig.add_subplot(221, projection='3d')
        
        # Plot surface where data exists
        mask = ~np.isnan(dim_grid)
        if np.any(mask):
            surf = ax1.plot_surface(time_grid, res_grid, dim_grid, cmap='viridis', alpha=0.8)
            ax1.set_xlabel('Time')
            ax1.set_ylabel('Total Cells')
            ax1.set_zlabel('Fractal Dimension')
            ax1.set_title('D(Time, Resolution) Surface')
            fig.colorbar(surf, ax=ax1, shrink=0.5)
        
    except ImportError:
        # Fallback: 2D heatmap
        print("   3D plotting not available, using 2D heatmap")
        
        # Create pivot table
        pivot_data = successful_df.pivot_table(
            values='fractal_dim', 
            index='resolution_str', 
            columns='actual_time', 
            aggfunc='mean'
        )
        
        im = ax1.imshow(pivot_data.values, cmap='viridis', aspect='auto')
        ax1.set_xticks(range(len(pivot_data.columns)))
        ax1.set_xticklabels([f"{t:.1f}" for t in pivot_data.columns])
        ax1.set_yticks(range(len(pivot_data.index)))
        ax1.set_yticklabels(pivot_data.index)
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Resolution')
        ax1.set_title('Fractal Dimension Heatmap')
        plt.colorbar(im, ax=ax1)
    
    # Plot 2: Fractal dimension evolution for different resolutions
    colors = plt.cm.viridis(np.linspace(0, 1, len(resolutions)))
    
    for i, resolution_str in enumerate(resolutions):
        res_data = successful_df[successful_df['resolution_str'] == resolution_str]
        nx, ny = parse_grid_resolution(resolution_str)
        grid_str = format_grid_resolution(nx, ny)
        
        if len(res_data) > 0:
            ax2.errorbar(res_data['actual_time'], res_data['fractal_dim'], 
                        yerr=res_data['fd_error'], fmt='o-', capsize=3, 
                        color=colors[i], linewidth=2, markersize=6, 
                        label=grid_str)
    
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Fractal Dimension')
    ax2.set_title('Fractal Dimension Evolution (Multiple Resolutions)')
    ax2.grid(True, alpha=0.7)
    ax2.legend()
    
    # Plot 3: Resolution convergence at different times
    if len(times) > 1:
        time_colors = plt.cm.plasma(np.linspace(0, 1, min(len(times), 5)))  # Limit colors
        time_subset = times[::max(1, len(times)//5)]  # Show up to 5 times
        
        for i, target_time in enumerate(time_subset):
            time_data = []
            for resolution_str in resolutions:
                res_time_data = successful_df[
                    (successful_df['resolution_str'] == resolution_str) & 
                    (abs(successful_df['actual_time'] - target_time) < 0.1)
                ]
                if len(res_time_data) > 0:
                    closest_match = res_time_data.iloc[
                        np.argmin(abs(res_time_data['actual_time'] - target_time))
                    ]
                    time_data.append(closest_match)
            
            if time_data:
                time_df = pd.DataFrame(time_data)
                ax3.errorbar(time_df['effective_resolution'], time_df['fractal_dim'], 
                           yerr=time_df['fd_error'], fmt='s-', capsize=3,
                           color=time_colors[i], linewidth=2, markersize=6,
                           label=f't ≈ {target_time:.1f}')
        
        ax3.set_xscale('log', base=2)
        ax3.set_xlabel('Effective Resolution')
        ax3.set_ylabel('Fractal Dimension')
        ax3.set_title('Resolution Convergence at Different Times')
        ax3.grid(True, alpha=0.7)
        ax3.legend()
    else:
        # Single time - show segment scaling
        ax3.loglog(successful_df['effective_resolution'], successful_df['segments'], 
                  'mo-', linewidth=2, markersize=8, label='Interface Segments')
        ax3.set_xlabel('Effective Resolution')
        ax3.set_ylabel('Number of Interface Segments')
        ax3.set_title('Interface Complexity Scaling')
        ax3.grid(True, alpha=0.7)
        ax3.legend()
    
    # Plot 4: Analysis quality matrix
    # Create quality score based on R² and segment count
    successful_df['quality_score'] = successful_df['fd_r_squared'] * np.log10(successful_df['segments'])
    
    for i, resolution_str in enumerate(resolutions):
        res_data = successful_df[successful_df['resolution_str'] == resolution_str]
        nx, ny = parse_grid_resolution(resolution_str)
        grid_str = format_grid_resolution(nx, ny)
        
        if len(res_data) > 0:
            ax4.plot(res_data['actual_time'], res_data['quality_score'], 
                    'o-', color=colors[i], linewidth=2, markersize=6,
                    label=grid_str)
    
    ax4.set_xlabel('Time')
    ax4.set_ylabel('Quality Score (R² × log₁₀(segments))')
    ax4.set_title('Analysis Quality Evolution')
    ax4.grid(True, alpha=0.7)
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'matrix_analysis{method_suffix}.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   ✅ Saved matrix analysis plots")

def create_hybrid_plots(df, output_dir, analysis_params):
    """Create appropriate plots based on analysis mode."""
    method_name, method_suffix, method_description = get_method_info(analysis_params)
    mode = df['analysis_mode'].iloc[0] if 'analysis_mode' in df else 'unknown'
    
    if mode == 'temporal_evolution':
        create_temporal_evolution_plots(df, output_dir, method_suffix)
    elif mode == 'convergence_study':
        create_convergence_plots(df, output_dir, method_suffix)
    elif mode == 'matrix_analysis':
        create_matrix_plots(df, output_dir, method_suffix)
    else:
        print(f"⚠️  Unknown analysis mode: {mode}")

def main():
    """Main function with comprehensive argument parsing."""
    parser = argparse.ArgumentParser(
        description='Hybrid Parallel Resolution Analyzer - Comprehensive tool for temporal evolution and convergence analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
🎯 ANALYSIS MODES (automatically detected):
  Temporal Evolution: Single resolution, multiple times
  Convergence Study:  Multiple resolutions, single time  
  Matrix Analysis:    Multiple resolutions, multiple times

📐 GRID SUPPORT:
  Square grids:       --resolutions 200 400 800
  Rectangular grids:  --resolutions 160x200 320x400 640x800
  Mixed grids:        --resolutions 200 160x200 400x400 320x400

⚡ INTERFACE EXTRACTION METHODS:
  Standard:           Default scikit-image method
  Precision:          --use-conrec (CONREC algorithm)
  Theoretical:        --use-plic (PLIC reconstruction)

📊 EXAMPLES:

# Temporal evolution - single rectangular grid
python hybrid_parallel_analyzer.py \\
  --data-dirs ~/RT/160x200 \\
  --resolutions 160x200 \\
  --target-times 1.0 2.0 3.0 4.0 5.0 \\
  --use-plic

# Convergence study - multiple square grids  
python hybrid_parallel_analyzer.py \\
  --data-dirs ~/RT/100 ~/RT/200 ~/RT/400 ~/RT/800 \\
  --resolutions 100 200 400 800 \\
  --target-times 9.0 \\
  --use-conrec

# Matrix analysis - mixed grid types and times
python hybrid_parallel_analyzer.py \\
  --data-dirs ~/RT/160x200 ~/RT/200 ~/RT/320x400 ~/RT/400 \\
  --resolutions 160x200 200 320x400 400 \\
  --target-times 2.0 4.0 6.0 8.0 \\
  --processes 8

# Dalziel validation - rectangular grid series
python hybrid_parallel_analyzer.py \\
  --data-dirs ~/RT/160x200 ~/RT/320x400 ~/RT/640x800 ~/RT/1280x1600 \\
  --resolutions 160x200 320x400 640x800 1280x1600 \\
  --target-times 9.0 \\
  --mixing-method dalziel --use-conrec
""")
    
    # Required arguments
    parser.add_argument('--data-dirs', nargs='+', required=True,
                       help='Data directories (one per resolution)')
    parser.add_argument('--resolutions', nargs='+', required=True,
                       help='Grid resolutions (e.g., "200" for square, "160x200" for rectangular)')
    parser.add_argument('--target-times', nargs='+', type=float, required=True,
                       help='Target simulation times for analysis')
    
    # Optional arguments
    parser.add_argument('--output-dir', default=None,
                       help='Output directory (default: auto-generated based on analysis mode)')
    parser.add_argument('--mixing-method', default='dalziel',
                       choices=['geometric', 'statistical', 'dalziel'],
                       help='Mixing thickness calculation method (default: dalziel)')
    parser.add_argument('--h0', type=float, default=0.5,
                       help='Initial interface position (default: 0.5)')
    parser.add_argument('--min-box-size', type=float, default=None,
                       help='Minimum box size for fractal analysis (default: auto-estimate)')
    parser.add_argument('--time-tolerance', type=float, default=0.5,
                       help='Maximum time difference allowed when finding files (default: 0.5)')
    
    # Parallel processing
    parser.add_argument('--processes', type=int, default=None,
                       help='Number of parallel processes (default: auto-detect based on mode)')
    
    # Interface extraction methods
    parser.add_argument('--use-conrec', action='store_true',
                       help='Use CONREC precision interface extraction')
    parser.add_argument('--use-plic', action='store_true',
                       help='Use PLIC theoretical interface reconstruction')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug output for extraction methods')
    
    # Output control
    parser.add_argument('--no-plots', action='store_true',
                       help='Skip plot generation')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output')
    
    args = parser.parse_args()
    
    # Validate arguments
    if len(args.data_dirs) != len(args.resolutions):
        print("❌ Number of data directories must match number of resolutions")
        return 1
    
    # Check data directories
    for data_dir in args.data_dirs:
        if not os.path.exists(data_dir):
            print(f"❌ Data directory not found: {data_dir}")
            return 1
    
    # Validate resolution formats
    try:
        for res_str in args.resolutions:
            nx, ny = parse_grid_resolution(res_str)
            if args.verbose:
                print(f"Parsed resolution {res_str}: {nx}×{ny}")
    except ValueError as e:
        print(f"❌ {e}")
        return 1
    
    # Validate extraction method conflicts
    if args.use_conrec and args.use_plic:
        print("⚠️  Both CONREC and PLIC specified. PLIC will take precedence.")
        args.use_conrec = False
    
    # Validate number of processes
    if args.processes is not None:
        if args.processes < 1:
            print(f"❌ Number of processes must be at least 1")
            return 1
        if args.processes > cpu_count():
            print(f"⚠️  Requested {args.processes} processes, but only {cpu_count()} CPUs available")
    
    # Set analysis parameters
    analysis_params = {
        'mixing_method': args.mixing_method,
        'h0': args.h0,
        'min_box_size': args.min_box_size,
        'time_tolerance': args.time_tolerance,
        'use_conrec': args.use_conrec,
        'use_plic': args.use_plic,
        'debug': args.debug,
        'verbose': args.verbose
    }
    
    # Run hybrid analysis
    df = run_hybrid_analysis(
        args.data_dirs,
        args.resolutions,
        args.target_times,
        args.output_dir,
        analysis_params,
        args.processes
    )
    
    if df is None:
        print("❌ Analysis failed")
        return 1
    
    # Create plots unless disabled
    if not args.no_plots:
        try:
            # Determine output directory from DataFrame if not explicitly set
            if 'analysis_mode' in df.columns:
                output_dir = args.output_dir
                if output_dir is None:
                    method_name, method_suffix, _ = get_method_info(analysis_params)
                    mode = df['analysis_mode'].iloc[0]
                    output_dir = create_output_directory_name(mode, args.resolutions, args.target_times, method_suffix)
                
                create_hybrid_plots(df, output_dir, analysis_params)
            else:
                print("⚠️  Could not determine analysis mode for plotting")
        except Exception as e:
            print(f"⚠️  Plot generation failed: {str(e)}")
            if args.verbose:
                import traceback
                traceback.print_exc()
    
    # Final summary
    successful_count = len(df[df['status'] == 'success']) if 'status' in df.columns else len(df)
    mode = df['analysis_mode'].iloc[0] if 'analysis_mode' in df.columns else 'unknown'
    method_name, _, _ = get_method_info(analysis_params)
    
    print(f"\n🎉 Hybrid analysis complete!")
    print(f"Mode: {mode.replace('_', ' ').title()}")
    print(f"Method: {method_name}")
    print(f"Success rate: {successful_count}/{len(df)}")
    
    # Grid type summary
    if 'is_rectangular' in df.columns:
        grid_info = analyze_grid_types(args.resolutions)
        if grid_info['has_mixed_types']:
            print(f"Grid types: {grid_info['square_count']} square, {grid_info['rectangular_count']} rectangular")
        elif grid_info['rectangular_count'] > 0:
            print(f"Grid types: All rectangular (max aspect ratio: {grid_info['max_aspect_ratio']:.2f})")
        else:
            print(f"Grid types: All square")
    
    print(f"Check the output directory for detailed results and plots.")
    
    return 0

if __name__ == "__main__":
    exit(main())

