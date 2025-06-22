# Enhanced validity check methods to add to RTAnalyzer class in rt_analyzer.py

def _estimate_grid_spacing(self, data):
    """
    Estimate grid spacing from VTK data structure.
    
    Args:
        data: VTK data dictionary containing x, y grids
        
    Returns:
        float: Estimated grid spacing (Δx)
    """
    try:
        # Extract x-coordinates from the first row
        x_coords = data['x'][:, 0] if len(data['x'].shape) > 1 else data['x']
        
        # Calculate spacing (should be uniform for structured grids)
        if len(x_coords) > 1:
            dx = np.abs(x_coords[1] - x_coords[0])
            
            # Verify uniformity (check a few more points)
            if len(x_coords) > 3:
                dx_check = np.abs(x_coords[2] - x_coords[1])
                if abs(dx - dx_check) / dx > 0.01:  # More than 1% difference
                    print(f"Warning: Non-uniform grid spacing detected")
                    print(f"  First spacing: {dx:.8f}, Second: {dx_check:.8f}")
            
            return dx
        else:
            print("Warning: Cannot determine grid spacing - insufficient grid points")
            return None
            
    except Exception as e:
        print(f"Warning: Error estimating grid spacing: {e}")
        return None

def check_analysis_validity(self, resolution=None, mixing_thickness=None, grid_spacing=None, 
                          time=None, interface_segments=None):
    """
    Enhanced validity check accounting for multi-scale RT physics and time evolution.
    
    Args:
        resolution: Grid resolution (e.g., 800 for 800x800)
        mixing_thickness: Current mixing layer thickness
        grid_spacing: Grid spacing (Δx) from VTK data
        time: Simulation time
        interface_segments: Interface segments for additional validation
        
    Returns:
        dict: Comprehensive validity assessment
    """
    print(f"\n🔍 ANALYSIS VALIDITY CHECK")
    print(f"=" * 50)
    
    validity_status = {
        'overall_valid': True,
        'warnings': [],
        'critical_issues': [],
        'recommendations': [],
        'metrics': {}
    }
    
    # === BASIC RESOLUTION CHECK ===
    if mixing_thickness is not None and grid_spacing is not None:
        cells_across_mixing = mixing_thickness / grid_spacing
        validity_status['metrics']['cells_across_mixing'] = cells_across_mixing
        
        print(f"📏 Resolution Analysis:")
        print(f"   Mixing thickness: {mixing_thickness:.6f}")
        print(f"   Grid spacing (Δx): {grid_spacing:.8f}")
        print(f"   Cells across mixing: {cells_across_mixing:.1f}")
        
        # Critical threshold
        if cells_across_mixing < 5:
            validity_status['critical_issues'].append(
                f"CRITICAL: Only {cells_across_mixing:.1f} cells across mixing layer")
            validity_status['overall_valid'] = False
            validity_status['recommendations'].append(
                "Use higher resolution or analyze earlier times")
        
        # Warning thresholds
        elif cells_across_mixing < 10:
            validity_status['warnings'].append(
                f"Marginal resolution: {cells_across_mixing:.1f} cells across mixing")
            validity_status['recommendations'].append(
                "Consider higher resolution for better accuracy")
        
        elif cells_across_mixing < 15:
            validity_status['warnings'].append(
                f"Adequate but not optimal resolution: {cells_across_mixing:.1f} cells")
    
    # === TIME-DEPENDENT VALIDITY ===
    if time is not None:
        validity_status['metrics']['simulation_time'] = time
        
        print(f"⏰ Time-Dependent Analysis:")
        print(f"   Simulation time: {time:.3f}")
        
        # Late-time requirements (turbulent cascade development)
        if time >= 3.0:  # Late time when full cascade develops
            if mixing_thickness is not None and grid_spacing is not None:
                required_cells_late = 20  # Higher requirement for late times
                
                if cells_across_mixing < required_cells_late:
                    validity_status['warnings'].append(
                        f"Late-time warning (t={time:.1f}): Need >{required_cells_late} cells "
                        f"for developed turbulence, have {cells_across_mixing:.1f}")
                    validity_status['recommendations'].append(
                        "Late-time analysis requires higher resolution to capture turbulent cascade")
        
        # Early time considerations
        elif time < 1.0:
            validity_status['recommendations'].append(
                "Early time analysis - interface may not have developed sufficient complexity")
    
    # === MULTI-SCALE PHYSICS CHECK ===
    if resolution is not None and time is not None:
        validity_status['metrics']['resolution'] = resolution
        
        print(f"🌊 Multi-Scale Physics Analysis:")
        print(f"   Resolution: {resolution}x{resolution}")
        
        # Resolution-time coupling effects (based on your convergence findings)
        if time >= 2.5:  # When scale-dependent effects emerge
            if resolution == 200:
                validity_status['warnings'].append(
                    "Resolution resonance regime: 200x200 may capture dominant plume scale optimally")
                validity_status['recommendations'].append(
                    "Compare with higher resolution results to verify scale-dependent behavior")
            
            elif resolution == 400:
                validity_status['warnings'].append(
                    "Transition regime: 400x400 may be between large plumes and fine turbulence")
                validity_status['recommendations'].append(
                    "Consider this may be in intermediate resolution regime")
        
        # Very high resolution considerations
        if resolution >= 1600:
            validity_status['recommendations'].append(
                "High resolution analysis - capturing finest scales but computationally expensive")
    
    # === FRACTAL DIMENSION SPECIFIC CHECKS ===
    if interface_segments is not None:
        segment_count = len(interface_segments)
        validity_status['metrics']['interface_segments'] = segment_count
        
        print(f"🔄 Fractal Analysis Readiness:")
        print(f"   Interface segments: {segment_count}")
        
        if segment_count < 100:
            validity_status['warnings'].append(
                f"Low segment count ({segment_count}) may limit fractal dimension accuracy")
            validity_status['recommendations'].append(
                "Consider higher interface resolution or different contour level")
        
        elif segment_count > 50000:
            validity_status['recommendations'].append(
                f"Very high segment count ({segment_count}) - analysis will be thorough but slow")
    
    # === BOX-COUNTING PREPARATION ===
    if grid_spacing is not None:
        # Estimate appropriate min_box_size for fractal analysis
        suggested_min_box = 4 * grid_spacing  # Conservative safety factor
        validity_status['metrics']['suggested_min_box_size'] = suggested_min_box
        
        print(f"📦 Box-Counting Preparation:")
        print(f"   Suggested min box size: {suggested_min_box:.8f} ({4:.1f}×Δx)")
        
        # Check scaling range
        domain_size = 1.0  # Assume unit domain
        max_box_size = domain_size / 2
        scaling_decades = np.log10(max_box_size / suggested_min_box)
        validity_status['metrics']['scaling_decades'] = scaling_decades
        
        print(f"   Expected scaling range: {scaling_decades:.2f} decades")
        
        if scaling_decades < 1.5:
            validity_status['warnings'].append(
                f"Limited scaling range: {scaling_decades:.2f} decades may affect accuracy")
            validity_status['recommendations'].append(
                "Consider smaller min_box_size if computationally feasible")
    
    # === OVERALL ASSESSMENT ===
    print(f"\n📋 VALIDITY SUMMARY:")
    
    if validity_status['overall_valid']:
        if len(validity_status['warnings']) == 0:
            print(f"   ✅ EXCELLENT: Analysis conditions are optimal")
        elif len(validity_status['warnings']) <= 2:
            print(f"   ✅ GOOD: Analysis is valid with minor considerations")
        else:
            print(f"   ⚠️  ACCEPTABLE: Analysis is valid but has multiple warnings")
    else:
        print(f"   ❌ PROBLEMATIC: Critical issues detected")
    
    # Print warnings and recommendations
    if validity_status['warnings']:
        print(f"\n⚠️  Warnings ({len(validity_status['warnings'])}):")
        for i, warning in enumerate(validity_status['warnings'], 1):
            print(f"   {i}. {warning}")
    
    if validity_status['critical_issues']:
        print(f"\n❌ Critical Issues ({len(validity_status['critical_issues'])}):")
        for i, issue in enumerate(validity_status['critical_issues'], 1):
            print(f"   {i}. {issue}")
    
    if validity_status['recommendations']:
        print(f"\n💡 Recommendations ({len(validity_status['recommendations'])}):")
        for i, rec in enumerate(validity_status['recommendations'], 1):
            print(f"   {i}. {rec}")
    
    print(f"=" * 50)
    
    return validity_status

def get_validity_summary_for_output(self, validity_status):
    """
    Generate a concise validity summary for inclusion in output files.
    
    Args:
        validity_status: Dict returned from check_analysis_validity()
        
    Returns:
        str: Formatted summary string
    """
    lines = []
    lines.append("# ANALYSIS VALIDITY SUMMARY")
    lines.append(f"# Overall Status: {'VALID' if validity_status['overall_valid'] else 'INVALID'}")
    
    if 'cells_across_mixing' in validity_status['metrics']:
        lines.append(f"# Cells across mixing layer: {validity_status['metrics']['cells_across_mixing']:.1f}")
    
    if 'scaling_decades' in validity_status['metrics']:
        lines.append(f"# Fractal scaling range: {validity_status['metrics']['scaling_decades']:.2f} decades")
    
    if validity_status['warnings']:
        lines.append(f"# Warnings: {len(validity_status['warnings'])}")
        for warning in validity_status['warnings']:
            lines.append(f"#   - {warning}")
    
    if validity_status['critical_issues']:
        lines.append(f"# Critical Issues: {len(validity_status['critical_issues'])}")
        for issue in validity_status['critical_issues']:
            lines.append(f"#   - {issue}")
    
    return '\n'.join(lines)