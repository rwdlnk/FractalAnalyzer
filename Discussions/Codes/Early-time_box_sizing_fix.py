# Add this method to RTAnalyzer class in rt_analyzer.py

def determine_early_time_min_box_size(self, mixing_thickness, grid_spacing, time, segments):
    """
    Special box sizing for early-time RT interfaces when mixing layer is very thin.
    
    Args:
        mixing_thickness: Current mixing layer thickness
        grid_spacing: Grid spacing (Δx)
        time: Simulation time
        segments: Interface segments
        
    Returns:
        float: Appropriate min_box_size for early-time analysis
    """
    print(f"\n🔬 EARLY-TIME BOX SIZING ANALYSIS")
    print(f"   Time: {time:.3f}")
    print(f"   Mixing thickness: {mixing_thickness:.6f}")
    print(f"   Grid spacing: {grid_spacing:.8f}")
    
    # Check if this is an early-time thin interface
    cells_across_mixing = mixing_thickness / grid_spacing if grid_spacing > 0 else 0
    
    # EARLY TIME CRITERIA (t < 3.0 AND thin mixing layer)
    is_early_time = time < 3.0 and cells_across_mixing < 10
    
    if is_early_time:
        print(f"   → EARLY TIME DETECTED: {cells_across_mixing:.1f} cells across mixing")
        
        # Strategy 1: Base on mixing thickness rather than grid spacing
        if mixing_thickness > 0:
            # Use a fraction of the mixing thickness to ensure we can capture the interface
            mixing_based_size = mixing_thickness / 8  # 8 boxes across mixing layer
            print(f"   → Mixing-based size: {mixing_based_size:.8f} ({mixing_thickness:.6f}/8)")
        else:
            mixing_based_size = grid_spacing  # Fallback for zero mixing
        
        # Strategy 2: Segment-based sizing for very thin interfaces
        if segments:
            # Find the scale of interface variations
            y_coords = []
            for (x1, y1), (x2, y2) in segments:
                y_coords.extend([y1, y2])
            
            if len(y_coords) > 2:
                y_range = max(y_coords) - min(y_coords)
                segment_based_size = y_range / 20  # 20 boxes across interface variations
                print(f"   → Segment-based size: {segment_based_size:.8f} (y_range={y_range:.6f}/20)")
            else:
                segment_based_size = mixing_based_size
        else:
            segment_based_size = mixing_based_size
        
        # Strategy 3: Grid-based minimum (but much smaller safety factor)
        grid_based_size = grid_spacing * 1.0  # Just 1× grid spacing for early times
        print(f"   → Grid-based size: {grid_based_size:.8f} (1.0×Δx)")
        
        # Choose the smallest reasonable size that's not too extreme
        candidate_sizes = [mixing_based_size, segment_based_size, grid_based_size]
        
        # Filter out any that are too small (less than 0.5× grid spacing)
        min_allowed = grid_spacing * 0.5
        valid_sizes = [s for s in candidate_sizes if s >= min_allowed]
        
        if valid_sizes:
            early_min_box_size = min(valid_sizes)
        else:
            # Fallback if all are too small
            early_min_box_size = min_allowed
        
        print(f"   → SELECTED early-time min_box_size: {early_min_box_size:.8f}")
        
        # Validate scaling range
        domain_size = 1.0
        max_box_size = domain_size / 2
        scaling_decades = np.log10(max_box_size / early_min_box_size)
        
        print(f"   → Early-time scaling range: {scaling_decades:.2f} decades")
        
        if scaling_decades < 1.5:
            # For very early times, this might be acceptable
            print(f"   → Warning: Limited scaling range for early time, but may be unavoidable")
        
        return early_min_box_size
    
    else:
        print(f"   → NOT early time - using standard approach")
        return None  # Use standard approach


# Modify the analyze_vtk_file method to use this for early times
# Replace the min_box_size determination section with:

if segments:
    # Check if this needs early-time special handling
    early_min_box_size = self.determine_early_time_min_box_size(
        mixing['h_total'], grid_spacing, data['time'], segments)
    
    if early_min_box_size is not None:
        # Use early-time sizing
        optimal_min_box_size = early_min_box_size
        print(f"Using early-time min_box_size: {optimal_min_box_size:.8f}")
    
    elif min_box_size is None and 'suggested_min_box_size' in validity_status['metrics']:
        # Use validity-recommended min_box_size for normal times
        optimal_min_box_size = validity_status['metrics']['suggested_min_box_size']
        print(f"Using validity-recommended min_box_size: {optimal_min_box_size:.8f}")
    
    else:
        # Use existing physics-based approach
        optimal_min_box_size = self.determine_optimal_min_box_size(
            vtk_file, segments, min_box_size)
    
    # Now compute fractal dimension with the optimal box size
    fd_results = self.compute_fractal_dimension(data, min_box_size=optimal_min_box_size)
