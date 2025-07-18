# Replace the physical constraints section in check_analysis_validity()

# Enhancement for fractal_analyzer.py analyze_linear_region method
# Find the window with dimension closest to theoretical or best R² WITH PHYSICAL CONSTRAINTS
if theoretical_dimension is not None:
    closest_idx = np.argmin(np.abs(np.array(dimensions) - theoretical_dimension))
else:
    # For RT interfaces, apply TIME-DEPENDENT physical constraints:
    
    valid_indices = []
    
    # EARLY TIME CONSTRAINTS (t < 2.0): Allow nearly smooth interfaces
    if time is not None and time < 2.0:
        print(f"Applying early-time constraints (t={time:.1f}):")
        for i, (window, dim, r2) in enumerate(zip(windows, dimensions, r_squared)):
            # Early time: Allow D from 1.0 to 1.5, lower R² requirements
            if 1.0 <= dim <= 1.5:
                if window >= 3:  # Reduced from 4
                    if r2 >= 0.95:  # Reduced from 0.99
                        valid_indices.append(i)
        
        print(f"  Early-time valid range: 1.0 ≤ D ≤ 1.5, R² ≥ 0.95, window ≥ 3")
    
    # INTERMEDIATE TIME CONSTRAINTS (2.0 ≤ t < 4.0): Developing complexity  
    elif time is not None and 2.0 <= time < 4.0:
        print(f"Applying intermediate-time constraints (t={time:.1f}):")
        for i, (window, dim, r2) in enumerate(zip(windows, dimensions, r_squared)):
            # Intermediate: Allow broader range as complexity develops
            if 1.0 <= dim <= 1.8:
                if window >= 4:
                    if r2 >= 0.98:  # Slightly relaxed
                        valid_indices.append(i)
        
        print(f"  Intermediate-time valid range: 1.0 ≤ D ≤ 1.8, R² ≥ 0.98, window ≥ 4")
    
    # LATE TIME CONSTRAINTS (t ≥ 4.0): Full turbulent cascade
    else:
        print(f"Applying late-time constraints (t={time:.1f if time else 'unknown'}):")
        for i, (window, dim, r2) in enumerate(zip(windows, dimensions, r_squared)):
            # Late time: Full physical range, strict quality requirements
            if 1.0 <= dim <= 2.0:
                if window >= 4:
                    if r2 >= 0.99:
                        valid_indices.append(i)
        
        print(f"  Late-time valid range: 1.0 ≤ D ≤ 2.0, R² ≥ 0.99, window ≥ 4")

    if valid_indices:
        # Among valid windows, prefer larger windows when R² is close
        # Sort by R² (descending), then by window size (descending) as tiebreaker
        valid_data = [(r_squared[i], windows[i], i) for i in valid_indices]
        valid_data.sort(key=lambda x: (x[0], x[1]), reverse=True)

        # Select the best valid window
        closest_idx = valid_data[0][2]

        print(f"  Valid windows: {[windows[i] for i in valid_indices]}")
        print(f"  Valid dimensions: {[f'{dimensions[i]:.3f}' for i in valid_indices]}")
        print(f"  Valid R²: {[f'{r_squared[i]:.6f}' for i in valid_indices]}")
        print(f"  Selected window {windows[closest_idx]} (D={dimensions[closest_idx]:.6f})")
    else:
        # Fallback: use best R² without constraints (but warn)
        closest_idx = np.argmax(r_squared)
        print(f"WARNING: No windows met time-dependent constraints. Using best R² = {r_squared[closest_idx]:.6f}")
        print(f"WARNING: Dimension {dimensions[closest_idx]:.6f} may need manual verification for t={time:.1f if time else 'unknown'}")
        
        # For very early times with simple interfaces, this might actually be correct
        if time is not None and time < 1.0:
            print(f"NOTE: For very early time (t={time:.1f}), simple interface with D≈{dimensions[closest_idx]:.3f} may be physically correct")
