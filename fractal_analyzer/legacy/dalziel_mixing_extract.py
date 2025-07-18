        def compute_mixing_thickness_dalziel_correct(self, data, h0):
            """
            Correct Dalziel mixing thickness implementation following JFM 1999 Equation 7.
    
            Finds positions where horizontally-averaged concentration crosses 0.05 and 0.95,
            then calculates h_{1,0} and h_{1,1} as per Dalziel et al. (1999).
    
            Args:
                data: VTK data dictionary containing 'f' (concentration) and 'y' (coordinates)
                h0: Initial interface position
        
            Returns:
                dict: Mixing thickness measurements with Dalziel-specific parameters
            """
            print(f"DEBUG: Computing Dalziel mixing thickness (CORRECTED)")
    
            # Horizontal average (along-tank averaging) - following Dalziel notation C̄(z)
            f_avg = np.mean(data['f'], axis=0)  # Average over x (first axis)
            y_values = data['y'][0, :]  # y-coordinates along first row
    
            print(f"DEBUG: f_avg shape: {f_avg.shape}")
            print(f"DEBUG: y_values shape: {y_values.shape}")
            print(f"DEBUG: f_avg range: [{np.min(f_avg):.3f}, {np.max(f_avg):.3f}]")
            print(f"DEBUG: y_values range: [{np.min(y_values):.3f}, {np.max(y_values):.3f}]")
            print(f"DEBUG: h0 = {h0:.6f}")
    
            # Dalziel thresholds
            lower_threshold = 0.05  # 5% threshold for h_{1,0}
            upper_threshold = 0.95  # 95% threshold for h_{1,1}
    
            # Find exact crossing points using interpolation
            try:
                crossings_005 = self.find_concentration_crossing(f_avg, y_values, lower_threshold)
                crossings_095 = self.find_concentration_crossing(f_avg, y_values, upper_threshold)
        
                print(f"DEBUG: Found {len(crossings_005)} crossings at f=0.05: {crossings_005}")
                print(f"DEBUG: Found {len(crossings_095)} crossings at f=0.95: {crossings_095}")
        
            except Exception as e:
                print(f"ERROR in crossing detection: {e}")
                return {
                    'ht': 0, 'hb': 0, 'h_total': 0,
                    'h_10': None, 'h_11': None,
                    'method': 'dalziel_error',
                    'error': str(e)
                }
    
            # Dalziel definition: h_{1,0} and h_{1,1}
            # h_{1,0}: Position where C̄ = 0.05 (lower boundary)
            # h_{1,1}: Position where C̄ = 0.95 (upper boundary)
    
            h_10 = None  # Lower boundary position
            h_11 = None  # Upper boundary position
    
            # Select appropriate crossings
            if crossings_005:
                # For RT mixing, we want the crossing closest to h0 from below
                # In case of multiple crossings, choose the one nearest to h0
                if len(crossings_005) == 1:
                    h_10 = crossings_005[0]
                else:
                    # Multiple crossings - select the one that makes physical sense
                    valid_crossings = [c for c in crossings_005 if c <= h0]  # Below initial interface
                    if valid_crossings:
                        h_10 = max(valid_crossings)  # Highest crossing below h0
                    else:
                        h_10 = min(crossings_005)  # Fallback to lowest crossing
        
                print(f"DEBUG: Selected h_10 = {h_10:.6f} (f=0.05 crossing)")
    
            if crossings_095:
                # For RT mixing, we want the crossing closest to h0 from above
                if len(crossings_095) == 1:
                    h_11 = crossings_095[0]
                else:
                    # Multiple crossings - select the one that makes physical sense
                    valid_crossings = [c for c in crossings_095 if c >= h0]  # Above initial interface
                    if valid_crossings:
                        h_11 = min(valid_crossings)  # Lowest crossing above h0
                    else:
                        h_11 = max(crossings_095)  # Fallback to highest crossing
        
                print(f"DEBUG: Selected h_11 = {h_11:.6f} (f=0.95 crossing)")
    
            # Calculate mixing thicknesses according to Dalziel methodology
            if h_10 is not None and h_11 is not None:
                # Upper mixing thickness: how far 95% contour extends above initial interface
                ht = max(0, h_11 - h0)
        
                # Lower mixing thickness: how far 5% contour extends below initial interface  
                hb = max(0, h0 - h_10)
        
                # Total mixing thickness
                h_total = ht + hb
        
                print(f"DEBUG: Dalziel mixing thickness calculation:")
                print(f"  h_10 (5% crossing): {h_10:.6f}")
                print(f"  h_11 (95% crossing): {h_11:.6f}")
                print(f"  h0 (initial interface): {h0:.6f}")
                print(f"  ht = max(0, {h_11:.6f} - {h0:.6f}) = {ht:.6f}")
                print(f"  hb = max(0, {h0:.6f} - {h_10:.6f}) = {hb:.6f}")
                print(f"  h_total = {ht:.6f} + {hb:.6f} = {h_total:.6f}")
        
                # Additional Dalziel-specific diagnostics
                mixing_zone_center = (h_10 + h_11) / 2
                mixing_zone_width = h_11 - h_10
                interface_offset = mixing_zone_center - h0
        
                # Mixing efficiency (fraction of domain that is mixed)
                mixing_region = (f_avg >= lower_threshold) & (f_avg <= upper_threshold)
                mixing_fraction = np.sum(mixing_region) / len(f_avg)
        
                return {
                    'ht': ht,
                    'hb': hb, 
                    'h_total': h_total,
                    'h_10': h_10,  # Position where C̄ = 0.05 (Dalziel h_{1,0})
                    'h_11': h_11,  # Position where C̄ = 0.95 (Dalziel h_{1,1})
                    'mixing_zone_center': mixing_zone_center,
                    'mixing_zone_width': mixing_zone_width,
                    'interface_offset': interface_offset,
                    'mixing_fraction': mixing_fraction,
                    'lower_threshold': lower_threshold,
                    'upper_threshold': upper_threshold,
                    'method': 'dalziel_corrected',
                    'crossings_005': crossings_005,  # All 5% crossings found
                    'crossings_095': crossings_095   # All 95% crossings found
                }
    
            else:
                # Handle case where crossings are not found
                print(f"WARNING: Could not find required concentration crossings")
                print(f"  5% crossings found: {len(crossings_005) if crossings_005 else 0}")
                print(f"  95% crossings found: {len(crossings_095) if crossings_095 else 0}")
        
                # Provide fallback using simple thresholding (closer to your original method)
                mixed_indices = np.where((f_avg >= lower_threshold) & (f_avg <= upper_threshold))[0]
        
                if len(mixed_indices) > 0:
                    # Fallback to extent-based calculation
                    mixed_y_min = y_values[mixed_indices[0]]
                    mixed_y_max = y_values[mixed_indices[-1]]
            
                    ht_fallback = max(0, mixed_y_max - h0)
                    hb_fallback = max(0, h0 - mixed_y_min)
                    h_total_fallback = ht_fallback + hb_fallback
            
                    print(f"  Using fallback extent method:")
                    print(f"    ht = {ht_fallback:.6f}, hb = {hb_fallback:.6f}")
            
                    return {
                        'ht': ht_fallback,
                        'hb': hb_fallback,
                        'h_total': h_total_fallback,
                        'h_10': mixed_y_min,  # Approximate
                        'h_11': mixed_y_max,  # Approximate
                        'mixing_fraction': len(mixed_indices) / len(f_avg),
                        'lower_threshold': lower_threshold,
                        'upper_threshold': upper_threshold,
                        'method': 'dalziel_fallback',
                        'warning': 'Used extent-based fallback due to missing crossings'
                    }
                else:
                    # No mixing detected at all
                    print(f"  No mixing zone detected between {lower_threshold} and {upper_threshold}")
                    return {
                        'ht': 0,
                        'hb': 0,
                        'h_total': 0,
                        'h_10': None,
                        'h_11': None,
                        'mixing_fraction': 0,
                        'lower_threshold': lower_threshold,
                        'upper_threshold': upper_threshold,
                        'method': 'dalziel_no_mixing'
                    }
