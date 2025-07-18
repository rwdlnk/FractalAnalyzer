    def compute_multifractal_spectrum(self, data, min_box_size=0.001, q_values=None, output_dir=None):
        """Compute multifractal spectrum of the interface.
        
        Args:
            data: Data dictionary containing VTK data
            min_box_size: Minimum box size for analysis (default: 0.001)
            q_values: List of q moments to analyze (default: -5 to 5 in 0.5 steps)
            output_dir: Directory to save results (default: None)
            
        Returns:
            dict: Multifractal spectrum results
        """
        if self.fractal_analyzer is None:
            print("Fractal analyzer not available. Skipping multifractal analysis.")
            return None
        
        # Set default q values if not provided
        if q_values is None:
            q_values = np.arange(-5, 5.1, 0.5)
        
        # Extract contours and convert to segments
        contours = self.extract_interface(data['f'], data['x'], data['y'])
        segments = self.convert_contours_to_segments(contours)
        
        if not segments:
            print("No interface segments found. Skipping multifractal analysis.")
            return None
        
        # Create output directory if specified and not existing
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Calculate extent for max box size
        min_x = min(min(s[0][0], s[1][0]) for s in segments)
        max_x = max(max(s[0][0], s[1][0]) for s in segments)
        min_y = min(min(s[0][1], s[1][1]) for s in segments)
        max_y = max(max(s[0][1], s[1][1]) for s in segments)
        
        extent = max(max_x - min_x, max_y - min_y)
        max_box_size = extent / 2
        
        print(f"Performing multifractal analysis with {len(q_values)} q-values")
        print(f"Box size range: {min_box_size:.6f} to {max_box_size:.6f}")
        
        # Generate box sizes
        box_sizes = []
        current_size = max_box_size
        box_size_factor = 1.5
        
        while current_size >= min_box_size:
            box_sizes.append(current_size)
            current_size /= box_size_factor
            
        box_sizes = np.array(box_sizes)
        num_box_sizes = len(box_sizes)
        
        print(f"Using {num_box_sizes} box sizes for analysis")
        
        # Use spatial index from BoxCounter to speed up calculations
        bc = self.fractal_analyzer.box_counter
        
        # Add small margin to bounding box
        margin = extent * 0.01
        min_x -= margin
        max_x += margin
        min_y -= margin
        max_y += margin
        
        # Create spatial index for segments
        start_time = time.time()
        print("Creating spatial index...")
        
        # Determine grid cell size for spatial index (use smallest box size)
        grid_size = min_box_size * 2
        segment_grid, grid_width, grid_height = bc.create_spatial_index(
            segments, min_x, min_y, max_x, max_y, grid_size)
        
        print(f"Spatial index created in {time.time() - start_time:.2f} seconds")
        
        # Initialize data structures for box counting
        all_box_counts = []
        all_probabilities = []
        
        # Analyze each box size
        for box_idx, box_size in enumerate(box_sizes):
            box_start_time = time.time()
            print(f"Processing box size {box_idx+1}/{num_box_sizes}: {box_size:.6f}")
            
            num_boxes_x = int(np.ceil((max_x - min_x) / box_size))
            num_boxes_y = int(np.ceil((max_y - min_y) / box_size))
            
            # Count segments in each box
            box_counts = np.zeros((num_boxes_x, num_boxes_y))
            
            for i in range(num_boxes_x):
                for j in range(num_boxes_y):
                    box_xmin = min_x + i * box_size
                    box_ymin = min_y + j * box_size
                    box_xmax = box_xmin + box_size
                    box_ymax = box_ymin + box_size
                    
                    # Find grid cells that might overlap this box
                    min_cell_x = max(0, int((box_xmin - min_x) / grid_size))
                    max_cell_x = min(int((box_xmax - min_x) / grid_size) + 1, grid_width)
                    min_cell_y = max(0, int((box_ymin - min_y) / grid_size))
                    max_cell_y = min(int((box_ymax - min_y) / grid_size) + 1, grid_height)
                    
                    # Get segments that might intersect this box
                    segments_to_check = set()
                    for cell_x in range(min_cell_x, max_cell_x):
                        for cell_y in range(min_cell_y, max_cell_y):
                            segments_to_check.update(segment_grid.get((cell_x, cell_y), []))
                    
                    # Count intersections (for multifractal, count each segment)
                    count = 0
                    for seg_idx in segments_to_check:
                        (x1, y1), (x2, y2) = segments[seg_idx]
                        if self.fractal_analyzer.base.liang_barsky_line_box_intersection(
                                x1, y1, x2, y2, box_xmin, box_ymin, box_xmax, box_ymax):
                            count += 1
                    
                    box_counts[i, j] = count
            
            # Keep only non-zero counts and calculate probabilities
            occupied_boxes = box_counts[box_counts > 0]
            total_segments = occupied_boxes.sum()
            
            if total_segments > 0:
                probabilities = occupied_boxes / total_segments
            else:
                probabilities = np.array([])
                
            all_box_counts.append(occupied_boxes)
            all_probabilities.append(probabilities)
            
            # Report statistics
            box_count = len(occupied_boxes)
            print(f"  Box size: {box_size:.6f}, Occupied boxes: {box_count}, Time: {time.time() - box_start_time:.2f}s")
        
        # Calculate multifractal properties
        print("Calculating multifractal spectrum...")
        
        taus = np.zeros(len(q_values))
        Dqs = np.zeros(len(q_values))
        r_squared = np.zeros(len(q_values))
        
        for q_idx, q in enumerate(q_values):
            print(f"Processing q = {q:.1f}")
            
            # Skip q=1 as it requires special treatment
            if abs(q - 1.0) < 1e-6:
                continue
                
            # Calculate partition function for each box size
            Z_q = np.zeros(num_box_sizes)
            
            for i, probabilities in enumerate(all_probabilities):
                if len(probabilities) > 0:
                    Z_q[i] = np.sum(probabilities ** q)
                else:
                    Z_q[i] = np.nan
            
            # Remove NaN values
            valid = ~np.isnan(Z_q)
            if np.sum(valid) < 3:
                print(f"Warning: Not enough valid points for q={q}")
                taus[q_idx] = np.nan
                Dqs[q_idx] = np.nan
                r_squared[q_idx] = np.nan
                continue
                
            log_eps = np.log(box_sizes[valid])
            log_Z_q = np.log(Z_q[valid])
            
            # Linear regression to find tau(q)
            slope, intercept, r_value, p_value, std_err = stats.linregress(log_eps, log_Z_q)
            
            # Calculate tau(q) and D(q)
            taus[q_idx] = slope
            Dqs[q_idx] = taus[q_idx] / (q - 1) if q != 1 else np.nan
            r_squared[q_idx] = r_value ** 2
            
            print(f"  τ({q}) = {taus[q_idx]:.4f}, D({q}) = {Dqs[q_idx]:.4f}, R² = {r_squared[q_idx]:.4f}")
        
        # Handle q=1 case (information dimension) separately
        q1_idx = np.where(np.abs(q_values - 1.0) < 1e-6)[0]
        if len(q1_idx) > 0:
            q1_idx = q1_idx[0]
            print(f"Processing q = 1.0 (information dimension)")
            
            # Calculate using L'Hôpital's rule
            mu_log_mu = np.zeros(num_box_sizes)
            
            for i, probabilities in enumerate(all_probabilities):
                if len(probabilities) > 0:
                    # Use -sum(p*log(p)) for information dimension
                    mu_log_mu[i] = -np.sum(probabilities * np.log(probabilities))
                else:
                    mu_log_mu[i] = np.nan
            
            # Remove NaN values
            valid = ~np.isnan(mu_log_mu)
            if np.sum(valid) >= 3:
                log_eps = np.log(box_sizes[valid])
                log_mu = mu_log_mu[valid]
                
                # Linear regression
                slope, intercept, r_value, p_value, std_err = stats.linregress(log_eps, log_mu)
                
                # Store information dimension
                taus[q1_idx] = -slope  # Convention: τ(1) = -D₁
                Dqs[q1_idx] = -slope   # Information dimension D₁
                r_squared[q1_idx] = r_value ** 2
                
                print(f"  τ(1) = {taus[q1_idx]:.4f}, D(1) = {Dqs[q1_idx]:.4f}, R² = {r_squared[q1_idx]:.4f}")
        
        # Calculate alpha and f(alpha) for multifractal spectrum
        alpha = np.zeros(len(q_values))
        f_alpha = np.zeros(len(q_values))
        
        print("Calculating multifractal spectrum f(α)...")
        
        for i, q in enumerate(q_values):
            if np.isnan(taus[i]):
                alpha[i] = np.nan
                f_alpha[i] = np.nan
                continue
                
            # Numerical differentiation for alpha
            if i > 0 and i < len(q_values) - 1:
                alpha[i] = -(taus[i+1] - taus[i-1]) / (q_values[i+1] - q_values[i-1])
            elif i == 0:
                alpha[i] = -(taus[i+1] - taus[i]) / (q_values[i+1] - q_values[i])
            else:
                alpha[i] = -(taus[i] - taus[i-1]) / (q_values[i] - q_values[i-1])
            
            # Calculate f(alpha)
            f_alpha[i] = q * alpha[i] + taus[i]
            
            print(f"  q = {q:.1f}, α = {alpha[i]:.4f}, f(α) = {f_alpha[i]:.4f}")
        
        # Calculate multifractal parameters
        valid_idx = ~np.isnan(Dqs)
        if np.sum(valid_idx) >= 3:
            D0 = Dqs[np.searchsorted(q_values, 0)] if 0 in q_values else np.nan
            D1 = Dqs[np.searchsorted(q_values, 1)] if 1 in q_values else np.nan
            D2 = Dqs[np.searchsorted(q_values, 2)] if 2 in q_values else np.nan
            
            # Width of multifractal spectrum
            valid = ~np.isnan(alpha)
            if np.sum(valid) >= 2:
                alpha_width = np.max(alpha[valid]) - np.min(alpha[valid])
            else:
                alpha_width = np.nan
            
            # Degree of multifractality: D(-∞) - D(+∞) ≈ D(-5) - D(5)
            if -5 in q_values and 5 in q_values:
                degree_multifractality = Dqs[np.searchsorted(q_values, -5)] - Dqs[np.searchsorted(q_values, 5)]
            else:
                degree_multifractality = np.nan
            
            print(f"Multifractal parameters:")
            print(f"  D(0) = {D0:.4f} (capacity dimension)")
            print(f"  D(1) = {D1:.4f} (information dimension)")
            print(f"  D(2) = {D2:.4f} (correlation dimension)")
            print(f"  α width = {alpha_width:.4f}")
            print(f"  Degree of multifractality = {degree_multifractality:.4f}")
        else:
            D0 = D1 = D2 = alpha_width = degree_multifractality = np.nan
            print("Warning: Not enough valid points to calculate multifractal parameters")
        
        # Plot results if output directory provided
        if output_dir:
            # Plot D(q) vs q
            plt.figure(figsize=(10, 6))
            valid = ~np.isnan(Dqs)
            plt.plot(q_values[valid], Dqs[valid], 'bo-', markersize=4)
            
            if 0 in q_values:
                plt.axhline(y=Dqs[np.searchsorted(q_values, 0)], color='r', linestyle='--', 
                           label=f"D(0) = {Dqs[np.searchsorted(q_values, 0)]:.4f}")
            
            plt.xlabel('q')
            plt.ylabel('D(q)')
            plt.title(f'Generalized Dimensions D(q) at t = {data["time"]:.2f}')
            plt.grid(True)
            plt.legend()
            plt.savefig(os.path.join(output_dir, "multifractal_dimensions.png"), dpi=300)
            plt.close()
            
            # Plot f(alpha) vs alpha (multifractal spectrum)
            plt.figure(figsize=(10, 6))
            valid = ~np.isnan(alpha) & ~np.isnan(f_alpha)
            plt.plot(alpha[valid], f_alpha[valid], 'bo-', markersize=4)
            
            # Add selected q values as annotations
            q_to_highlight = [-5, -2, 0, 2, 5]
            for q_val in q_to_highlight:
                if q_val in q_values:
                    idx = np.searchsorted(q_values, q_val)
                    if idx < len(q_values) and valid[idx]:
                        plt.annotate(f"q={q_values[idx]}", 
                                    (alpha[idx], f_alpha[idx]),
                                    xytext=(5, 0), textcoords='offset points')
            
            plt.xlabel('α')
            plt.ylabel('f(α)')
            plt.title(f'Multifractal Spectrum f(α) at t = {data["time"]:.2f}')
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, "multifractal_spectrum.png"), dpi=300)
            plt.close()
            
            # Plot R² values
            plt.figure(figsize=(10, 6))
            valid = ~np.isnan(r_squared)
            plt.plot(q_values[valid], r_squared[valid], 'go-', markersize=4)
            plt.xlabel('q')
            plt.ylabel('R²')
            plt.title(f'Fit Quality for Different q Values at t = {data["time"]:.2f}')
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, "multifractal_r_squared.png"), dpi=300)
            plt.close()
            
            # Save results to CSV
            import pandas as pd
            results_df = pd.DataFrame({
                'q': q_values,
                'tau': taus,
                'Dq': Dqs,
                'alpha': alpha,
                'f_alpha': f_alpha,
                'r_squared': r_squared
            })
            results_df.to_csv(os.path.join(output_dir, "multifractal_results.csv"), index=False)
            
            # Save multifractal parameters
            params_df = pd.DataFrame({
                'Parameter': ['Time', 'D0', 'D1', 'D2', 'alpha_width', 'degree_multifractality'],
                'Value': [data['time'], D0, D1, D2, alpha_width, degree_multifractality]
            })
            params_df.to_csv(os.path.join(output_dir, "multifractal_parameters.csv"), index=False)
        
        # Return results
        return {
            'q_values': q_values,
            'tau': taus,
            'Dq': Dqs,
            'alpha': alpha,
            'f_alpha': f_alpha,
            'r_squared': r_squared,
            'D0': D0,
            'D1': D1,
            'D2': D2,
            'alpha_width': alpha_width,
            'degree_multifractality': degree_multifractality,
            'time': data['time']
        }
    
    def analyze_multifractal_evolution(self, vtk_files, output_dir=None, q_values=None):
        """
        Analyze how multifractal properties evolve over time or across resolutions.
        
        Args:
            vtk_files: Dict mapping either times or resolutions to VTK files
                      e.g. {0.1: 'RT_t0.1.vtk', 0.2: 'RT_t0.2.vtk'} for time series
                      or {100: 'RT100x100.vtk', 200: 'RT200x200.vtk'} for resolutions
            output_dir: Directory to save results
            q_values: List of q moments to analyze (default: -5 to 5 in 0.5 steps)
            
        Returns:
            dict: Multifractal evolution results
        """
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Set default q values if not provided
        if q_values is None:
            q_values = np.arange(-5, 5.1, 0.5)
        
        # Determine type of analysis (time or resolution)
        keys = list(vtk_files.keys())
        is_time_series = all(isinstance(k, float) for k in keys)
        
        if is_time_series:
            print(f"Analyzing multifractal evolution over time series: {sorted(keys)}")
            x_label = 'Time'
            series_name = "time"
        else:
            print(f"Analyzing multifractal evolution across resolutions: {sorted(keys)}")
            x_label = 'Resolution'
            series_name = "resolution"
        
        # Initialize results storage
        results = []
        
        # Process each file
        for key, vtk_file in sorted(vtk_files.items()):
            print(f"\nProcessing {series_name} = {key}, file: {vtk_file}")
            
            try:
                # Read VTK file
                data = self.read_vtk_file(vtk_file)
                
                # Create subdirectory for this point
                if output_dir:
                    point_dir = os.path.join(output_dir, f"{series_name}_{key}")
                    os.makedirs(point_dir, exist_ok=True)
                else:
                    point_dir = None
                
                # Perform multifractal analysis
                mf_results = self.compute_multifractal_spectrum(data, q_values=q_values, output_dir=point_dir)
                
                if mf_results:
                    # Store results with the key (time or resolution)
                    mf_results[series_name] = key
                    results.append(mf_results)
                
            except Exception as e:
                print(f"Error processing {vtk_file}: {str(e)}")
                import traceback
                traceback.print_exc()
        
        # Create summary plots
        if results and output_dir:
            # Extract evolution of key parameters
            x_values = [res[series_name] for res in results]
            D0_values = [res['D0'] for res in results]
            D1_values = [res['D1'] for res in results]
            D2_values = [res['D2'] for res in results]
            alpha_width = [res['alpha_width'] for res in results]
            degree_mf = [res['degree_multifractality'] for res in results]
            
            # Plot generalized dimensions evolution
            plt.figure(figsize=(10, 6))
            plt.plot(x_values, D0_values, 'bo-', label='D(0) - Capacity dimension')
            plt.plot(x_values, D1_values, 'ro-', label='D(1) - Information dimension')
            plt.plot(x_values, D2_values, 'go-', label='D(2) - Correlation dimension')
            plt.xlabel(x_label)
            plt.ylabel('Generalized Dimensions')
            plt.title(f'Evolution of Generalized Dimensions with {x_label}')
            plt.grid(True)
            plt.legend()
            plt.savefig(os.path.join(output_dir, "dimensions_evolution.png"), dpi=300)
            plt.close()
            
            # Plot multifractal parameters evolution
            plt.figure(figsize=(10, 6))
            plt.plot(x_values, alpha_width, 'ms-', label='α width')
            plt.plot(x_values, degree_mf, 'cd-', label='Degree of multifractality')
            plt.xlabel(x_label)
            plt.ylabel('Parameter Value')
            plt.title(f'Evolution of Multifractal Parameters with {x_label}')
            plt.grid(True)
            plt.legend()
            plt.savefig(os.path.join(output_dir, "multifractal_params_evolution.png"), dpi=300)
            plt.close()
            
            # Create 3D surface plot of D(q) evolution if matplotlib supports it
            try:
                from mpl_toolkits.mplot3d import Axes3D
                
                # Prepare data for 3D plot
                X, Y = np.meshgrid(x_values, q_values)
                Z = np.zeros((len(q_values), len(x_values)))
                
                for i, result in enumerate(results):
                    for j, q in enumerate(q_values):
                        q_idx = np.where(result['q_values'] == q)[0]
                        if len(q_idx) > 0:
                            Z[j, i] = result['Dq'][q_idx[0]]
                
                # Create 3D plot
                fig = plt.figure(figsize=(12, 8))
                ax = fig.add_subplot(111, projection='3d')
                surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.8)
                
                ax.set_xlabel(x_label)
                ax.set_ylabel('q')
                ax.set_zlabel('D(q)')
                ax.set_title(f'Evolution of D(q) Spectrum with {x_label}')
                
                fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='D(q)')
                plt.savefig(os.path.join(output_dir, "Dq_evolution_3D.png"), dpi=300)
                plt.close()
                
            except Exception as e:
                print(f"Error creating 3D plot: {str(e)}")
            
            # Save summary CSV
            import pandas as pd
            summary_df = pd.DataFrame({
                series_name: x_values,
                'D0': D0_values,
                'D1': D1_values,
                'D2': D2_values,
                'alpha_width': alpha_width,
                'degree_multifractality': degree_mf
            })
            summary_df.to_csv(os.path.join(output_dir, "multifractal_evolution_summary.csv"), index=False)
        
        return results
