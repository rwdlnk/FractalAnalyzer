# Fractal Analyzer

A comprehensive Python toolkit for fractal dimension analysis of fluid interfaces and mathematical fractals, with specialized tools for Rayleigh-Taylor instability simulations.

## Overview

This package provides advanced tools for computing fractal dimensions using the box-counting method with sophisticated optimization techniques. Originally developed for analyzing mixing layer growth in Rayleigh-Taylor instability simulations, it includes general-purpose fractal analysis capabilities for mathematical fractals like Koch curves, Sierpinski triangles, and more.

## Key Features

### Fractal Dimension Analysis
- **Advanced Box-Counting**: Grid optimization for improved accuracy
- **Sliding Window Analysis**: Automatic optimal scaling region detection
- **Multiple Fractal Types**: Koch, Sierpinski, Minkowski, Hilbert, Dragon curves
- **Boundary Artifact Removal**: Enhanced detection and correction of edge effects
- **Publication-Quality Plots**: Professional visualization tools

### Rayleigh-Taylor Simulation Analysis
- **VTK File Support**: Direct reading of simulation output files
- **Multiple Mixing Methods**: Geometric, statistical, and Dalziel methodologies
- **Temporal Evolution**: Time series analysis of mixing layer growth
- **Resolution Convergence**: Grid independence studies
- **Interface Extraction**: Automated contour detection and processing

### Advanced Capabilities
- **Multifractal Analysis**: Complete spectrum calculation D(q), f(α)
- **Resolution Studies**: Systematic convergence analysis
- **Batch Processing**: Automated analysis of simulation series
- **Command-Line Tools**: Ready-to-use scripts for common workflows

## Installation

### From Source (Recommended for Development)
```bash
git clone https://github.com/yourusername/Fractal_Analyzer.git
cd Fractal_Analyzer
pip install -e .
```

### From PyPI (Once Published)
```bash
pip install fractal-analyzer
```

### Requirements
- Python ≥ 3.8
- NumPy ≥ 1.20.0
- SciPy ≥ 1.7.0
- Matplotlib ≥ 3.3.0
- pandas ≥ 1.3.0
- scikit-image ≥ 0.18.0
- numba ≥ 0.56.0

## Quick Start

### Basic Fractal Analysis
```python
from fractal_analyzer import FractalAnalyzer

# Generate and analyze a Koch curve
analyzer = FractalAnalyzer('koch')
points, segments = analyzer.generate_fractal('koch', level=5)

# Calculate fractal dimension with automatic optimization
results = analyzer.analyze_linear_region(
    segments, 
    plot_results=True,
    use_grid_optimization=True
)

print(f"Fractal dimension: {results[5]:.6f}")
```

### Rayleigh-Taylor Simulation Analysis
```python
from fractal_analyzer import RTAnalyzer

# Initialize analyzer
rt = RTAnalyzer('./output')

# Analyze a single VTK file
result = rt.analyze_vtk_file('RT800x800-5000.vtk', mixing_method='dalziel')

print(f"Time: {result['time']:.3f}")
print(f"Fractal dimension: {result['fractal_dim']:.6f}")
print(f"Mixing thickness: {result['h_total']:.6f}")
```

### Command-Line Usage
```bash
# Generate and analyze fractals
fractal-analyze --generate koch --level 5 --analyze_linear_region

# Temporal evolution analysis
rt-analyze --resolutions 400 800 1600 --auto-times --output ./results

# Resolution convergence study
rt-convergence --time 5.0 --base-dir ./simulations --output ./convergence
```

## Documentation

### Core Modules
- **FractalAnalyzer**: General fractal dimension analysis
- **RTAnalyzer**: Rayleigh-Taylor simulation processing
- **Box Counting**: Optimized algorithms with grid positioning
- **Interface Extraction**: Contour detection and segment conversion

### Analysis Methods
- **Linear Region Selection**: Sliding window optimization
- **Mixing Thickness**: Geometric, statistical, and Dalziel methods
- **Multifractal Spectrum**: Complete D(q) and f(α) analysis
- **Resolution Convergence**: Richardson extrapolation

### File Formats
- **VTK Rectilinear Grid**: Direct support for simulation output
- **Line Segments**: Text format for interface data
- **CSV Results**: Structured output for further analysis

## Examples

The `examples/` directory contains:
- `koch_curve_demo.py`: Mathematical fractal generation and analysis
- `rt_simulation_demo.py`: Complete Rayleigh-Taylor workflow
- `temporal_analysis_example.py`: Time series processing
- `resolution_study_example.py`: Grid convergence analysis

## Scripts

Pre-built analysis scripts in `scripts/`:
- `analyze_temporal_improved.py`: Temporal evolution with file detection
- `basic_resolution_convergence.py`: Resolution convergence analysis
- `batch_analysis.py`: Automated processing of simulation series

## Scientific Applications

This toolkit has been used for:

### Rayleigh-Taylor Instability Studies
- Mixing layer growth rate analysis
- Interface complexity quantification
- Grid resolution requirements
- Comparison with experimental data (Dalziel et al. 1999)

### Mathematical Fractal Validation
- Koch curve dimension verification (theoretical: 1.2619)
- Sierpinski triangle analysis (theoretical: 1.5850)
- Multifractal spectrum characterization

### Computational Fluid Dynamics
- Interface tracking in multiphase flows
- Turbulent mixing quantification
- Mesh independence verification

## Contributing

We welcome contributions! Please see `CONTRIBUTING.md` for guidelines.

### Development Setup
```bash
git clone https://github.com/yourusername/Fractal_Analyzer.git
cd Fractal_Analyzer
pip install -e ".[dev]"
pytest tests/
```

### Code Style
- Black formatting: `black fractal_analyzer/`
- Type hints encouraged
- Docstring documentation required

## Citation

If you use this software in your research, please cite:

```bibtex
@software{fractal_analyzer,
  title={Fractal Analyzer: Advanced Tools for Fractal Dimension Analysis},
  author={Rod Douglass},
  year={2024},
  url={https://github.com/rwdlnk/Fractal_Analyzer},
  version={1.0.0}
}
```

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.

## Acknowledgments

- Box-counting optimization techniques inspired by computational geometry research
- Rayleigh-Taylor analysis methods based on Dalziel et al. (1999)
- Multifractal algorithms following Chhabra & Jensen (1989)

## Related Work

- **Original Fortran77 Code**: This package represents a modernized Python implementation
- **Mixing Layer Studies**: Designed for quantitative comparison with experimental data
- **CFD Post-Processing**: Integrates with standard simulation output formats

## Support

- **Issues**: Report bugs and request features via GitHub Issues
- **Discussions**: Ask questions in GitHub Discussions
- **Documentation**: Full API reference at [link-to-docs]

---

**Keywords**: fractal dimension, box counting, Rayleigh-Taylor instability, mixing layer, CFD post-processing, interface analysis, multifractal, Python, VTK
