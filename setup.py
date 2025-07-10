#!/usr/bin/env python3
"""
Setup configuration for the Fractal Analyzer package.
"""

from setuptools import setup, find_packages
import os

# Read the README file for long description
here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, "README.md"), "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements from requirements.txt
with open(os.path.join(here, "requirements.txt"), "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="fractal-analyzer",
    version="2.1.0",  # UPDATED: Increment version for CONREC + parallel features
    author="Rod Douglass",
    author_email="rwdlanm@gmail.com",
    description="Advanced fractal dimension analysis for fluid interfaces with CONREC precision extraction and parallel processing",  # UPDATED
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rwdlnk/Fractal_Analyzer",
    project_urls={
        "Bug Tracker": "https://github.com/rwdlnk/Fractal_Analyzer/issues",
        "Documentation": "https://github.com/rwdlnk/Fractal_Analyzer/wiki",
        "Source Code": "https://github.com/rwdlnk/Fractal_Analyzer",
    },
    
    # UPDATED: Include scripts directory packages
    packages=find_packages(exclude=["tests", "tests.*", "docs", "docs.*"]) + ['scripts', 'scripts.optimized'],
    
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Visualization",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Environment :: Console",  # NEW: Added for command-line tools
    ],
    keywords=[
        "fractal", "dimension", "box-counting", "rayleigh-taylor", 
        "mixing", "cfd", "interface", "multifractal", "physics", "simulation",
        "conrec", "contouring", "parallel-processing", "temporal-evolution"  # NEW: CONREC + parallel keywords
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.10",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.900",
            "isort>=5.0",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=1.0",
            "myst-parser>=0.15",
            "nbsphinx>=0.8",
        ],
        "jupyter": [
            "jupyter>=1.0",
            "notebook>=6.0",
            "ipywidgets>=7.6",
        ],
        "parallel": [  # NEW: Parallel processing extras
            "psutil>=5.8.0",
        ],
        "all": [
            # Development tools
            "pytest>=6.0", "pytest-cov>=2.10", "black>=21.0", "flake8>=3.8", "mypy>=0.900",
            # Documentation
            "sphinx>=4.0", "sphinx-rtd-theme>=1.0", "myst-parser>=0.15",
            # Jupyter
            "jupyter>=1.0", "notebook>=6.0", "ipywidgets>=7.6",
            # Parallel processing  # NEW
            "psutil>=5.8.0",
        ],
    },
    entry_points={
        "console_scripts": [
            # Core analysis tools
            "fractal-analyze=fractal_analyzer.core.fractal_analyzer:main",
            "rt-analyze=fractal_analyzer.core.rt_analyzer:main",  # UPDATED: Now supports CONREC

            "rt-convergence=scripts.basic_resolution_convergence:main",  # Keep existing
            "rt-temporal=scripts.analyze_temporal_improved:main",  # Your existing script
        
            # NEW: Add serial analysis scripts
            "temporal-evolution=fractal_analyzer.optimized.temporal_evolution_analyzer:main",
            "resolution-comparison=fractal_analyzer.optimized.resolution_comparison:main",
        
            # NEW: Add parallel analysis scripts (these were missing!)
            "parallel-temporal=fractal_analyzer.optimized.parallel_temporal_evolution_analyzer:main",
            "parallel-resolution=fractal_analyzer.optimized.parallel_resolution_comparison:main",
        ],
    },
    include_package_data=True,
    package_data={
        "fractal_analyzer": [
            "examples/data/*.txt",
            "examples/data/*.vtk", 
            "tests/test_data/*.txt",
            "tests/test_data/*.vtk",
        ],
        "scripts": [  # NEW: Include scripts in package data
            "*.py",
            "optimized/*.py",
        ],
    },
    zip_safe=False,  # For numba compatibility
    
    # Additional metadata
    platforms=["any"],
    
    # Long description content type is already set above
    # Ensure we can find the package
    package_dir={"": "."},
)
