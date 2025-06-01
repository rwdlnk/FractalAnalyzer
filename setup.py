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
    version="1.0.0",
    author="Rod Douglass",  # Replace with your actual name
    author_email="rwdlanm@gmail.com",  # Replace with your email
    description="Advanced fractal dimension analysis for fluid interfaces and mathematical fractals",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rwdlnk/Fractal_Analyzer",  # Replace with your GitHub URL
    project_urls={
        "Bug Tracker": "https://github.com/rwdlnk/Fractal_Analyzer/issues",
        "Documentation": "https://github.com/rwdlnk/Fractal_Analyzer/wiki",
        "Source Code": "https://github.com/rwdlnk/Fractal_Analyzer",
    },
    packages=find_packages(exclude=["tests", "tests.*", "docs", "docs.*"]),
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
    ],
    keywords=[
        "fractal", "dimension", "box-counting", "rayleigh-taylor", 
        "mixing", "cfd", "interface", "multifractal", "physics", "simulation"
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
        "all": [
            # Development tools
            "pytest>=6.0", "pytest-cov>=2.10", "black>=21.0", "flake8>=3.8", "mypy>=0.900",
            # Documentation
            "sphinx>=4.0", "sphinx-rtd-theme>=1.0", "myst-parser>=0.15",
            # Jupyter
            "jupyter>=1.0", "notebook>=6.0", "ipywidgets>=7.6",
        ],
    },
    entry_points={
        "console_scripts": [
            "fractal-analyze=fractal_analyzer.core.fractal_analyzer:main",
            "rt-analyze=scripts.analyze_temporal_improved:main",
            "rt-convergence=scripts.basic_resolution_convergence:main",
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
    },
    zip_safe=False,  # For numba compatibility
    
    # Additional metadata
    platforms=["any"],
    
    # Long description content type is already set above
    # Ensure we can find the package
    package_dir={"": "."},
)
