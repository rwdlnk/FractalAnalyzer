#!/bin/bash
# activate_fractal.sh - Activate the FractalAnalyzer virtual environment

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Activate the virtual environment
source "$SCRIPT_DIR/fractal_env/bin/activate"

# Optional: Change to the project directory
cd "$SCRIPT_DIR"

# Confirm activation
echo "✅ FractalAnalyzer environment activated!"
echo "📁 Current directory: $(pwd)"
echo "🐍 Python path: $(which python)"
echo ""
echo "Ready to analyze fractals! 🔬"
