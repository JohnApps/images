#!/bin/bash
# NumPy Compatibility Fix Script
# Resolves "NumPy 1.x vs 2.x" compatibility issues

echo "================================================"
echo "  NumPy Compatibility Fix"
echo "================================================"
echo ""
echo "This script will fix NumPy version conflicts by:"
echo "1. Uninstalling NumPy 2.x"
echo "2. Installing NumPy 1.26.x (last stable 1.x version)"
echo "3. Reinstalling all dependencies"
echo ""
read -p "Press Enter to continue or Ctrl+C to cancel..."
echo ""

echo "Step 1: Uninstalling current NumPy..."
pip uninstall -y numpy

echo ""
echo "Step 2: Installing NumPy 1.26.4 (compatible version)..."
pip install "numpy>=1.24.0,<2.0.0"

echo ""
echo "Step 3: Reinstalling dependencies..."
pip install --force-reinstall --no-cache-dir -r requirements.txt

echo ""
echo "================================================"
echo "  Fix Complete!"
echo "================================================"
echo ""
echo "You can now run the visualization tools:"
echo "  - Static generator: python visualize_duckdb_tables.py"
echo "  - Streamlit dashboard: streamlit run streamlit_duckdb_dashboard.py"
echo ""
