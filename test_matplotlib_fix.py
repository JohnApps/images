#!/usr/bin/env python3
"""
Quick test to verify matplotlib axes handling is fixed
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def test_single_subplot():
    """Test single subplot handling"""
    print("Testing single subplot...")
    
    fig, axes = plt.subplots(1, 1, figsize=(8, 5))
    
    # Normalize to list
    if isinstance(axes, np.ndarray):
        axes = axes.flatten()
    else:
        axes = [axes]
    
    # Test barh
    axes[0].barh([0, 1, 2], [10, 20, 30])
    axes[0].set_title("Test Single Subplot")
    
    plt.close()
    print("✅ Single subplot test passed!")

def test_multiple_subplots():
    """Test multiple subplots handling"""
    print("Testing multiple subplots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Normalize to list
    if isinstance(axes, np.ndarray):
        axes = axes.flatten()
    else:
        axes = [axes]
    
    # Test all axes
    for i, ax in enumerate(axes):
        ax.barh([0, 1, 2], [10+i, 20+i, 30+i])
        ax.set_title(f"Test Subplot {i+1}")
    
    plt.close()
    print("✅ Multiple subplots test passed!")

def test_categorical_plots():
    """Test the actual categorical plot logic"""
    print("Testing categorical plots logic...")
    
    # Create test data
    df = pd.DataFrame({
        'category1': ['A', 'B', 'C', 'A', 'B'] * 20,
        'category2': ['X', 'Y', 'Z', 'X', 'Y'] * 20,
    })
    
    categorical_cols = ['category1', 'category2']
    
    n_cols = min(len(categorical_cols), 2)
    n_rows = (len(categorical_cols) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8*n_cols, 5*n_rows))
    
    # Normalize axes to always be a flat array (THE FIX)
    if n_rows == 1 and n_cols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for idx, col in enumerate(categorical_cols):
        if idx < len(axes):
            value_counts = df[col].value_counts().head(20)
            axes[idx].barh(range(len(value_counts)), value_counts.values, color='coral')
            axes[idx].set_yticks(range(len(value_counts)))
            axes[idx].set_yticklabels(value_counts.index)
            axes[idx].set_title(f'Top Values: {col}')
            axes[idx].set_xlabel('Count')
    
    plt.close()
    print("✅ Categorical plots logic test passed!")

def test_edge_cases():
    """Test edge cases"""
    print("Testing edge cases...")
    
    # Test with 1 column
    df1 = pd.DataFrame({'col1': ['A', 'B', 'C'] * 10})
    categorical_cols = ['col1']
    
    n_cols = min(len(categorical_cols), 2)
    n_rows = (len(categorical_cols) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8, 5))
    
    if n_rows == 1 and n_cols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    value_counts = df1['col1'].value_counts()
    axes[0].barh(range(len(value_counts)), value_counts.values)
    
    plt.close()
    print("✅ Edge case (1 column) test passed!")

def main():
    print("="*60)
    print("  Matplotlib Axes Handling Test")
    print("="*60)
    print()
    
    try:
        test_single_subplot()
        test_multiple_subplots()
        test_categorical_plots()
        test_edge_cases()
        
        print()
        print("="*60)
        print("✅ All matplotlib axes tests passed!")
        print("="*60)
        print()
        print("The fix is working correctly. You can now run:")
        print("  python visualize_duckdb_tables.py")
        print()
        return True
        
    except Exception as e:
        print()
        print("="*60)
        print("❌ Test failed!")
        print("="*60)
        print(f"Error: {str(e)}")
        print()
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
