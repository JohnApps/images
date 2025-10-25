#!/usr/bin/env python3
"""
DuckDB Streamlit Setup Verification Script
Tests all dependencies and database connectivity before running the dashboard
"""

import sys
from pathlib import Path

def print_header(text):
    """Print formatted header"""
    print("\n" + "="*60)
    print(f"  {text}")
    print("="*60)

def test_imports():
    """Test if all required packages can be imported"""
    print_header("Testing Package Imports")
    
    packages = [
        ('duckdb', 'DuckDB'),
        ('pandas', 'Pandas'),
        ('numpy', 'NumPy'),
        ('matplotlib', 'Matplotlib'),
        ('seaborn', 'Seaborn'),
        ('streamlit', 'Streamlit'),
        ('plotly', 'Plotly')
    ]
    
    all_ok = True
    for module, name in packages:
        try:
            imported = __import__(module)
            version = getattr(imported, '__version__', 'unknown')
            print(f"  ✅ {name:15} v{version}")
        except ImportError as e:
            print(f"  ❌ {name:15} NOT FOUND - {str(e)}")
            all_ok = False
    
    return all_ok

def test_numpy_version():
    """Check NumPy version compatibility"""
    print_header("NumPy Version Check")
    
    try:
        import numpy as np
        version = np.__version__
        major_version = int(version.split('.')[0])
        
        if major_version >= 2:
            print(f"  ⚠️  NumPy {version} detected (2.x)")
            print(f"  ⚠️  This may cause compatibility issues")
            print(f"  💡 Recommended: Run fix_numpy_compatibility.sh/.bat")
            return False
        else:
            print(f"  ✅ NumPy {version} (1.x) - Good!")
            return True
    except Exception as e:
        print(f"  ❌ Error checking NumPy: {str(e)}")
        return False

def test_database_connection(db_path="images.db"):
    """Test DuckDB database connectivity"""
    print_header("Database Connection Test")
    
    if not Path(db_path).exists():
        print(f"  ⚠️  Database '{db_path}' not found")
        print(f"  💡 Make sure the database file is in the current directory")
        print(f"  📁 Current directory: {Path.cwd()}")
        return False
    
    try:
        import duckdb
        conn = duckdb.connect(db_path, read_only=True)
        
        # Test basic query
        tables = conn.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'main'
        """).fetchall()
        
        print(f"  ✅ Connected to database: {db_path}")
        print(f"  📊 Found {len(tables)} tables:")
        for table in tables:
            # Get row count
            try:
                count = conn.execute(f"SELECT COUNT(*) FROM {table[0]}").fetchone()[0]
                print(f"      - {table[0]:20} ({count:,} rows)")
            except:
                print(f"      - {table[0]:20} (count failed)")
        
        conn.close()
        return True
        
    except Exception as e:
        print(f"  ❌ Database error: {str(e)}")
        return False

def test_dataframe_serialization():
    """Test DataFrame JSON serialization (the main issue)"""
    print_header("DataFrame Serialization Test")
    
    try:
        import pandas as pd
        import numpy as np
        
        # Create test DataFrame with object dtype
        df = pd.DataFrame({
            'text_col': ['a', 'b', 'c'],
            'num_col': [1, 2, 3],
            'mixed_col': ['x', 1, None]
        })
        
        print(f"  Original dtypes:")
        for col, dtype in df.dtypes.items():
            print(f"    {col:15} {dtype}")
        
        # Convert object columns to string (the fix)
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].astype(str)
        
        print(f"\n  After conversion:")
        for col, dtype in df.dtypes.items():
            print(f"    {col:15} {dtype}")
        
        # Try to serialize
        json_str = df.to_json()
        print(f"\n  ✅ JSON serialization successful!")
        print(f"  📦 Serialized size: {len(json_str)} bytes")
        return True
        
    except Exception as e:
        print(f"  ❌ Serialization error: {str(e)}")
        return False

def test_streamlit_cache():
    """Test Streamlit caching mechanism"""
    print_header("Streamlit Cache Test")
    
    try:
        import streamlit as st
        import pandas as pd
        
        # Test simple caching
        @st.cache_data(ttl=60, show_spinner=False)
        def test_func():
            df = pd.DataFrame({'col': ['a', 'b', 'c']})
            df['col'] = df['col'].astype(str)
            return df
        
        result = test_func()
        print(f"  ✅ Streamlit caching works!")
        print(f"  📊 Test DataFrame shape: {result.shape}")
        return True
        
    except Exception as e:
        print(f"  ❌ Caching error: {str(e)}")
        return False

def test_plotting():
    """Test plotting libraries"""
    print_header("Plotting Libraries Test")
    
    try:
        import matplotlib.pyplot as plt
        import plotly.express as px
        import pandas as pd
        
        # Test matplotlib
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3])
        plt.close()
        print(f"  ✅ Matplotlib works!")
        
        # Test plotly
        df = pd.DataFrame({'x': [1, 2, 3], 'y': [1, 2, 3]})
        fig = px.scatter(df, x='x', y='y')
        print(f"  ✅ Plotly works!")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Plotting error: {str(e)}")
        return False

def test_python_version():
    """Check Python version"""
    print_header("Python Version Check")
    
    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"  ⚠️  Python {version_str}")
        print(f"  ⚠️  Python 3.8+ recommended")
        return False
    else:
        print(f"  ✅ Python {version_str}")
        return True

def main():
    """Run all tests"""
    print("\n" + "🔬 DuckDB Streamlit Setup Verification".center(60))
    print("This script will test your environment and diagnose issues\n")
    
    tests = [
        ("Python Version", test_python_version),
        ("Package Imports", test_imports),
        ("NumPy Compatibility", test_numpy_version),
        ("DataFrame Serialization", test_dataframe_serialization),
        ("Streamlit Caching", test_streamlit_cache),
        ("Plotting Libraries", test_plotting),
        ("Database Connection", lambda: test_database_connection("images.db"))
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n  ❌ Unexpected error in {name}: {str(e)}")
            results.append((name, False))
    
    # Summary
    print_header("Test Summary")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status}  {name}")
    
    print("\n" + "-"*60)
    print(f"  Results: {passed}/{total} tests passed")
    print("-"*60)
    
    if passed == total:
        print("\n🎉 All tests passed! You're ready to run the dashboard:")
        print("   streamlit run streamlit_duckdb_dashboard.py")
    else:
        print("\n⚠️  Some tests failed. Please fix the issues above.")
        print("\n💡 Quick fixes:")
        print("   - Missing packages: pip install -r requirements.txt")
        print("   - NumPy 2.x issue: ./fix_numpy_compatibility.sh")
        print("   - Database not found: Check database path")
    
    print("\n📚 Documentation:")
    print("   - NUMPY_FIX_GUIDE.md - Fix NumPy issues")
    print("   - JSON_SERIALIZATION_FIX.md - Fix caching issues")
    print("   - README.md - Complete usage guide")
    print()
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
