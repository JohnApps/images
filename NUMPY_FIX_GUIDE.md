# NumPy Compatibility Fix Guide

## Problem
```
A module that was compiled using NumPy 1.x cannot be run in
NumPy 2.3.4 as it may crash. To support both 1.x and 2.x
```

## Quick Fix (Recommended)

### Linux:
```bash
./fix_numpy_compatibility.sh
```

### Windows 11:
```cmd
fix_numpy_compatibility.bat
```

## Manual Fix

### Option 1: Downgrade NumPy to 1.x (Recommended for Stability)
```bash
# Uninstall NumPy 2.x
pip uninstall -y numpy

# Install NumPy 1.26.4 (last stable 1.x)
pip install "numpy>=1.24.0,<2.0.0"

# Reinstall all packages
pip install --force-reinstall --no-cache-dir -r requirements.txt
```

### Option 2: Upgrade All Packages to NumPy 2.x Compatible Versions
```bash
# Upgrade pip first
pip install --upgrade pip

# Upgrade all packages to latest versions
pip install --upgrade duckdb pandas matplotlib seaborn numpy streamlit plotly

# Verify versions
pip list | grep -E "numpy|pandas|matplotlib|seaborn|duckdb|streamlit|plotly"
```

## Why This Happens

NumPy 2.0 introduced breaking changes in the C API. Packages compiled against NumPy 1.x 
(like older versions of pandas, matplotlib, etc.) cannot run with NumPy 2.x without recompilation.

## Verification

After applying the fix, verify your NumPy version:
```bash
python -c "import numpy; print(f'NumPy version: {numpy.__version__}')"
```

Should show: `NumPy version: 1.26.x` or `1.24.x` (not 2.x)

## Testing the Fix

Run a quick test:
```bash
python -c "import numpy as np; import pandas as pd; print('✅ All imports successful!')"
```

If you see "✅ All imports successful!" - you're good to go!

## For Your Use Case

Since you're focused on:
- OLAP and real-time OLAP analysis
- Testing large datasets with DuckDB
- Running benchmarks on Linux and Windows 11

**Recommendation:** Use NumPy 1.26.x for maximum compatibility and stability.

NumPy 1.26.4 is the last 1.x release and is:
- Battle-tested and stable
- Compatible with all major data science packages
- Sufficient for all DuckDB operations
- Consistent across Linux and Windows 11

## Package Version Compatibility Matrix

| Package    | NumPy 1.26.x | NumPy 2.x |
|------------|--------------|-----------|
| DuckDB     | ✅ Yes       | ✅ Yes    |
| Pandas 2.0 | ✅ Yes       | ⚠️ 2.1+  |
| Matplotlib | ✅ Yes       | ⚠️ 3.9+  |
| Seaborn    | ✅ Yes       | ⚠️ 0.13+ |
| Streamlit  | ✅ Yes       | ✅ Yes    |
| Plotly     | ✅ Yes       | ✅ Yes    |

✅ = Fully compatible
⚠️ = Requires newer version

## Additional Resources

- [NumPy 2.0 Migration Guide](https://numpy.org/devdocs/numpy_2_0_migration_guide.html)
- [DuckDB Python API](https://duckdb.org/docs/api/python/overview)

## Still Having Issues?

If you continue to experience problems:

1. **Create a fresh virtual environment:**
   ```bash
   python -m venv venv_duckdb
   source venv_duckdb/bin/activate  # Linux
   # or
   venv_duckdb\Scripts\activate  # Windows
   
   pip install -r requirements.txt
   ```

2. **Check for conflicting packages:**
   ```bash
   pip list | grep numpy
   pip show numpy
   ```

3. **Clear pip cache:**
   ```bash
   pip cache purge
   pip install --no-cache-dir -r requirements.txt
   ```

---

**Last Updated:** October 2025
**NumPy Recommended Version:** 1.26.4
**Status:** ✅ Tested on Linux and Windows 11
