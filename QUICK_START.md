# 🚀 Quick Start Guide - DuckDB Visualization Tools

Complete setup and troubleshooting guide for getting your DuckDB dashboard running.

## ⚡ Quick Setup (5 minutes)

### Step 1: Verify Your Setup
```bash
python test_setup.py
```

This will check:
- ✅ Python version (3.8+)
- ✅ All required packages
- ✅ NumPy compatibility (1.x vs 2.x)
- ✅ Database connectivity
- ✅ JSON serialization
- ✅ Streamlit caching

### Step 2: Fix Any Issues

If you see **NumPy 2.x warning**:
```bash
./fix_numpy_compatibility.sh        # Linux
fix_numpy_compatibility.bat         # Windows
```

If **packages are missing**:
```bash
pip install -r requirements.txt
```

### Step 3: Run the Dashboard
```bash
./launch_dashboard.sh               # Linux
launch_dashboard.bat                # Windows

# Or manually:
streamlit run streamlit_duckdb_dashboard.py
```

---

## 🔧 Common Errors & Quick Fixes

### Error 1: "'numpy.ndarray' object has no attribute 'barh'"

**Status:** ✅ **FIXED** in latest version

**What happened:** Matplotlib axes handling was incorrect for single subplots.

**Fix Applied:**
- Updated `visualize_duckdb_tables.py` to properly handle single and multiple subplots
- Added individual error handling for each visualization type
- Script now continues even if one plot type fails

**Test the fix:**
```bash
python test_matplotlib_fix.py
```

**Details:** See `MATPLOTLIB_AXES_FIX.md`

---

### Error 2: "Object of type ObjectDType is not JSON serializable"

**Status:** ✅ **FIXED** in latest version

**What happened:** Streamlit's caching couldn't serialize pandas object dtypes.

**Fix Applied:**
- Updated `streamlit_duckdb_dashboard.py` to automatically convert object columns to strings
- Added error handling for type conversions
- Improved caching configuration

**If still occurring:**
```bash
# Clear Streamlit cache
rm -rf ~/.streamlit/cache

# Re-run the dashboard
streamlit run streamlit_duckdb_dashboard.py
```

**Details:** See `JSON_SERIALIZATION_FIX.md`

---

### Error 3: "NumPy 1.x cannot be run in NumPy 2.3.4"

**Status:** ✅ **FIXED** in requirements.txt

**What happened:** NumPy 2.x has breaking changes that cause compatibility issues.

**Quick Fix:**
```bash
./fix_numpy_compatibility.sh        # Linux
fix_numpy_compatibility.bat         # Windows
```

**Manual Fix:**
```bash
pip uninstall -y numpy
pip install "numpy>=1.24.0,<2.0.0"
pip install --force-reinstall -r requirements.txt
```

**Details:** See `NUMPY_FIX_GUIDE.md`

---

### Error 4: "Database 'images.db' not found"

**Quick Fix:**
```bash
# Check current directory
pwd  # or 'cd' on Windows

# List files
ls -l images.db  # or 'dir images.db' on Windows

# If in wrong directory, copy database or change path in Streamlit sidebar
```

---

### Error 5: "Port 8501 already in use"

**Quick Fix:**
```bash
# Option 1: Kill existing Streamlit process
pkill -f streamlit  # Linux/Mac
taskkill /F /IM streamlit.exe  # Windows

# Option 2: Use different port
streamlit run streamlit_duckdb_dashboard.py --server.port 8502
```

---

### Error 6: "ImportError: cannot import name 'X'"

**Quick Fix:**
```bash
# Reinstall all packages
pip install --force-reinstall --no-cache-dir -r requirements.txt

# If that doesn't work, create fresh virtual environment
python -m venv venv_new
source venv_new/bin/activate  # Linux
venv_new\Scripts\activate  # Windows
pip install -r requirements.txt
```

---

## 📋 Pre-Flight Checklist

Before running the dashboard, verify:

- [ ] Python 3.8+ installed (`python --version`)
- [ ] NumPy 1.x installed (`python -c "import numpy; print(numpy.__version__)"`)
- [ ] All packages installed (`pip list | grep -E "duckdb|pandas|streamlit"`)
- [ ] Database file exists (`ls images.db`)
- [ ] No port conflicts (`lsof -i :8501` on Linux/Mac)

---

## 🎯 Two Tools Available

### 1. Static Graph Generator
**Best for:** Reports, documentation, batch processing

```bash
python visualize_duckdb_tables.py
```

**Output:** 
- Creates `graphs/` directory
- PNG files for each visualization
- `index.html` dashboard

**Use when:**
- Creating reports or presentations
- Need static images
- Processing multiple databases
- Automated pipelines

---

### 2. Streamlit Interactive Dashboard
**Best for:** Real-time exploration, OLAP analysis

```bash
streamlit run streamlit_duckdb_dashboard.py
```

**Features:**
- 📊 Interactive charts with zoom/pan
- 🔍 Dynamic filtering
- 💻 Custom SQL queries
- 📈 Real-time analysis
- 📥 CSV export

**Use when:**
- Exploring data interactively
- Running ad-hoc queries
- Need drill-down analysis
- Collaborative review

---

## 💡 Pro Tips

### Performance Optimization

**For Large Tables (>1M rows):**
```python
# In Streamlit: Use row limit slider (start with 1,000)
# In static generator: Edit limit in code (line ~XX)
```

**For Many Columns (>50):**
- Select specific columns in Data Explorer tab
- Static generator auto-limits to 15 columns for correlation

### OLAP Queries

**Aggregation Example:**
```sql
SELECT 
    DATE_TRUNC('month', date_column) as month,
    camera_model,
    COUNT(*) as photos,
    AVG(focal_length) as avg_focal
FROM images
GROUP BY month, camera_model
ORDER BY month DESC
```

**Time Series Analysis:**
```sql
SELECT 
    date_column,
    SUM(size) OVER (ORDER BY date_column) as cumulative_size
FROM images
ORDER BY date_column
```

### Benchmarking DuckDB

For your TPC-DS benchmarking with different scale factors:

```python
# Create comparison queries in SQL Query tab
SELECT 
    query_name,
    execution_time_ms,
    rows_returned,
    scale_factor
FROM benchmark_results
ORDER BY execution_time_ms DESC
```

---

## 🔍 Debug Mode

Enable detailed logging:

```bash
# Streamlit with debug output
streamlit run streamlit_duckdb_dashboard.py --logger.level=debug

# Python with verbose errors
python -u visualize_duckdb_tables.py 2>&1 | tee output.log
```

---

## 📁 File Reference

| File | Purpose |
|------|---------|
| `streamlit_duckdb_dashboard.py` | Interactive dashboard |
| `visualize_duckdb_tables.py` | Static graph generator |
| `requirements.txt` | Python dependencies |
| `test_setup.py` | Setup verification |
| `fix_numpy_compatibility.sh/.bat` | NumPy fix script |
| `launch_dashboard.sh/.bat` | Easy launcher |
| `README.md` | Complete documentation |
| `NUMPY_FIX_GUIDE.md` | NumPy troubleshooting |
| `JSON_SERIALIZATION_FIX.md` | Caching issues |

---

## 🆘 Still Having Issues?

### 1. Run Diagnostics
```bash
python test_setup.py
```

### 2. Check Logs
Look for error details in terminal output

### 3. Create Fresh Environment
```bash
python -m venv venv_fresh
source venv_fresh/bin/activate  # Linux
venv_fresh\Scripts\activate  # Windows
pip install -r requirements.txt
```

### 4. Platform-Specific Issues

**Linux:**
- Check file permissions: `chmod +x *.sh`
- Check Python path: `which python3`

**Windows 11:**
- Use Command Prompt (not PowerShell if issues)
- Check Python in PATH: `where python`
- Run as Administrator if permission errors

---

## 📊 Expected Performance

### Static Generator:
- **Small DB** (<10K rows): ~5 seconds
- **Medium DB** (10K-100K): ~30 seconds  
- **Large DB** (>100K): 1-2 minutes

### Streamlit Dashboard:
- **Initial Load**: 2-5 seconds
- **Queries** (<10K rows): Instant
- **Queries** (10K-100K): 1-3 seconds
- **Custom Queries**: Depends on complexity

---

## ✅ Success Criteria

You'll know everything is working when:

1. ✅ `python test_setup.py` shows all tests passed
2. ✅ Static generator creates `graphs/index.html` successfully
3. ✅ Streamlit dashboard opens in browser at `http://localhost:8501`
4. ✅ Can view tables, run queries, and export data
5. ✅ No error messages in terminal

---

## 🎓 Next Steps

Once running:

1. **Explore your EXIF data** in the Overview tab
2. **Create custom queries** for benchmarking in SQL Query tab
3. **Compare performance** across Linux and Windows 11
4. **Generate reports** using the static generator
5. **Share insights** using the HTML dashboard

---

## 📚 Learning Resources

- **DuckDB:** https://duckdb.org/docs/
- **Streamlit:** https://docs.streamlit.io/
- **Pandas:** https://pandas.pydata.org/docs/
- **OLAP:** https://en.wikipedia.org/wiki/Online_analytical_processing

---

**Last Updated:** October 2025  
**Status:** ✅ Both tools fully functional  
**Tested:** Linux & Windows 11 with DuckDB 0.9+

---

## 🎉 Quick Win

Run this right now to see if everything works:

```bash
# 1. Test setup
python test_setup.py

# 2. If all pass, launch dashboard
streamlit run streamlit_duckdb_dashboard.py

# 3. Open browser to http://localhost:8501

# 4. Enjoy! 🦆📊
```
