# 🚨 TROUBLESHOOTING QUICK REFERENCE

## Error: "Object of type ObjectDType is not JSON serializable" (Plotly Chart)

### ✅ FIXED in latest version!

**Symptoms:** Error when rendering charts in Streamlit dashboard  
**Location:** `st.plotly_chart(fig, use_container_width=True)`

**What to do:**
```bash
# 1. Make sure you have the latest version
cp streamlit_duckdb_dashboard.py .

# 2. Clear Streamlit cache
rm -rf ~/.streamlit/cache

# 3. Run the dashboard
streamlit run streamlit_duckdb_dashboard.py
```

**If still having issues:**
- See `PLOTLY_SERIALIZATION_FIX.md` for full details

---

## Error: "Object of type ObjectDType is not JSON serializable" (Streamlit Cache)

### ✅ FIXED in latest version!

**Symptoms:** Error when loading data or caching  
**Location:** During data queries and caching

**What to do:**
```bash
# 1. Make sure you have the latest version
cp streamlit_duckdb_dashboard.py .

# 2. Clear Streamlit cache
rm -rf ~/.streamlit/cache

# 3. Run the dashboard
streamlit run streamlit_duckdb_dashboard.py
```

**If still having issues:**
- See `CACHING_FIX_COMPLETE.md` for full details

---

## Error: "'numpy.ndarray' object has no attribute 'barh'"

### ✅ FIXED in latest version!

**What to do:**
```bash
# 1. Make sure you have the latest version
cp visualize_duckdb_tables.py .

# 2. Test the fix
python test_matplotlib_fix.py

# 3. Run the generator
python visualize_duckdb_tables.py
```

**If still having issues:**
- See `MATPLOTLIB_AXES_FIX.md` for full details

---

## Error: "NumPy 1.x cannot be run in NumPy 2.x"

### ✅ FIXED in requirements.txt!

**Quick fix:**
```bash
./fix_numpy_compatibility.sh        # Linux
fix_numpy_compatibility.bat         # Windows
```

**Manual fix:**
```bash
pip uninstall -y numpy
pip install "numpy>=1.24.0,<2.0.0"
pip install --force-reinstall -r requirements.txt
```

**If still having issues:**
- See `NUMPY_FIX_GUIDE.md` for full details

---

## 🎯 Quick Start (No Errors)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Verify setup
python test_setup.py

# 3. Run static generator
python visualize_duckdb_tables.py

# 4. Run dashboard
streamlit run streamlit_duckdb_dashboard.py
```

---

## 📚 Documentation Index

| File | When to Read |
|------|-------------|
| **QUICK_START.md** | 👈 **Start here!** First time setup |
| **PLOTLY_SERIALIZATION_FIX.md** | Plotly chart rendering errors |
| **CACHING_FIX_COMPLETE.md** | Streamlit caching/JSON errors |
| **MATPLOTLIB_AXES_FIX.md** | Matplotlib/numpy.ndarray errors |
| **NUMPY_FIX_GUIDE.md** | NumPy version issues |
| **FIX_SUMMARY.md** | Overview of all fixes |
| **README.md** | Feature documentation |

---

## 🔧 Nuclear Option (If Nothing Works)

```bash
# 1. Delete everything
rm -rf venv __pycache__ .streamlit/cache ~/.streamlit/cache

# 2. Fresh virtual environment
python -m venv venv_new
source venv_new/bin/activate  # Linux
venv_new\Scripts\activate      # Windows

# 3. Fresh install
pip install -r requirements.txt

# 4. Verify
python test_setup.py

# 5. Run
streamlit run streamlit_duckdb_dashboard.py
```

---

## 🆘 Still Stuck?

1. **Check which error** you're getting (exact message)
2. **Read the specific guide** for that error
3. **Clear all caches** (Streamlit, Python, pip)
4. **Verify versions**: `python test_setup.py`
5. **Try nuclear option** (fresh venv)

---

## ✅ Success Indicators

You'll know it's working when:

- ✅ `test_setup.py` shows 7/7 tests passed
- ✅ No errors when running either tool
- ✅ Dashboard opens in browser
- ✅ Can navigate all tabs
- ✅ Can view and query data

---

## 📊 Expected Behavior

### Static Generator:
```
📊 Processing table: table_name
  Rows: 1234, Columns: 8
  ✓ Created summary statistics
  ✓ Created distribution plots
  ✓ Created categorical plots
  ✓ Created correlation heatmap
✅ Visualization complete!
```

### Streamlit Dashboard:
- Opens browser automatically to http://localhost:8501
- Shows database connection status
- All 6 tabs load without errors
- Data displays in tables
- Charts render properly

---

**All fixes applied! Both tools ready to use!** 🦆📊
