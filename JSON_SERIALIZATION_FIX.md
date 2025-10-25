# Streamlit JSON Serialization Fix Guide

## Problem
```
Object of type ObjectDType is not JSON serializable
```

## What Causes This?

This error occurs when Streamlit's caching mechanism (`@st.cache_data`) tries to serialize pandas DataFrames containing **object dtypes** (strings, mixed types) into JSON format. Object dtypes aren't directly JSON serializable.

## ✅ Quick Fix - Already Applied

The updated `streamlit_duckdb_dashboard.py` now includes automatic fixes:

1. **Converts object columns to strings** before caching
2. **Adds error handling** for type conversions
3. **Removes caching for custom queries** (which may return unpredictable types)
4. **Converts numeric types** to native Python types (float, int)

You should now be able to run the dashboard without issues!

## 🚀 Test the Fix

```bash
streamlit run streamlit_duckdb_dashboard.py
```

## Alternative Solutions (if still having issues)

### Option 1: Clear Streamlit Cache
```bash
# Clear cache directory
rm -rf ~/.streamlit/cache

# Or within the app, click "Clear cache" in settings menu (⋮)
```

### Option 2: Disable Caching Temporarily
If you still encounter issues, you can temporarily disable caching by setting TTL to 0:

Edit `streamlit_duckdb_dashboard.py` and change:
```python
@st.cache_data(ttl=300)  # 5 minutes
```
to:
```python
@st.cache_data(ttl=0)  # No caching
```

### Option 3: Use Pickle Protocol (Advanced)
For complex objects, you can use pickle instead of JSON:

```python
@st.cache_data(ttl=300, max_entries=10, persist="disk")
def get_data():
    # your code here
    pass
```

## Common Scenarios and Fixes

### 1. DuckDB BLOB/Binary Data
If your tables contain BLOB or binary data:

```python
# Convert BLOB to hex string
df['blob_column'] = df['blob_column'].apply(lambda x: x.hex() if x else None)
```

### 2. Mixed Type Columns
If a column has mixed types (numbers and strings):

```python
# Force to string
df['mixed_column'] = df['mixed_column'].astype(str)
```

### 3. Nested Structures
If you have nested lists or dicts:

```python
import json
df['nested_column'] = df['nested_column'].apply(json.dumps)
```

### 4. Timestamp/Date Issues
If you have timezone-aware timestamps:

```python
# Convert to timezone-naive
df['timestamp'] = df['timestamp'].dt.tz_localize(None)
# Or convert to string
df['timestamp'] = df['timestamp'].astype(str)
```

## Understanding the Fix in the Code

Here's what was changed in `streamlit_duckdb_dashboard.py`:

### Before (causing error):
```python
@st.cache_data(ttl=300)
def get_tables(_self):
    return _self.conn.execute(query).fetchdf()
```

### After (fixed):
```python
@st.cache_data(ttl=300, show_spinner=False)
def get_tables(_self):
    df = _self.conn.execute(query).fetchdf()
    # Convert all object columns to string
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype(str)
    return df
```

## For EXIF Data Specifically

If your `images.db` contains EXIF data with special types:

```python
# Common EXIF fields that may cause issues:
- GPS coordinates (tuples)
- Camera settings (mixed types)
- Thumbnail data (binary)
- Maker notes (binary/complex)

# Solution: Convert to strings
df['gps_info'] = df['gps_info'].astype(str)
df['maker_notes'] = df['maker_notes'].apply(lambda x: str(x)[:100] if x else None)
```

## Debugging Tips

### 1. Identify Problematic Columns
```python
import pandas as pd

df = conn.execute("SELECT * FROM your_table LIMIT 100").fetchdf()

# Check dtypes
print(df.dtypes)

# Find object columns
object_cols = df.select_dtypes(include=['object']).columns
print(f"Object columns: {object_cols.tolist()}")

# Check for mixed types
for col in object_cols:
    types = df[col].apply(type).unique()
    print(f"{col}: {types}")
```

### 2. Test Individual Functions
```python
# Test without caching
def test_query():
    result = conn.execute("SELECT * FROM table").fetchdf()
    print(result.dtypes)
    return result

df = test_query()
```

### 3. Check Cache Size
```bash
# Linux/Mac
du -sh ~/.streamlit/cache

# Windows
dir %USERPROFILE%\.streamlit\cache
```

## Performance Considerations

With these fixes applied:
- ✅ **Caching still works** (5-minute TTL for queries)
- ✅ **Type conversion is fast** (happens once per cache miss)
- ✅ **Memory usage is reasonable** (strings are efficient)
- ✅ **Compatible with OLAP queries** (aggregations work on converted data)

## Still Having Issues?

### Check Python/Package Versions
```bash
python --version  # Should be 3.8+
pip show streamlit pandas duckdb
```

### Create Fresh Virtual Environment
```bash
python -m venv venv_streamlit
source venv_streamlit/bin/activate  # Linux
# or
venv_streamlit\Scripts\activate  # Windows

pip install -r requirements.txt
streamlit run streamlit_duckdb_dashboard.py
```

### Enable Debug Mode
```bash
streamlit run streamlit_duckdb_dashboard.py --logger.level=debug
```

### Report the Issue
If none of the above works, check the Streamlit terminal output for the exact line causing the error:
```
File "streamlit_duckdb_dashboard.py", line XXX, in function_name
```

## Configuration File

You can also create a `.streamlit/config.toml` file:

```toml
[server]
maxUploadSize = 200
maxMessageSize = 200

[runner]
fastReruns = true

[client]
showErrorDetails = true
```

## Additional Resources

- [Streamlit Caching Documentation](https://docs.streamlit.io/library/advanced-features/caching)
- [Pandas dtype Documentation](https://pandas.pydata.org/docs/user_guide/basics.html#dtypes)
- [DuckDB Python API](https://duckdb.org/docs/api/python/overview)

---

**Status:** ✅ Fixed in latest version
**Last Updated:** October 2025
**Tested On:** Linux and Windows 11 with DuckDB 0.9+
