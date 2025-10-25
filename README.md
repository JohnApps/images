# 🦆 DuckDB Visualization Tools

Two powerful Python tools for visualizing and analyzing DuckDB databases:
1. **Static Graph Generator** - Creates PNG graphs and HTML dashboard
2. **Streamlit Interactive Dashboard** - Real-time interactive web interface

## 📦 Installation

```bash
# Install required dependencies
pip install -r requirements.txt
```

## 🎨 Tool 1: Static Graph Generator

### Features
- Automatically generates visualizations for all tables
- Creates distribution plots, correlation heatmaps, and categorical charts
- Generates summary statistics tables
- Outputs an HTML dashboard to browse all graphs
- Perfect for reports and documentation

### Usage

```bash
python visualize_duckdb_tables.py
```

### Output
- Creates a `graphs/` directory with all PNG files
- Generates `graphs/index.html` - open this in your browser

### Customization
Edit these parameters in the code:
- `limit=10000` - Change row limit for large tables
- Graph styles, colors, and sizes
- Output directory location

---

## 🌐 Tool 2: Streamlit Interactive Dashboard

### Features
- **Real-time interactive exploration** of DuckDB tables
- **6 powerful tabs:**
  1. 📊 **Overview** - Data preview and summary statistics
  2. 🔍 **Data Explorer** - Filter and export data
  3. 📈 **Visualizations** - Interactive column distributions and scatter plots
  4. 🔗 **Relationships** - Correlation heatmaps and scatter matrices
  5. 💻 **SQL Query** - Execute custom DuckDB queries
  6. 📋 **Schema** - View table structure and column details

- **Interactive features:**
  - Dynamic filtering with sliders and multi-select
  - Zoom, pan, and hover on all charts
  - Export filtered data to CSV
  - Custom SQL query interface
  - Real-time column statistics

### Usage

```bash
streamlit run streamlit_duckdb_dashboard.py
```

This will open your browser automatically to `http://localhost:8501`

### Configuration
- Change database path in the sidebar
- Select different tables from dropdown
- Adjust row limits with slider
- Write custom SQL queries

### Pro Tips
- Use the **Data Explorer** tab to filter and export specific data subsets
- The **SQL Query** tab supports all DuckDB SQL features
- Charts are interactive - click legend items to toggle series
- Right-click charts to download as PNG

---

## 📊 Use Cases

### Static Generator Best For:
- Creating reports and documentation
- Batch processing multiple databases
- Generating charts for presentations
- Automated analysis pipelines
- Embedding visualizations in documents

### Streamlit Dashboard Best For:
- Real-time data exploration
- Interactive filtering and analysis
- Ad-hoc SQL queries
- Collaborative data review sessions
- Quick insights without coding
- OLAP-style drill-down analysis

---

## 🔧 Performance Tips

### For Large Tables (>1M rows):
```python
# Static Generator: Adjust sample size
df = self.get_table_data(table_name, limit=50000)

# Streamlit: Use the row limit slider (sidebar)
# Start with 1,000 rows, increase as needed
```

### For Many Columns (>50):
- Static generator automatically limits correlation matrices to 15 columns
- Streamlit lets you select specific columns to visualize

### DuckDB Optimization:
```sql
-- Create indexes for frequently queried columns
CREATE INDEX idx_column ON table_name(column_name);

-- Use columnar storage for OLAP workloads (DuckDB default)
-- Query only needed columns
SELECT col1, col2 FROM table LIMIT 1000;
```

---

## 🎯 OLAP Analysis Features

Both tools support typical OLAP operations:

### Aggregation
- Summary statistics (mean, median, std dev)
- Group by analysis via SQL queries
- Correlation analysis

### Drill-Down
- Filter data by dimensions
- View distribution at different granularities
- Time series analysis (if datetime columns present)

### Slice & Dice
- Select specific columns
- Filter by multiple dimensions
- Export filtered subsets

### Pivot Analysis
- Custom SQL queries for pivoting
- Cross-tabulation via pandas in Streamlit

---

## 📝 Example Workflows

### 1. Initial Exploration
```bash
# Start with static generator for overview
python visualize_duckdb_tables.py

# Review graphs/index.html to identify interesting patterns
# Then launch Streamlit for deep dive
streamlit run streamlit_duckdb_dashboard.py
```

### 2. EXIF Data Analysis
If your `images.db` contains EXIF data:
```python
# In Streamlit SQL Query tab:
SELECT 
    camera_make,
    camera_model,
    COUNT(*) as photo_count,
    AVG(focal_length) as avg_focal_length
FROM images
GROUP BY camera_make, camera_model
ORDER BY photo_count DESC
LIMIT 20;
```

### 3. Time-Based Analysis
```python
# Find patterns by capture date
SELECT 
    DATE_TRUNC('month', capture_date) as month,
    COUNT(*) as photos,
    AVG(exposure_time) as avg_exposure
FROM images
GROUP BY month
ORDER BY month;
```

---

## 🚀 Advanced Features

### Streamlit Caching
The dashboard uses `@st.cache_data` to speed up repeated queries:
- Schema information cached for 5 minutes
- Query results cached for 1 minute
- Clear cache with "Refresh Data" button

### Custom Visualizations
Both tools can be extended:
- Add new plot types in the visualization functions
- Implement domain-specific metrics
- Add export formats (PDF, Excel, etc.)

---

## 🐛 Troubleshooting

### "Database not found"
```bash
# Check current directory
pwd
# Verify database exists
ls -l images.db
# Or provide full path
python streamlit_duckdb_dashboard.py
# Then enter full path in sidebar: /path/to/images.db
```

### "Port 8501 already in use"
```bash
# Use different port
streamlit run streamlit_duckdb_dashboard.py --server.port 8502
```

### Memory Issues with Large Tables
```bash
# Reduce row limits
# Static: Edit limit=1000 in code
# Streamlit: Use slider to load fewer rows
```

---

## 📚 Resources

- [DuckDB Documentation](https://duckdb.org/docs/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Plotly Python Documentation](https://plotly.com/python/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)

---

## 🎓 Learning More

### DuckDB OLAP Features
- Use `QUALIFY` for window function filtering
- Leverage columnar storage for analytical queries
- Use `PIVOT` and `UNPIVOT` for data reshaping

### Streamlit Tips
- Use `st.cache_data` for expensive operations
- Session state for maintaining user selections
- Custom components for advanced interactions

---

## 📄 License

These tools are provided as-is for data analysis and visualization purposes.

## 🤝 Contributing

Feel free to extend these tools for your specific use cases:
- Add new visualization types
- Implement additional statistical tests
- Create domain-specific analysis modules
- Add export formats (PDF reports, Excel workbooks)

---

**Happy Analyzing! 🦆📊**
