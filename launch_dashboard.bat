@echo off
REM Launch script for DuckDB Streamlit Dashboard (Windows)

echo.
echo ============================================================
echo   DuckDB Interactive Dashboard
echo ============================================================
echo.
echo The dashboard will open in your default browser.
echo If it doesn't open automatically, visit:
echo http://localhost:8501
echo.
echo Press Ctrl+C to stop the server
echo ============================================================
echo.

streamlit run streamlit_duckdb_dashboard.py

pause
