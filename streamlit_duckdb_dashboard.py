#!/usr/bin/env python3
"""
DuckDB Interactive Dashboard with Streamlit
Real-time interactive visualization and analysis of DuckDB databases
"""

import streamlit as st
import duckdb
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from pathlib import Path
from datetime import datetime
import json

# Page configuration
st.set_page_config(
    page_title="DuckDB Analytics Dashboard",
    page_icon="🦆",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #4CAF50;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
</style>
""", unsafe_allow_html=True)


class DuckDBExplorer:
    """Interactive DuckDB database explorer
    
    Note: Caching is handled at the connection level only to avoid
    JSON serialization issues with pandas ObjectDType. Query results
    are not cached to prevent serialization errors.
    """
    
    def __init__(self, db_path):
        """Initialize database connection"""
        self.db_path = db_path
    
    @st.cache_resource
    def _get_connection(_db_path):
        """Create cached database connection"""
        return duckdb.connect(_db_path, read_only=True)
    
    @property
    def conn(self):
        """Get database connection"""
        return DuckDBExplorer._get_connection(self.db_path)
    
    def get_tables(self):
        """Get list of all tables"""
        query = """
        SELECT table_name, 
               (SELECT COUNT(*) FROM information_schema.columns c 
                WHERE c.table_name = t.table_name) as column_count
        FROM information_schema.tables t
        WHERE table_schema = 'main' 
        AND table_type = 'BASE TABLE'
        ORDER BY table_name
        """
        df = self.conn.execute(query).fetchdf()
        # Convert all object columns to string to avoid serialization issues
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].astype(str)
        return df
    
    def get_table_schema(self, table_name):
        """Get schema information for a table"""
        query = f"""
        SELECT column_name, data_type, is_nullable
        FROM information_schema.columns 
        WHERE table_name = '{table_name}'
        ORDER BY ordinal_position
        """
        df = self.conn.execute(query).fetchdf()
        # Convert all object columns to string
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].astype(str)
        return df
    
    def get_table_stats(self, table_name):
        """Get basic statistics for a table"""
        row_count = self.conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        
        return {
            'row_count': row_count,
            'column_count': len(self.get_table_schema(table_name))
        }
    
    def query_table(self, table_name, limit=1000, offset=0, where_clause=None):
        """Query table data with optional filtering"""
        query = f"SELECT * FROM {table_name}"
        if where_clause:
            query += f" WHERE {where_clause}"
        query += f" LIMIT {limit} OFFSET {offset}"
        
        try:
            df = self.conn.execute(query).fetchdf()
            # Convert object dtypes to string to avoid serialization issues
            for col in df.select_dtypes(include=['object']).columns:
                try:
                    df[col] = df[col].astype(str)
                except:
                    # If conversion fails, keep as is
                    pass
            return df
        except Exception as e:
            st.error(f"Query error: {str(e)}")
            return pd.DataFrame()
    
    def execute_custom_query(_self, query):
        """Execute custom SQL query (no caching for custom queries)"""
        try:
            df = _self.conn.execute(query).fetchdf()
            # Convert object dtypes to string to avoid serialization issues
            for col in df.select_dtypes(include=['object']).columns:
                try:
                    df[col] = df[col].astype(str)
                except:
                    pass
            return df
        except Exception as e:
            st.error(f"Query error: {str(e)}")
            return pd.DataFrame()
    
    def get_column_stats(self, table_name, column_name, data_type):
        """Get statistics for a specific column"""
        stats = {}
        
        try:
            if 'INT' in data_type.upper() or 'DOUBLE' in data_type.upper() or 'DECIMAL' in data_type.upper() or 'FLOAT' in data_type.upper():
                query = f"""
                SELECT 
                    MIN({column_name}) as min_val,
                    MAX({column_name}) as max_val,
                    AVG({column_name}) as avg_val,
                    STDDEV({column_name}) as std_val,
                    COUNT(DISTINCT {column_name}) as distinct_count,
                    COUNT(*) - COUNT({column_name}) as null_count
                FROM {table_name}
                """
                result = self.conn.execute(query).fetchone()
                stats = {
                    'min': float(result[0]) if result[0] is not None else None,
                    'max': float(result[1]) if result[1] is not None else None,
                    'avg': float(result[2]) if result[2] is not None else None,
                    'std': float(result[3]) if result[3] is not None else None,
                    'distinct': int(result[4]) if result[4] is not None else 0,
                    'nulls': int(result[5]) if result[5] is not None else 0
                }
            else:
                query = f"""
                SELECT 
                    COUNT(DISTINCT {column_name}) as distinct_count,
                    COUNT(*) - COUNT({column_name}) as null_count
                FROM {table_name}
                """
                result = self.conn.execute(query).fetchone()
                stats = {
                    'distinct': int(result[0]) if result[0] is not None else 0,
                    'nulls': int(result[1]) if result[1] is not None else 0
                }
        except Exception as e:
            st.warning(f"Could not compute stats for {column_name}: {str(e)}")
        
        return stats


def create_distribution_plot(df, column):
    """Create interactive distribution plot"""
    if pd.api.types.is_numeric_dtype(df[column]):
        # Ensure numeric data is clean
        data = df[column].dropna()
        fig = px.histogram(data, x=data, nbins=50, 
                          title=f'Distribution of {column}',
                          color_discrete_sequence=['#4CAF50'])
        fig.update_layout(showlegend=False, height=400)
        return fig
    else:
        # For categorical, show top 20 values
        value_counts = df[column].value_counts().head(20)
        # Convert index to list of strings to avoid object dtype issues
        categories = [str(x) for x in value_counts.index]
        values = value_counts.values.tolist()
        
        fig = px.bar(x=values, y=categories, 
                    orientation='h',
                    title=f'Top 20 Values: {column}',
                    color_discrete_sequence=['#FF6B6B'])
        fig.update_layout(showlegend=False, height=max(400, len(value_counts) * 25))
        fig.update_xaxes(title='Count')
        fig.update_yaxes(title='')
        return fig


def create_correlation_heatmap(df):
    """Create correlation heatmap for numeric columns"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) < 2:
        return None
    
    corr_matrix = df[numeric_cols].corr()
    
    # Convert column names to strings to avoid serialization issues
    col_names = [str(col) for col in corr_matrix.columns]
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values.tolist(),
        x=col_names,
        y=col_names,
        colorscale='RdBu',
        zmid=0,
        text=corr_matrix.values.round(2).tolist(),
        texttemplate='%{text}',
        textfont={"size": 10},
        colorbar=dict(title="Correlation")
    ))
    
    fig.update_layout(
        title='Correlation Matrix',
        height=600,
        xaxis={'side': 'bottom'}
    )
    
    return fig


def create_scatter_matrix(df, columns):
    """Create scatter plot matrix"""
    if len(columns) < 2:
        return None
    
    # Ensure columns are strings and limit to 5
    col_list = [str(col) for col in columns[:5]]
    
    # Create a clean DataFrame with only numeric data
    plot_df = df[columns[:5]].copy()
    plot_df.columns = col_list
    
    fig = px.scatter_matrix(plot_df, dimensions=col_list, 
                           title='Scatter Matrix (first 5 numeric columns)')
    fig.update_traces(diagonal_visible=False, showupperhalf=False)
    fig.update_layout(height=800)
    
    return fig


def create_time_series_plot(df, date_col, value_cols):
    """Create time series plot"""
    fig = go.Figure()
    
    for col in value_cols[:5]:  # Limit to 5 series
        # Convert to lists to avoid serialization issues
        x_data = df[date_col].tolist()
        y_data = df[col].tolist()
        
        fig.add_trace(go.Scatter(
            x=x_data, 
            y=y_data,
            mode='lines+markers',
            name=str(col),
            marker=dict(size=4)
        ))
    
    fig.update_layout(
        title=f'Time Series Analysis',
        xaxis_title=str(date_col),
        yaxis_title='Value',
        height=500,
        hovermode='x unified'
    )
    
    return fig


def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<p class="main-header">🦆 DuckDB Interactive Analytics Dashboard</p>', 
                unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Database selection
        db_path = st.text_input("Database Path", value="images.db")
        
        if not Path(db_path).exists():
            st.error(f"❌ Database '{db_path}' not found!")
            st.stop()
        
        # Initialize explorer
        explorer = DuckDBExplorer(db_path)
        
        # Get tables
        tables_df = explorer.get_tables()
        
        if tables_df.empty:
            st.error("No tables found in database!")
            st.stop()
        
        st.success(f"✅ Connected to database")
        st.metric("Tables", len(tables_df))
        
        # Table selection
        st.subheader("📋 Select Table")
        table_name = st.selectbox(
            "Table",
            options=tables_df['table_name'].tolist(),
            format_func=lambda x: f"{x} ({tables_df[tables_df['table_name']==x]['column_count'].values[0]} cols)"
        )
        
        # Data limit
        st.subheader("🔢 Data Limits")
        row_limit = st.slider("Rows to load", 100, 10000, 1000, 100)
        
        # Refresh button
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
    
    # Main content
    if table_name:
        # Get table statistics
        stats = explorer.get_table_stats(table_name)
        schema = explorer.get_table_schema(table_name)
        
        # Display metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📊 Total Rows", f"{stats['row_count']:,}")
        with col2:
            st.metric("📐 Columns", stats['column_count'])
        with col3:
            st.metric("💾 Rows Loaded", f"{min(row_limit, stats['row_count']):,}")
        
        # Load data
        with st.spinner("Loading data..."):
            df = explorer.query_table(table_name, limit=row_limit)
        
        if df.empty:
            st.warning("No data available")
            st.stop()
        
        # Create tabs
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📊 Overview", 
            "🔍 Data Explorer", 
            "📈 Visualizations",
            "🔗 Relationships",
            "💻 SQL Query",
            "📋 Schema"
        ])
        
        # Tab 1: Overview
        with tab1:
            st.subheader("Data Preview")
            st.dataframe(df.head(100), use_container_width=True, height=400)
            
            st.subheader("Summary Statistics")
            numeric_df = df.select_dtypes(include=[np.number])
            if not numeric_df.empty:
                st.dataframe(numeric_df.describe(), use_container_width=True)
            else:
                st.info("No numeric columns to summarize")
            
            # Data types
            st.subheader("Column Types")
            type_counts = df.dtypes.value_counts()
            col1, col2 = st.columns(2)
            
            with col1:
                # Convert dtype objects to strings for plotting
                type_labels = [str(dtype) for dtype in type_counts.index]
                type_values = type_counts.values.tolist()
                
                fig = px.pie(values=type_values, names=type_labels,
                           title='Column Type Distribution')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.dataframe(pd.DataFrame({
                    'Data Type': [str(dtype) for dtype in type_counts.index],
                    'Count': type_counts.values.tolist()
                }), use_container_width=True, hide_index=True)
        
        # Tab 2: Data Explorer
        with tab2:
            st.subheader("Interactive Data Explorer")
            
            # Column selector
            all_columns = df.columns.tolist()
            selected_columns = st.multiselect(
                "Select columns to display",
                options=all_columns,
                default=all_columns[:10]
            )
            
            if selected_columns:
                # Filters
                st.subheader("Filters")
                filter_col = st.selectbox("Filter by column", ["None"] + all_columns)
                
                filtered_df = df.copy()
                
                if filter_col != "None":
                    if pd.api.types.is_numeric_dtype(df[filter_col]):
                        min_val = float(df[filter_col].min())
                        max_val = float(df[filter_col].max())
                        range_vals = st.slider(
                            f"Range for {filter_col}",
                            min_val, max_val, (min_val, max_val)
                        )
                        filtered_df = filtered_df[
                            (filtered_df[filter_col] >= range_vals[0]) & 
                            (filtered_df[filter_col] <= range_vals[1])
                        ]
                    else:
                        unique_vals = df[filter_col].unique()[:50]  # Limit to 50 unique values
                        selected_vals = st.multiselect(
                            f"Select values for {filter_col}",
                            options=unique_vals,
                            default=list(unique_vals[:5])
                        )
                        if selected_vals:
                            filtered_df = filtered_df[filtered_df[filter_col].isin(selected_vals)]
                
                # Display filtered data
                st.dataframe(filtered_df[selected_columns], use_container_width=True, height=500)
                
                # Download button
                csv = filtered_df[selected_columns].to_csv(index=False)
                st.download_button(
                    label="📥 Download CSV",
                    data=csv,
                    file_name=f"{table_name}_export.csv",
                    mime="text/csv"
                )
        
        # Tab 3: Visualizations
        with tab3:
            st.subheader("Column Distributions")
            
            # Column selector for visualization
            viz_column = st.selectbox("Select column to visualize", df.columns.tolist())
            
            if viz_column:
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    fig = create_distribution_plot(df, viz_column)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.subheader("Column Statistics")
                    col_stats = explorer.get_column_stats(
                        table_name, 
                        viz_column,
                        schema[schema['column_name'] == viz_column]['data_type'].values[0]
                    )
                    
                    for key, value in col_stats.items():
                        if value is not None:
                            if isinstance(value, float):
                                st.metric(key.title(), f"{value:.2f}")
                            else:
                                st.metric(key.title(), f"{value:,}")
            
            # Multiple column comparison
            st.subheader("Multi-Column Analysis")
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numeric_cols) >= 2:
                col1, col2 = st.columns(2)
                with col1:
                    x_col = st.selectbox("X-axis", numeric_cols, key='x')
                with col2:
                    y_col = st.selectbox("Y-axis", numeric_cols, 
                                        index=min(1, len(numeric_cols)-1), key='y')
                
                if x_col and y_col:
                    # Create clean data for plotting - only numeric values
                    plot_df = df[[x_col, y_col]].copy()
                    plot_df = plot_df.dropna()
                    
                    fig = px.scatter(plot_df, x=x_col, y=y_col, 
                                   title=f'{x_col} vs {y_col}',
                                   color_discrete_sequence=['#4CAF50'])
                    fig.update_traces(marker=dict(size=8, opacity=0.6))
                    st.plotly_chart(fig, use_container_width=True)
        
        # Tab 4: Relationships
        with tab4:
            st.subheader("Data Relationships")
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numeric_cols) >= 2:
                # Correlation heatmap
                fig = create_correlation_heatmap(df)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                
                # Scatter matrix
                if len(numeric_cols) >= 2:
                    with st.expander("🔍 Scatter Matrix", expanded=False):
                        selected_cols = st.multiselect(
                            "Select columns (max 5)",
                            numeric_cols,
                            default=numeric_cols[:min(4, len(numeric_cols))]
                        )
                        if len(selected_cols) >= 2:
                            fig = create_scatter_matrix(df, selected_cols)
                            if fig:
                                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Need at least 2 numeric columns for relationship analysis")
            
            # Time series
            datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
            if datetime_cols and numeric_cols:
                st.subheader("Time Series Analysis")
                date_col = st.selectbox("Select date column", datetime_cols)
                value_cols = st.multiselect("Select value columns (max 5)", 
                                          numeric_cols,
                                          default=numeric_cols[:min(3, len(numeric_cols))])
                
                if value_cols:
                    fig = create_time_series_plot(df, date_col, value_cols)
                    st.plotly_chart(fig, use_container_width=True)
        
        # Tab 5: SQL Query
        with tab5:
            st.subheader("Custom SQL Query")
            st.info("Write custom DuckDB SQL queries to analyze your data")
            
            # Query templates
            templates = {
                "Select all": f"SELECT * FROM {table_name} LIMIT 100",
                "Count rows": f"SELECT COUNT(*) as total_rows FROM {table_name}",
                "Group by": f"SELECT column_name, COUNT(*) as count FROM {table_name} GROUP BY column_name ORDER BY count DESC LIMIT 10",
                "Custom": ""
            }
            
            template_choice = st.selectbox("Query Template", list(templates.keys()))
            
            query = st.text_area(
                "SQL Query",
                value=templates[template_choice],
                height=150,
                help="Enter your DuckDB SQL query"
            )
            
            col1, col2 = st.columns([1, 4])
            with col1:
                execute_btn = st.button("▶️ Execute Query", use_container_width=True)
            
            if execute_btn and query:
                with st.spinner("Executing query..."):
                    result = explorer.execute_custom_query(query)
                    
                    if not result.empty:
                        st.success(f"Query returned {len(result)} rows")
                        st.dataframe(result, use_container_width=True, height=400)
                        
                        # Export results
                        csv = result.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Results",
                            data=csv,
                            file_name="query_results.csv",
                            mime="text/csv"
                        )
                    else:
                        st.warning("Query returned no results")
        
        # Tab 6: Schema
        with tab6:
            st.subheader("Table Schema")
            
            # Display schema
            st.dataframe(schema, use_container_width=True, hide_index=True)
            
            # Export schema
            col1, col2 = st.columns(2)
            with col1:
                if st.button("📋 Copy CREATE TABLE"):
                    create_stmt = f"CREATE TABLE {table_name} (\n"
                    for idx, row in schema.iterrows():
                        nullable = "" if row['is_nullable'] == 'YES' else "NOT NULL"
                        create_stmt += f"  {row['column_name']} {row['data_type']} {nullable},\n"
                    create_stmt = create_stmt.rstrip(',\n') + "\n);"
                    st.code(create_stmt, language='sql')
            
            with col2:
                # Column details
                st.subheader("Column Details")
                for idx, row in schema.iterrows():
                    with st.expander(f"📊 {row['column_name']}"):
                        st.write(f"**Type:** {row['data_type']}")
                        st.write(f"**Nullable:** {row['is_nullable']}")
                        
                        stats = explorer.get_column_stats(
                            table_name, 
                            row['column_name'],
                            row['data_type']
                        )
                        if stats:
                            for key, val in stats.items():
                                if val is not None:
                                    st.write(f"**{key.title()}:** {val}")


if __name__ == "__main__":
    main()
