#!/usr/bin/env python3
"""
DuckDB Table Visualization Generator
Automatically creates visualizations for all tables in a DuckDB database
"""

import duckdb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


class DuckDBVisualizer:
    def __init__(self, db_path):
        """Initialize connection to DuckDB database"""
        self.db_path = db_path
        self.conn = duckdb.connect(db_path, read_only=True)
        self.output_dir = Path("graphs")
        self.output_dir.mkdir(exist_ok=True)
        
    def get_all_tables(self):
        """Get list of all tables in the database"""
        query = """
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'main' 
        AND table_type = 'BASE TABLE'
        ORDER BY table_name
        """
        result = self.conn.execute(query).fetchall()
        return [row[0] for row in result]
    
    def get_table_info(self, table_name):
        """Get column information for a table"""
        query = f"""
        SELECT column_name, data_type 
        FROM information_schema.columns 
        WHERE table_name = '{table_name}'
        ORDER BY ordinal_position
        """
        return self.conn.execute(query).fetchdf()
    
    def get_table_data(self, table_name, limit=10000):
        """Get data from a table with optional limit"""
        query = f"SELECT * FROM {table_name} LIMIT {limit}"
        return self.conn.execute(query).fetchdf()
    
    def identify_column_types(self, df):
        """Classify columns by type for appropriate visualization"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
        
        # Limit categorical columns to those with reasonable cardinality
        categorical_cols = [col for col in categorical_cols 
                           if df[col].nunique() < 50 and df[col].nunique() > 1]
        
        return {
            'numeric': numeric_cols,
            'categorical': categorical_cols,
            'datetime': datetime_cols
        }
    
    def create_distribution_plots(self, df, table_name, numeric_cols):
        """Create distribution plots for numeric columns"""
        if not numeric_cols:
            return
        
        n_cols = min(len(numeric_cols), 4)
        n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        
        # Normalize axes to always be a flat array
        if n_rows == 1 and n_cols == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for idx, col in enumerate(numeric_cols):
            if idx < len(axes):
                data = df[col].dropna()
                if len(data) > 0:
                    axes[idx].hist(data, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
                    axes[idx].set_title(f'Distribution: {col}')
                    axes[idx].set_xlabel(col)
                    axes[idx].set_ylabel('Frequency')
                    axes[idx].grid(True, alpha=0.3)
        
        # Hide unused subplots
        for idx in range(len(numeric_cols), len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f"{table_name}_distributions.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Created distribution plots")
    
    def create_categorical_plots(self, df, table_name, categorical_cols):
        """Create bar plots for categorical columns"""
        if not categorical_cols:
            return
        
        n_cols = min(len(categorical_cols), 2)
        n_rows = (len(categorical_cols) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(8*n_cols, 5*n_rows))
        
        # Normalize axes to always be a flat array
        if n_rows == 1 and n_cols == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for idx, col in enumerate(categorical_cols):
            if idx < len(axes):
                value_counts = df[col].value_counts().head(20)  # Top 20 categories
                axes[idx].barh(range(len(value_counts)), value_counts.values, color='coral')
                axes[idx].set_yticks(range(len(value_counts)))
                axes[idx].set_yticklabels(value_counts.index)
                axes[idx].set_title(f'Top Values: {col}')
                axes[idx].set_xlabel('Count')
                axes[idx].grid(True, alpha=0.3, axis='x')
        
        # Hide unused subplots
        for idx in range(len(categorical_cols), len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f"{table_name}_categories.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Created categorical plots")
    
    def create_correlation_heatmap(self, df, table_name, numeric_cols):
        """Create correlation heatmap for numeric columns"""
        if len(numeric_cols) < 2:
            return
        
        # Limit to first 15 numeric columns for readability
        numeric_cols = numeric_cols[:15]
        correlation_matrix = df[numeric_cols].corr()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                   center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
        plt.title(f'Correlation Matrix: {table_name}')
        plt.tight_layout()
        plt.savefig(self.output_dir / f"{table_name}_correlation.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Created correlation heatmap")
    
    def create_time_series_plots(self, df, table_name, datetime_cols, numeric_cols):
        """Create time series plots if datetime columns exist"""
        if not datetime_cols or not numeric_cols:
            return
        
        for dt_col in datetime_cols[:1]:  # Use first datetime column
            # Select up to 4 numeric columns for time series
            plot_cols = numeric_cols[:4]
            
            fig, axes = plt.subplots(len(plot_cols), 1, figsize=(12, 3*len(plot_cols)))
            
            # Normalize axes to always be a list
            if len(plot_cols) == 1:
                axes = [axes]
            
            df_sorted = df.sort_values(dt_col)
            
            for idx, num_col in enumerate(plot_cols):
                axes[idx].plot(df_sorted[dt_col], df_sorted[num_col], 
                             marker='o', markersize=2, linewidth=1)
                axes[idx].set_title(f'{num_col} over time')
                axes[idx].set_xlabel(dt_col)
                axes[idx].set_ylabel(num_col)
                axes[idx].grid(True, alpha=0.3)
                axes[idx].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            plt.savefig(self.output_dir / f"{table_name}_timeseries.png", dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  ✓ Created time series plots")
            break
    
    def create_summary_stats(self, df, table_name):
        """Create a visualization of summary statistics"""
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.empty:
            return
        
        stats = numeric_df.describe().T
        
        fig, ax = plt.subplots(figsize=(12, max(6, len(stats) * 0.5)))
        
        # Create table
        cell_text = []
        for idx, row in stats.iterrows():
            cell_text.append([f'{val:.2f}' if not pd.isna(val) else 'N/A' 
                            for val in row])
        
        table = ax.table(cellText=cell_text,
                        rowLabels=stats.index,
                        colLabels=stats.columns,
                        cellLoc='center',
                        loc='center')
        
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        
        # Style header
        for i in range(len(stats.columns)):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        ax.axis('off')
        plt.title(f'Summary Statistics: {table_name}', fontsize=14, weight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(self.output_dir / f"{table_name}_summary.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Created summary statistics table")
    
    def visualize_table(self, table_name):
        """Create all appropriate visualizations for a table"""
        print(f"\n📊 Processing table: {table_name}")
        
        try:
            # Get table data
            df = self.get_table_data(table_name)
            
            if df.empty:
                print(f"  ⚠ Table {table_name} is empty, skipping...")
                return
            
            print(f"  Rows: {len(df)}, Columns: {len(df.columns)}")
            
            # Identify column types
            col_types = self.identify_column_types(df)
            
            # Create various visualizations with individual error handling
            try:
                self.create_summary_stats(df, table_name)
            except Exception as e:
                print(f"  ⚠ Could not create summary stats: {str(e)}")
            
            try:
                self.create_distribution_plots(df, table_name, col_types['numeric'])
            except Exception as e:
                print(f"  ⚠ Could not create distribution plots: {str(e)}")
            
            try:
                self.create_categorical_plots(df, table_name, col_types['categorical'])
            except Exception as e:
                print(f"  ⚠ Could not create categorical plots: {str(e)}")
            
            try:
                self.create_correlation_heatmap(df, table_name, col_types['numeric'])
            except Exception as e:
                print(f"  ⚠ Could not create correlation heatmap: {str(e)}")
            
            try:
                self.create_time_series_plots(df, table_name, col_types['datetime'], col_types['numeric'])
            except Exception as e:
                print(f"  ⚠ Could not create time series plots: {str(e)}")
                
        except Exception as e:
            print(f"  ❌ Error loading table data: {str(e)}")
            import traceback
            print(f"  Traceback: {traceback.format_exc()}")
    
    def generate_index_html(self, tables):
        """Generate an HTML index page to view all graphs"""
        html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>DuckDB Visualization Dashboard</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }
        h1 {
            color: #333;
            border-bottom: 3px solid #4CAF50;
            padding-bottom: 10px;
        }
        .table-section {
            background: white;
            margin: 20px 0;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .table-name {
            color: #4CAF50;
            font-size: 24px;
            margin-bottom: 15px;
        }
        .graph-container {
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
        }
        .graph-item {
            flex: 1 1 45%;
            min-width: 400px;
        }
        img {
            width: 100%;
            border: 1px solid #ddd;
            border-radius: 4px;
            cursor: pointer;
            transition: transform 0.2s;
        }
        img:hover {
            transform: scale(1.02);
        }
        .timestamp {
            color: #666;
            font-size: 14px;
            margin-top: 10px;
        }
    </style>
</head>
<body>
    <h1>🦆 DuckDB Database Visualization Dashboard</h1>
    <p class="timestamp">Generated: """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """</p>
"""
        
        for table in tables:
            html_content += f"""
    <div class="table-section">
        <h2 class="table-name">📋 {table}</h2>
        <div class="graph-container">
"""
            # List all possible graph types
            graph_types = [
                (f"{table}_summary.png", "Summary Statistics"),
                (f"{table}_distributions.png", "Distributions"),
                (f"{table}_categories.png", "Categorical Analysis"),
                (f"{table}_correlation.png", "Correlation Matrix"),
                (f"{table}_timeseries.png", "Time Series")
            ]
            
            for filename, title in graph_types:
                if (self.output_dir / filename).exists():
                    html_content += f"""
            <div class="graph-item">
                <h3>{title}</h3>
                <img src="{filename}" alt="{title}" onclick="window.open('{filename}', '_blank')">
            </div>
"""
            
            html_content += """
        </div>
    </div>
"""
        
        html_content += """
</body>
</html>
"""
        
        index_path = self.output_dir / "index.html"
        with open(index_path, 'w') as f:
            f.write(html_content)
        
        print(f"\n✅ Generated HTML dashboard: {index_path}")
        return index_path
    
    def visualize_all(self):
        """Main method to visualize all tables in the database"""
        print("="*60)
        print("🦆 DuckDB Table Visualizer")
        print("="*60)
        print(f"Database: {self.db_path}")
        print(f"Output directory: {self.output_dir}")
        
        # Get all tables
        tables = self.get_all_tables()
        print(f"\nFound {len(tables)} tables: {', '.join(tables)}")
        
        if not tables:
            print("⚠ No tables found in database!")
            return
        
        # Process each table
        successful = 0
        failed = 0
        for table in tables:
            try:
                self.visualize_table(table)
                successful += 1
            except Exception as e:
                print(f"  ❌ Error processing {table}: {str(e)}")
                import traceback
                print(f"  Full error: {traceback.format_exc()}")
                failed += 1
        
        # Generate HTML index
        try:
            self.generate_index_html(tables)
        except Exception as e:
            print(f"  ⚠ Error generating HTML index: {str(e)}")
        
        print("\n" + "="*60)
        print("✅ Visualization complete!")
        print(f"📊 Successfully processed: {successful}/{len(tables)} tables")
        if failed > 0:
            print(f"⚠️  Failed: {failed}/{len(tables)} tables")
        print(f"📁 All graphs saved to: {self.output_dir.absolute()}")
        print(f"🌐 Open index.html to view all visualizations")
        print("="*60)
    
    def close(self):
        """Close database connection"""
        self.conn.close()


def main():
    # Path to your DuckDB database
    db_path = "images.db"
    
    # Check if database exists
    if not Path(db_path).exists():
        print(f"❌ Error: Database '{db_path}' not found!")
        print(f"Current directory: {Path.cwd()}")
        return
    
    # Create visualizer and generate all graphs
    visualizer = DuckDBVisualizer(db_path)
    try:
        visualizer.visualize_all()
    finally:
        visualizer.close()


if __name__ == "__main__":
    main()
