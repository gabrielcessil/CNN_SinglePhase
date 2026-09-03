import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import ScalarFormatter

def plot_error_vs_geometry_correlations(csv_path: str, output_dir: str):
    """
    Reads the comprehensive sample metrics CSV, generates a correlation heatmap,
    and creates scatter plots for each error metric vs geometrical property.
    """
    # 1. Load Data
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Could not find {csv_path}. Run the extraction script first.")
        
    df = pd.read_csv(csv_path)
    
    # Drop columns that are entirely NaN
    df = df.dropna(axis=1, how='all')
    
    # 2. Dynamically Identify Columns
    # Include Tortuosity, but explicitly exclude any column with 'Error' to avoid 
    # grabbing 'Tortuosity_Error_[%]' as a geometrical property.
    geom_keywords = ['Porosity', 'Permeability', 'Thickness', 'Minkowski', 'Tortuosity']
    geom_cols = [col for col in df.columns if any(kw in col for kw in geom_keywords) and 'Error' not in col]
    
    # Error columns are whatever is left (excluding the 'Dataset' label)
    error_cols = [col for col in df.columns if col not in geom_cols and col != 'Dataset']
    
    os.makedirs(output_dir, exist_ok=True)
    
    # ==========================================
    # 3. GLOBAL PLOT STYLING (Academic)
    # ==========================================
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'Bitstream Vera Serif'],
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': '--'
    })

    # ==========================================
    # 4. CROSS-CORRELATION HEATMAP
    # ==========================================
    print("Generating Cross-Correlation Heatmap...")
    
    # Use Spearman correlation to account for non-linear relationships
    corr_matrix = df[geom_cols + error_cols].corr(method='spearman')
    
    # Isolate the intersection of Geometry (rows) vs Error (columns)
    cross_corr = corr_matrix.loc[geom_cols, error_cols]
    
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        cross_corr, 
        annot=True, 
        cmap='coolwarm', 
        fmt=".2f", 
        cbar_kws={'label': 'Spearman Correlation'},
        vmin=-1, vmax=1,
        ax=ax,
        square=True,
        linewidths=0.5
    )
    plt.title("Correlation: Geometrical Properties vs. Model Errors", pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    heatmap_path = os.path.join(output_dir, "Heatmap_Geometry_vs_Error.png")
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    plt.close()

    # ==========================================
    # 5. SCATTER PLOT GRIDS (One Figure per Error Metric)
    # ==========================================
    print("Generating Scatter Plots per Error Metric...")
    
    n_geom = len(geom_cols)
    cols_per_row = 3
    rows = int(np.ceil(n_geom / cols_per_row))
    
    # Define a consistent color palette based on unique datasets
    datasets = df['Dataset'].unique()
    palette = sns.color_palette("tab10", n_colors=len(datasets))
    dataset_colors = dict(zip(datasets, palette))

    for error_metric in error_cols:
        print(f"  Plotting {error_metric}...")
        
        fig, axes = plt.subplots(rows, cols_per_row, figsize=(cols_per_row * 6, rows * 5))
        axes = axes.flatten()
        
        for i, geom_prop in enumerate(geom_cols):
            ax = axes[i]
            
            # Scatter plot
            sns.scatterplot(
                data=df, 
                x=geom_prop, 
                y=error_metric, 
                hue='Dataset', 
                palette=dataset_colors,
                alpha=0.7, 
                edgecolor='black',
                s=50,
                ax=ax,
                legend=(i == 0) # Only put the legend on the first subplot
            )
            
            # Apply Formatting based on property type
            if "Permeability" in geom_prop:
                ax.set_xscale('log') # Use logarithmic scale for permeability
            elif "Volume" in geom_prop:
                formatter = ScalarFormatter(useMathText=True)
                formatter.set_scientific(True)
                formatter.set_powerlimits((-3, 3))
                ax.xaxis.set_major_formatter(formatter)

        # Hide any unused subplots
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])
            
        # Move legend outside the first plot for better visibility
        if len(datasets) > 0:
            axes[0].legend(title="Dataset", bbox_to_anchor=(1.05, 1), loc='upper left')

        fig.suptitle(f'Impact of Geometry on {error_metric.replace("_", " ")}', fontsize=20, y=1.02)
        plt.tight_layout()
        
        # Save figure safely
        safe_name = error_metric.replace('/', '').replace('\\', '').replace('[', '').replace(']', '').replace('%', 'Perc')
        plot_path = os.path.join(output_dir, f"Scatter_{safe_name}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

    print(f"\n✅ Plots saved successfully in: {output_dir}")

# =======================================================
# EXECUTION
# =======================================================
if __name__ == "__main__":
    
    CSV_FILE = "../NN_Results/Comprehensive_Sample_Metrics.csv"
    OUT_DIR  = "../NN_Results/Geometry_Correlations/"
    
    plot_error_vs_geometry_correlations(
        csv_path=CSV_FILE, 
        output_dir=OUT_DIR
    )