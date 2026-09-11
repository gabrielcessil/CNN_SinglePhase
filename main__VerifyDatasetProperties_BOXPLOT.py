import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def analyze_and_plot_from_csv(csv_path: str, base_save_plot_path: str = None):
    """
    Reads a pre-calculated CSV of geometric properties, dynamically identifies 
    all variables, and plots/saves individual jittered boxplots for each property.
    """
    
    # ==========================================
    # 1. LOAD DATA
    # ==========================================
    print(f"--- Loading Data from {csv_path} ---")
    df_samples = pd.read_csv(csv_path, sep=",", decimal=".")
    
    # Revert underscores back to spaces for clean plotting labels
    df_samples.columns = df_samples.columns.str.replace('_', ' ')
    if "Dataset" in df_samples.columns:
        df_samples["Dataset"] = df_samples["Dataset"].str.replace('_', ' ')

    # ==========================================
    # 2. IDENTIFY VARIABLES TO PLOT
    # ==========================================
    # Exclude non-metric columns and grab all numeric columns dynamically
    exclude_cols = ["Dataset", "Sample Index"]
    variables = [
        col for col in df_samples.columns 
        if col not in exclude_cols and pd.api.types.is_numeric_dtype(df_samples[col])
    ]
    print(f"Found {len(variables)} properties to plot: {variables}")

    # ==========================================
    # 3. BOXPLOT GENERATION (Academic Style)
    # ==========================================
    print("\nGenerating Academic Boxplots...")
    
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'Bitstream Vera Serif'],
        'axes.labelsize': 16,
        'axes.titlesize': 18,
        'xtick.labelsize': 11,
        'ytick.labelsize': 14,
        'legend.fontsize': 14,
        'figure.titlesize': 20,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': '--'
    })
    
    datasets_labels = df_samples["Dataset"].unique()
    
    # Format labels to jump lines by replacing spaces with newline characters
    formatted_labels = [str(label).replace(" ", "\n") for label in datasets_labels]
    
    colors = plt.cm.tab10.colors  
    
    # Setup directory and base filename for saving
    if base_save_plot_path:
        out_dir = os.path.dirname(base_save_plot_path) or "."
        os.makedirs(out_dir, exist_ok=True)
        base_name, ext = os.path.splitext(os.path.basename(base_save_plot_path))
    
    for var in variables:
        fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
        
        # Group data per dataset for the boxplot list format
        data_list = [df_samples[df_samples["Dataset"] == ds][var].dropna().values for ds in datasets_labels]
        
        # Create base boxplot using the newly formatted multi-line labels
        bplot = ax.boxplot(data_list, tick_labels=formatted_labels, patch_artist=True, 
                           showfliers=False, widths=0.5, zorder=2)
        
        # Style boxes
        for j, patch in enumerate(bplot['boxes']):
            patch.set_facecolor('white')
            patch.set_edgecolor(colors[j % len(colors)])
            patch.set_linewidth(1.5)
        for median in bplot['medians']:
            median.set_color('black')
            median.set_linewidth(2)
        for element in ['whiskers', 'caps']:
            for line in bplot[element]:
                line.set_color('black')
                line.set_linewidth(1)
        
        # Add Jittered Scatter Dots
        for j, data in enumerate(data_list):
            if len(data) > 0:
                x = np.random.uniform(j + 1 - 0.15, j + 1 + 0.15, size=len(data))
                ax.scatter(x, data, alpha=0.6, facecolors='none', edgecolors='black', 
                           s=30, linewidths=1.0, zorder=3)
            
        ax.set_ylabel(var)
        ax.set_axisbelow(True) 
        
        # Apply Logarithmic scale for Permeability (or any specific variable if needed)
        if var == "Permeability":
            ax.set_yscale('log')

        # Save or Show
        if base_save_plot_path:
            # Create a safe, unique filename for this variable
            safe_var_name = var.replace(" ", "_").replace(".", "")
            current_save_path = os.path.join(out_dir, f"{base_name}_{safe_var_name}{ext}")
            
            plt.savefig(current_save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {current_save_path}")
        else:
            plt.show()
            
        # Close the figure to free up memory before the next loop iteration
        plt.close(fig)

# ==========================================
# RUN BLOCK
# ==========================================
if __name__ == "__main__":
    # Point this to your actual properties CSV
    csv_input_file = "./Tables/Dataset_Properties_Table_Test.csv" 
    
    analyze_and_plot_from_csv(
        csv_path=csv_input_file, 
        base_save_plot_path="./Plots/Dataset_Properties_Boxplot.png"
    )