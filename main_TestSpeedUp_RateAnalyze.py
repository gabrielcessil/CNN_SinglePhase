import os
import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde

# ==============================================================================
# 1. CONFIGURATION & HELPERS
# ==============================================================================
"""
main_folders = {
    "Tol 1%":                   "../TestSpeedUp_Simulations_1e2/Test_Oliveira_BereaUpperGray_SAug_DNorm/",
    "Tol 0.01%":                "../TestSpeedUp_Simulations_1e4/Test_Oliveira_BereaUpperGray_SAug_DNorm/",
    "Tol 0.0001%":              "../TestSpeedUp_Simulations_1e6/Test_Oliveira_BereaUpperGray_SAug_DNorm/",
}
"""

"""
main_folders = {
    "Spherical Pores":          "../TestSpeedUp_Simulations_CrossDatasets/Test_Silveira_SphPore_SAug_DNorm/",
    "Spherical Grains":         "../TestSpeedUp_Simulations_CrossDatasets/Test_Silveira_SphGrain_SAug_DNorm/",
    "Leopard":                  "../TestSpeedUp_Simulations_CrossDatasets/Test_Oliveira_Leopard_SAug_DNorm/",
    "CastleGate":               "../TestSpeedUp_Simulations_CrossDatasets/Test_Oliveira_CastleGate_SAug_DNorm/",
    "Berea Upper Gray":         "../TestSpeedUp_Simulations_CrossDatasets/Test_Oliveira_BereaUpperGray_SAug_DNorm/",
    "Berea Sinter Gray":        "../TestSpeedUp_Simulations_CrossDatasets/Test_Oliveira_BereaSinterGray_SAug_DNorm/",
    "Berea Buff":               "../TestSpeedUp_Simulations_CrossDatasets/Test_Oliveira_BereaBuff_SAug_DNorm/",
    "Berea":                    "../TestSpeedUp_Simulations_CrossDatasets/Test_Oliveira_Berea_SAug_DNorm/",
    "Bentheimer":               "../TestSpeedUp_Simulations_CrossDatasets/Test_Oliveira_Bentheimer_SAug_DNorm/",
}
"""

main_folders = {
    "256 cubic":                   "../TestSpeedUp_Simulations_BiggerCrops/DRP247_256_256_256/",
    "512 cubic":                   "../TestSpeedUp_Simulations_BiggerCrops/DRP247_512_512_512/",
    }
remove_outliers = False
logscale        = True
dataset_colors = {}
for key in main_folders.keys():
    if key not in dataset_colors:
        dataset_colors[key] = "black"
        

def get_max_timestep_from_vis(folder_path):
    p = Path(folder_path)
    if not p.exists():
        return None

    vis_pattern = re.compile(r'^vis(\d+)$')
    max_ts = 0
    found_vis = False
    
    for item in p.iterdir():
        if item.is_dir():
            match = vis_pattern.match(item.name)
            if match:
                found_vis = True
                ts = int(match.group(1))
                if ts > max_ts:
                    max_ts = ts
                    
    return max_ts if found_vis else None

# ==============================================================================
# 2. DATA EXTRACTION
# ==============================================================================
results = []

for dataset_name, dataset_path in main_folders.items():
    dataset_p = Path(dataset_path)
    if not dataset_p.exists():
        continue
        
    for sample_folder in dataset_p.glob("Sample_*"):
        sample_name = sample_folder.name
        
        run_dir = sample_folder / "lbpm_grad_run"
        started_dir = sample_folder / "lbpm_nn_run"
        
        ts_standard = get_max_timestep_from_vis(run_dir)
        ts_started = get_max_timestep_from_vis(started_dir)
        
        results.append({
            "Dataset": dataset_name,
            "Sample": sample_name,
            "Standard_Timesteps": ts_standard,
            "NN_Started_Timesteps": ts_started
        })

df = pd.DataFrame(results)

# Clean up data: Drop rows where simulations failed (NaN) or took 0 timesteps
df = df.dropna(subset=["Standard_Timesteps", "NN_Started_Timesteps"])
df = df[df["NN_Started_Timesteps"] > 0]

# Calculate Speedup Ratio
df['Speedup_Ratio'] = df['Standard_Timesteps'] / df['NN_Started_Timesteps']

print("\nConvergence Timesteps & Speedup Comparison:")
print("-" * 75)
print(df.to_string(index=False))
print("-" * 75)

df.to_csv("timesteps_comparison.csv", index=False)
print("\nResults saved to timesteps_comparison.csv")

# ==============================================================================
# 3. GLOBAL ACADEMIC STYLING CONFIGURATION
# ==============================================================================
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 11,
        "axes.labelsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "axes.linewidth": 1.0,
    }
)

# ==============================================================================
# 4. PLOT GENERATION FUNCTION (Density-Proportional Scatter Boxplot)
# ==============================================================================
def plot_error_boxplots(df: pd.DataFrame, error_cols: list, output_dir: str, dataset_colors: dict, remove_outliers: bool = False, logscale = False):
    """
    Generates box plots overlaid with density-calculated data points.
    Synchronizes the scatter points precisely with the mathematical bounds of the boxplot whiskers.
    """
    print(f"\n--- Starting Speedup Boxplots (Outliers Removed from Scatter: {remove_outliers}) ---")
    os.makedirs(output_dir, exist_ok=True)
    
    datasets = df['Dataset'].unique()

    for error_metric in error_cols:
        print(f"  Plotting boxplot for {error_metric}...")
        
        fig, ax = plt.subplots(figsize=(9, 6), dpi=300)

        # 1. The Boxplot (Always uses the full DataFrame for stable IQR math)
        sns.boxplot(
            data=df,
            x='Dataset',
            y=error_metric,
            palette=dataset_colors,
            width=0.45,
            showfliers=False, # Hides seaborn's native outliers so we can draw our own density scatter
            zorder=4,
            boxprops=dict(linewidth=1.5, edgecolor='black', alpha=0.35, zorder=4),
            medianprops=dict(linewidth=2.0, color='black', zorder=5),
            whiskerprops=dict(linewidth=1.5, color='black', zorder=4),
            capprops=dict(linewidth=1.5, color='black', zorder=4),
            ax=ax,
            legend=False
        )

        # 2. Density-Proportional Scatter & Annotations
        for i, dataset in enumerate(datasets):
            y_vals = df[df['Dataset'] == dataset][error_metric].dropna().values
            base_color = dataset_colors.get(dataset, '#4c72b0')
            
            if len(y_vals) == 0:
                continue
            
            # --- Calculate Quartiles and Median for Annotations ---
            q1 = np.percentile(y_vals, 25)
            median = np.percentile(y_vals, 50)
            q3 = np.percentile(y_vals, 75)

            # --- Annotate Q3, Median, and Q1 beside the box in grey ---
            ax.text(i + 0.24, q3, f" {q3:.2f}", va='center', ha='left', fontsize=9, color='gray')
            ax.text(i + 0.24, median, f" {median:.2f}", va='center', ha='left', fontsize=10, color='gray', fontweight='bold')
            ax.text(i + 0.24, q1, f" {q1:.2f}", va='center', ha='left', fontsize=9, color='gray')
            
            # --- Filter scatter dots to perfectly match the boxplot whiskers ---
            if remove_outliers:
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                
                mask = (y_vals >= lower_bound) & (y_vals <= upper_bound)
                y_vals = y_vals[mask]
                
                if len(y_vals) == 0:
                    continue

            # --- Density Calculation (Log Space) ---
            if len(y_vals) > 2:
                try:
                    y_vals_safe = np.clip(y_vals, a_min=1e-12, a_max=None)
                    y_vals_log = np.log10(y_vals_safe)
                    
                    kde = gaussian_kde(y_vals_log)
                    density = kde(y_vals_log)
                    
                    density_norm = density / density.max()
                except np.linalg.LinAlgError:
                    density_norm = np.ones_like(y_vals)
            else:
                density_norm = np.ones_like(y_vals)

            # --- Apply Horizontal Jitter ---
            max_jitter = 0.25 # slightly reduced to accommodate text labels on the right
            jitter = np.random.uniform(-1, 1, size=len(y_vals)) * max_jitter * density_norm
            x_vals = i + jitter
            
            # Plot the scatter behind the boxplot (zorder=2)
            ax.scatter(
                x_vals, 
                y_vals, 
                color=base_color, 
                s=20, 
                alpha=0.75, 
                edgecolor='black', 
                linewidths=0.4, 
                zorder=2
            )

        # 3. Formatting and Scientific Frame
        ax.set_ylabel("Speed-up Ratio", fontweight="bold")
        ax.set_xlabel("Tolerance Configuration", fontweight="bold")
        
        if logscale: ax.set_yscale('log')

        # Add a horizontal reference line at Speedup = 1.0 (baseline comparison)
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=1.0, alpha=0.7, label="No Speedup (1.0x)", zorder=1)

        ax.set_xticks(range(len(datasets)))
        ax.set_xticklabels(datasets, rotation=30, ha='right', rotation_mode='anchor')

        ax.tick_params(axis='both', which='major', direction='in', top=True, right=True, length=6, width=1.5)
        ax.tick_params(axis='y', which='minor', direction='in', right=True, length=3, width=1.0)

        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.5)

        ax.grid(True, alpha=0.3, which="both", ls="--", axis="y", zorder=1)
        ax.set_axisbelow(True)

        plt.tight_layout()
        
        safe_name = error_metric.replace('/', '').replace('\\', '')
        suffix = "_NoOutliers" if remove_outliers else ""
        
        plt.savefig(os.path.join(output_dir, f"Boxplot_{safe_name}{suffix}.pdf"), bbox_inches='tight')
        plt.savefig(os.path.join(output_dir, f"Boxplot_{safe_name}{suffix}.png"), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved Boxplot_{safe_name}{suffix}.pdf/.png to {output_dir}")

# Execute the plotting function for Speedup_Ratio
plot_error_boxplots(
    df=df,
    error_cols=["Speedup_Ratio"],
    output_dir="./Plots/",
    dataset_colors=dataset_colors,
    remove_outliers=remove_outliers,
    logscale=logscale
)