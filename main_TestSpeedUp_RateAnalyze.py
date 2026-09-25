import os
import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
from scipy.stats import gaussian_kde

# ==============================================================================
# 1. CONFIGURATION & HELPERS
# ==============================================================================
"""
file = "SpeedUp_tolerance"
main_folders = {
    "Tol 1%":                   "../TestSpeedUp_Simulations_1e2/Test_Oliveira_BereaUpperGray_SAug_DNorm/",
    "Tol 0.01%":                "../TestSpeedUp_Simulations_1e4/Test_Oliveira_BereaUpperGray_SAug_DNorm/",
    "Tol 0.0001%":              "../TestSpeedUp_Simulations_1e6/Test_Oliveira_BereaUpperGray_SAug_DNorm/",
}
"""

"""
file = "SpeedUp_crossDataset"
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

#"""
file = "SpeedUp_256"
main_folders = {
    "256³ Sandstone":                  "../TestSpeedUp_Simulations_BiggerCrops/DRP247_256_256_256/",
    "256³ Pre-salt":                   "../TestSpeedUp_Simulations_BiggerCrops/DRP503_256_256_256_sw02/",
    "256³ I.C Doddington":             "../TestSpeedUp_Simulations_BiggerCrops/IC_Doddington_256_256_256/",
    "256³ I.C Estaillades":            "../TestSpeedUp_Simulations_BiggerCrops/IC_Estaillades_256_256_256/",
    "256³ I.C Ketton":                 "../TestSpeedUp_Simulations_BiggerCrops/IC_Ketton_256_256_256/",
}  
#"""
"""
file = "SpeedUp_512"
main_folders = {  
    "512³ Sandstone":                  "../TestSpeedUp_Simulations_BiggerCrops/DRP247_512_512_512/",
    "512³ Pre-salt":                   "../TestSpeedUp_Simulations_BiggerCrops/DRP503_512_512_512_sw02/",
    "500³ I.C Doddington":             "../TestSpeedUp_Simulations_BiggerCrops/IC_Doddington_500_500_500/",
    "500³ I.C Estaillades":            "../TestSpeedUp_Simulations_BiggerCrops/IC_Estaillades_500_500_500/",
    "500³ I.C Ketton":                 "../TestSpeedUp_Simulations_BiggerCrops/IC_Ketton_500_500_500/",
}
"""

remove_outliers = False
logscale        = True
dataset_colors = {}
for key in main_folders.keys():
    if key not in dataset_colors:
        dataset_colors[key] = "black"

# ---------------------------------------------------------
# NEW: Convergence Analysis Settings
# ---------------------------------------------------------
analysis_interval = 200  # The timestep interval between permeability records
convergence_tolerance = 1.0  # X% difference allowed from the reference permeability


def get_permeability_data(folder_path):
    """
    Reads the Permeability.csv file from the simulation folder and 
    returns the entire DataFrame (useful for historical tracking).
    """
    p = Path(folder_path)
    if not p.exists():
        return None
        
    perm_file = p / "Permeability.csv"
    
    if perm_file.exists():
        try:
            # The CSV data is whitespace-separated
            df_perm = pd.read_csv(perm_file, sep=r'\s+')
            if 'absperm(mDa)' in df_perm.columns:
                return df_perm
        except Exception as e:
            print(f"Warning: Could not read {perm_file} due to {e}")
            
    return None

def get_convergence_step(df_sim, k_ref, tol=1.0, interval=200):
    """
    Calculates the first timestep of the final uninterrupted sequence 
    where the simulation permeability stays within `tol` % of `k_ref`.
    """
    if df_sim is None or len(df_sim) == 0:
        return np.nan
        
    k_array = df_sim['absperm(mDa)'].values
    
    # Safely infer the real timesteps based on the df columns
    if 'Step' in df_sim.columns:
        steps = df_sim['Step'].values
        if np.max(steps) <= len(steps):
            steps = steps * interval
    elif 'Time(s)' in df_sim.columns:
        steps = df_sim['Time(s)'].values
    else:
        steps = np.arange(1, len(k_array) + 1) * interval
        
    # Calculate percentage error relative to the reference value
    errors = 100.0 * np.abs(k_array - k_ref) / (np.abs(k_ref) + 1e-15)
    
    conv_step = np.nan
    
    # Iterate backwards from the end of the simulation.
    # As long as the error is <= tol, we keep moving back.
    # The moment we hit an error > tol, we break. The last valid step 
    # we saw is the "entering instant" of the final uninterrupted sequence.
    for i in range(len(errors) - 1, -1, -1):
        if errors[i] <= tol:
            conv_step = steps[i]
        else:
            break
            
    return conv_step

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
        
        df_grad = get_permeability_data(run_dir)
        df_nn = get_permeability_data(started_dir)
        
        if df_grad is None or df_nn is None:
            continue
            
        k_array_grad = df_grad['absperm(mDa)'].values
        k_array_nn = df_nn['absperm(mDa)'].values
        
        if len(k_array_grad) == 0 or len(k_array_nn) == 0:
            continue
            
        # 1. Establish the reference permeability (Final value of Grad run)
        k_ref = k_array_grad[-1]
        
        # 2, 3, 4. Find the convergence entering instant for both runs based on the tolerance
        ts_standard = get_convergence_step(df_grad, k_ref, tol=convergence_tolerance, interval=analysis_interval)
        ts_started = get_convergence_step(df_nn, k_ref, tol=convergence_tolerance, interval=analysis_interval)
        
        # Keep final permeabilities to check if the end state had structural issues
        k_standard = k_array_grad[-1]
        k_started = k_array_nn[-1]
        
        results.append({
            "Dataset": dataset_name,
            "Sample": sample_name,
            "Standard_Timesteps": ts_standard,
            "NN_Started_Timesteps": ts_started,
            "Standard_Permeability": k_standard,
            "NN_Permeability": k_started
        })

df = pd.DataFrame(results)

# Clean up data: Drop rows where simulations didn't hit the convergence threshold before finishing
dropped_unconverged = df['Standard_Timesteps'].isna() | df['NN_Started_Timesteps'].isna()
if dropped_unconverged.any():
    print(f"Warning: Dropped {dropped_unconverged.sum()} samples that failed to reach {convergence_tolerance}% convergence.")

df = df.dropna(subset=["Standard_Timesteps", "NN_Started_Timesteps"])
df = df[df["NN_Started_Timesteps"] > 0]

# Calculate Speedup Ratio (using the convergence steps) & Permeability Relative Error
df['Speedup_Ratio'] = df['Standard_Timesteps'] / df['NN_Started_Timesteps']
df['Perm_Error [%]'] = 100 * np.abs(df['NN_Permeability'] - df['Standard_Permeability']) / np.abs(df['Standard_Permeability'])

print("\nConvergence Timesteps & Speedup Comparison:")
print("-" * 75)
print(df[['Dataset', 'Sample', 'Speedup_Ratio', 'Perm_Error [%]']].to_string(index=False))
print("-" * 75)

os.makedirs("./Tables", exist_ok=True)
df.to_csv("./Tables/"+file+".csv", index=False)
print("\nResults saved to ./Tables/"+file+".csv")

# ==============================================================================
# 3. GLOBAL ACADEMIC STYLING CONFIGURATION & PLOTTING
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

def plot_prop_boxplots(df: pd.DataFrame, prop_cols: list, output_dir: str, suffix:str, dataset_colors: dict, remove_outliers: bool = False, logscale = False):
    print(f"\n--- Starting Speedup Boxplots (Outliers Removed from Scatter: {remove_outliers}) ---")
    os.makedirs(output_dir, exist_ok=True)
    
    datasets = df['Dataset'].unique()

    for prop in prop_cols:
        print(f"  Plotting boxplot for {prop}...")
        
        fig, ax = plt.subplots(figsize=(9, 6), dpi=300)

        # 1. The Boxplot (Always uses the full DataFrame for stable IQR math)
        sns.boxplot(
            data=df,
            x='Dataset',
            y=prop,
            palette=dataset_colors,
            width=0.45,
            showfliers=False,
            zorder=4,
            boxprops=dict(linewidth=1.5, edgecolor='black', alpha=0.35, zorder=4),
            medianprops=dict(linewidth=2.0, color='black', zorder=5),
            whiskerprops=dict(linewidth=1.5, color='black', zorder=4),
            capprops=dict(linewidth=1.5, color='black', zorder=4),
            ax=ax,
            legend=False
        )

        # Boolean flags to track which highlights were actually plotted
        has_red = False
        has_yellow = False
        has_green = False

        # 2. Density-Proportional Scatter & Annotations
        for i, dataset in enumerate(datasets):
            subset = df[df['Dataset'] == dataset].dropna(subset=[prop, 'Perm_Error [%]'])
            y_vals = subset[prop].values
            perm_errors = subset['Perm_Error [%]'].values
            
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
                perm_errors = perm_errors[mask] # Mask perm_errors concurrently to keep synced
                
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
            max_jitter = 0.25 
            jitter = np.random.uniform(-1, 1, size=len(y_vals)) * max_jitter * density_norm
            x_vals = i + jitter
            
            # --- Split data by FINAL Permeability Error for styling ---
            mask_red = perm_errors > 5
            mask_yellow = (perm_errors > 2) & (perm_errors <= 5)
            mask_green = (perm_errors > 1) & (perm_errors <= 2)
            mask_normal = perm_errors <= 1
            
            # Update global tracking flags
            if np.any(mask_red): has_red = True
            if np.any(mask_yellow): has_yellow = True
            if np.any(mask_green): has_green = True
            
            # Plot normal items (<= 1% diff)
            if np.any(mask_normal):
                ax.scatter(
                    x_vals[mask_normal], 
                    y_vals[mask_normal], 
                    color=base_color, 
                    s=20, 
                    alpha=0.75, 
                    edgecolor='black', 
                    linewidths=0.4, 
                    zorder=2
                )
                
            # Plot Green squares (1% < diff <= 2%)
            if np.any(mask_green):
                ax.scatter(
                    x_vals[mask_green], 
                    y_vals[mask_green], 
                    color='green', 
                    marker='s', 
                    s=80, 
                    alpha=1.0, 
                    edgecolor='black', 
                    linewidths=0.3, 
                    zorder=3
                )
                
            # Plot Yellow triangles (2% < diff <= 5%)
            if np.any(mask_yellow):
                ax.scatter(
                    x_vals[mask_yellow], 
                    y_vals[mask_yellow], 
                    color='gold', 
                    marker='^', 
                    s=80, 
                    alpha=1.0, 
                    edgecolor='black', 
                    linewidths=0.3, 
                    zorder=3
                )

            # Plot Red stars (> 5% diff)
            if np.any(mask_red):
                ax.scatter(
                    x_vals[mask_red], 
                    y_vals[mask_red], 
                    color='red', 
                    marker='*', 
                    s=80, 
                    alpha=1.0, 
                    edgecolor='black', 
                    linewidths=0.3, 
                    zorder=4
                )

        # 3. Formatting and Scientific Frame
        ax.set_ylabel("Speed-up Ratio", fontweight="bold")
        ax.set_xlabel("Tolerance Configuration", fontweight="bold")
        
        if logscale: ax.set_yscale('log')

        # Add horizontal reference line
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

        # Add custom Legend for Reference Line and Dynamic Highlighting Shapes
        handles, labels = ax.get_legend_handles_labels()
        
        if has_red:
            handles.append(Line2D([0], [0], marker='*', color='w', markerfacecolor='red', markersize=12, label='Final Perm. Diff > 5%'))
        if has_yellow:
            handles.append(Line2D([0], [0], marker='^', color='w', markerfacecolor='gold', markersize=12, label='2% < Final Perm. Diff $\leq$ 5%'))
        if has_green:
            handles.append(Line2D([0], [0], marker='s', color='w', markerfacecolor='green', markersize=12, label='1% < Final Perm. Diff $\leq$ 2%'))
        
        ax.legend(handles=handles, loc='upper left', frameon=True, edgecolor='black', framealpha=0.9, fancybox=False)

        plt.tight_layout()
        
        plt.savefig(os.path.join(output_dir, f"Boxplot_{suffix}.pdf"), bbox_inches='tight')
        plt.savefig(os.path.join(output_dir, f"Boxplot_{suffix}.png"), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved Boxplot_{suffix}.pdf/.png to {output_dir}")

# Execute the plotting function
plot_prop_boxplots(
    df=df,
    prop_cols=["Speedup_Ratio"],
    output_dir="./Plots/",
    suffix=file,
    dataset_colors=dataset_colors,
    remove_outliers=remove_outliers,
    logscale=logscale
)