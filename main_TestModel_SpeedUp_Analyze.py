import os
import re
from pathlib import Path
import numpy as np
import pandas as pd
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
OUTPUT_CSV_DIR  = "./Tables/SpeedUp/"
file            = "SpeedUp_120"

"""
main_folders = {
    "Premature Convergence": "../TestSpeedUp_Simulations_PrematureConvergence/",
}
"""

#"""
main_folders = {
    "SphPore 120": "../TestSpeedUp_Simulations_120/Test_Silveira_SphPore_SAug_DNorm/",
    "SphGrain 120": "../TestSpeedUp_Simulations_120/Test_Silveira_SphGrain_SAug_DNorm/",
    "Leopard 120": "../TestSpeedUp_Simulations_120/Test_Oliveira_Leopard_SAug_DNorm/",
    "CastleGate 120": "../TestSpeedUp_Simulations_120/Test_Oliveira_CastleGate_SAug_DNorm/",
    "BereaUpperGray 120": "../TestSpeedUp_Simulations_120/Test_Oliveira_BereaUpperGray_SAug_DNorm/",
    "BereaSinterGray 120": "../TestSpeedUp_Simulations_120/Test_Oliveira_BereaSinterGray_SAug_DNorm/",
    "BereaBuff 120": "../TestSpeedUp_Simulations_120/Test_Oliveira_BereaBuff_SAug_DNorm/",
    "Berea 120": "../TestSpeedUp_Simulations_120/Test_Oliveira_Berea_SAug_DNorm/",
    "Bentheimer 120": "../TestSpeedUp_Simulations_120/Test_Oliveira_Bentheimer_SAug_DNorm/",
}
#"""
"""
main_folders = {
    "Doddington 256³": "../TestSpeedUp_Simulations_BiggerCrops/IC_Doddington_256_256_256/",
    "Ketton 256³": "../TestSpeedUp_Simulations_BiggerCrops/IC_Ketton_256_256_256/",
    "Estaillades 256³": "../TestSpeedUp_Simulations_BiggerCrops/IC_Estaillades_256_256_256/"

    }
"""




# ---------------------------------------------------------
# Convergence Analysis Settings
# ---------------------------------------------------------

remove_outliers = False
logscale        = True
dataset_colors  = {}

analysis_interval       = 2000        
totaL_timesteps         = 1000000
self_convergence_tail   = 0.25   # Evaluate the last 25% of the simulation steps
self_convergence_tol    = 10.0   # Must stay within 10% of its OWN final value
ref_convergence_tol     = 2.0    # Must reach within 2.5% of the GRAD final value

# ---------------------------------------------------------
# AUXILIARY FUNCTIONS
# ---------------------------------------------------------

for key in main_folders.keys():
    if key not in dataset_colors:
        dataset_colors[key] = "black"

def get_permeability_data(folder_path):
    p = Path(folder_path)
    if not p.exists():
        return None
        
    perm_file = p / "Permeability.csv"
    if perm_file.exists():
        try:
            df_perm = pd.read_csv(perm_file, sep=r'\s+')
            if 'absperm(mDa)' in df_perm.columns:
                return df_perm
        except Exception as e:
            print(f"Warning: Could not read {perm_file} due to {e}")
            
    return None

def check_self_convergence(df_sim, tol=1.0, tail_frac=0.10):
    """
    Checks if the simulation converged into itself.
    Returns (is_converged, max_error) in the last `tail_frac` window.
    """
    if df_sim is None or len(df_sim) == 0:
        return False, np.nan
        
    k_array = df_sim['absperm(mDa)'].values
    if len(k_array) < 2:
        return False, np.nan
        
    k_final = k_array[-1]
    
    # Determine the index for the last X% of the data
    tail_start_idx = int(len(k_array) * (1.0 - tail_frac))
    if tail_start_idx == len(k_array): 
        tail_start_idx = len(k_array) - 1 # Ensure at least one point is checked
        
    k_tail = k_array[tail_start_idx:]
    
    # Calculate error relative to its own final value
    errors = 100.0 * np.abs(k_tail - k_final) / (np.abs(k_final) + 1e-15)
    max_err = np.max(errors)
    
    # If any error in the tail exceeds tolerance, it has not stably converged
    return np.all(errors <= tol), max_err

def get_convergence_step(df_sim, k_ref, tol=1.0, interval=200):
    if df_sim is None or len(df_sim) == 0:
        return np.nan, np.nan
        
    k_array = df_sim['absperm(mDa)'].values
    
    if 'Step' in df_sim.columns:
        steps = df_sim['Step'].values
        if np.max(steps) <= len(steps):
            steps = steps * interval
    elif 'Time(s)' in df_sim.columns:
        steps = df_sim['Time(s)'].values
    else:
        steps = np.arange(1, len(k_array) + 1) * interval
        
    errors = 100.0 * np.abs(k_array - k_ref) / (np.abs(k_ref) + 1e-15)
    
    # NEW: Get the error at the exact final timestep of the simulation
    final_err = errors[-1] 
    
    conv_step = np.nan
    for i in range(len(errors) - 1, -1, -1):
        if errors[i] <= tol:
            conv_step = steps[i]
        else:
            break
            
    return conv_step, final_err

# ==============================================================================
# 2. DATA EXTRACTION
# ==============================================================================
results = []

for dataset_name, dataset_path in main_folders.items():
    dataset_p = Path(dataset_path)
    if not dataset_p.exists():
        continue
        
    for sample_folder in dataset_p.glob("*Sample_*"):
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
            
        # 1. Evaluate self-convergence first (unpack the new max_err)
        grad_self_conv, grad_max_err = check_self_convergence(df_grad, tol=self_convergence_tol, tail_frac=self_convergence_tail)
        nn_self_conv, nn_max_err = check_self_convergence(df_nn, tol=self_convergence_tol, tail_frac=self_convergence_tail)
        
        # 2. Establish the target reference permeability (Final value of Grad run)
        k_ref = k_array_grad[-1]
        
        # 3. Find the instant each curve reaches the reference tolerance zone without leaving
        ts_reference, _ = get_convergence_step(df_grad, k_ref, tol=ref_convergence_tol, interval=analysis_interval)
        ts_started, nn_min_ref_err = get_convergence_step(df_nn, k_ref, tol=ref_convergence_tol, interval=analysis_interval)
        
        k_reference = k_array_grad[-1]
        k_started = k_array_nn[-1]
        
        expected_total_steps = totaL_timesteps/analysis_interval - 1

        results.append({
            "Dataset": dataset_name,
            "Sample": sample_name,
            "Reference_Timesteps": ts_reference,
            "NN_Started_Timesteps": ts_started,
            "NN_Min_Ref_Error": nn_min_ref_err,              
            "Reference_Permeability": k_reference,
            "NN_Permeability": k_started,
            "Grad_Self_Converged": grad_self_conv,
            "Grad_Tail_Max_Error": grad_max_err,
            "NN_Self_Converged": nn_self_conv,
            "NN_Tail_Max_Error": nn_max_err,
            "Is_Complete": (len(k_array_grad) >= expected_total_steps) and (len(k_array_nn) >= expected_total_steps)
        })

df = pd.DataFrame(results)

if df.empty:
    print("Error: No data extracted.")
    exit()

# Clean up data: Drop rows that are incomplete (still running)
incomplete_runs = ~df['Is_Complete']
if incomplete_runs.any():
    initial_total_samples = len(df)
    print(f"Warning: Dropped {incomplete_runs.sum()}/{initial_total_samples} samples because they are still running (fewer than {expected_total_steps} timesteps).")
    
    # Iterate through failing rows and print dataset and sample
    for _, row in df[incomplete_runs].iterrows():
        print(f"  -> [{row['Dataset']}] {row['Sample']}")
    print("") # Formatting newline

df = df[~incomplete_runs]

# Clean up data: Drop rows that failed the self-convergence check
failed_self_conv = ~(df['Grad_Self_Converged'] & df['NN_Self_Converged'])

# Clean up data: Drop rows that failed the self-convergence check
failed_self_conv = ~(df['Grad_Self_Converged'] & df['NN_Self_Converged'])
if failed_self_conv.any():
    initial_total_samples = len(df)
    print(f"Warning: Dropped {failed_self_conv.sum()}/{initial_total_samples} samples because they failed the {self_convergence_tol}% self-convergence check over their last {self_convergence_tail*100}% steps.")
    
    # Iterate through failing rows and print the dataset, sample, and exact max differences
    for _, row in df[failed_self_conv].iterrows():
        print(f"  -> [{row['Dataset']}] {row['Sample']} (Max Tail Diff - Grad: {row['Grad_Tail_Max_Error']:.2f}%, NN: {row['NN_Tail_Max_Error']:.2f}%)")
    print("") # Formatting newline

df = df[~failed_self_conv]

# Clean up data: Drop rows where simulations didn't hit the reference threshold
dropped_unconverged = df['Reference_Timesteps'].isna() | df['NN_Started_Timesteps'].isna()
if dropped_unconverged.any():
    total_samples = len(df)
    print(f"Warning: Dropped {dropped_unconverged.sum()}/{total_samples} samples that failed to reach within {ref_convergence_tol}% of the reference grad permeability.")
    
    # Iterate through failing rows and print the dataset, sample, and minimum difference reached
    for _, row in df[dropped_unconverged].iterrows():
        print(f"  -> [{row['Dataset']}] {row['Sample']} (Closest Diff: {row['NN_Min_Ref_Error']:.2f}%)")
    print("") # Formatting newline

df = df.dropna(subset=["Reference_Timesteps", "NN_Started_Timesteps"])
df = df[df["NN_Started_Timesteps"] > 0]
# Calculate Speedup Ratio (using the convergence entering instant) & Permeability Relative Error
df['Speedup_Ratio'] = df['Reference_Timesteps'] / df['NN_Started_Timesteps']
df['Perm_Error [%]'] = 100 * np.abs(df['NN_Permeability'] - df['Reference_Permeability']) / np.abs(df['Reference_Permeability'])

print("\nConvergence Timesteps & Speedup Comparison:")
print("-" * 75)
print(df[['Dataset', 'Sample', 'Speedup_Ratio', 'Perm_Error [%]']].to_string(index=False))
print("-" * 75)

os.makedirs(OUTPUT_CSV_DIR, exist_ok=True)
csv_output_path = os.path.join(OUTPUT_CSV_DIR, f"{file}.csv")
df.to_csv(csv_output_path, index=False)

print(f"\n Speed-up evaluation saved to: {csv_output_path}")

# ==============================================================================
# 3. GLOBAL ACADEMIC STYLING CONFIGURATION & PLOTTING
# ==============================================================================
# [The plotting functionality remains exactly the same below]
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

        # 1. The Boxplot
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

       
        # 2. Density-Proportional Scatter & Annotations
        for i, dataset in enumerate(datasets):
            subset = df[df['Dataset'] == dataset].dropna(subset=[prop, 'Perm_Error [%]'])
            y_vals = subset[prop].values
            perm_errors = subset['Perm_Error [%]'].values
            
            base_color = dataset_colors.get(dataset, '#4c72b0')
            
            if len(y_vals) == 0:
                continue
            
            q1 = np.percentile(y_vals, 25)
            median = np.percentile(y_vals, 50)
            q3 = np.percentile(y_vals, 75)

            ax.text(i + 0.24, q3, f" {q3:.2f}", va='center', ha='left', fontsize=9, color='gray')
            ax.text(i + 0.24, median, f" {median:.2f}", va='center', ha='left', fontsize=10, color='gray', fontweight='bold')
            ax.text(i + 0.24, q1, f" {q1:.2f}", va='center', ha='left', fontsize=9, color='gray')
            
            if remove_outliers:
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                
                mask = (y_vals >= lower_bound) & (y_vals <= upper_bound)
                y_vals = y_vals[mask]
                perm_errors = perm_errors[mask] 
                
                if len(y_vals) == 0:
                    continue

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

            max_jitter = 0.25 
            jitter = np.random.uniform(-1, 1, size=len(y_vals)) * max_jitter * density_norm
            x_vals = i + jitter
            
            ax.scatter(
                x_vals, 
                y_vals, 
                color=base_color, 
                s=25,          # Dot size
                alpha=0.75,    # Transparency
                edgecolor='white', 
                linewidths=0.4, 
                zorder=2
            )
            
        ax.set_ylabel("Speed-up Ratio", fontweight="bold")
        ax.set_xlabel("Tolerance Configuration", fontweight="bold")
        
        if logscale: ax.set_yscale('log')

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

        handles, labels = ax.get_legend_handles_labels()
        
       
        ax.legend(handles=handles, loc='upper left', frameon=True, edgecolor='black', framealpha=0.9, fancybox=False)

        plt.tight_layout()
        
        plt.savefig(os.path.join(output_dir, f"Boxplot_{suffix}.pdf"), bbox_inches='tight')
        plt.savefig(os.path.join(output_dir, f"Boxplot_{suffix}.png"), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved Boxplot_{suffix}.pdf/.png to {output_dir}")


plot_prop_boxplots(
    df=df,
    prop_cols=["Speedup_Ratio"],
    output_dir="./Plots/",
    suffix=file,
    dataset_colors=dataset_colors,
    remove_outliers=remove_outliers,
    logscale=logscale
)

