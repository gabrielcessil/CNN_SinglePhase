import os
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.metrics import r2_score

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
extracted_csv_path  = "./Tables/PermeabilityExtraction/Extracted_Permeability_Predictions.csv"
output_plot_dir     = "./Plots/PermeabilityComparison"

N_samples_plotted   = 20        # Limit plotting points per dataset (None for all samples)
random_seed         = 42        # Seed for reproducible random sampling/shuffling of plotted points
use_log_plot        = True      # True: Plot on log10 axes | False: Plot on linear axes
use_log_metrics     = False     # True: Calculate Pearson/R² in log10 space | False: Linear space

# Optional: List dataset patterns to exclude
datasets_to_exclude = []

models_to_include = ["PE", "PE-GE-R"]
# ---------------------------------------------------------
# 1. LOAD AND PREPROCESS EXTRACTED PREDICTIONS
# ---------------------------------------------------------
if not os.path.exists(extracted_csv_path):
    raise FileNotFoundError(f"Could not find extracted CSV file at {extracted_csv_path}")

# Read raw CSV
df_all = pd.read_csv(extracted_csv_path, sep=",")
df_all.columns = df_all.columns.str.strip().str.replace(' ', '_')

# Convert string numbers with comma decimals ("1308,81...") into float64 (1308.81...)
numeric_cols = ["Permeability", "Predicted_Permeability"]
for col in numeric_cols:
    if col in df_all.columns:
        df_all[col] = (
            df_all[col]
            .astype(str)
            .str.replace(',', '.', regex=False)
        )
        df_all[col] = pd.to_numeric(df_all[col], errors='coerce')

# Exclude datasets if requested
if datasets_to_exclude:
    exclude_pattern = "|".join(datasets_to_exclude)
    df_all = df_all[~df_all["Dataset"].astype(str).str.contains(exclude_pattern, case=False, na=False)].reset_index(drop=True)

# Drop any rows where numeric parsing produced NaN
df_all = df_all.dropna(subset=numeric_cols).reset_index(drop=True)

# Include only specified models if requested
if models_to_include:
    df_all = df_all[df_all["Model"].isin(models_to_include)].reset_index(drop=True)

# Drop any rows where numeric parsing produced NaN
df_all = df_all.dropna(subset=numeric_cols).reset_index(drop=True)

if df_all.empty:
    raise ValueError("No data left after filtering models and datasets. Check 'models_to_include' or 'datasets_to_exclude'.")

# ---------------------------------------------------------
# Determine model list & explicit display order
# ---------------------------------------------------------
present_models = set(df_all["Model"].unique())

if models_to_include:
    # Preserve the exact sequence defined in models_to_include
    unique_models = [m for m in models_to_include if m in present_models]
else:
    # Fall back to CSV appearance order if models_to_include is empty
    unique_models = list(df_all["Model"].unique())

# Split into separate dataframes per model preserving ordered sequence
merged_dfs = {m: df_all[df_all["Model"] == m].copy() for m in unique_models}


# ---------------------------------------------------------
# STEP A: Build Plotting Dataframe Subset (Randomly Sampled Plotted Points)
# ---------------------------------------------------------
plot_dfs = {}
for model_name, df_m in merged_dfs.items():
    # 1. Shuffle/randomly sample up to N samples per dataset
    if N_samples_plotted is not None:
        # Sample indices per group to avoid dropping the 'Dataset' column or triggering deprecation warnings
        sampled_indices = (
            df_m.groupby("Dataset")
            .apply(
                lambda g: g.sample(n=min(len(g), N_samples_plotted), random_state=random_seed).index,
                include_groups=False
            )
        )
        flat_indices = np.concatenate(sampled_indices.values)
        sub_df = df_m.loc[flat_indices].reset_index(drop=True)
    else:
        sub_df = df_m.copy()

    # 2. Filter strictly renderable points for log scale (>0)
    if use_log_plot:
        sub_df = sub_df[(sub_df["Permeability"] > 0) & (sub_df["Predicted_Permeability"] > 0)].reset_index(drop=True)

    plot_dfs[model_name] = sub_df

# ---------------------------------------------------------
# STEP B: Determine Global Axis Limits strictly from Plotted Subset
# ---------------------------------------------------------
all_true_plotted = np.concatenate([df["Permeability"].values for df in plot_dfs.values()])
all_pred_plotted = np.concatenate([df["Predicted_Permeability"].values for df in plot_dfs.values()])

if use_log_plot:
    log_min = min(np.log10(all_true_plotted).min(), np.log10(all_pred_plotted).min())
    log_max = max(np.log10(all_true_plotted).max(), np.log10(all_pred_plotted).max())

    log_margin = (log_max - log_min) * 0.03
    axis_min = 10 ** (log_min - log_margin)
    axis_max = 10 ** (log_max + log_margin)
else:
    val_min = min(all_true_plotted.min(), all_pred_plotted.min())
    val_max = max(all_true_plotted.max(), all_pred_plotted.max())
    margin = (val_max - val_min) * 0.03
    axis_min = val_min - margin
    axis_max = val_max + margin

# ---------------------------------------------------------
# STEP C: Generate Side-by-Side Plots with Statistics
# ---------------------------------------------------------
plt.rcParams['font.family'] = 'serif'
num_models = len(plot_dfs)

fig, axes = plt.subplots(
    1, num_models,
    figsize=(4.8 * num_models + 1.8, 4.8),
    sharey=True, sharex=True
)

if num_models == 1:
    axes = [axes]

for i, (model_name, df_plot) in enumerate(plot_dfs.items()):
    ax = axes[i]
    df_full_m = merged_dfs[model_name]  # Full dataset for this model

    # Metrics calculation calculated on ALL data in the CSV for this model
    if use_log_metrics:
        clean_df_calc = df_full_m[(df_full_m["Permeability"] > 0) & (df_full_m["Predicted_Permeability"] > 0)]
        x_calc = np.log10(clean_df_calc["Permeability"].values)
        y_calc = np.log10(clean_df_calc["Predicted_Permeability"].values)
        metric_label_suffix = ""
    else:
        x_calc = df_full_m["Permeability"].values
        y_calc = df_full_m["Predicted_Permeability"].values
        metric_label_suffix = ""

    r_pearson, _ = pearsonr(x_calc, y_calc)
    r2 = r2_score(x_calc, y_calc)

    # Plot 1:1 Parity Line
    ax.plot([axis_min, axis_max], [axis_min, axis_max], color='gray', linestyle='--', linewidth=1.2, label='1:1')

    # Scatter plot per dataset (sampled subset)
    sns.scatterplot(
        data=df_plot,
        x="Permeability",
        y="Predicted_Permeability",
        hue="Dataset",
        s=45,
        linewidth=1.3,
        ax=ax,
        alpha=0.85
    )

    # Transparent marker interiors with colored borders
    for collection in ax.collections:
        collection.set_edgecolor(collection.get_facecolor())
        collection.set_facecolor('none')

    # Apply axis scaling
    if use_log_plot:
        ax.set_xscale('log')
        ax.set_yscale('log')

    ax.set_xlim(axis_min, axis_max)
    ax.set_ylim(axis_min, axis_max)

    # Annotate metric box
    stats_text = f"$Pearson{metric_label_suffix} = {r_pearson:.3f}$\n$R²{metric_label_suffix} = {r2:.3f}$"
    ax.text(
        0.05, 0.93, stats_text,
        transform=ax.transAxes,
        fontsize=9.5,
        verticalalignment='top',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.85, edgecolor='gray')
    )

    ax.set_xlabel("True Permeability", fontsize=11)
    if i == 0:
        ax.set_ylabel("Predicted Permeability", fontsize=11)

    ax.set_title(model_name, fontsize=12, pad=8)

    # Single unified external legend handling
    if i < num_models - 1:
        if ax.get_legend() is not None:
            ax.get_legend().remove()
    else:
        ax.legend(
            bbox_to_anchor=(1.03, 1),
            loc='upper left',
            fontsize=9,
            title_fontsize=10,
            ncol=1,
            markerscale=0.85,
            labelspacing=0.35,
            borderpad=0.4,
            frameon=True,
            framealpha=0.9
        )

scale_title = "Log Scale" if use_log_plot else "Linear Scale"
fig.suptitle(f"True Permeability vs. Predicted Permeability ({scale_title})", fontsize=13, y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.95])

# Save output image
os.makedirs(output_plot_dir, exist_ok=True)
filename = os.path.join(
    output_plot_dir,
    "PermeabilityPrediction.png"
)
filename = os.path.join(
    output_plot_dir,
    "PermeabilityPrediction.pdf"
)
plt.savefig(filename, dpi=300, bbox_inches="tight")
plt.close()

print(f"Plot saved successfully to: {filename}")