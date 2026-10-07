import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

N_samples = 50

# 1. Define model names and CSV paths
model_files = {
    "PE":       "./Tables/ErrorByModel/ErrorMetrics_Comp0_Ko_et_al_D0_N4.csv", 
    "PE-GE-R":  "./Tables/ErrorByModel/ErrorMetrics_Comp0_Ko_et_al_D3_N4.csv",
    
    #"PE-GE-R":   "./Tables/ErrorByModel/ErrorMetrics_Comp5_Ko_et_al_D3.csv"
}

# 2. Define datasets/patterns to exclude and columns for log scaling
datasets_to_exclude = []#["Leopard", "Berea", "Berea Buff", "Berea Upper Gray"]

log_x_cols = ["Permeability", "Porosity", "M._Surface_Area"]
log_y_cols = ["Magnitude_Error_", "Bias_Error_"]

# 3. Load and clean geometry dataset
df_geom = pd.read_csv("./Tables/DatasetProperties/Dataset_Properties_Table_Test.csv", sep=",", decimal=".")
df_geom.columns = df_geom.columns.str.strip().str.replace(' ', '_')

geom_cols = [
    "Porosity", "Tortuosity", "Permeability", "Q1_Local_Thickness", 
    "Mean_Local_Thickness", "Max_Local_Thickness", "M._Volume", 
    "M._Surface_Area", "M._Mean_Curvature", "M._Euler_Char"
]
error_cols = ["Bias_Error_", "Prediction_Correlation"]
              
#error_cols = ["Bias_Error_", "Prediction_Correlation", "Angular_Error_Deg", 
#              "Flux_Error", "Tortuosity_Error_", "Divergent_Residual_" ]

for col in geom_cols:
    df_geom[col] = pd.to_numeric(df_geom[col], errors='coerce')

# --- EXCLUDE LIST OF DATASETS ---
if datasets_to_exclude:
    exclude_pattern = "|".join(datasets_to_exclude)
    df_geom = df_geom[~df_geom["Dataset"].astype(str).str.contains(exclude_pattern, case=False, na=False)].reset_index(drop=True)

# 4. Load, clean, and merge datasets for each model
merged_dfs = {}

for model_name, file_path in model_files.items():
    df_err = pd.read_csv(file_path, sep=",", decimal=".")
    df_err.columns = df_err.columns.str.strip().str.replace(' ', '_')
    
    if "Correlation" in df_err.columns:
        df_err = df_err.rename(columns={"Correlation": "Prediction_Correlation"})
        
    df_m = pd.merge(df_geom, df_err, on=["Dataset", "Sample_Index"], how="inner")
    
    for col in error_cols:
        df_m[col] = pd.to_numeric(df_m[col], errors='coerce')
        
    df_m = df_m.groupby("Dataset").head(N_samples).reset_index(drop=True)
    merged_dfs[model_name] = df_m

# 5. Generate side-by-side scatter plots
plt.rcParams['font.family'] = 'serif'
num_models = len(model_files)

for target_error in error_cols:
    for feature in geom_cols:
        # Extra horizontal width (+1.8 in) so the external legend does not squish the subplots
        fig, axes = plt.subplots(
            1, num_models, 
            figsize=(4.8 * num_models + 1.8, 4.8), 
            sharey=True
        )
        
        if num_models == 1:
            axes = [axes]

        for i, (model_name, df_m) in enumerate(merged_dfs.items()):
            ax = axes[i]
            
            # Plot scatter
            sns.scatterplot(
                data=df_m, 
                x=feature, 
                y=target_error, 
                hue="Dataset",
                s=45,
                linewidth=1.3,
                ax=ax
            )
            
            # Make point interior transparent and transfer dataset hue to the outline
            for collection in ax.collections:
                collection.set_edgecolor(collection.get_facecolor())
                collection.set_facecolor('none')

            # Apply log scale conditionally
            if feature in log_x_cols:
                ax.set_xscale('log')
            if target_error in log_y_cols:
                ax.set_yscale('log')
            
            ax.set_title(model_name, fontsize=12, pad=8)
            
            if i < num_models - 1:
                if ax.get_legend() is not None:
                    ax.get_legend().remove()
            else:
                # CLEAN EXTERNAL LEGEND (1 Column)
                ax.legend(
                    bbox_to_anchor=(1.03, 1),
                    loc='upper left',
                    fontsize=9,
                    title_fontsize=10,
                    ncol=1,                # Single column is narrower and cleaner
                    markerscale=0.85,
                    labelspacing=0.35,
                    borderpad=0.4,
                    frameon=True,
                    framealpha=0.9
                )

        fig.suptitle(f"{feature} vs {target_error}", fontsize=13, y=0.98)
        
        # Reserve top space for suptitle without creating huge vertical gaps
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        filename = f"./Plots/ErrorCorrelations/Scatter_{feature}_vs_{target_error}_ModelsComparison.png"
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close()