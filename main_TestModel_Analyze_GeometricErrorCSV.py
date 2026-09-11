import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

N_samples = 50

df_errors = pd.read_csv("./Tables/ErrorMetrics_Comp0_Ko_et_al_Etapa_3.csv", sep=",", decimal=".")
df_geom   = pd.read_csv("./Tables/Dataset_Properties_Table_Test.csv", sep=",", decimal=".")

# 1. Strip leading/trailing spaces, then replace any internal spaces with underscores
df_errors.columns = df_errors.columns.str.strip().str.replace(' ', '_')
df_geom.columns   = df_geom.columns.str.strip().str.replace(' ', '_')

print("Error columns:", df_errors.columns.tolist())
print("Geom columns:", df_geom.columns.tolist())

if "Correlation" in df_errors.columns:
    df_errors = df_errors.rename(columns={"Correlation": "Prediction_Correlation"})

# ==========================================
# 2. MERGE DATASETS
# ==========================================
df_merged = pd.merge(
    df_geom, 
    df_errors, 
    on=["Dataset", "Sample_Index"], 
    how="inner" 
)

geom_cols = [
    "Porosity", "Tortuosity", "Permeability", "Q1_Local_Thickness", 
    "Mean_Local_Thickness", "Max_Local_Thickness", "M._Volume", 
    "M._Surface_Area", "M._Mean_Curvature", "M._Euler_Char"
]
error_cols = ["Bias_Error_", "Magnitude_Error_", "Prediction_Correlation"]

# ==========================================
# 3. PREPROCESS AND CALCULATE QUARTILES (UNFILTERED)
# ==========================================
for col in geom_cols + error_cols:
    df_merged[col] = pd.to_numeric(df_merged[col], errors='coerce')

# Calculate quartiles for EACH error metric on the full dataset
for err in error_cols:
    df_merged[f'{err}_Quartile'] = pd.qcut(df_merged[err], q=4, labels=[1, 2, 3, 4], duplicates='drop').astype(float)

# ==========================================
# 4. DOWNSAMPLE DATASETS
# ==========================================
df_merged = df_merged.groupby("Dataset").head(N_samples).reset_index(drop=True)

# ==========================================
# 5. SCATTER PLOTS: GEOMETRY VS ERROR
# ==========================================
features_to_plot = geom_cols 
plt.rcParams['font.family'] = 'serif'

for target_error in error_cols:
    fig, axes = plt.subplots(1, len(features_to_plot), figsize=(4 * len(features_to_plot), 5))
    
    if len(features_to_plot) == 1:
        axes = [axes]

    for i, feature in enumerate(features_to_plot):
        sns.scatterplot(
            data=df_merged, 
            x=feature, 
            y=target_error, 
            hue="Dataset",
            size=f"{target_error}_Quartile",  
            sizes=(20, 200),          
            alpha=0.7,
            ax=axes[i]
        )
        axes[i].set_title(f"{feature} vs {target_error}")
        
        # --- NEW: Calculate and annotate correlation ---
        corr_val = df_merged[feature].corr(df_merged[target_error])
        axes[i].annotate(
            f"r = {corr_val:.2f}", 
            xy=(0.05, 0.95), 
            xycoords='axes fraction', 
            ha='left', 
            va='top', 
            fontsize=12,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
        )
        # -----------------------------------------------

        if i < len(features_to_plot) - 1:
            if axes[i].get_legend() is not None:
                axes[i].get_legend().remove()
        else:
            axes[i].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f"Scatter_Geom_vs_{target_error}.png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

# ==========================================
# 6. PCA OF GEOMETRICAL PROPERTIES
# ==========================================
scaler = StandardScaler()
geom_scaled = scaler.fit_transform(df_merged[geom_cols].fillna(0))

pca = PCA(n_components=2)
pca_results = pca.fit_transform(geom_scaled)

df_merged['PC1'] = pca_results[:, 0]
df_merged['PC2'] = pca_results[:, 1]

var_pc1 = pca.explained_variance_ratio_[0] * 100
var_pc2 = pca.explained_variance_ratio_[1] * 100

for target_error in error_cols:
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        data=df_merged,
        x='PC1',
        y='PC2',
        hue='Dataset',
        size=f'{target_error}_Quartile',   
        sizes=(30, 250),         
        alpha=0.5,               
        palette='tab10'          
    )

    plt.title(f"PCA of Geometrical Properties\nPoint Size = Global Error Quartile ({target_error})\nExplained Variance: PC1 ({var_pc1:.1f}%), PC2 ({var_pc2:.1f}%)", pad=15)
    plt.xlabel(f"Principal Component 1 ({var_pc1:.1f}%)")
    plt.ylabel(f"Principal Component 2 ({var_pc2:.1f}%)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
    plt.tight_layout()
    plt.savefig(f"PCA_Geometries_{target_error}Size.png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()