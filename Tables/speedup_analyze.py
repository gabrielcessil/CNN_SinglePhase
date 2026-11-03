import numpy as np
import pandas as pd
import re
import unicodedata
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import percentileofscore, pearsonr

# 1. Leitura dos arquivos
df_speedup = pd.read_csv("SpeedUp_ID.csv")
df_properties = pd.read_csv("Dataset_Properties_Table_Test.csv")
df_errors = pd.read_csv("ErrorMetrics_Comp5_Ko_et_al_D3_PONTO.csv")

# Converter Speedup_Ratio para numérico
if df_speedup["Speedup_Ratio"].dtype == object:
    df_speedup["Speedup_Ratio"] = df_speedup["Speedup_Ratio"].astype(str).str.replace(",", ".").astype(float)
else:
    df_speedup["Speedup_Ratio"] = pd.to_numeric(df_speedup["Speedup_Ratio"], errors="coerce")

# Funções auxiliares de padronização
def standardize_sample_id(val):
    if pd.isna(val): return val
    if isinstance(val, str):
        numbers = re.findall(r'\d+', val)
        if numbers: return int(numbers[-1])
    try: return int(val)
    except ValueError: return val

def normalize_dataset_name(name):
    if not isinstance(name, str): return ""
    nfkd = unicodedata.normalize('NFKD', name)
    cleaned = "".join([c for c in nfkd if not unicodedata.combining(c)]).lower()
    cleaned = cleaned.replace(" ", "").replace("-", "").replace("120³", "")
    return cleaned

mapping_ds = {'porosesfericos': 'sphericalpores', 'graosesfericos': 'sphericalgrains'}

df_speedup["Sample_Clean_ID"] = df_speedup["Sample"].apply(standardize_sample_id)
df_speedup["Dataset_Key_Base"] = df_speedup["Dataset"].apply(normalize_dataset_name).replace(mapping_ds)

df_properties["Sample_Clean_ID"] = df_properties["Sample Index"].apply(standardize_sample_id)
df_properties["Dataset_Key_Base"] = df_properties["Dataset"].apply(normalize_dataset_name)

df_errors["Sample_Clean_ID"] = df_errors["Sample_Index"].apply(standardize_sample_id)
df_errors["Dataset_Key_Base"] = df_errors["Dataset"].apply(normalize_dataset_name).str.replace('1203', '').str.replace('2563', '').str.replace('sphpore(koetal)', 'sphericalpores')

# 2. Mesclagem de todas as tabelas utilizando as chaves padronizadas
merged_all = df_speedup.merge(df_properties, on=["Dataset_Key_Base", "Sample_Clean_ID"], how="inner")
merged_all = merged_all.merge(df_errors, on=["Dataset_Key_Base", "Sample_Clean_ID"], how="inner", suffixes=('', '_err'))

# Definir colunas numéricas de interesse
geometric_cols = ['Porosity', 'Tortuosity', 'Permeability', 'Q1 Local Thickness', 
                  'Mean Local Thickness', 'Max Local Thickness', 'M. Volume', 
                  'M. Surface Area', 'M. Mean Curvature', 'M. Euler Char']

error_cols = ['Perm_Error [%]', 'Bias_Error_', 'Magnitude_Error_', 'Correlation', 
              'Angular_Error_Deg', 'Flux_Error', 'Tortuosity_Error_', 'Divergent_Residual_']

geometric_cols = [c for c in geometric_cols if c in merged_all.columns]
error_cols = [c for c in error_cols if c in merged_all.columns]

# 3. Cálculo dos percentis GLOBAIS (em relação a TODAS as amostras combinadas)
for col in geometric_cols + error_cols:
    if col in merged_all.columns:
        # População global combinada de valores válidos
        pop_global = merged_all[col].dropna()
        merged_all[f"{col}_Global_Percentile"] = merged_all[col].apply(
            lambda val: percentileofscore(pop_global, val, kind='mean') if pd.notna(val) else np.nan
        )

# Criar coluna indicadora para destacar amostras com slowdown (< 1)
merged_all["Slowdown"] = merged_all["Speedup_Ratio"] < 1.0

# Salvar tabela consolidada global
merged_all.to_csv("Tabela_Percentis_Global_Todas_Amostras.csv", index=False)
print("Tabela global salva com sucesso como 'Tabela_Percentis_Global_Todas_Amostras.csv'.")

# 4. Geração de Gráficos de Dispersão com todas as amostras e correlação global
sns.set_theme(style="whitegrid")
all_metrics_to_plot = geometric_cols + error_cols

for metric in all_metrics_to_plot:
    percentile_col_name = f"{metric}_Global_Percentile"
    if percentile_col_name in merged_all.columns:
        plt.figure(figsize=(8, 6))
        
        x = merged_all[percentile_col_name]
        y = merged_all["Speedup_Ratio"]
        
        # Correlação de Pearson global
        valid_idx = ~(x.isna() | y.isna())
        if valid_idx.sum() > 1:
            corr, p_val = pearsonr(x[valid_idx], y[valid_idx])
        else:
            corr, p_val = 0.0, 1.0
            
        # Scatter plot global
        sns.scatterplot(
            data=merged_all, x=percentile_col_name, y="Speedup_Ratio",
            hue="Dataset_x", style="Slowdown", palette="tab10", s=80, alpha=0.85
        )
        sns.regplot(x=x, y=y, scatter=False, color="gray", line_kws={"linestyle": "--"})
        
        plt.axhline(1.0, color="red", linestyle=":", linewidth=1.5, label="Limite Speed-up = 1.0")
        
        plt.title(f"Speed-up vs. {metric} (Percentil Global)", fontsize=12, fontweight="bold")
        plt.xlabel(f"Percentil Global da Amostra ({metric})", fontsize=10)
        plt.ylabel("Speed-up Ratio", fontsize=10)
        
        # Anotação da correlação global
        plt.annotate(f"r = {corr:.2f} (p = {p_val:.3f})", 
                     xy=(0.05, 0.90), xycoords='axes fraction',
                     fontsize=11, fontweight='bold',
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", lw=1))
        
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        plt.tight_layout()
        
        # Salvar imagem
        safe_name = metric.replace(" ", "_").replace("%", "pct").replace("[", "").replace("]", "")
        filename = f"plot_global_percentile_vs_{safe_name}.png"
        plt.savefig(filename, dpi=300)
        plt.close()

print("Gráficos de dispersão com percentil global gerados com sucesso!")