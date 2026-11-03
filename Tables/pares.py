import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Rectangle
from scipy.stats import pearsonr

# ============================================================
# 1. Carregar a tabela consolidada global
# ============================================================

merged_all = pd.read_csv("Tabela_Percentis_Global_Todas_Amostras.csv")

# Converter colunas que venham com vírgula decimal para float
for col in merged_all.columns:
    if merged_all[col].dtype == object:
        converted = (
            merged_all[col]
            .astype(str)
            .str.replace(".", "", regex=False)
            .str.replace(",", ".", regex=False)
        )

        numeric_series = pd.to_numeric(converted, errors="coerce")

        if numeric_series.notna().sum() > 0.5 * len(merged_all):
            merged_all[col] = numeric_series


# ============================================================
# 2. Configuração visual
# ============================================================

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 14,
        "axes.labelsize": 18,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "axes.linewidth": 1.0,
    }
)


# ============================================================
# 3. Definição das propriedades e métricas de erro
# ============================================================

geometric_cols = [
    "Porosity",
    "Tortuosity",
    "Permeability",
    "Q1 Local Thickness",
    "Mean Local Thickness",
    "Max Local Thickness",
    "M. Surface Area",
    "M. Mean Curvature",
    "M. Euler Char",
]

error_cols = [
    "Bias_Error_",
    "Correlation",
    "Angular_Error_Deg",
    "Flux_Error",
    "Tortuosity_Error_",
    "Divergent_Residual_",
]


# ============================================================
# 4. Configurações
# ============================================================

# True  = utilizar percentis globais
# False = utilizar valores reais
percentil_based = False

# Limite para exibir as anotações de correlação
correlation_threshold = 0.5

if percentil_based:
    geom_cols = [
        f"{c}_Global_Percentile"
        for c in geometric_cols
        if f"{c}_Global_Percentile" in merged_all.columns
    ]

    err_cols = [
        f"{c}_Global_Percentile"
        for c in error_cols
        if f"{c}_Global_Percentile" in merged_all.columns
    ]

    file_suffix = "percentil"

else:
    geom_cols = [
        c for c in geometric_cols
        if c in merged_all.columns
    ]

    err_cols = [
        c for c in error_cols
        if c in merged_all.columns
    ]

    file_suffix = "valores_reais"


# Apenas propriedades geométricas e métricas de erro.
# Speedup_Ratio será analisado separadamente.
all_corr_cols = geom_cols + err_cols

if not all_corr_cols:
    raise ValueError(
        "Nenhuma propriedade geométrica ou métrica de erro "
        "predefinida foi encontrada no CSV."
    )

print(f"Propriedades geométricas: {geom_cols}")
print(f"Métricas de erro: {err_cols}")


# ============================================================
# 5. Função para criar anotações seletivas
# ============================================================

def create_correlation_annotations(corr_matrix, threshold=0.5):
    """
    Retorna uma matriz de strings para anotar somente correlações
    estritamente maiores que threshold ou menores que -threshold.

    Correlações fracas, valores NaN e valores exatamente iguais
    aos limites não recebem anotação.
    """

    annotations = corr_matrix.copy().astype(object)

    significant_mask = (
        (corr_matrix > threshold)
        | (corr_matrix < -threshold)
    )

    for row in range(corr_matrix.shape[0]):
        for col in range(corr_matrix.shape[1]):

            value = corr_matrix.iloc[row, col]

            if (
                significant_mask.iloc[row, col]
                and pd.notna(value)
            ):
                annotations.iloc[row, col] = f"{value:.2f}"
            else:
                annotations.iloc[row, col] = ""

    return annotations


# ============================================================
# 6. Função para gerar heatmaps de correlação
# ============================================================

def plot_correlation_heatmap(
    df_subset,
    title_suffix,
    filename_suffix,
):

    # Selecionar somente as colunas predefinidas disponíveis
    available_cols = [
        col for col in all_corr_cols
        if col in df_subset.columns
    ]

    df_ml = df_subset[available_cols].dropna()

    if len(df_ml) < 2:
        print(
            f"Aviso: Dados insuficientes para o heatmap "
            f"de {filename_suffix}."
        )
        return

    # Remover colunas constantes, cuja correlação é indefinida
    variable_cols = [
        col for col in df_ml.columns
        if df_ml[col].nunique() > 1
    ]

    df_ml = df_ml[variable_cols]

    if df_ml.shape[1] < 2:
        print(
            f"Aviso: Menos de duas variáveis não constantes "
            f"para o heatmap de {filename_suffix}."
        )
        return

    # Matriz de correlação
    corr_matrix = df_ml.corr()

    # Anotações seletivas
    annotations = create_correlation_annotations(
        corr_matrix,
        threshold=correlation_threshold,
    )

    # Máscara para exibir somente o triângulo inferior
    mask = np.triu(
        np.ones_like(corr_matrix, dtype=bool)
    )

    # Heatmap
    plt.figure(figsize=(14, 12))

    ax = sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=annotations,
        fmt="",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        linewidths=0.5,
    )

    # Identificar quantas propriedades geométricas e métricas
    # de erro estão efetivamente presentes na matriz
    n_geom = sum(
        col in corr_matrix.columns
        for col in geom_cols
    )

    n_err = sum(
        col in corr_matrix.columns
        for col in err_cols
    )

    # Bloco 1: Propriedades geométricas x propriedades geométricas
    if n_geom > 0:
        ax.add_patch(
            Rectangle(
                (0, 0),
                n_geom,
                n_geom,
                linewidth=2,
                edgecolor="blue",
                facecolor="none",
                linestyle=":",
            )
        )

    # Bloco 2: Propriedades geométricas x métricas de erro
    if n_geom > 0 and n_err > 0:
        ax.add_patch(
            Rectangle(
                (0, n_geom),
                n_geom,
                n_err,
                linewidth=2.5,
                edgecolor="black",
                facecolor="none",
                linestyle="--",
            )
        )

    # Bloco 3: Métricas de erro x métricas de erro
    if n_err > 0:
        ax.add_patch(
            Rectangle(
                (n_geom, n_geom),
                n_err,
                n_err,
                linewidth=2,
                edgecolor="green",
                facecolor="none",
                linestyle=":",
            )
        )

    plt.title(
        f"Correlation Matrix ({title_suffix})",
        fontweight="bold",
        pad=15,
    )

    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    plt.tight_layout()

    output_file = (
        f"heatmap_correlacao_{filename_suffix}_{file_suffix}.pdf"
    )

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(f"Gerado: '{output_file}'")


# ============================================================
# 7. Heatmap global
# ============================================================

plot_correlation_heatmap(
    merged_all,
    "Total — Todos os Datasets",
    "total",
)


# ============================================================
# 8. Identificar a coluna de identificação dos datasets
# ============================================================

dataset_col = None

for col in [
    "Dataset_x",
    "Dataset",
    "Dataset_Key_Base",
]:
    if col in merged_all.columns:
        dataset_col = col
        break


# ============================================================
# 9. Heatmaps individuais por dataset
# ============================================================

if dataset_col is not None:

    datasets = merged_all[dataset_col].dropna().unique()

    for ds in datasets:

        df_ds = merged_all[
            merged_all[dataset_col] == ds
        ]

        safe_ds_name = (
            str(ds)
            .replace(" ", "_")
            .replace("/", "_")
            .replace("\\", "_")
            .lower()
        )

        plot_correlation_heatmap(
            df_ds,
            f"Dataset: {ds}",
            f"dataset_{safe_ds_name}",
        )


    # ========================================================
    # 10. Heatmap de correlação com Speedup_Ratio
    # ========================================================

    if "Speedup_Ratio" in merged_all.columns:

        speedup_corr_data = {}

        for ds in datasets:

            df_ds = merged_all[
                merged_all[dataset_col] == ds
            ]

            row_corrs = {}

            for feat in all_corr_cols:

                # Ignorar variáveis não disponíveis
                if feat not in df_ds.columns:
                    continue

                # Converter para valores numéricos, se necessário
                x = pd.to_numeric(
                    df_ds[feat],
                    errors="coerce",
                )

                y = pd.to_numeric(
                    df_ds["Speedup_Ratio"],
                    errors="coerce",
                )

                # Selecionar pares válidos
                valid_idx = (
                    x.notna()
                    & y.notna()
                    & np.isfinite(x)
                    & np.isfinite(y)
                )

                x_valid = x[valid_idx]
                y_valid = y[valid_idx]

                # Pearson exige pelo menos dois pontos
                # e variáveis não constantes
                if (
                    len(x_valid) > 1
                    and x_valid.nunique() > 1
                    and y_valid.nunique() > 1
                ):

                    corr, _ = pearsonr(
                        x_valid,
                        y_valid,
                    )

                    row_corrs[feat] = corr

                else:
                    row_corrs[feat] = np.nan

            speedup_corr_data[str(ds)] = row_corrs

        df_speedup_matrix = pd.DataFrame.from_dict(
            speedup_corr_data,
            orient="index",
        )

        # Anotações seletivas para Speedup_Ratio
        annotations_speedup = create_correlation_annotations(
            df_speedup_matrix,
            threshold=correlation_threshold,
        )

        # Figura
        plt.figure(
            figsize=(
                14,
                max(4, 0.8 * len(datasets)),
            )
        )

        ax = sns.heatmap(
            df_speedup_matrix,
            annot=annotations_speedup,
            fmt="",
            cmap="coolwarm",
            vmin=-1,
            vmax=1,
            linewidths=0.5,
            cbar_kws={
                "label": "Correlação de Pearson com Speedup_Ratio"
            },
        )

        plt.title(
            "Correlação de Cada Atributo/Erro com Speedup_Ratio por Dataset",
            fontweight="bold",
            pad=15,
        )

        plt.xlabel(
            "Atributos Geométricos e Métricas de Erro",
            labelpad=10,
        )

        plt.ylabel(
            "Datasets",
            labelpad=10,
        )

        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)

        plt.tight_layout()

        speedup_output_file = (
            f"heatmap_speedup_por_dataset_{file_suffix}.pdf"
        )

        plt.savefig(
            speedup_output_file,
            dpi=300,
            bbox_inches="tight",
        )

        plt.close()

        print(
            "Gerado heatmap de Speedup por dataset: "
            f"'{speedup_output_file}'"
        )

    else:
        print(
            "Aviso: A coluna 'Speedup_Ratio' não foi encontrada."
        )

else:
    print(
        "Aviso: Nenhuma coluna de identificação de dataset "
        "foi encontrada para gerar os heatmaps individuais "
        "e o heatmap de speedup."
    )
