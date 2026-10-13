import os
import pandas as pd


def clean_numeric_columns(df, text_columns=["Dataset"]):
    """Converts all columns except text_columns to numeric types.

    Handles string numbers, extra whitespace, and comma decimal separators
    (e.g., '1,23' -> 1.23).
    """
    for col in df.columns:
        if col in text_columns:
            continue
        if df[col].dtype == object:
            # Strip whitespace, replace European decimal commas with dots, convert to numeric
            cleaned = df[col].astype(str).str.strip().str.replace(",", ".")
            df[col] = pd.to_numeric(cleaned, errors="coerce")
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def format_median_q1_q3(series):
    """Calcula a Mediana, Q1 e Q3 e retorna uma string formatada: "Mediana (Q1-Q3)" """
    s = series.dropna()
    if len(s) == 0:
        return pd.NA

    median_val = s.median()
    Q1 = s.quantile(0.25)
    Q3 = s.quantile(0.75)

    return f"{median_val:.2f} ({Q1:.2f}-{Q3:.2f})"


def format_median_whiskers(series):
    """Calculates the Median and the actual Box Plot Whiskers (excluding outliers),

    returning a formatted string: "Median (LowerWhisker - UpperWhisker)"
    """
    s = series.dropna()
    if len(s) == 0:
        return pd.NA

    median_val = s.median()
    Q1 = s.quantile(0.25)
    Q3 = s.quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    lower_whisker = s[s >= lower_bound].min()
    upper_whisker = s[s <= upper_bound].max()

    return f"{median_val:.2f} ({lower_whisker:.2f}-{upper_whisker:.2f})"


def generate_metric_summaries(
    group_name,
    file_list,
    input_dir="./Tables",
    output_dir="./Tables_Summary",
    dataset_mapping=None,
):
    """Lê uma lista de CSVs, agrupa por Dataset (ou por categorias se dataset_mapping for fornecido),

    calcula as métricas e exporta para CSV e LaTeX.
    """
    os.makedirs(output_dir, exist_ok=True)
    metric_dfs = {}

    for file in file_list:
        filepath = os.path.join(input_dir, file)

        if not os.path.exists(filepath):
            print(f"Warning: File not found -> {filepath}")
            continue

        model_col_name = file.replace("ErrorMetrics_", "").replace(".csv", "")

        df = pd.read_csv(filepath)

        # Drop index column first if present
        if "Sample_Index" in df.columns:
            df = df.drop(columns=["Sample_Index"])

        # Convert all metric columns to numeric type
        df = clean_numeric_columns(df, text_columns=["Dataset"])

        if dataset_mapping:
            df["Dataset"] = df["Dataset"].map(
                lambda x: dataset_mapping.get(x, x)
            )

        summary = df.groupby("Dataset").agg(format_median_q1_q3)

        for metric in summary.columns:
            if metric not in metric_dfs:
                metric_dfs[metric] = pd.DataFrame()

            metric_dfs[metric][model_col_name] = summary[metric]

    for metric, metric_df in metric_dfs.items():
        safe_metric_name = metric.replace(" ", "_").replace("/", "")

        csv_filename = os.path.join(
            output_dir, f"{group_name}_{safe_metric_name}.csv"
        )
        metric_df.to_csv(csv_filename)

        tex_filename = os.path.join(
            output_dir, f"{group_name}_{safe_metric_name}.tex"
        )
        with open(tex_filename, "w") as f:
            f.write(metric_df.to_latex())

        print(f"Exported: {csv_filename} & .tex")


def generate_combined_model_metric_summary(
    group_name,
    file_list,
    input_dir="./Tables",
    output_dir="./Tables_Summary",
    dataset_mapping=None,
):
    """Lê uma lista de CSVs, agrupa por Dataset (ou categorias), calcula métricas e

    mescla tudo em UMA ÚNICA tabela onde Modelos são as colunas principais.
    """
    os.makedirs(output_dir, exist_ok=True)
    combined_dfs = []

    for file in file_list:
        filepath = os.path.join(input_dir, file)

        if not os.path.exists(filepath):
            print(f"Warning: File not found -> {filepath}")
            continue

        model_col_name = file.replace("ErrorMetrics_", "").replace(".csv", "")

        df = pd.read_csv(filepath)

        if "Sample_Index" in df.columns:
            df = df.drop(columns=["Sample_Index"])

        # Convert all metric columns to numeric type
        df = clean_numeric_columns(df, text_columns=["Dataset"])

        if dataset_mapping:
            df["Dataset"] = df["Dataset"].map(
                lambda x: dataset_mapping.get(x, x)
            )

        summary = df.groupby("Dataset").agg(format_median_q1_q3)

        summary.columns = pd.MultiIndex.from_product(
            [[model_col_name], summary.columns]
        )
        combined_dfs.append(summary)

    if not combined_dfs:
        print(f"No valid data found for group: {group_name}")
        return

    final_df = pd.concat(combined_dfs, axis=1)

    csv_filename = os.path.join(output_dir, f"{group_name}.csv")
    final_df.to_csv(csv_filename)

    tex_filename = os.path.join(output_dir, f"{group_name}.tex")
    with open(tex_filename, "w") as f:
        f.write(final_df.to_latex(multicolumn=True, multirow=True))

    print(f"Exported Combined Table: {csv_filename} & .tex")


INPUT_DIRECTORY = "./Tables/ErrorByModel/"
OUTPUT_DIRECTORY = "./Tables/ErrorSummaries/"

DATASET_CATEGORY_MAPPING = {
    "Bentheimer 120³": "Rocks 120³ (ID)",
    "Berea 120³": "Rocks 120³ (ID)",
    "Berea Buff 120³": "Rocks 120³ (ID)",
    "Berea Sinter Gray 120³": "Rocks 120³ (ID)",
    "Berea Upper Gray 120³": "Rocks 120³ (ID)",
    "Castle Gate 120³": "Rocks 120³ (ID)",
    "Leopard 120³": "Rocks 120³ (ID)",
    "Ketton 256³": "Rocks 256³ (OOD T-S)",
    "Estaillades 256³": "Rocks 256³ (OOD T-S)",
    "Doddington 256³": "Rocks 256³ (OOD T-S)",
}

# 1) Compare Ko et al (Component 0) with increasing N samples
group_1_files = [
    "ErrorMetrics_Comp0_Ko_et_al_D3_N0.csv",
    "ErrorMetrics_Comp0_Ko_et_al_D3_N1.csv",
    "ErrorMetrics_Comp0_Ko_et_al_D3_N2.csv",
    "ErrorMetrics_Comp0_Ko_et_al_D3_N3.csv",
    "ErrorMetrics_Comp0_Ko_et_al_D3_N4.csv",
]
print("\nProcessing Group 1: Ko et al (Increasing N)")
generate_metric_summaries(
    "Group1_Ko_IncreasingN", group_1_files, INPUT_DIRECTORY, OUTPUT_DIRECTORY
)
generate_metric_summaries(
    "Group1_Ko_IncreasingN_Summarized",
    group_1_files,
    INPUT_DIRECTORY,
    OUTPUT_DIRECTORY,
    dataset_mapping=DATASET_CATEGORY_MAPPING,
)

# 2) Compare Ko et al (Component 0) with decreasing Diversity (D)
group_2_files = [
    "ErrorMetrics_Comp0_Ko_et_al_D3_N4.csv",
    "ErrorMetrics_Comp0_Ko_et_al_D2_N4.csv",
    "ErrorMetrics_Comp0_Ko_et_al_D1_N4.csv",
    "ErrorMetrics_Comp0_Ko_et_al_D0_N4.csv",
]
print("\nProcessing Group 2: Ko et al (Decreasing Diversity)")
generate_metric_summaries(
    "Group2_Ko_DecreasingD", group_2_files, INPUT_DIRECTORY, OUTPUT_DIRECTORY
)
generate_metric_summaries(
    "Group2_Ko_DecreasingD_Summarized",
    group_2_files,
    INPUT_DIRECTORY,
    OUTPUT_DIRECTORY,
    dataset_mapping=DATASET_CATEGORY_MAPPING,
)

# 3) Compare Javier Santos (Component 0) with increasing Diversity (D)
group_3_files = [
    "ErrorMetrics_Comp0_Javier_Santos_D0_N4.csv",
    "ErrorMetrics_Comp0_Javier_Santos_D1_N4.csv",
    "ErrorMetrics_Comp0_Javier_Santos_D2_N4.csv",
    "ErrorMetrics_Comp0_Javier_Santos_D3_N4.csv",
]
print("\nProcessing Group 3: Javier Santos (Increasing Diversity)")
generate_metric_summaries(
    "Group3_Santos_IncreasingD_Summarized",
    group_3_files,
    INPUT_DIRECTORY,
    OUTPUT_DIRECTORY,
    dataset_mapping=DATASET_CATEGORY_MAPPING,
)

# 4) Combined table for models across different components
group_4_files = [
    "ErrorMetrics_Comp2_Ko_et_al_D3.csv",
    "ErrorMetrics_Comp3_Ko_et_al_D3.csv",
]
print("\nProcessing Group 4: Separate Components (Combined Table)")
generate_combined_model_metric_summary(
    "Group4_Ko_Components", group_4_files, INPUT_DIRECTORY, OUTPUT_DIRECTORY
)

group_5_files = ["ErrorMetrics_Comp5_Ko_et_al_D3.csv"]
print("\nProcessing Group 5: Main model")
generate_combined_model_metric_summary(
    "Group5_Ko_Main", group_5_files, INPUT_DIRECTORY, OUTPUT_DIRECTORY
)

print("\nAll summary tables generated successfully!")