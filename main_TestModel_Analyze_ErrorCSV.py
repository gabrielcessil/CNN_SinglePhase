import pandas as pd
from pathlib import Path

# Path to your CSV
csv_path = "./Tables/ErrorMetrics_Comp3_Ko_et_al_Etapa_3.csv"

# Read CSV
df = pd.read_csv(csv_path)

# Metrics to average
metrics = [
    "Bias_Error_",
    "Magnitude_Error_",
    "Correlation"
]

# Calculate average for each dataset
summary = (
    df.groupby("Dataset")[metrics]
      .mean()
      .reset_index()
)

# Calculate median across datasets
median_row = pd.DataFrame([{
    "Dataset": "Median",
    "Bias_Error_": summary["Bias_Error_"].median(),
    "Magnitude_Error_": summary["Magnitude_Error_"].median(),
    "Correlation": summary["Correlation"].median()
}])

# Add median as the bottom row
summary = pd.concat([summary, median_row], ignore_index=True)

# Round values
summary[metrics] = summary[metrics].round(3)

# Display
print(summary.to_string(index=False))

# Save using original filename + "_summary"
input_path = Path(csv_path)
output_path = input_path.parent / f"{input_path.stem}_summary.csv"

summary.to_csv(output_path, index=False)
print(f"\nSummary CSV saved to: {output_path}")


# ==========================================
# CREATE LATEX TABLE
# ==========================================
latex_path = input_path.parent / f"{input_path.stem}_summary.tex"

# Clean up headers for the LaTeX table (remove trailing underscores, replace with spaces)
latex_df = summary.copy()
latex_df.columns = latex_df.columns.str.replace('_', ' ').str.strip()

# Generate the core LaTeX table string
latex_body = latex_df.to_latex(
    index=False,
    column_format="l" + "c" * len(metrics),
    float_format="%.3f"
)

# Insert a horizontal line before the Median row for visual separation
latex_body = latex_body.replace("Median", "\\hline\nMedian")

# Wrap in standard table environment
wrapped_latex = (
    "\\begin{table}[h!]\n"
    "    \\centering\n"
    "    \\caption{Summary of Error Metrics}\n"
    "    \\label{tab:metrics_summary}\n"
    + latex_body +
    "\\end{table}\n"
)

# Save to .tex file
with open(latex_path, "w") as f:
    f.write(wrapped_latex)

print(f"LaTeX table saved to: {latex_path}")