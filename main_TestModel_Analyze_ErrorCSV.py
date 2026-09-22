import os
import pandas as pd

def generate_metric_summaries(group_name, file_list, input_dir="./Tables", output_dir="./Tables_Summary"):
    """
    Reads a list of model CSVs, groups by Dataset, calculates the mean for each metric, 
    rounds to 3 decimal places, and exports a separate CSV for every metric.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Dictionary to store a DataFrame for each unique metric
    metric_dfs = {}
    
    for file in file_list:
        filepath = os.path.join(input_dir, file)
        
        if not os.path.exists(filepath):
            print(f"Warning: File not found -> {filepath}")
            continue
            
        # Extract model name for the column header (removes 'ErrorMetrics_' and '.csv')
        model_col_name = file.replace("ErrorMetrics_", "").replace(".csv", "")
        
        df = pd.read_csv(filepath)
        
        # Drop the sample index to aggregate purely by dataset
        if "Sample_Index" in df.columns:
            df = df.drop(columns=["Sample_Index"])
            
        # Group by Dataset, calculate the mean, and round to 3 decimal places
        summary = df.groupby("Dataset").mean().round(3)
        
        # Pivot the data into the metric_dfs dictionary
        for metric in summary.columns:
            if metric not in metric_dfs:
                metric_dfs[metric] = pd.DataFrame()
            
            # Assign the model's mean values for this metric as a new column
            metric_dfs[metric][model_col_name] = summary[metric]

    # Export each metric as a separate table
    for metric, metric_df in metric_dfs.items():
        # Clean metric name for the filename
        safe_metric_name = metric.replace(" ", "_").replace("/", "")
        output_filename = os.path.join(output_dir, f"{group_name}_{safe_metric_name}.csv")
        
        metric_df.to_csv(output_filename)
        print(f"Exported: {output_filename}")


if __name__ == "__main__":
    
    INPUT_DIRECTORY = "./Tables"
    OUTPUT_DIRECTORY = "./Tables/ErrorSummaries/"

    # 1) Compare Ko et al (Component 0) with increasing N samples
    group_1_files = [
        "ErrorMetrics_Comp0_Ko_et_al_D3_N0.csv",
        "ErrorMetrics_Comp0_Ko_et_al_D3_N1.csv",
        "ErrorMetrics_Comp0_Ko_et_al_D3_N2.csv",
        "ErrorMetrics_Comp0_Ko_et_al_D3_N3.csv",
        "ErrorMetrics_Comp0_Ko_et_al_D3_N4.csv"
    ]
    print("\nProcessing Group 1: Ko et al (Increasing N)")
    generate_metric_summaries("Group1_Ko_IncreasingN", group_1_files, INPUT_DIRECTORY, OUTPUT_DIRECTORY)

    # 2) Compare Ko et al (Component 0) with decreasing Diversity (D)
    group_2_files = [
        "ErrorMetrics_Comp0_Ko_et_al_D3_N4.csv",
        "ErrorMetrics_Comp0_Ko_et_al_D2_N4.csv",
        "ErrorMetrics_Comp0_Ko_et_al_D1_N4.csv",
        "ErrorMetrics_Comp0_Ko_et_al_D0_N4.csv"
    ]
    print("\nProcessing Group 2: Ko et al (Decreasing Diversity)")
    generate_metric_summaries("Group2_Ko_DecreasingD", group_2_files, INPUT_DIRECTORY, OUTPUT_DIRECTORY)

    # 3) Compare Javier Santos (Component 0) with increasing Diversity (D)
    group_3_files = [
        "ErrorMetrics_Comp0_Javier_Santos_D0_N4.csv",
        "ErrorMetrics_Comp0_Javier_Santos_D1_N4.csv",
        "ErrorMetrics_Comp0_Javier_Santos_D2_N4.csv",
        "ErrorMetrics_Comp0_Javier_Santos_D3_N4.csv"
    ]
    print("\nProcessing Group 3: Javier Santos (Increasing Diversity)")
    generate_metric_summaries("Group3_Santos_IncreasingD", group_3_files, INPUT_DIRECTORY, OUTPUT_DIRECTORY)

    # 4) Separate tables for individual models across different components
    group_4_files = [
        "ErrorMetrics_Comp2_Ko_et_al_D3.csv",
        "ErrorMetrics_Comp3_Ko_et_al_D3.csv",
        "ErrorMetrics_Comp5_Ko_et_al_D3.csv"
    ]
    print("\nProcessing Group 4: Separate Components")
    for file in group_4_files:
        comp_name = file.split("_")[1]
        generate_metric_summaries(f"Group4_Isolated_{comp_name}", [file], INPUT_DIRECTORY, OUTPUT_DIRECTORY)

    print("\nAll summary tables generated successfully!")