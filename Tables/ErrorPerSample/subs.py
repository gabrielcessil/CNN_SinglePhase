 
import os
import pandas as pd


def update_dataset_column(
    file_list, dataset_mapping, input_dir="./", output_dir=None
):
    
    # Use input_dir if output_dir is not specified (overwrites original files)
    if output_dir is None:
        output_dir = input_dir
    else:
        os.makedirs(output_dir, exist_ok=True)

    for filename in file_list:
        filepath = os.path.join(input_dir, filename)

        if not os.path.exists(filepath):
            print(f"Warning: File not found -> {filepath}")
            continue

        # Load CSV
        df = pd.read_csv(filepath)

        if "Dataset" in df.columns:
            # Replace values based on dictionary mapping
            # Unmapped values remain unchanged
            df["Dataset"] = df["Dataset"].replace(dataset_mapping)

            # Save modified DataFrame back to CSV
            out_filepath = os.path.join(output_dir, filename)
            df.to_csv(out_filepath, index=False)
            print(f"Updated and saved: {out_filepath}")
        else:
            print(f"Skipped (no 'Dataset' column found): {filepath}")


# ==========================================
# Example Usage
# ==========================================
if __name__ == "__main__":
    # 1. Define your mapping dictionary
    mapping_dict = {
        "Spherical Pores":      "Spherical Pores 120³",
        "Spherical Grains":     "Spherical Grains 120³",
        "Cylindrical Pores":     "Cylindrical Pores 120³",
        "Cylindrical Grains":     "Cylindrical Grains 120³",
        "Leopard":              "Leopard 120³",
        "Castle Gate":          "Castle Gate 120³",
        "Berea Upper Gray":     "Berea Upper Gray 120³",
        "Berea Sinter Gray":    "Berea Sinter Gray 120³",
        "Berea Buff":           "Berea Buff 120³",
        "Berea":                "Berea 120³",
        "Bentheimer":           "Bentheimer 120³",
    }

    # 2. List of CSV files to process
    files_to_process = [
        "ErrorMetrics_Comp0_Ko_et_al_D3_N0.csv",
        "ErrorMetrics_Comp0_Ko_et_al_D3_N1.csv",
        "ErrorMetrics_Comp0_Ko_et_al_D3_N2.csv",
        "ErrorMetrics_Comp0_Ko_et_al_D3_N3.csv",
        "ErrorMetrics_Comp0_Ko_et_al_D3_N4.csv",
        
        "ErrorMetrics_Comp0_Ko_et_al_D0_N4.csv",
        "ErrorMetrics_Comp0_Ko_et_al_D1_N4.csv",
        "ErrorMetrics_Comp0_Ko_et_al_D2_N4.csv",
        
        "ErrorMetrics_Comp0_Javier_Santos_D0_N4.csv",
        "ErrorMetrics_Comp0_Javier_Santos_D1_N4.csv",
        "ErrorMetrics_Comp0_Javier_Santos_D2_N4.csv",  
        "ErrorMetrics_Comp0_Javier_Santos_D3_N4.csv",

        "ErrorMetrics_Comp2_Ko_et_al_D3.csv",
        "ErrorMetrics_Comp3_Ko_et_al_D3.csv",
        "ErrorMetrics_Comp5_Ko_et_al_D3.csv"
    ]

    # Directories
    INPUT_FOLDER = "./"
    OUTPUT_FOLDER = (
        "./DEBUG/"  # Set to None to overwrite original files
    )

    # Execute
    update_dataset_column(
        files_to_process,
        mapping_dict,
        input_dir=INPUT_FOLDER,
        output_dir=OUTPUT_FOLDER,
    )