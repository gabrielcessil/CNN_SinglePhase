" UPDATE BOX-PLOT SCRIPT"
" UPDATE UMAP SCRIPT"


import os
import torch
import numpy as np
import pandas as pd
import porespy as ps
import quantimpy.minkowski as mk
from torch.utils.data import DataLoader

# Import local utilities
from Utilities import dataset_reader as dr
from Utilities import velocity_usage as vu

def analyze_and_save_properties(datasets_dict: dict, batch_size: int = 4, save_csv_path: str = "Dataset_Properties_Table.csv"):
    """
    Extracts Porosity, Permeability, Q1/Mean/Max Local Thickness, and 
    Minkowski Functionals per sample, and generates tabular summary CSVs.
    """
    
    # ==========================================
    # 1. DATA EXTRACTION (Create Per-Sample Table)
    # ==========================================
    records = []
    
    print("--- Starting Extraction ---")
    for dataset_name, datapath in datasets_dict.items():
        print(f"Processing Dataset: {dataset_name}")
        
        # Load Dataset
        dataset = dr.LazyDatasetTorch(h5_path=datapath, list_ids=None, 
                                      x_dtype=torch.float32, y_dtype=torch.float32)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        sample_index = 0  # Track the sample index per dataset
        
        with torch.no_grad():
            for batch_inp, batch_tar in loader:
                B = batch_inp.shape[0]
                
                for b in range(B):
                    # Extract porous mask and convert to uint8 for Quantimpy
                    mask_np = (batch_inp[b, 0] > 0).cpu().numpy()
                    mask_int = mask_np.astype(np.uint8)
                    
                    # Calculate Porosity
                    porosity = np.mean(mask_np)
                    
                    # Calculate Permeability
                    denorm_tar = vu.tensor_denorm(batch_tar[b:b+1], batch_inp[b:b+1])
                    perm = vu.permeability_calculation(denorm_tar, 
                                                       batch_inp[b:b+1], 
                                                       tau=1.5,   
                                                       Re=0.1, 
                                                       dens=1.0,
                                                       denorm=False).item()
                    
                    # Calculate Tortuosity
                    uz = denorm_tar[0, 0].cpu().numpy()[mask_np]
                    uy = denorm_tar[0, 1].cpu().numpy()[mask_np]
                    ux = denorm_tar[0, 2].cpu().numpy()[mask_np]
                    mean_uz = np.mean(uz) if len(uz) > 0 else 0
                    mag = np.sqrt(uz**2 + uy**2 + ux**2)
                    tortuosity = np.mean(mag) / mean_uz if mean_uz != 0 else np.nan
                        
                    # Calculate Local Thickness
                    thick = ps.filters.local_thickness(mask_np)
                    valid_thick = thick[mask_np]
                    
                    if valid_thick.size > 0:
                        q1_thick   = np.percentile(valid_thick, 25)
                        mean_thick = valid_thick.mean()
                        max_thick  = valid_thick.max()
                    else:
                        q1_thick = mean_thick = max_thick = np.nan
                        
                    # Calculate Minkowski Functionals (W0, W1, W2, W3 for 3D)
                    try:
                        mink_vals = mk.functionals(mask_int.astype(bool))
                        w0, w1, w2, w3 = mink_vals[0], mink_vals[1], mink_vals[2], mink_vals[3]
                    except Exception as e:
                        raise (f"  Warning: Minkowski calculation failed on sample {sample_index}: {e}")
                        w0 = w1 = w2 = w3 = np.nan
                        
                    # Append sample data
                    records.append({
                        "Dataset":              dataset_name,
                        "Sample Index":         sample_index,
                        "Porosity":             porosity,
                        "Tortuosity":           tortuosity,
                        "Permeability":         perm,
                        "Q1 Local Thickness":   q1_thick,
                        "Mean Local Thickness": mean_thick,
                        "Max Local Thickness":  max_thick,
                        "M. Volume":            w0,
                        "M. Surface Area":      w1,
                        "M. Mean Curvature":    w2,
                        "M. Euler Char":        w3
                    })
                    
                    sample_index += 1

    # Create the complete per-sample table
    df_samples = pd.DataFrame(records)    
    
    # Save the per-sample data
    if save_csv_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_csv_path) or ".", exist_ok=True)
        df_samples.to_csv(save_csv_path, index=False)
        print(f"\nPer-sample table saved to {save_csv_path}")

    # Create the summary table
    print("\n--- Summary Table (Mean per Dataset) ---")
    
    # Drop "Sample Index" since averaging it doesn't make sense for a summary
    df_summary = df_samples.drop(columns=["Sample Index"]).groupby("Dataset", as_index=False).mean()
    
    # Print summary to console
    try:
        print(df_summary.to_markdown(index=False, tablefmt="grid"))
    except ImportError:
        print(df_summary.to_string(index=False, justify='left', col_space=18))
    
    # Save the summary data
    if save_csv_path:
        summary_csv = save_csv_path.replace(".csv", "_Summary.csv")
        df_summary.to_csv(summary_csv, index=False)
        print(f"\nSummary table saved to {summary_csv}")
        
    print("-" * 40)



table_name      = "Dataset_Properties_Table_Test.csv"

# Define datasets mapping
dataset_folder  = "../NN_Datasets_Grad_Dist_40_5_55/"
datasets        = {
    "Spherical Pores":   dataset_folder+"Test_Silveira_SphPore_SAug_DNorm.h5",
    "Spherical Grains":  dataset_folder+"Test_Silveira_SphGrain_SAug_DNorm.h5",
    #"Cylindrical Pores": dataset_folder+"Train_Silveira_CylinPore_SAug_DNorm.h5",
    #"Cylindrical Grains":dataset_folder+"Train_Silveira_CylinGrain_SAug_DNorm.h5",
    "Leopard":           dataset_folder+"Test_Oliveira_Leopard_SAug_DNorm.h5",
    "Castle Gate":       dataset_folder+"Test_Oliveira_CastleGate_SAug_DNorm.h5",
    "Berea Upper Gray":  dataset_folder+"Test_Oliveira_BereaUpperGray_SAug_DNorm.h5",
    "Berea Sinter Gray": dataset_folder+"Test_Oliveira_BereaSinterGray_SAug_DNorm.h5",
    "Berea Buff":        dataset_folder+"Test_Oliveira_BereaBuff_SAug_DNorm.h5",
    "Berea":             dataset_folder+"Test_Oliveira_Berea_SAug_DNorm.h5",
    "Bentheimer":        dataset_folder+"Test_Oliveira_Bentheimer_SAug_DNorm.h5",
}

# Run execution
analyze_and_save_properties(
    datasets_dict=datasets, 
    batch_size=8,
    save_csv_path=dataset_folder+table_name
)