import os
import torch
import pandas as pd
from torch.utils.data import DataLoader, Subset

# Import local architectures
from Architectures.Unet import Extended_DannyKo

# Import local utilities
from Utilities import dataset_reader as dr
from Utilities import error_metrics as em 
from Utilities import start_handler as sh

# =======================================================
# CONFIGURATION
# =======================================================
component = 0      # Uz
batch_size = 30    # Increased to process multiple samples at once
N_samples = 30     # Process 2 batches of 4 per dataset
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Defining Datasets
datasets = {
    "Spherical Pores":      "../NN_Datasets_Grad_Dist_40_5_55/Test_Silveira_SphPore_SAug_DNorm.h5",
    "Spherical Grains":     "../NN_Datasets_Grad_Dist_40_5_55/Test_Silveira_SphGrain_SAug_DNorm.h5",
    "Cylindrical Pores":    "../NN_Datasets_Grad_Dist_40_5_55/Test_Silveira_CylinPore_SAug_DNorm.h5",
    "Cylindrical Grains":   "../NN_Datasets_Grad_Dist_40_5_55/Test_Silveira_CylinGrain_SAug_DNorm.h5",
    "Bentheimer":           "../NN_Datasets_Grad_Dist_40_5_55/Test_Oliveira_Bentheimer_SAug_DNorm.h5",
    "Berea Buff":           "../NN_Datasets_Grad_Dist_40_5_55/Test_Oliveira_BereaBuff_SAug_DNorm.h5",
    "Leopard":              "../NN_Datasets_Grad_Dist_40_5_55/Test_Oliveira_Leopard_SAug_DNorm.h5",
    "Castle Gate":          "../NN_Datasets_Grad_Dist_40_5_55/Test_Oliveira_CastleGate_SAug_DNorm.h5",
    "Berea Upper Gray":     "../NN_Datasets_Grad_Dist_40_5_55/Test_Oliveira_BereaUpperGray_SAug_DNorm.h5",
    "Berea Sinter Gray":    "../NN_Datasets_Grad_Dist_40_5_55/Test_Oliveira_BereaSinterGray_SAug_DNorm.h5",
    "Berea":                "../NN_Datasets_Grad_Dist_40_5_55/Test_Oliveira_Berea_SAug_DNorm.h5",
}

# =======================================================
# MODEL INITIALIZATION: Ko et al (D3 N4)
# =======================================================
print("Initializing Ko et al (D3 N4)...")
danny_model_base3 = Extended_DannyKo()
model_z_3 = danny_model_base3.z_model

model_path = "./Trained_Models/NN_Trainning_26_August_2026_03-45PM_Job27376/model_LowerValidationLoss.pth"
model_z_3.load_state_dict(torch.load(model_path, map_location=torch.device(device), weights_only=True))
model_z_3.bin_input = True
model_z_3.eval()
model_z_3.to(device)

results = []

# =======================================================
# EVALUATION LOOP
# =======================================================
for dataset_name, datapath in datasets.items():
    print(f"\nEvaluating {dataset_name}...")
    
    # Load Dataset
    dataset = dr.LazyDatasetTorch(
        h5_path=datapath, 
        list_ids=None, 
        x_dtype=torch.float32, 
        y_dtype=torch.float32,
        component=component
    )
    
    # Subset
    subset = Subset(dataset, range(N_samples))
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False)
    
    sample_idx = 0
    
    with torch.no_grad():
        for batch_inputs, batch_targets in loader:
            B = batch_inputs.shape[0]
            print(f"  Processing Batch (Size: {B})...")
            
            batch_inputs  = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)
            
            # ---------------------------------------------------------
            # 1. PREDICTION WITHOUT PADDING
            # ---------------------------------------------------------
            pred_nopad = model_z_3.predict(batch_inputs)
            
            # ---------------------------------------------------------
            # 2. PREDICTION WITH PADDING
            # ---------------------------------------------------------
            padded_inputs_list = []
            
            # Unpack the 5D tensor (B, C, Z, Y, X) to process 3D volumes (Z, Y, X)
            for i in range(B):
                inp_3d = batch_inputs[i, 0].cpu().numpy()
                pad_inp_3d = sh.pad_geometry(inp_3d, shape=(160,160,160))
                padded_inputs_list.append(torch.from_numpy(pad_inp_3d).unsqueeze(0).to(torch.float32))
                
            batch_inputs_padded = torch.stack(padded_inputs_list).to(device)
            
            # Predict on padded batch
            pred_pad = model_z_3.predict(batch_inputs_padded)
            
            # Unpad predictions back to original shape
            original_spatial_shape = batch_inputs.shape[2:] 
            unpadded_preds_list = []
            
            for i in range(B):
                pred_3d = pred_pad[i, 0].cpu().numpy()
                unpad_pred_3d = sh.unpad_geometry(pred_3d, original_spatial_shape)
                unpadded_preds_list.append(torch.from_numpy(unpad_pred_3d).unsqueeze(0).to(torch.float32))
                
            pred_pad_unpadded = torch.stack(unpadded_preds_list).to(device)
            
            # ---------------------------------------------------------
            # 3. METRICS CALCULATION
            # ---------------------------------------------------------
            # Metrics now return arrays/lists of length B
            bias_nopad = em.Bias_Comparison(batch_inputs, pred_nopad, batch_targets)
            corr_nopad = em.Correlation_Comparison(batch_inputs, pred_nopad, batch_targets)
            
            bias_pad = em.Bias_Comparison(batch_inputs, pred_pad_unpadded, batch_targets)
            corr_pad = em.Correlation_Comparison(batch_inputs, pred_pad_unpadded, batch_targets)
            
            # Iterate through the batch to append individual results
            for b in range(B):
                results.append({
                    "Dataset": dataset_name,
                    "Sample": sample_idx,
                    "Bias Error (Base)": float(bias_nopad[b]),
                    "Bias Error (Padded)": float(bias_pad[b]),
                    "Correlation (Base)": float(corr_nopad[b]),
                    "Correlation (Padded)": float(corr_pad[b]),
                })
                sample_idx += 1

# =======================================================
# PRINT RESULTS
# =======================================================
df_results = pd.DataFrame(results)
df_medians = df_results.drop(columns=["Sample"]).groupby("Dataset").median().reset_index()

# Calculate percentage changes (using absolute value for denominator to handle negative errors correctly)
bias_base = df_medians["Bias Error (Base)"]
bias_pad  = df_medians["Bias Error (Padded)"]
df_medians["Bias Change"] = (((bias_pad - bias_base) / bias_base.abs()) * 100).apply(lambda x: f"{x:+.2f}%")

corr_base = df_medians["Correlation (Base)"]
corr_pad  = df_medians["Correlation (Padded)"]
df_medians["Corr Change"] = (((corr_pad - corr_base) / corr_base.abs()) * 100).apply(lambda x: f"{x:+.2f}%")

# Reorder columns for readability
df_medians = df_medians[[
    "Dataset", 
    "Bias Error (Base)", "Bias Error (Padded)", "Bias Change",
    "Correlation (Base)", "Correlation (Padded)", "Corr Change"
]]

print("\n" + "="*105)
print("IMPACT OF PADDING ON PREDICTIONS - Ko et al (D3 N4) [Comp 0]")
print("="*105)
print(df_medians.to_string(index=False))