import os
import csv
import torch
from torch.utils.data import DataLoader

# Import local utilities
from Utilities import dataset_reader as dr
from Utilities import velocity_usage as vu
from Architectures.Unet import Extended_DannyKo

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# ---------------------------------------------------------
# 1. DEFINE DATASETS & OUTPUT PATH
# ---------------------------------------------------------
dataset_folder = "../NN_Datasets_Grad_Dist_40_5_55/"
datasets = {
    "Spherical Pores":   dataset_folder + "Test_Silveira_SphPore_SAug_DNorm.h5",
    "Spherical Grains":  dataset_folder + "Test_Silveira_SphGrain_SAug_DNorm.h5",
    "Leopard":           dataset_folder + "Test_Oliveira_Leopard_SAug_DNorm.h5",
    "Castle Gate":       dataset_folder + "Test_Oliveira_CastleGate_SAug_DNorm.h5",
    "Berea Upper Gray":  dataset_folder + "Test_Oliveira_BereaUpperGray_SAug_DNorm.h5",
    "Berea Sinter Gray": dataset_folder + "Test_Oliveira_BereaSinterGray_SAug_DNorm.h5",
    "Berea Buff":        dataset_folder + "Test_Oliveira_BereaBuff_SAug_DNorm.h5",
    "Berea":             dataset_folder + "Test_Oliveira_Berea_SAug_DNorm.h5",
    "Bentheimer":        dataset_folder + "Test_Oliveira_Bentheimer_SAug_DNorm.h5",
}

out_dir = "./Tables/PermeabilityExtraction/"
os.makedirs(out_dir, exist_ok=True)
csv_path = os.path.join(out_dir, "Extracted_Permeability_Predictions.csv")

# ---------------------------------------------------------
# 2. INSTANTIATE AND LOAD SINGLE-COMPONENT Z-MODELS
# ---------------------------------------------------------
models_dict = {}

# Model 1: 
danny_model_base1 = Extended_DannyKo()
model_z_1 = danny_model_base1.z_model
model_z_1.load_state_dict(
    torch.load("./Trained_Models/NN_Trainning_26_August_2026_03-45PM_Job27376/model_LowerValidationLoss.pth", 
               map_location=torch.device(device), 
               weights_only=True)
)
model_z_1.bin_input = True
model_z_1.to(device)
model_z_1.eval()
models_dict["PE-GE-R"] = model_z_1

# Model 2:
danny_model_base2 = Extended_DannyKo()
model_z_2 = danny_model_base2.z_model
model_z_2.load_state_dict(
    torch.load("./Trained_Models/NN_Trainning_9_September_2026_01-49PM_Job28304/model_LowerValidationLoss.pth", 
               map_location=torch.device(device), 
               weights_only=True)
)
model_z_2.bin_input = True
model_z_2.to(device)
model_z_2.eval()
models_dict["PE"] = model_z_2

# ---------------------------------------------------------
# 3. SAMPLE-BY-SAMPLE METRIC COLLECTION & LINE-BY-LINE WRITING
# ---------------------------------------------------------
fieldnames = ["Model", "Dataset", "Sample_Index", "Permeability", "Predicted_Permeability"]

# Open with default buffering (Python handles optimal chunking to disk)
with open(csv_path, mode='w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()

    for dataset_name, datapath in datasets.items():
        if not os.path.exists(datapath):
            print(f"Skipping {dataset_name}: File not found ({datapath})")
            continue

        print(f"Processing Dataset: {dataset_name}")
        dataset = dr.LazyDatasetTorch(h5_path=datapath, list_ids=None, x_dtype=torch.float32, y_dtype=torch.float32)
        loader = DataLoader(dataset, batch_size=1, shuffle=False)

        for sample_idx, (batch_inp, batch_tar) in enumerate(loader):
            batch_inp = batch_inp.to(device)
            batch_tar = batch_tar.to(device)

            # Denormalize target and compute ground-truth permeability
            denorm_tar = vu.tensor_denorm(batch_tar, batch_inp)
            true_perm = vu.permeability_calculation(
                denorm_tar, batch_inp, tau=1.5, Re=0.1, dens=1.0, denorm=False
            ).item()

            # Predict and calculate permeability for each model
            for model_name, model in models_dict.items():
                with torch.no_grad():
                    pred_z = model(batch_inp)
                    
                    if pred_z.shape[1] == 1:
                        zeros = torch.zeros_like(pred_z)
                        pred_full = torch.cat([pred_z, zeros, zeros], dim=1)
                    else:
                        pred_full = pred_z

                    denorm_pred = vu.tensor_denorm(out=pred_full, inp=batch_inp)

                pred_perm = vu.permeability_calculation(
                    denorm_pred, batch_inp, tau=1.5, Re=0.1, dens=1.0, denorm=False
                ).item()

                # Stream directly to CSV line-by-line
                writer.writerow({
                    "Model": model_name,
                    "Dataset": dataset_name,
                    "Sample_Index": sample_idx,
                    "Permeability": true_perm,
                    "Predicted_Permeability": pred_perm
                })
                
print(f"\nExtraction completed successfully! Results streamed directly to: {csv_path}")