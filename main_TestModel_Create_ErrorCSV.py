import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Subset

# Import local architectures
from Architectures.Unet import Extended_DannyKo
from Architectures.MSnet import Extended_JavierSantos

from Architectures.Models import SubModels_Composition

# Import local utilities
from Utilities import dataset_reader as dr
from Utilities import error_metrics as em 
from Utilities import velocity_usage as vu
from Utilities import model_handler as mh


def generate_metrics_per_model(
    datasets_dict: dict, 
    models_dict: dict, 
    component: int = 5, 
    batch_size: int = 4, 
    N_samples: int = None,
    output_dir: str = "./Model_Error_Outputs"
):
    """
    Runs model predictions and error metrics batch-by-batch, saving only 
    the Dataset Name, Sample Index, and error metrics into a separate CSV for each model.
    """
    # Ensure all models are in evaluation mode
    for model_name, model in models_dict.items():
        if hasattr(model, 'eval'):
            model.eval()

    os.makedirs(output_dir, exist_ok=True)

    # Dictionary to store records per model
    model_records = {model_name: [] for model_name in models_dict.keys()}
    
    # Global sample tracker across batches per dataset
    global_sample_indices = {dataset_name: 0 for dataset_name in datasets_dict.keys()}

    for dataset_name, datapath in datasets_dict.items():
        print(f"\nProcessing Dataset: {dataset_name}")
        
        # Load Dataset
        dataset = dr.LazyDatasetTorch(
            h5_path=datapath, 
            list_ids=None, 
            x_dtype=torch.float32, 
            y_dtype=torch.float32,
            component=component
        )
        
        # Subset if testing on a smaller scale
        if N_samples is not None:
            N = min(N_samples, len(dataset))
            dataset = Subset(dataset, range(N))
            
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        with torch.no_grad():
            for batch_idx, (batch_inputs, batch_targets) in enumerate(loader):
                B = batch_inputs.shape[0]
                print(f"  Processing Batch {batch_idx+1} (Size: {B})...")
                
                batch_inputs  = batch_inputs.clone().detach().to(dtype=torch.float32)
                batch_targets = batch_targets.clone().detach().to(dtype=torch.float32)

                # Loop through each model for the current batch
                for model_name, model in models_dict.items():
                    print(f"    Evaluating {model_name}")
                    batch_outputs = model.predict(batch_inputs)
                    batch_outputs = batch_outputs.clone().detach().to(dtype=torch.float32)

                    b_metrics, m_metrics, a_metrics, c_metrics = [], [], [], []
                    f_metrics, t_metrics, d_metrics = [], [], []

                    if component is None or component == 5:
                        b_metrics = em.Bias_Comparison(batch_inputs, batch_outputs, batch_targets)
                        m_metrics = em.Magnitude_Comparison(batch_inputs, batch_outputs, batch_targets)
                        a_metrics = em.Angular_Comparison(batch_inputs, batch_outputs, batch_targets)
                        c_metrics = em.Correlation_Comparison(batch_inputs, batch_outputs, batch_targets)
                        f_metrics = em.Flux_Comparison(batch_inputs, batch_outputs, batch_targets)
                        t_metrics = em.Tortuosity_Comparison(batch_inputs, batch_outputs, batch_targets)
                        d_metrics = em.Divergent_Residual(batch_inputs, batch_outputs)
                        
                    elif component in [0, 1, 2, 3]:
                        if component == 1 or component == 2:
                            batch_outputs_aux = batch_outputs.abs()
                            batch_targets_aux = batch_targets.abs()
                            b_metrics = em.Bias_Comparison(batch_inputs, batch_outputs_aux, batch_targets_aux)
                            m_metrics = em.Magnitude_Comparison(batch_inputs, batch_outputs_aux, batch_targets_aux)
                        else:
                            b_metrics = em.Bias_Comparison(batch_inputs, batch_outputs, batch_targets)
                            m_metrics = em.Magnitude_Comparison(batch_inputs, batch_outputs, batch_targets)
                            
                        c_metrics = em.Correlation_Comparison(batch_inputs, batch_outputs, batch_targets)

                    # Assemble records storing only dataset name, sample index, and error metrics
                    for b in range(B):
                        current_sample_idx = global_sample_indices[dataset_name] + b

                        record = {
                            "Dataset": dataset_name,
                            "Sample_Index": current_sample_idx,
                        }
                        
                        # Safely append error metrics
                        if b < len(b_metrics): record["Bias Error [%]"] = b_metrics[b]
                        if b < len(m_metrics): record["Magnitude Error [%]"] = m_metrics[b]
                        if b < len(c_metrics): record["Correlation"] = c_metrics[b]
                        
                        if component is None or component == 5:
                            if b < len(a_metrics): record["Angular Error [Deg]"] = a_metrics[b]
                            if b < len(f_metrics): record["Flux Error"] = f_metrics[b]
                            if b < len(t_metrics): record["Tortuosity Error [%]"] = t_metrics[b]
                            if b < len(d_metrics): record["Divergent Residual [%]"] = d_metrics[b]

                        model_records[model_name].append(record)

                # Increment global sample tracker for this dataset after processing all models for the batch
                global_sample_indices[dataset_name] += B

    # ==========================================
    # EXPORT RESULTS SEPARATELY PER MODEL
    # ==========================================
    saved_files = {}
    for model_name, recs in model_records.items():
        df_model = pd.DataFrame(recs)
        
        # Clean headers
        df_model.columns = df_model.columns.str.replace(' ', '_')
        df_model.columns = df_model.columns.str.replace(r'[\[\]\%]', '', regex=True)
        
        safe_model_name = model_name.replace(" ", "_").replace("(", "").replace(")", "").replace(".", "")
        csv_filename = os.path.join(output_dir, f"ErrorMetrics_Comp{component}_{safe_model_name}.csv")
        
        df_model.to_csv(csv_filename, index=False)
        saved_files[model_name] = csv_filename
        print(f"Saved metrics for '{model_name}' to {csv_filename}")

    return saved_files


# =======================================================
# MAIN SETUP
# =======================================================
component   = 5 # Uz =0, Ux=2, P=3, (Uz,Uy,Ux)=5
batch_size  = 50
N_samples   = None 
device      = 'cuda'

output_results_dir = "./Tables/"

# Define Datasets
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

print("Initializing models...")

models_dict = {}

if component == 0:
    #==========================================================================
    # Increasing Diversity - Javier Santos
    javier_model_base3 = Extended_JavierSantos()
    model_z_3 = javier_model_base3.z_model
    model_z_3.load_state_dict(torch.load("./Trained_Models/NN_Trainning_9_September_2026_06-38PM_Job28309/model_LowerValidationLoss.pth", map_location=torch.device(device), weights_only=True))
    model_z_3.bin_input = False
    models_dict["Javier Santos (D0 N4)"] = model_z_3
    
    javier_model_base3 = Extended_JavierSantos()
    model_z_3 = javier_model_base3.z_model
    model_z_3.load_state_dict(torch.load("./Trained_Models/NN_Trainning_9_September_2026_01-53PM_Job28306/model_LowerValidationLoss.pth", map_location=torch.device(device), weights_only=True))
    model_z_3.bin_input = False
    models_dict["Javier Santos (D1 N4)"] = model_z_3
    
    javier_model_base3 = Extended_JavierSantos()
    model_z_3 = javier_model_base3.z_model
    model_z_3.load_state_dict(torch.load("./Trained_Models/NN_Trainning_10_September_2026_12-32PM_Job28376/model_LowerValidationLoss.pth", map_location=torch.device(device), weights_only=True))
    model_z_3.bin_input = False
    models_dict["Javier Santos (D2 N4)"] = model_z_3
    
    javier_model_base3 = Extended_JavierSantos()
    model_z_3 = javier_model_base3.z_model
    model_z_3.load_state_dict(torch.load("./Trained_Models/NN_Trainning_10_September_2026_12-33PM_Job28377/model_LowerValidationLoss.pth", map_location=torch.device(device), weights_only=True))
    model_z_3.bin_input = False
    models_dict["Javier Santos (D3 N4)"] = model_z_3
    #==========================================================================
    
    
    #==========================================================================
    # Increasing Diversity - Danny Ko
    danny_model_base3 = Extended_DannyKo()
    model_z_3 = danny_model_base3.z_model
    model_z_3.load_state_dict(torch.load("./Trained_Models/NN_Trainning_9_September_2026_02-04PM_Job28308/model_LowerValidationLoss.pth", map_location=torch.device(device), weights_only=True))
    model_z_3.bin_input = True
    models_dict["Ko et al (D0 N4)"] = model_z_3
    
    danny_model_base3 = Extended_DannyKo()
    model_z_3 = danny_model_base3.z_model
    model_z_3.load_state_dict(torch.load("./Trained_Models/NN_Trainning_9_September_2026_01-49PM_Job28304/model_LowerValidationLoss.pth", map_location=torch.device(device), weights_only=True))
    model_z_3.bin_input = True
    models_dict["Ko et al (D1 N4)"] = model_z_3
    
    danny_model_base3 = Extended_DannyKo()
    model_z_3 = danny_model_base3.z_model
    model_z_3.load_state_dict(torch.load("./Trained_Models/NN_Trainning_9_September_2026_01-51PM_Job28305/model_LowerValidationLoss.pth", map_location=torch.device(device), weights_only=True))
    model_z_3.bin_input = True
    models_dict["Ko et al (D2 N4)"] = model_z_3
    
    danny_model_base3 = Extended_DannyKo()
    model_z_3 = danny_model_base3.z_model
    model_z_3.load_state_dict(torch.load("./Trained_Models/NN_Trainning_26_August_2026_03-45PM_Job27376/model_LowerValidationLoss.pth", map_location=torch.device(device), weights_only=True))
    model_z_3.bin_input = True
    models_dict["Ko et al (D3 N4)"] = model_z_3
    #==========================================================================
    
    
    #==========================================================================
    # Increasing number of samples but keeping diversity - Danny Ko
    danny_model_base3 = Extended_DannyKo()
    model_z_3 = danny_model_base3.z_model
    model_z_3.load_state_dict(torch.load("./Trained_Models/NN_Trainning_11_September_2026_03-56PM_Job28438/model_LowerValidationLoss.pth", map_location=torch.device(device), weights_only=True))
    model_z_3.bin_input = True
    models_dict["Ko et al (D3 N0)"] = model_z_3
    
    danny_model_base3 = Extended_DannyKo() # 0.125
    model_z_3 = danny_model_base3.z_model
    model_z_3.load_state_dict(torch.load("./Trained_Models/NN_Trainning_11_September_2026_03-55PM_Job28437/model_LowerValidationLoss.pth", map_location=torch.device(device), weights_only=True))
    model_z_3.bin_input = True
    models_dict["Ko et al (D3 N1)"] = model_z_3
    
    danny_model_base3 = Extended_DannyKo()
    model_z_3 = danny_model_base3.z_model
    model_z_3.load_state_dict(torch.load("./Trained_Models/NN_Trainning_11_September_2026_03-53PM_Job28436/model_LowerValidationLoss.pth", map_location=torch.device(device), weights_only=True))
    model_z_3.bin_input = True
    models_dict["Ko et al (D3 N2)"] = model_z_3
    
    danny_model_base3 = Extended_DannyKo()
    model_z_3 = danny_model_base3.z_model
    model_z_3.load_state_dict(torch.load("./Trained_Models/NN_Trainning_11_September_2026_03-53PM_Job28435/model_LowerValidationLoss.pth", map_location=torch.device(device), weights_only=True))
    model_z_3.bin_input = True
    models_dict["Ko et al (D3 N3)"] = model_z_3
    #==========================================================================
    
elif component == 2:
    danny_model_base3 = Extended_DannyKo()
    model_x_3 = danny_model_base3.x_model
    model_x_3.load_state_dict(torch.load("./Trained_Models/NN_Trainning_26_August_2026_06-21PM_Job27380/model_LowerValidationLoss.pth", map_location=torch.device(device), weights_only=True))
    model_x_3.bin_input = True
    models_dict["Ko et al (D3)"] = model_x_3
   
elif component == 3:
    danny_model_base3 = Extended_DannyKo()
    model_p_3 = danny_model_base3.p_model
    model_p_3.load_state_dict(torch.load("./Trained_Models/NN_Trainning_26_August_2026_03-47PM_Job27377/model_LowerValidationLoss.pth", map_location=torch.device(device), weights_only=True))
    model_p_3.bin_input = True
    models_dict["Ko et al (D3)"] = model_p_3
    
elif component == 5:
    print("Initializing Neural Network Models...")
    danny_model = Extended_DannyKo() 
    model_full_z_name = "./Trained_Models/NN_Trainning_26_August_2026_03-45PM_Job27376/model_LowerValidationLoss.pth"
    model_full_x_name = "./Trained_Models/NN_Trainning_26_August_2026_06-21PM_Job27380/model_LowerValidationLoss.pth"
    model_full_p_name = "./Trained_Models/NN_Trainning_26_August_2026_03-47PM_Job27377/model_LowerValidationLoss.pth"

    model = SubModels_Composition( 
        main_model=danny_model,  
        z_name=model_full_z_name, 
        x_name=model_full_x_name,  
        p_name=model_full_p_name,  
        device=device,  
        is_eval=True 
    ) 
    models_dict["Ko et al (D3)"] = model
    
else:
    raise ValueError(f"Component {component} is not configured in the main block.")

print("Models initialized successfully. Starting evaluation...")

# Run evaluation directly saving Dataset name and Sample Index + metrics
output_files = generate_metrics_per_model(
    datasets_dict=datasets,
    models_dict=models_dict,
    component=component,
    batch_size=batch_size,
    N_samples=N_samples,
    output_dir=output_results_dir
)

print("\nExtraction complete! Generated files:")
for m_name, path in output_files.items():
    print(f"- {m_name}: {path}")