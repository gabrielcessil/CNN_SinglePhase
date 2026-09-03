import os
import torch
import numpy as np
import pandas as pd
import porespy as ps
from torch.utils.data import DataLoader, Subset
import quantimpy.minkowski as mk

# Import local architectures
from Architectures.Unet   import Extended_DannyKo
from Architectures.Models import SubModels_Composition

# Import local utilities
from Utilities            import dataset_reader as dr
from Utilities            import error_metrics as em 
from Utilities            import velocity_usage as vu
from Utilities            import model_handler as mh


def generate_comprehensive_sample_metrics(
    datasets_dict: dict, 
    models_dict: dict,    # <-- Now accepts a dictionary of models
    component: int = 5, 
    batch_size: int = 4, 
    N_samples: int = None,
    save_csv_path: str = None
):
    """
    Extracts geometric properties (Porosity, Permeability, Local Thickness, Minkowski Functionals)
    and maps them to error metrics (Bias, Magnitude, Correlation, etc.) sample by sample 
    for MULTIPLE models.
    """
    records = []
    
    print("--- Starting Sample-by-Sample Extraction ---")
    
    # Ensure all models are in evaluation mode
    for model_name, model in models_dict.items():
        if hasattr(model, 'eval'):
            model.eval()

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

                # ---------------------------------------------------------
                # 1. GEOMETRICAL PROPERTIES (Model-Independent)
                # Compute this only ONCE per batch to save time
                # ---------------------------------------------------------
                batch_geom_props = []
                for b in range(B):
                    mask_np = (batch_inputs[b, 0] > 0).cpu().numpy()
                    
                    # 1a. Porosity
                    porosity = np.mean(mask_np)
                    
                    # 1b. Permeability
                    denorm_tar = vu.tensor_denorm(batch_targets[b:b+1], batch_inputs[b:b+1])
                    perm = vu.permeability_calculation(
                        denorm_tar, batch_inputs[b:b+1], tau=1.5, Re=0.1, dens=1.0, denorm=False
                    ).item()
                    
                    # 1c. Local Thickness
                    thick = ps.filters.local_thickness(mask_np)
                    valid_thick = thick[mask_np]
                    if valid_thick.size > 0:
                        min_thick  = valid_thick.min()
                        mean_thick = valid_thick.mean()
                        max_thick  = valid_thick.max()
                    else:
                        min_thick = mean_thick = max_thick = np.nan
                        
                    # 1d. Topology / Pseudo-Minkowski
                    mink = mk.functionals(mask_np)
                    m_vol, m_sa, m_curv, m_euler = mink[0], mink[1], mink[2], mink[3]
                    
                    batch_geom_props.append({
                        "Porosity":                        porosity,
                        "Permeability":                    perm,
                        "Min_Local_Thickness":             min_thick,
                        "Mean_Local_Thickness":            mean_thick,
                        "Max_Local_Thickness":             max_thick,
                        "Minkowski_Volume":                m_vol,
                        "Minkowski_Surface_Area":          m_sa,
                        "Minkowski_Mean_Curvature":        m_curv,
                        "Minkowski_Euler_Characteristic":  m_euler,
                    })

                # ---------------------------------------------------------
                # 2. MODEL PREDICTIONS & ERROR METRICS
                # Loop through all provided models for the same batch
                # ---------------------------------------------------------
                for model_name, model in models_dict.items():
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

                    # ---------------------------------------------------------
                    # 3. RECORD ASSEMBLY
                    # ---------------------------------------------------------
                    for b in range(B):
                        record = {
                            "Model": model_name,  # <-- Added Model column
                            "Dataset": dataset_name,
                            **batch_geom_props[b], # Unpack the precalculated geometry
                        }
                        
                        # Safely append general metrics
                        if b < len(b_metrics): record["Bias Error [%]"] = b_metrics[b]
                        if b < len(m_metrics): record["Magnitude Error [%]"] = m_metrics[b]
                        if b < len(c_metrics): record["Correlation"] = c_metrics[b]
                        
                        # Safely append physics metrics (if applicable)
                        if component is None or component == 5:
                            if b < len(a_metrics): record["Angular Error [Deg]"] = a_metrics[b]
                            if b < len(f_metrics): record["Flux Error"] = f_metrics[b]
                            if b < len(t_metrics): record["Tortuosity Error [%]"] = t_metrics[b]
                            if b < len(d_metrics): record["Divergent Residual [%]"] = d_metrics[b]

                        records.append(record)

    # ==========================================
    # 4. CLEANUP LABELS & EXPORT RESULTS
    # ==========================================
    df_samples = pd.DataFrame(records)
    
    # Clean headers
    df_samples.columns = df_samples.columns.str.replace(' ', '_')
    df_samples.columns = df_samples.columns.str.replace(r'[\[\]\%]', '', regex=True)
    
    if save_csv_path:
        os.makedirs(os.path.dirname(save_csv_path) or ".", exist_ok=True)
        df_samples.to_csv(save_csv_path, index=False)
        print(f"\nTable saved to {save_csv_path}")

    return df_samples


# =======================================================
# MAIN SETUP
# =======================================================
component   = 0
batch_size  = 10
N_samples   = None 
device      = 'cpu'

csv_out_dir = f"./Error_vs_Geometry_bySample_{component}.csv"

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

models = {}

if component == 0:
    # Example setup mapping multiple versions of the model (Etapa 0, Etapa 1, etc.)
    danny_model_base1 = Extended_DannyKo()
    model_z_1 = danny_model_base1.z_model
    model_z_1.load_state_dict(torch.load("./Trained_Models/NN_Trainning_6_July_2026_01-12PM_Job26188/model_LowerValidationLoss.pth", map_location=torch.device(device), weights_only=True))
    model_z_1.bin_input = True
    models_dict["Ko et. al (Etapa 0)"] = model_z_1

    danny_model_base2 = Extended_DannyKo()
    model_z_2 = danny_model_base2.z_model
    model_z_2.load_state_dict(torch.load("./Trained_Models/NN_Trainning_13_July_2026_06-02PM_Job26267/model_LowerValidationLoss.pth", map_location=torch.device(device), weights_only=True))
    model_z_2.bin_input = True
    models_dict["Ko et. al (Etapa 3)"] = model_z_2

elif component == 5:
    danny_model = Extended_DannyKo()
    concat_model = SubModels_Composition(
        main_model=danny_model, 
        z_name="./Trained_Models/NN_Trainning_13_July_2026_06-02PM_Job26267/model_LowerValidationLoss.pth",
        x_name="./Trained_Models/NN_Trainning_15_July_2026_03-59PM_Job26381/model_LowerValidationLoss.pth", 
        p_name="./Trained_Models/NN_Trainning_21_July_2026_05-22PM_Job26505/model_LowerValidationLoss.pth", 
        device=device, 
        is_eval=True
    )
    models_dict["Ko et. al (Etapas 3)"] = concat_model
    
else:
    raise ValueError(f"Component {component} is not configured in the main block.")

print("Models initialized successfully. Starting evaluation...")

# Run the comprehensive metrics extraction passing the dictionary
df = generate_comprehensive_sample_metrics(
    datasets_dict=datasets,
    models_dict=models, # <-- Passed the dictionary here
    component=component,
    batch_size=batch_size,
    N_samples=N_samples,
    save_csv_path=csv_out_dir
)

print("\nExtraction complete! Data sample:")
print(df.head())