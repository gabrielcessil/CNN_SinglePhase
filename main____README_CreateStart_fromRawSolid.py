import numpy as np
from scipy.ndimage import distance_transform_edt as edt
import torch
import matplotlib.pyplot as plt

from Architectures.Unet   import Extended_DannyKo
from Architectures.MSnet  import JavierSantos_Extended
from Architectures.Models import SubModels_Composition

from Utilities import start_handler as sh
from Utilities import velocity_usage as vu



paths = [
      "./Example_Bentheimer/"
]
shape = (120,120,120)
raw_file    = "domain.raw"
device      = "cpu"



# ==============================================================================
# LOADING MODELs
# ============================================================================== 
danny_model         = Extended_DannyKo()
# Z- component
model_full_z_name = "./Trained_Models/NN_Trainning_26_August_2026_03-45PM_Job27376/model_LowerValidationLoss.pth"
# X- component
model_full_x_name = "./Trained_Models/NN_Trainning_26_August_2026_06-21PM_Job27380/model_LowerValidationLoss.pth"
# P- component
model_full_p_name = "./Trained_Models/NN_Trainning_26_August_2026_03-47PM_Job27377/model_LowerValidationLoss.pth"

# Concatenation model (no main model)
model             = SubModels_Composition(main_model=danny_model, 
                                          z_name=model_full_z_name,
                                          x_name=model_full_x_name, 
                                          p_name=model_full_p_name, 
                                          device=device, 
                                          is_eval=True)


# ==============================================================================
# MAIN
# ============================================================================== 

    
  
for path in paths: 
            
    geometry     = (np.fromfile(path+raw_file, dtype=np.uint8).reshape(shape)>0)
    geometry_edt = edt(geometry).astype("float32")
    
    
    
    # Convert numpy array (Z,Y,X) to tensor (B=1,C=1, Z,Y,X)
    geometry_edt = torch.from_numpy(geometry_edt).unsqueeze(0).unsqueeze(0)
    
    # Make prediction
    print(f"Creating prediction for {path}{raw_file}:")
    pred    = model.predict(geometry_edt)
    uz      = pred[0,0].numpy()
    uy      = pred[0,1].numpy()
    ux      = pred[0,2].numpy()
    pr      = pred[0,3].numpy()
    
    # Denormalize predictions
    pred    = vu.tensor_denorm(out=pred, inp=geometry_edt)
    
    # Prepare data for start file
    uz      = pred[0,0].numpy()
    uy      = pred[0,1].numpy()
    ux      = pred[0,2].numpy()
    pr      = pred[0,3].numpy()
            
    # Sanity Checks
    #  - Shape Matching
    if not (uz.shape==shape and uy.shape==shape and ux.shape==shape and pr.shape==shape): 
        raise Exception("Prediction dont match specified .raw shape.")
    #  - NaN and Inf presence check
    if np.isnan(pred.numpy()).any() or np.isinf(pred.numpy()).any():
        raise ValueError(f"Model predicted NaN or Inf values!")
    #  - Solid Matching (No-Slip Condition)
    solid_vel_mag =  np.sqrt(ux[~geometry]**2 + uy[~geometry]**2 + uz[~geometry]**2)
    
    #  - LBM Stability Check (Max Velocity)
    max_v   = np.max(np.sqrt(ux**2 + uy**2 + uz**2))
    if max_v > 0.7:
        raise ValueError(f"   Model predicted a max velocity of {max_v:.4f}. LBPM may be unstable due to Mach limit.")
        
    # Write start file 
    print(f"   -> Creating Start.00000 file")
    sh.write_start_raw(
        filename = path+"Start.00000",
        ux=ux, uy=uy, uz=uz, pr=pr
    )
    
    # Write the .db
    print(f"   -> Creating .db file")
    tau         = 1.5
    Re          = 0.1
    Dens        = 1.0
    p_drop      = vu.pressure_calculation(geometry, tau=tau, Re=Re, Dens=Dens)
    sh.write_lbpm_db(
        db_name = path+"start_pressure.db",
        path    = "",
        tau     = tau,
        bc      = 3,
        din     = 1.0,
        dout    = 1.0-3*p_drop,
        nproc   = (1, 1, 1),
        n       = shape,
        N       = shape,
        analysis_interval = 1000,
        tolerance         = 1e-6,
        out_format        = "silo",
        Start             = True
    )
    
    # Rewrite .raw
    geometry.astype(np.uint8).tofile(path+raw_file)
    
    # Use prediction and show primary statistics
    pred_perm = vu.permeability_calculation(pred, geometry_edt, denorm=False)
    perm_val = float(pred_perm)
    print(f"   -> Perm | {perm_val:.6e}")
    print(f"   -> Uz   | max: {uz.max():>13.6e} | mean: {uz.mean():>13.6e} | min: {uz.min():>13.6e}")
    print(f"   -> Uy   | max: {uy.max():>13.6e} | mean: {uy.mean():>13.6e} | min: {uy.min():>13.6e}")
    print(f"   -> Ux   | max: {ux.max():>13.6e} | mean: {ux.mean():>13.6e} | min: {ux.min():>13.6e}")
    print(f"   -> Pr   | max: {pr.max():>13.6e} | mean: {pr.mean():>13.6e} | min: {pr.min():>13.6e}")
    print()
