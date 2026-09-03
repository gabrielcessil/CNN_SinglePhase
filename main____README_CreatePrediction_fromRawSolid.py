import numpy as np
from scipy.ndimage import distance_transform_edt as edt
import torch
import os
import pyvista as pv

# Adjust these imports according to your folder structure
from Architectures.Unet   import Extended_DannyKo
from Architectures.MSnet  import JavierSantos_Extended
from Architectures.Models import SubModels_Composition
from Utilities import start_handler as sh
from Utilities import velocity_usage as vu

def save_vti(filename, geometry, ux, uy, uz):
    """
    Exports a .vti file so ParaView can read both Density (Geometry) and Velocity.
    ParaView script thresholds Density <= 0.0 to render the solid.
    """
    Nz, Ny, Nx = geometry.shape
    grid = pv.ImageData(dimensions=(Nx, Ny, Nz))
    
    # Map geometry: Solid (-1.0) and Pore (1.0). ParaView thresholds < 0.0 as solid.
    density_array = np.where(geometry, 1.0, -1.0)
    grid.point_data["Density"] = density_array.flatten(order="C")
    
    # Stack velocity vectors
    grid.point_data["Velocity"] = np.column_stack((
        ux.flatten(order="C"), 
        uy.flatten(order="C"), 
        uz.flatten(order="C")
    ))
    
    grid.save(filename)



paths = [
    "./Example_Bentheimer/",    
]

raw_file    = "domain.raw"
shape       = (120, 120, 120)
device      = "cpu"

danny_model = Extended_DannyKo()

# Z- component
model_full_z_name = "./Trained_Models/NN_Trainning_26_August_2026_03-45PM_Job27376/model_LowerValidationLoss.pth"
# X- component
model_full_x_name = "./Trained_Models/NN_Trainning_26_August_2026_06-21PM_Job27380/model_LowerValidationLoss.pth"
# P- component
model_full_p_name = "./Trained_Models/NN_Trainning_26_August_2026_03-47PM_Job27377/model_LowerValidationLoss.pth"

# Concatenation model
concat_model = SubModels_Composition(main_model=danny_model, 
                                     z_name=model_full_z_name,
                                     x_name=model_full_x_name, 
                                     p_name=model_full_p_name, 
                                     device=device, 
                                     is_eval=True)

# Force evaluation mode for deterministic predictions
concat_model.eval()
models = {"Composed Model ": concat_model}

for path in paths:
    for model_name, model in models.items():
        
        # Read Geometry
        geometry     = (np.fromfile(os.path.join(path, raw_file), dtype=np.uint8).reshape(shape) > 0)
        geometry_edt = edt(geometry).astype("float32")
        
        # Convert numpy array (Z,Y,X) to tensor (B=1,C=1, Z,Y,X)
        geometry_edt_tensor = torch.from_numpy(geometry_edt).unsqueeze(0).unsqueeze(0)
        
        # Make prediction
        print(f"Creating prediction with {model_name.strip()}")
        pred = model.predict(geometry_edt_tensor)
        
        # Denormalize predictions
        pred = vu.tensor_denorm(out=pred, inp=geometry_edt_tensor)
        
        # Extract components
        uz = pred[0,0].numpy()
        uy = pred[0,1].numpy()
        ux = pred[0,2].numpy()
        pr = pred[0,3].numpy()
                
        # Sanity Checks
        if not (uz.shape==shape and uy.shape==shape and ux.shape==shape and pr.shape==shape): 
            raise Exception("Prediction doesn't match specified .raw shape.")
        if np.isnan(pred.numpy()).any() or np.isinf(pred.numpy()).any():
            raise ValueError(f"Model {model_name} predicted NaN or Inf values!")
            
        solid_vel_mag = np.sqrt(ux[~geometry]**2 + uy[~geometry]**2 + uz[~geometry]**2)
        if np.any(solid_vel_mag > 1e-6):
            print(f"   [!] WARNING: Predicted velocity inside solid! Forcing to 0.0.")
            ux[~geometry] = 0.0
            uy[~geometry] = 0.0
            uz[~geometry] = 0.0
            
        if np.max(np.abs(uz)) == 0.0 and np.max(np.abs(uy)) == 0.0 and np.max(np.abs(ux)) == 0.0:
            print(f"   [!] WARNING: Predicted a completely ZERO velocity field.")
            
        max_v = np.max(np.sqrt(ux**2 + uy**2 + uz**2))
        if max_v > 0.7:
            print(f"   [!] DANGER: Max velocity is {max_v:.4f}. LBPM may be unstable.")
            
        # Create output directory
        out_dir = os.path.join(path)
        os.makedirs(out_dir, exist_ok=True)
        
        # Calculate and print stats
        pred_perm = vu.permeability_calculation(pred, geometry_edt_tensor, denorm=False)
        print(f"   -> Perm | {float(pred_perm):.6e}")
        print(f"   -> Uz   | max: {uz.max():>13.6e} | mean: {uz.mean():>13.6e} | min: {uz.min():>13.6e}")
        
        # Write VTI for ParaView
        vti_path = os.path.join(out_dir, "output_data.vti")
        save_vti(vti_path, geometry, ux, uy, uz)
        print(f"   -> Saved VTI for ParaView: {vti_path}\n")