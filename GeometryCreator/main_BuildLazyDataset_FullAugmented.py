import os
import re
from typing import List, Tuple
import torch
import numpy as np
import pyvista as pv
from scipy.ndimage import distance_transform_edt
import h5py
from numpy.random import default_rng
import utils
import matplotlib.pyplot as plt
# -------------------------------------------------------------------
# 1) Helpers and Augmentation Functions
# -------------------------------------------------------------------

def list_sample_dirs(base_dir: str, sample_dir_pattern: str) -> List[str]:
    pattern = re.compile(sample_dir_pattern)
    samples: List[Tuple[int, str]] = []
    for name in os.listdir(base_dir):
        full_path = os.path.join(base_dir, name)
        if not os.path.isdir(full_path): continue
        m = pattern.match(name)
        if m:
            num_part = int(m.group(1))
            samples.append((num_part, name))
    samples.sort(key=lambda t: t[0])
    return [name for _, name in samples]

def read_raw_volume(raw_path: str, shape: Tuple[int, int, int], dtype: np.dtype, order: str = "C") -> np.ndarray:
    flat = np.fromfile(raw_path, dtype=dtype)
    return flat.reshape(shape, order=order)

def get_latest_vis_summary_path(sample_dir: str) -> str:
    vis_pattern = re.compile(r"^vis(\d+)$")
    vis_candidates: List[Tuple[int, str]] = []
    for name in os.listdir(sample_dir):
        full_path = os.path.join(sample_dir, name)
        if os.path.isdir(full_path):
            m = vis_pattern.match(name)
            if m: vis_candidates.append((int(m.group(1)), full_path))
    if not vis_candidates: raise RuntimeError(f"No 'visY' in: {sample_dir}")
    vis_candidates.sort(key=lambda t: t[0])
    return os.path.join(vis_candidates[-1][1], "summary.pvti")

def read_summary_pvti(summary_path: str) -> pv.DataSet:
    return pv.read(summary_path)

# --- Danny Ko's Augmentation Logic ---


def mirror_y_augmentation(solid, uz, uy, ux, pr, shift):
   
    # Aux data
    aux_so        = np.zeros_like(solid)
    aux_uz        = np.zeros_like(uz)
    aux_uy        = np.zeros_like(uy)
    aux_ux        = np.zeros_like(ux)
    aux_pr        = np.zeros_like(pr)

    # --- Shift True Y (axis=1) ---
    if shift < 0:
    
        aux_so[:, :shift, :]          = solid[:, -1*shift:, :]    
        aux_uz[:, :shift, :]          = uz[:, -1*shift:, :]
        aux_uy[:, :shift, :]          = uy[:, -1*shift:, :]
        aux_ux[:, :shift, :]          = ux[:, -1*shift:, :]
        aux_pr[:, :shift, :]          = pr[:, -1*shift:, :]
        
        aux_so[:, shift:, :]          = np.flip(solid, axis=1) [:,:-1*shift,:]
        aux_ux[:, shift:, :]          = np.flip(ux,    axis=1 )[:,:-1*shift,:] # Ux
        aux_uy[:, shift:, :]          = np.flip(-1*uy, axis=1 )[:,:-1*shift,:] # Uy
        aux_uz[:, shift:, :]          = np.flip(uz,    axis=1 )[:,:-1*shift,:] # Uz
        aux_pr[:, shift:, :]          = np.flip(pr,    axis=1 )[:,:-1*shift,:] # Pr
        
    elif shift > 0:
        
        aux_so[:, :shift, :]          = np.flip(solid, axis=1) [:,-1*shift:,:] # 
        aux_ux[:, :shift, :]          = np.flip(ux,    axis=1 )[:,-1*shift:,:] # Ux
        aux_uy[:, :shift, :]          = np.flip(-1*uy, axis=1 )[:,-1*shift:,:] # Uy
        aux_uz[:, :shift, :]          = np.flip(uz,    axis=1 )[:,-1*shift:,:] # Uz
        aux_pr[:, :shift, :]          = np.flip(pr,    axis=1 )[:,-1*shift:,:] # Pr
        
        aux_so[:, shift:, :]          = solid[:, :-1*shift, :]    
        aux_uz[:, shift:, :]          = uz   [:, :-1*shift, :]
        aux_uy[:, shift:, :]          = uy   [:, :-1*shift, :]
        aux_ux[:, shift:, :]          = ux   [:, :-1*shift, :]
        aux_pr[:, shift:, :]          = pr   [:, :-1*shift, :]
        
    return aux_so, aux_uz, aux_uy, aux_ux, aux_pr
    
def mirror_x_augmentation(solid, uz, uy, ux, pr, shift):
   
    # Aux data
    aux_so        = np.zeros_like(solid)
    aux_uz        = np.zeros_like(uz)
    aux_uy        = np.zeros_like(uy)
    aux_ux        = np.zeros_like(ux)
    aux_pr        = np.zeros_like(pr)
    
    if shift < 0:
    
        aux_so[:, :, :shift]          = solid[:, :, -1*shift:]    
        aux_uz[:, :, :shift]          = uz[:, :, -1*shift:]
        aux_uy[:, :, :shift]          = uy[:, :, -1*shift:]
        aux_ux[:, :, :shift]          = ux[:, :, -1*shift:]
        aux_pr[:, :, :shift]          = pr[:, :, -1*shift:]
        
        aux_so[:, :, shift:]          = np.flip(solid, axis=2) [:,:-1*shift]
        aux_ux[:, :, shift:]          = np.flip(-1*ux, axis=2) [:,:-1*shift] # Ux
        aux_uy[:, :, shift:]          = np.flip(uy,    axis=2) [:,:-1*shift] # Uy
        aux_uz[:, :, shift:]          = np.flip(uz,    axis=2) [:,:-1*shift] # Uz
        aux_pr[:, :, shift:]          = np.flip(pr,    axis=2) [:,:-1*shift] # Pr
        
    elif shift > 0:
        
        aux_so[:, :, :shift]          = np.flip(solid, axis=2) [:,-1*shift:]
        aux_ux[:, :, :shift]          = np.flip(-1*ux, axis=2) [:,-1*shift:] # Ux
        aux_uy[:, :, :shift]          = np.flip(uy,    axis=2) [:,-1*shift:] # Uy
        aux_uz[:, :, :shift]          = np.flip(uz,    axis=2) [:,-1*shift:] # Uz
        aux_pr[:, :, :shift]          = np.flip(pr,    axis=2) [:,-1*shift:] # Pr
        
        aux_so[:, :, shift:]          = solid[:, :, :-1*shift]   
        aux_uz[:, :, shift:]          = uz   [:, :, :-1*shift]
        aux_uy[:, :, shift:]          = uy   [:, :, :-1*shift]
        aux_ux[:, :, shift:]          = ux   [:, :, :-1*shift]
        aux_pr[:, :, shift:]          = pr   [:, :, :-1*shift]

    return aux_so, aux_uz, aux_uy, aux_ux, aux_pr

def flip_x_augmentation(solid, uz, uy, ux, pr):
    
    aux_so = np.flip(solid, axis=2)
    aux_ux = np.flip(-1*ux, axis=2)
    aux_uy = np.flip(uy,    axis=2)
    aux_uz = np.flip(uz,    axis=2)
    aux_pr = np.flip(pr,    axis=2)
    
    return aux_so, aux_uz, aux_uy, aux_ux, aux_pr


def flip_y_augmentation(solid, uz, uy, ux, pr):
    
    aux_so = np.flip(solid, axis=1)
    aux_ux = np.flip(ux,    axis=1)
    aux_uy = np.flip(-1*uy, axis=1)
    aux_uz = np.flip(uz,    axis=1)
    aux_pr = np.flip(pr,    axis=1)
    
    return aux_so, aux_uz, aux_uy, aux_ux, aux_pr

def rotate_z_augmentation(solid, uz, uy, ux, pr, seed):
    # Change signals 
    if seed > 0:
        k_val   = 1
        base_ux = -1 * uy
        base_uy = ux
        
    elif seed < 0:
        k_val   = -1
        base_ux = uy
        base_uy = -1 * ux
        
    else:
        return solid, uz, uy, ux, pr
    
    # Attributes which the signal are not influencied by rotation
    aux_so = np.rot90(solid,    k=k_val, axes=(1, 2))
    aux_pr = np.rot90(pr,       k=k_val, axes=(1, 2))
    aux_uz = np.rot90(uz,       k=k_val, axes=(1, 2)) 
    
    # Attributes which the signal are influencied by rotation
    aux_ux = np.rot90(base_ux,  k=k_val, axes=(1, 2))
    aux_uy = np.rot90(base_uy,  k=k_val, axes=(1, 2))

    return aux_so, aux_uz, aux_uy, aux_ux, aux_pr

# -------------------------------------------------------------------
# 2) Main Builder with HDF5 and Augmentation
# -------------------------------------------------------------------

output_path         = "../NN_Datasets/Train_Danny_120_120_120_Pressure_Aug.h5"
simulations_folder  = "./Train_Danny_120_120_120_Pressure/"
sample_dir_pattern  = r"^domain_(\d+)$"
raw_name            = "domain.raw"
raw_shape           = (120, 120, 120)
raw_dtype           = np.uint8

# Augmentation Parameters
augment             = True
augGen_seed         = 10
aug_iter            = 30
shift_range         = 2
flip_range          = 5
# Normalization Parameters
norm_cte            = 0.2
tau                 = 1.5
Re                  = 0.1

base_dir            = os.path.join(os.getcwd(), simulations_folder)
sample_dirs         = list_sample_dirs(base_dir, sample_dir_pattern)
rnd_num_gen         = default_rng(augGen_seed)

min_values  = []
max_values  = []
mean_values = []

z_values = np.array([])
y_values = np.array([])
x_values = np.array([])

output_dir = os.path.dirname(output_path)
if output_dir: os.makedirs(output_dir, exist_ok=True)

with h5py.File(output_path, "w") as f:
    D, H, W = raw_shape
    max_points = int(D * H * W) # Adjust if you want a lower fraction

    # Dataset Initialization (expandable)
    vel_x_ds = f.create_dataset("vel_x", (0, max_points), maxshape=(None, max_points), dtype="float32", chunks=(1, max_points))
    vel_y_ds = f.create_dataset("vel_y", (0, max_points), maxshape=(None, max_points), dtype="float32", chunks=(1, max_points))
    vel_z_ds = f.create_dataset("vel_z", (0, max_points), maxshape=(None, max_points), dtype="float32", chunks=(1, max_points))
    press_ds = f.create_dataset("press", (0, max_points), maxshape=(None, max_points), dtype="float32", chunks=(1, max_points))
    coorX_ds = f.create_dataset("coorX", (0, max_points), maxshape=(None, max_points), dtype="uint8", chunks=(1, max_points))
    coorY_ds = f.create_dataset("coorY", (0, max_points), maxshape=(None, max_points), dtype="uint8", chunks=(1, max_points))
    coorZ_ds = f.create_dataset("coorZ", (0, max_points), maxshape=(None, max_points), dtype="uint8", chunks=(1, max_points))
    edt_ds   = f.create_dataset("edt",   (0, max_points), maxshape=(None, max_points), dtype="float32", chunks=(1, max_points))
    n_valid_ds = f.create_dataset("n_valid", (0,), maxshape=(None,), dtype="int64")
    sample_names_ds = f.create_dataset("sample_names", (0,), maxshape=(None,), dtype=h5py.string_dtype())

    global_idx = 0

    for sample_name in sample_dirs:
        sample_dir = os.path.join(base_dir, sample_name)
        raw_path   = os.path.join(sample_dir, raw_name)
        
        try:
            f.attrs["raw_shape"]   = raw_shape
            f.attrs["vel_dtype"]   = "float32"
            f.attrs["coorX_dtype"] = "uint8"
            f.attrs["coorY_dtype"] = "uint8"
            f.attrs["coorZ_dtype"] = "uint8"
            f.attrs["edt_dtype"]   = "float32"
            f.attrs["max_points"]  = max_points
            
            # 1. Load Original Data
            vol_orig        = read_raw_volume(raw_path, raw_shape, raw_dtype)
            summary_path    = get_latest_vis_summary_path(sample_dir)
            mesh            = read_summary_pvti(summary_path)
            
            vx_orig = mesh["Velocity_x"].reshape(raw_shape, order="C")
            vy_orig = mesh["Velocity_y"].reshape(raw_shape, order="C")
            vz_orig = mesh["Velocity_z"].reshape(raw_shape, order="C")
            if "Pressure" in mesh.array_names:
                print("Mesh contains pressure data.")
                pr_orig = mesh["Pressure"].reshape(raw_shape, order="C")
            else:
                print("Pressure data not found.")
                pr_orig        = np.zeros_like(vz_orig)
            
            porous_mask_orig = (vol_orig == 1) 

            # Velocity Normalization
            visc    = (tau-0.5)/3
            force   = utils.force_calculation(porous_mask_orig, tau=tau, Re=Re)
            perm_est= (2*0.65*np.max(distance_transform_edt(porous_mask_orig).astype("float32")))**2
            vx_norm = vx_orig*visc / (force*norm_cte*perm_est)
            vy_norm = vy_orig*visc / (force*norm_cte*perm_est)
            vz_norm = vz_orig*visc / (force*norm_cte*perm_est)
            # Pressure Normalization
            delta_p     = utils.pressure_calculation(porous_mask_orig, tau=tau, Re=Re)
            p_mean      = (2+3*delta_p)/6
            delta_p_new = 0.2
            p_mean_new  = 0.15
            pr_norm     = ((pr_orig -p_mean)/delta_p)*delta_p_new + p_mean_new            
            print("Applied force: ", force)
            print("Mins:  ",np.min(np.concatenate([vx_norm, vy_norm, vz_norm])))
            print("Means: ",np.mean(np.concatenate([vx_norm, vy_norm, vz_norm])))
            print("Devs:  ",np.std(np.concatenate([vx_norm, vy_norm, vz_norm])))
            print("Maxs:  ",np.max(np.concatenate([vx_norm, vy_norm, vz_norm])))
            print("-------------------------------------------------------------")            
        
        
            # Generate random augmentation parameters for this sample
            shift_val = D // shift_range  # D=120 -> max shift is 60
            
            # 1. Shifts: Random integers from -shift_val to +shift_val
            rnd_shifts_x = rnd_num_gen.integers(low=-shift_val, high=shift_val, size=aug_iter, endpoint=True)
            rnd_shifts_y = rnd_num_gen.integers(low=-shift_val, high=shift_val, size=aug_iter, endpoint=True)
            
            # 2. Flips: 50% chance to flip (0 = No Flip, 1 = Flip)
            rnd_flips_x = rnd_num_gen.integers(low=0, high=1, size=aug_iter, endpoint=True)
            rnd_flips_y = rnd_num_gen.integers(low=0, high=1, size=aug_iter, endpoint=True)
            
            # 3. Rotations: Choose between -1 (-90 deg), 0 (0 deg), and 1 (+90 deg)
            rnd_rotations = rnd_num_gen.integers(low=-1, high=1, size=aug_iter, endpoint=True)
                        
            print(f"Processing {sample_name} with {aug_iter} augmentations...")
            
            for j in range(aug_iter):
                # Apply Shift X and Y for 
                
                a_vol, a_uz, a_uy, a_ux, a_pr = mirror_x_augmentation( vol_orig, 
                                                                       vz_norm, 
                                                                       vy_norm, 
                                                                       vx_norm, 
                                                                       pr_norm, 
                                                                       rnd_shifts_x[j])
                
                a_vol, a_uz, a_uy, a_ux, a_pr = mirror_y_augmentation( a_vol, 
                                                                       a_uz, 
                                                                       a_uy, 
                                                                       a_ux, 
                                                                       a_pr, 
                                                                       rnd_shifts_y[j])
                
                
                if rnd_flips_x[j] == 1:
                    a_vol, a_uz, a_uy, a_ux, a_pr = flip_x_augmentation(a_vol, 
                                                                          a_uz, 
                                                                          a_uy, 
                                                                          a_ux,
                                                                          a_pr)
                                                                          
                
                if rnd_flips_y[j] == 1:
                    a_vol, a_uz, a_uy, a_ux, a_pr = flip_y_augmentation(a_vol, 
                                                                          a_uz, 
                                                                          a_uy, 
                                                                          a_ux, 
                                                                          a_pr) 
                                                                          
                    
                    
                    a_vol, a_uz, a_uy, a_ux, a_pr = rotate_z_augmentation(a_vol,
                                                                          a_uz,
                                                                          a_uy, 
                                                                          a_ux, 
                                                                          a_pr, 
                                                                          rnd_rotations[j])
                
                
                # 4. Geometry-based calculations (EDT and Mask) on augmented volume
                porous_mask = (a_vol == 1)
                if not np.any(porous_mask): continue

                edt_full = distance_transform_edt(porous_mask).astype("float32")
                coords_k, coords_j, coords_i,  = np.where(porous_mask)
                N_points = coords_k.size

                # 5. Flatten and Pad
                # (Re-using your padding logic)
                vx_row = np.zeros(max_points, dtype="float32")
                vy_row = np.zeros(max_points, dtype="float32")
                vz_row = np.zeros(max_points, dtype="float32")
                pr_row = np.zeros(max_points, dtype="float32")
                cX_row = np.zeros(max_points, dtype="uint8")
                cY_row = np.zeros(max_points, dtype="uint8")
                cZ_row = np.zeros(max_points, dtype="uint8")
                ed_row = np.zeros(max_points, dtype="float32")

                vx_row[:N_points] = a_ux[porous_mask]
                vy_row[:N_points] = a_uy[porous_mask]
                vz_row[:N_points] = a_uz[porous_mask]
                pr_row[:N_points] = a_pr[porous_mask]
                cX_row[:N_points] = coords_i.astype(np.uint8)
                cY_row[:N_points] = coords_j.astype(np.uint8)
                cZ_row[:N_points] = coords_k.astype(np.uint8)
                ed_row[:N_points] = edt_full[porous_mask]

                # 6. Save to HDF5
                for ds, data in zip([vel_x_ds, vel_y_ds, vel_z_ds, press_ds, coorX_ds, coorY_ds, coorZ_ds, edt_ds],
                                    [vx_row,   vy_row,   vz_row,   pr_row,   cX_row,   cY_row,   cZ_row,   ed_row]):
                    ds.resize((global_idx + 1, max_points))
                    ds[global_idx, :] = data
                
                n_valid_ds.resize((global_idx + 1,))
                sample_names_ds.resize((global_idx + 1,))
                n_valid_ds[global_idx] = N_points
                sample_names_ds[global_idx] = f"{sample_name}_aug_{j}"
                
                global_idx += 1

        except Exception as e:
            print(f"[FAIL] {sample_name}: {e}")
    
   
    f.attrs['norm_type']    = "v = v*visc/(force*norm_cte)"
    f.attrs['norm_cte']     = norm_cte
    f.attrs['tau_used']     = tau
    f.attrs['re_used']      = Re

    print(f"Finished. Total augmented samples written: {global_idx}")
    


