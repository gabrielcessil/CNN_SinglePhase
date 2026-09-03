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
import augmentations as aug

# -------------------------------------------------------------------
# 1) Helpers
# -------------------------------------------------------------------

def export_to_paraview(filename: str, mask: np.ndarray, vx: np.ndarray, vy: np.ndarray, vz: np.ndarray, pr: np.ndarray):
    """
    Exporta os campos 3D da simulação para um arquivo .vti legível pelo ParaView.
    """
    grid = pv.ImageData()
    grid.dimensions = np.array(mask.shape)
    
    velocity = np.stack((vx, vy, vz), axis=-1)
    
    grid.point_data["Uz"] =  vz.flatten(order="C")
    grid.point_data["Uy"] =  vy.flatten(order="C")
    grid.point_data["Ux"] =  vx.flatten(order="C")
    grid.point_data["Pressure"] = pr.flatten(order="C")
    grid.point_data["Porous_Mask"] = mask.flatten(order="C").astype(np.uint8)
    grid.save(filename)
    
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

def read_latest_vti_path(sample_dir: str) -> pv.DataSet:
    vti_pattern = re.compile(r"^output_(\d+)\.vti$")
    candidates: List[Tuple[int, str]] = []
    for name in os.listdir(sample_dir):
        if os.path.isfile(os.path.join(sample_dir, name)):
            m = vti_pattern.match(name)
            if m: candidates.append((int(m.group(1)), name))
    if not candidates: raise RuntimeError(f"No 'output_XXXXXXX.vti' files found in: {sample_dir}")
    candidates.sort(key=lambda t: t[0])
    summary_path = os.path.join(sample_dir, candidates[-1][1])
    return pv.read(summary_path)

def find_directional_local_maxima(dist_transform):
    """
    Finds local maxima in a 3D distance transform array.
    """
    dt = np.asarray(dist_transform)
    
    max_z = np.zeros_like(dt, dtype=bool)  # Axis 0
    max_y = np.zeros_like(dt, dtype=bool)  # Axis 1
    max_x = np.zeros_like(dt, dtype=bool)  # Axis 2
    
    max_z[1:-1, :, :] = (dt[1:-1, :, :] > dt[:-2, :, :]) & (dt[1:-1, :, :] > dt[2:, :, :])
    max_y[:, 1:-1, :] = (dt[:, 1:-1, :] > dt[:, :-2, :]) & (dt[:, 1:-1, :] > dt[:, 2:, :])
    max_x[:, :, 1:-1] = (dt[:, :, 1:-1] > dt[:, :, :-2]) & (dt[:, :, 1:-1] > dt[:, :, 2:])
    
    max_count = max_z.astype(np.int8) + max_y.astype(np.int8) + max_x.astype(np.int8)
    local_maxima_mask = (max_count >= 2) & (dt > 0)
    
    return local_maxima_mask    
    
# -------------------------------------------------------------------
# 2) Main Builder with HDF5 and Augmentation
# -------------------------------------------------------------------

infos = {
    "/home/gabriel/remote/hal/dissertacao/Simulations/Valid_Danny_SphPore_120_120_120/":
    "../../NN_Datasets/Valid_Danny_SphPore_SAug_SNorm.h5",
    
    # Add more datasets here to batch process them:
    # "/home/gabriel/remote/hal/dissertacao/Simulations/Train_Danny_SphPore_120_120_120/":
    # "../../NN_Datasets/Train_Danny_SphPore_SAug_SNorm.h5",
}

sample_dir_pattern  = r"^Sample_(\d+)$"
raw_name            = "mod_domain.raw"
raw_shape           = (120, 120, 120)
raw_dtype           = np.uint8
source              = "lbpm" # One of "lbpm", "gradlbm"
saved_fraction      = 1.0

# Normalization Parameters
tau                 = 1.5
Re                  = 0.1

for simulations_folder, output_path in infos.items():
    print(f"\n\n{'='*70}\nCREATING: {output_path}\n{'='*70}")
    
    base_dir   = os.path.join(os.getcwd(), simulations_folder)
    sample_dirs= list_sample_dirs(base_dir, sample_dir_pattern)
    
    output_dir = os.path.dirname(output_path)
    if output_dir: os.makedirs(output_dir, exist_ok=True)

    with h5py.File(output_path, "w") as f:
        
        # Auxiliary information
        f.attrs['norm']     = "v = v*visc/(force*k_eq)"
        f.attrs['k_eq']     = "R_max ^2 / 5"
        f.attrs['tau_used'] = tau
        f.attrs['re_used']  = Re
        
        D, H, W = raw_shape
        max_points = int(D * H * W * saved_fraction)

        # Dataset Initialization
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
                vol_orig = read_raw_volume(raw_path, raw_shape, raw_dtype)
                porous_mask = (vol_orig == 1) 
                
                if source == "lbpm":
                    summary_path = get_latest_vis_summary_path(sample_dir)
                    mesh         = read_summary_pvti(summary_path)
                    
                    vx_orig = mesh["Velocity_x"].reshape(raw_shape, order="C")
                    vy_orig = mesh["Velocity_y"].reshape(raw_shape, order="C")
                    vz_orig = mesh["Velocity_z"].reshape(raw_shape, order="C")
                    
                    if "Pressure" in mesh.array_names:
                        pr_orig = mesh["Pressure"].reshape(raw_shape, order="C")
                    else:
                        pr_orig = np.zeros_like(vz_orig)
                        
                elif source == "gradlbm":
                    mesh = read_latest_vti_path(sample_dir)
                    velocity_data = mesh["Velocity"]
                    
                    vx_orig = velocity_data[:, 0].reshape(raw_shape, order="C")
                    vy_orig = velocity_data[:, 1].reshape(raw_shape, order="C")
                    vz_orig = velocity_data[:, 2].reshape(raw_shape, order="C")
                    
                    if "Density" in mesh.array_names:
                        pr_orig = mesh["Density"].reshape(raw_shape, order="C") / 3
                    else:
                        pr_orig = np.zeros_like(vz_orig)

                # 2. Velocity and Pressure Normalization (Silveira)
                vx_norm = utils.silveira_normalization_vel(vx_orig, porous_mask)
                vy_norm = utils.silveira_normalization_vel(vy_orig, porous_mask)
                vz_norm = utils.silveira_normalization_vel(vz_orig, porous_mask)
                pr_norm = utils.silveira_normalization_pres(pr_orig, porous_mask)
                
                # Print perfectly aligned statistics
                print(f"\n{'Field':<7} | {'Min':>15} | {'Mean':>15} | {'Max':>15} | {'Std Dev':>15}")
                print("-" * 77)
                for name, field in [("vx", vx_norm), ("vy", vy_norm), ("vz", vz_norm), ("press", pr_norm)]:
                    data = field[porous_mask]
                    if data.size > 0:
                        print(
                            f"{name:<7} | "
                            f"{np.min(data):>15.6e} | "
                            f"{np.mean(data):>15.6e} | "
                            f"{np.max(data):>15.6e} | "
                            f"{np.std(data):>15.6e}"
                        )
                print("-" * 77)            
                
                print(f"Processing {sample_name} with rotations...")
                
                # 3. Rotate the sample 4 times cumulatively
                for rot in range(4):
                    porous_mask, vz_norm, vy_norm, vx_norm, pr_norm = aug.rotate_z_augmentation(
                        porous_mask,
                        vz_norm,
                        vy_norm, 
                        vx_norm, 
                        pr_norm, 
                        direc=1
                    )
                    
                    # Optional: Export debug visualization
                    export_to_paraview(
                        filename=f"debug_{sample_name}_rot_{rot}.vti", 
                        mask=porous_mask, 
                        vx=vx_norm, 
                        vy=vy_norm,
                        vz=vz_norm, 
                        pr=pr_norm
                    )
                    
                    # 4. Geometry-based calculations (EDT and Mask) on augmented volume
                    if not np.any(porous_mask): continue

                    edt_full = distance_transform_edt(porous_mask).astype("float32")
                    coords_k, coords_j, coords_i,  = np.where(porous_mask)
                    N_points = coords_k.size

                    # 5. Flatten and Pad
                    vx_row = np.zeros(max_points, dtype="float32")
                    vy_row = np.zeros(max_points, dtype="float32")
                    vz_row = np.zeros(max_points, dtype="float32")
                    pr_row = np.zeros(max_points, dtype="float32")
                    cX_row = np.zeros(max_points, dtype="uint8")
                    cY_row = np.zeros(max_points, dtype="uint8")
                    cZ_row = np.zeros(max_points, dtype="uint8")
                    ed_row = np.zeros(max_points, dtype="float32")

                    vx_row[:N_points] = vx_norm[porous_mask]
                    vy_row[:N_points] = vy_norm[porous_mask]
                    vz_row[:N_points] = vz_norm[porous_mask]
                    pr_row[:N_points] = pr_norm[porous_mask]
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
                    sample_names_ds[global_idx] = f"{sample_name}_rot_{rot}"
                    
                    global_idx += 1

            except Exception as e:
                print(f"[FAIL] {sample_name}: {e}")
    
        print(f"Finished writing {output_path}. Total augmented samples written: {global_idx}")