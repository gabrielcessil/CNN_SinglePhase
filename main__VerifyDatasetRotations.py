import torch
import numpy as np
import pyvista as pv
import h5py
import os
import warnings
from torch.utils.data import Dataset
from Utilities import dataset_reader as dr

"""
This code intends to verify the rotation augmentation applied in the dataset creations.
The expected behavior is:
    
Baseline: ux, uy

90º:
    uy' = ux
    ux' = -uy

180:
    uy'' = -uy
    ux'' = -ux
    
270º:
    uy''' = -ux
    ux''' = uy
"""
def verify_rotations_and_export(h5_filepaths):
    """
    Receives a list of .h5 dataset paths, loads the first 4 samples,
    checks the rotation logic quantitatively, and saves .vti files for Paraview.
    """
    
    for h5_path in h5_filepaths:
        print(f"\n{'='*50}")
        print(f"Processing Dataset: {h5_path}")
        print(f"{'='*50}")
        
        # Which sample in the dataset
        samp_idx = 10
        
        # 1. Initialize dataset for the first 4 samples (0°, 90°, 180°, 270°)
        dataset = dr.LazyDatasetTorch(
            h5_path=h5_path, 
            list_ids=np.arange(samp_idx*4, samp_idx*4+4), 
            x_dtype=torch.float32, 
            y_dtype=torch.float32
        )
        
        
        samples_X = []
        samples_Y = []
        for i in range(4):
            X, Y = dataset[i]
            samples_X.append(X)
            samples_Y.append(Y)

        # Baseline (0 degrees)
        X0, Y0 = samples_X[0], samples_Y[0] # Eliminate batch dimension
        # Collecting components
        edt_0 = X0[0]
        uz_0, uy_0, ux_0, pr_0 = Y0[0], Y0[1], Y0[2], Y0[3]
        print("\n--- 1. Quantitative Verification ---")
        print(f"{'Sample':<14} | {'Uy+ mean':>14} | {'Uy- mean':>14} | {'Ux+ mean':>14} | {'Ux- mean':>14}")       
        for k in range(4):
            Xk, Yk = samples_X[k], samples_Y[k]
            
            edt_k = Xk[0]
            uz_k, uy_k, ux_k, pr_k = Yk[0], Yk[1], Yk[2], Yk[3]
            
            label = "Baseline (0°)" if k == 0 else f"Rot {k} ({k*90}°)"
            uy_pos = (uy_k[uy_k > 0]).mean().item()
            uy_neg = (uy_k[uy_k < 0]).mean().item()
            ux_pos = (ux_k[ux_k > 0]).mean().item()
            ux_neg = (ux_k[ux_k < 0]).mean().item()
            print(
                f"{label:<14} | "
                f"{uy_pos:>14.6e} | "
                f"{uy_neg:>14.6e} | "
                f"{ux_pos:>14.6e} | "
                f"{ux_neg:>14.6e}"
            )
            
            if k!=0:
                # Calculate mathematically expected spatial rotations (plane Y-X is dims 1 and 2)
                exp_edt = torch.rot90(edt_0, k=k, dims=(1, 2))
                exp_uz  = torch.rot90(uz_0,  k=k, dims=(1, 2))
                exp_pr  = torch.rot90(pr_0,  k=k, dims=(1, 2))
    
                # Calculate mathematically expected vector transformations
                if k == 1:   # 90 degrees
                    exp_ux = torch.rot90(-uy_0, k=k, dims=(1, 2))
                    exp_uy = torch.rot90(ux_0,  k=k, dims=(1, 2))
                elif k == 2: # 180 degrees
                    exp_ux = torch.rot90(-ux_0, k=k, dims=(1, 2))
                    exp_uy = torch.rot90(-uy_0, k=k, dims=(1, 2))
                elif k == 3: # 270 degrees
                    exp_ux = torch.rot90(uy_0,  k=k, dims=(1, 2))
                    exp_uy = torch.rot90(-ux_0, k=k, dims=(1, 2))
    
                # Calculate Max Absolute Errors
                err_edt = (edt_k - exp_edt).abs().max().item()
                err_uz  = (uz_k - exp_uz).abs().max().item()
                err_ux  = (ux_k - exp_ux).abs().max().item()
                err_uy  = (uy_k - exp_uy).abs().max().item()
                err_pr  = (pr_k - exp_pr).abs().max().item()
                
                #print(f"Sample {k} (Rotated {k*90}°):")
                #print(f"  Max Error -> EDT: {err_edt:.1e} | Press: {err_pr:.1e} | Uz: {err_uz:.1e} | Uy: {err_uy:.1e} | Ux: {err_ux:.1e}")
                
                if sum([err_edt, err_uz, err_ux, err_uy, err_pr]) != 0:
                    print("  Augmentation applied uncorrectly.")
                else: print("  Augmentation applied correctly.")

        print("\n--- 2. Exporting to Paraview (.vti) ---")
        export_dir = "paraview_checks"
        os.makedirs(export_dir, exist_ok=True)

        for idx in range(4):
            X, Y = samples_X[idx], samples_Y[idx]
            
            # Convert to Numpy
            edt   = X[0].numpy()
            vel_z = Y[0].numpy()
            vel_y = Y[1].numpy()
            vel_x = Y[2].numpy()
            press = Y[3].numpy()

            D, H, W = edt.shape
            grid = pv.ImageData()
            grid.dimensions = (W, H, D)
            grid.spacing = (1.0, 1.0, 1.0)
            grid.point_data["EDT"] = edt.flatten()
            grid.point_data["Pressure"] = press.flatten()
            grid.point_data["Velocity"] = np.column_stack((
                vel_x.flatten(),
                vel_y.flatten(),
                vel_z.flatten()
            ))
            base_name = os.path.basename(h5_path).replace('.h5', '')
            filename = os.path.join(export_dir, f"{base_name}_sample_rot_{idx*90}.vti")
            grid.save(filename)
            print(f"  Saved: {filename}")



datasets_to_test = [
    "../NN_Datasets_Grad_2/Train_Silveira_SphPore_SAug_DNorm.h5", 
    "../NN_Datasets_Grad_2/Train_Silveira_SphGrain_SAug_DNorm.h5",
    "../NN_Datasets_Grad_2/Train_Oliveira_Bentheimer_SAug_DNorm.h5"
]

verify_rotations_and_export(datasets_to_test)