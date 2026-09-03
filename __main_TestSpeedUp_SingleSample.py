 
import os
import glob
import shutil
import subprocess
import re
import csv
import numpy as np
from pathlib import Path
from scipy.ndimage import distance_transform_edt as edt
import torch
import matplotlib.pyplot as plt

# Custom Utilities
from Architectures.Unet import Extended_DannyKo
from Architectures.Models import SubModels_Composition
from Utilities import start_handler as sh
from Utilities import velocity_usage as vu

# ==============================================================================
# CONFIGURATION & PATHS
# ==============================================================================
datasets = [
    "./Example_Bentheimer_2/",
]

raw_file = "domain.raw"
shape   = (120, 120, 120)
device  = "cpu"

# Exact MPI and LBPM Binary Paths from your scripts
MPI_PATH = "/home/gabriel/Desktop/LBPM_Install/mpi/bin/mpirun"
LBPM_EXEC = "/home/gabriel/Desktop/LBPM_Install/LBPM_dir/tests/lbpm_single_phase"

# ==============================================================================
# MODEL INITIALIZATION
# ==============================================================================
danny_model = Extended_DannyKo()

model_full_z_name = "./Trained_Models/NN_Trainning_13_July_2026_06-02PM_Job26267/model_LowerValidationLoss.pth"
model_full_x_name = "./Trained_Models/NN_Trainning_15_July_2026_03-59PM_Job26381/model_LowerValidationLoss.pth"
model_full_p_name = "./Trained_Models/NN_Trainning_21_July_2026_05-22PM_Job26505/model_LowerValidationLoss.pth"

concat_model = SubModels_Composition(
    main_model=danny_model, 
    z_name=model_full_z_name,
    x_name=model_full_x_name, 
    p_name=model_full_p_name, 
    device=device, 
    is_eval=True
)

# ==============================================================================
# HELPER FUNCTIONS (Refined to match your vis<number> folder structure)
# ==============================================================================
def get_max_timestep_from_vis(folder_path):
    """
    Finds the highest timestep from subdirectories named 'vis<number>'
    as structured by LBPM output logs.
    """
    p = Path(folder_path)
    if not p.exists():
        return 0

    vis_pattern = re.compile(r'^vis(\d+)$')
    max_ts = 0
    
    for item in p.iterdir():
        if item.is_dir():
            match = vis_pattern.match(item.name)
            if match:
                ts = int(match.group(1))
                if ts > max_ts:
                    max_ts = ts
                    
    return max_ts

def cleanup_vis_folders(base_path):
    """Removes old vis folders before running a new simulation."""
    p = Path(base_path)
    if not p.exists():
        return
    vis_pattern = re.compile(r'^vis(\d+)$')
    for folder in p.iterdir():
        if folder.is_dir() and vis_pattern.match(folder.name):
            shutil.rmtree(folder)

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
results = {}

for dataset_path in datasets:
    sample_folders = glob.glob(os.path.join(dataset_path, "Sample_*"))
    
    for sample_path in sample_folders:
        sample_name = os.path.basename(sample_path)
        print(f"\nProcessing: {dataset_path} -> {sample_name}")
        
        # 1. Target Directory Setup
        started_dir = os.path.join(sample_path, "Started_Sim")
        os.makedirs(started_dir, exist_ok=True)
        cleanup_vis_folders(started_dir)
        
        # 2. Copy Domain
        source_raw = os.path.join(sample_path, raw_file)
        dest_raw = os.path.join(started_dir, raw_file)
        shutil.copy(source_raw, dest_raw)
        
        # 3. Geometry & Prediction
        geometry = (np.fromfile(dest_raw, dtype=np.uint8).reshape(shape) > 0)
        geometry_edt = edt(geometry).astype("float32")
        geometry_edt = torch.from_numpy(geometry_edt).unsqueeze(0).unsqueeze(0)
        
        pred = concat_model.predict(geometry_edt)
        pred = vu.tensor_denorm(out=pred, inp=geometry_edt)
        
        uz = pred[0,0].numpy().astype(np.float64)
        uy = pred[0,1].numpy().astype(np.float64)
        ux = pred[0,2].numpy().astype(np.float64)
        pr = pred[0,3].numpy().astype(np.float64)
    
            
        # Write Start.00000.raw
        sh.write_start_raw(
            filename=os.path.join(started_dir, "Start.00000"),
            ux=ux, uy=uy, uz=uz, pr=pr
        )
        
        # Write Database Configuration
        p_drop = vu.pressure_calculation(geometry, tau=1.5, Re=0.1, Dens=1.0)
        db_filename = "start_pressure.db"
        sh.write_lbpm_db(
            db_name = os.path.join(started_dir, db_filename),
            path    = "",
            tau     = 1.5,
            bc      = 3,
            din     = 1.0,
            dout    = 1.0 - 3*p_drop,
            nproc   = (1, 1, 1),
            n       = shape,
            N       = shape,
            analysis_interval = 1000,
            tolerance         = 1e-6,
            out_format        = "silo",
            Start             = True
        )
        
        # 4. Run LBPM via Subprocess using exact parameters
        print("   -> Launching LBPM via subprocess...")
        try:
            subprocess.run(
                [
                    MPI_PATH, 
                    "-np", "1", 
                    LBPM_EXEC, 
                    db_filename, 
                    "--init", "1"  # Equilibrium initialization flag
                ],
                cwd=started_dir,
                check=True,
                capture_output=True,
                text=True
            )
        except subprocess.CalledProcessError as e:
            print(f"   [!] Error executing LBPM for {sample_name}:\n{e.stderr}")
            continue
            
        # 5. Extract Timesteps from vis<number> directories
        ts_original = get_max_timestep_from_vis(sample_path)
        ts_initiated = get_max_timestep_from_vis(started_dir)
        
        print(f"   -> Max Timesteps: Original ({ts_original}) vs NN-Initiated ({ts_initiated})")
        
        dict_key = f"{os.path.basename(os.path.normpath(dataset_path))}_{sample_name}"
        results[dict_key] = (ts_original, ts_initiated)

# ==============================================================================
# SAVE RESULTS & PLOT
# ==============================================================================
csv_filename = "simulation_convergence_comparison.csv"
with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Sample_ID", "Timesteps_Original", "Timesteps_Initiated"])
    for key, (orig, init) in results.items():
        writer.writerow([key, orig, init])

if results:
    orig_vals = [v[0] for v in results.values()]
    init_vals = [v[1] for v in results.values()]
    
    plt.figure(figsize=(7, 7), dpi=300)
    plt.scatter(orig_vals, init_vals, color='#16A085', edgecolor='black', alpha=0.8, s=50)
    
    max_val = max(max(orig_vals), max(init_vals)) if orig_vals else 1000
    plt.plot([0, max_val], [0, max_val], 'r--', label='y = x (No Acceleration)')
    
    plt.xlabel('Original Timesteps')
    plt.ylabel('NN-Initiated Timesteps')
    plt.title('LBPM Convergence Speedup')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("convergence_comparison.png", bbox_inches='tight')
    plt.show()