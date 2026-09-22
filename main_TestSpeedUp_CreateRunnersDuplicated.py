import os
import glob
import numpy as np
import pandas as pd
import torch
import math
import stat
from pathlib import Path
from scipy.ndimage import distance_transform_edt as edt

from Architectures.Unet import Extended_DannyKo
from Architectures.Models import SubModels_Composition
from Utilities import start_handler as sh
from Utilities import velocity_usage as vu

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================
# File paths
TABLE_PATH          = "./Tables/SpeedUp_crossDataset.csv"  # Adjust to your specific table file
NEW_RESULTS_DIR     = "../TestSpeedUp_Simulations_Reruns_1e4/"

 
old_main_folders = {
    "CastleGate":               "../TestSpeedUp_Simulations_CrossDatasets/Test_Oliveira_CastleGate_SAug_DNorm/",
    "Spherical Pores":          "../TestSpeedUp_Simulations_CrossDatasets/Test_Silveira_SphPore_SAug_DNorm/",
    "Spherical Grains":         "../TestSpeedUp_Simulations_CrossDatasets/Test_Silveira_SphGrain_SAug_DNorm/",
    "Leopard":                  "../TestSpeedUp_Simulations_CrossDatasets/Test_Oliveira_Leopard_SAug_DNorm/",
    "Berea Upper Gray":         "../TestSpeedUp_Simulations_CrossDatasets/Test_Oliveira_BereaUpperGray_SAug_DNorm/",
    "Berea Sinter Gray":        "../TestSpeedUp_Simulations_CrossDatasets/Test_Oliveira_BereaSinterGray_SAug_DNorm/",
    "Berea Buff":               "../TestSpeedUp_Simulations_CrossDatasets/Test_Oliveira_BereaBuff_SAug_DNorm/",
    "Berea":                    "../TestSpeedUp_Simulations_CrossDatasets/Test_Oliveira_Berea_SAug_DNorm/",
    "Bentheimer":               "../TestSpeedUp_Simulations_CrossDatasets/Test_Oliveira_Bentheimer_SAug_DNorm/",

    "256³ Sandstone":           "../TestSpeedUp_Simulations_BiggerCrops/DRP247_256_256_256/",
    "256³ Pre-salt":            "../TestSpeedUp_Simulations_BiggerCrops/DRP503_256_256_256_sw02/",
    "256³ I.C Doddington":      "../TestSpeedUp_Simulations_BiggerCrops/IC_Doddington_256_256_256/",
    "256³ I.C Estaillades":     "../TestSpeedUp_Simulations_BiggerCrops/IC_Estaillades_256_256_256/",
    "256³ I.C Ketton":          "../TestSpeedUp_Simulations_BiggerCrops/IC_Ketton_256_256_256/",
    
    "512³ Sandstone":           "../TestSpeedUp_Simulations_BiggerCrops/DRP247_512_512_512/",
    "512³ Pre-salt":            "../TestSpeedUp_Simulations_BiggerCrops/DRP503_512_512_512_sw02/",
    "500³ I.C Doddington":      "../TestSpeedUp_Simulations_BiggerCrops/IC_Doddington_500_500_500/",
    "500³ I.C Estaillades":     "../TestSpeedUp_Simulations_BiggerCrops/IC_Estaillades_500_500_500/",
    "500³ I.C Ketton":          "../TestSpeedUp_Simulations_BiggerCrops/IC_Ketton_500_500_500/",
}

# New parameters
perm_diff_criteria  = 5 # [%] difference between permeability to be a premature case
tolerance           = 1e-4
nproc               = (2, 2, 2)
NTASKS              = nproc[0] * nproc[1] * nproc[2]
visualization_interval = 1000000000
raw_file            = "domain.raw"
device              = "cpu"

# SLURM & Job Settings
LBPM_VERSION        = "lbpm/cpu/lbpm_init_07f0eef"
PARTITION           = "close_cpu"
GRES_STR            = ""
MPI_PATH            = "mpirun"
LBPM_EXEC           = "lbpm_permeability_simulator"
analysis_interval   = 200

os.makedirs(NEW_RESULTS_DIR, exist_ok=True)

# ==============================================================================
# 2. READ AND FILTER TABLE
# ==============================================================================
print(f"Reading table from {TABLE_PATH}...")
df = pd.read_csv(TABLE_PATH)

# Filter condition: Permeability difference higher than 10%
df_filtered = df[df['Perm_Error [%]'] > perm_diff_criteria].copy()
total_samples = len(df_filtered)

if total_samples == 0:
    print(f"No samples found with a Permeability difference > {perm_diff_criteria}%. Exiting.")
    exit()

print(f"Found {total_samples}/{len(df)} samples with >{perm_diff_criteria}% permeability error. Preparing reruns...\n")

A=B
# ==============================================================================
# 3. MODEL INITIALIZATION
# ==============================================================================
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

# ==============================================================================
# 4. MASTER SCRIPT SETUP
# ==============================================================================
master_submit_path = os.path.join(NEW_RESULTS_DIR, "submit_all_jobs.sh")
with open(master_submit_path, "w") as m_f:
    m_f.write("#!/bin/bash\n\n")
    m_f.write(f"# =========================================================\n")
    m_f.write(f"# GLOBAL RUN SETTINGS (Reruns: Tol 1e-4, nproc 2x2x2)\n")
    m_f.write(f"# =========================================================\n")
    m_f.write(f"export LBPM_VERSION=\"{LBPM_VERSION}\"\n")
    m_f.write(f"PARTITION=\"{PARTITION}\"\n")
    m_f.write(f"GRES_STR=\"{GRES_STR}\"\n\n")
    m_f.write("GRES_FLAG=\"\"\n")
    m_f.write("if [ ! -z \"$GRES_STR\" ]; then\n")
    m_f.write("    GRES_FLAG=\"--gres=$GRES_STR\"\n")
    m_f.write("fi\n\n")
    m_f.write("echo \"=== Starting Bulk Submissions ===\"\n\n")

# ==============================================================================
# 5. GENERATE INDIVIDUAL SIMULATIONS & SCRIPTS
# ==============================================================================
samples_to_process = df_filtered.to_dict('records')

for row in samples_to_process:
    dataset_name = row['Dataset']
    sample_name = row['Sample']
    
    # Identify original folder to copy the raw geometry
    if dataset_name not in old_main_folders:
        print(f"Warning: {dataset_name} not found in old_main_folders mapping. Skipping {sample_name}.")
        continue
        
    old_dataset_path = old_main_folders[dataset_name]
    old_sample_dir = os.path.join(old_dataset_path, sample_name)
    
    domain_file_path = glob.glob(os.path.join(old_sample_dir, "domain.raw"))[0]

    # Dynamically infer the shape of the cube based on file size
    file_size = os.path.getsize(domain_file_path)
    dim = int(round(file_size ** (1/3.0)))
    shape = (dim, dim, dim)
    
    if dim**3 != file_size:
        print(f"Warning: {domain_file_path} is not a perfect cube (Size: {file_size}). Skipping.")
        continue
        
    # Calculate required memory based on geometry
    mem = (30 * 8 * dim**3) // (1024**3)
    mem = max(4, mem) # Ensure a minimum of 4GB so SLURM doesn't fail with --mem=0G for very small cubes
    
    print(f"-> Setting up: {dataset_name} - {sample_name} | Shape: {shape} | RAM: {mem}GB")
    
    # Create isolated script name for this specific sample
    safe_dataset_name = dataset_name.replace(" ", "_")
    safe_sample_name = sample_name.replace(" ", "_")
    job_script_name = f"run_{safe_dataset_name}_{safe_sample_name}.sh"
    job_script_path = os.path.join(NEW_RESULTS_DIR, job_script_name)
    
    # Write SLURM Header for this specific sample
    with open(job_script_path, "w") as c_f:
        c_f.write("#!/bin/bash\n\n")
        c_f.write("# ---------------- SLURM Job Settings ----------------\n")
        c_f.write("#SBATCH --oversubscribe\n")
        c_f.write(f"#SBATCH --job-name=Rerun_{safe_sample_name}\n")
        c_f.write("#SBATCH -t 7-0:00\n")
        c_f.write(f"#SBATCH -o rerun_{safe_sample_name}_%j.out\n")
        c_f.write(f"#SBATCH -e rerun_{safe_sample_name}_%j.err\n")
        c_f.write(f"#SBATCH --ntasks={NTASKS}\n")
        c_f.write("#SBATCH --nodelist=node[008-020]\n")
        c_f.write("#SBATCH --cpus-per-task=4\n\n")
        c_f.write(f"#SBATCH --mem={mem}G\n") 
        c_f.write("# ---------------- Environment Setup ----------------\n")
        c_f.write("module load $LBPM_VERSION\n\n")
        c_f.write(f"echo \"=== Starting Simulation for {dataset_name} / {sample_name} ===\"\n\n")

    # Create new directories
    current_results_dir = os.path.join(NEW_RESULTS_DIR, dataset_name, sample_name)
    grad_dir = os.path.join(current_results_dir, "lbpm_grad_run")
    nn_dir = os.path.join(current_results_dir, "lbpm_nn_run")
    os.makedirs(grad_dir, exist_ok=True)
    os.makedirs(nn_dir, exist_ok=True)
    
    # Read geometry & binarize
    x_numpy = np.fromfile(domain_file_path, dtype=np.uint8).reshape(shape)
    geometry_bool = (x_numpy > 0)
    geometry_uint8 = geometry_bool.astype(np.uint8)
    
    # Save raw to new location
    source_raw = os.path.join(current_results_dir, raw_file)
    geometry_uint8.tofile(source_raw)
    
    p_drop = vu.pressure_calculation(geometry_bool, tau=1.5, Re=0.1, Dens=1.0)
    
    # 1. Gradient Setup
    uz_null = np.zeros(shape, dtype=np.float64)
    uy_null = np.zeros(shape, dtype=np.float64)
    ux_null = np.zeros(shape, dtype=np.float64)
    pr_grad = np.zeros(shape, dtype=np.float64)
    
    z_steps = np.linspace(1.0/3.0, (1.0/3.0) - p_drop, shape[0])    
    for i in range(shape[0]):
        pr_grad[i, :, :] = z_steps[i]
        
    uz_null[~geometry_bool] = 0.0
    uy_null[~geometry_bool] = 0.0
    ux_null[~geometry_bool] = 0.0
    pr_grad[~geometry_bool] = 0.0
    
    sh.write_start_raw(dirpath=grad_dir, ux=ux_null, uy=uy_null, uz=uz_null, pr=pr_grad, nproc=nproc)
    sh.write_lbpm_db(
        path=grad_dir, db_name="lbpm.db", domain_filename=f"../{raw_file}",
        Start=True, tau=1.5, bc=3, din=1.0, dout=1.0 - 3*p_drop,
        nproc=nproc, 
        n=(int(shape[2]/nproc[0]), int(shape[1]/nproc[1]), int(shape[0]/nproc[2])), 
        N=shape, 
        analysis_interval=analysis_interval, visualization_interval=visualization_interval,
        tolerance=tolerance, out_format="vtk"
    )
    
    # 2. Neural Network Setup
    geometry_uint8_padded = sh.pad_geometry(geometry_uint8) 
    geometry_edt = edt(geometry_uint8_padded).astype("float32")
    geometry_edt = torch.from_numpy(geometry_edt).unsqueeze(0).unsqueeze(0) 
    
    pred = model.predict(geometry_edt)
    pred = vu.tensor_denorm(out=pred, inp=geometry_edt)
    pred = sh.unpad_geometry(pred, shape)
    
    uz_nn = pred[0,0].detach().cpu().numpy().astype(np.float64)
    uy_nn = pred[0,1].detach().cpu().numpy().astype(np.float64)
    ux_nn = pred[0,2].detach().cpu().numpy().astype(np.float64)
    pr_nn = pred[0,3].detach().cpu().numpy().astype(np.float64)

    sh.write_start_raw(dirpath=nn_dir, ux=ux_nn, uy=uy_nn, uz=uz_nn, pr=pr_nn, nproc=nproc)
    sh.write_lbpm_db(
        path=nn_dir, db_name="lbpm.db", domain_filename=f"../{raw_file}",
        Start=True, tau=1.5, bc=3, din=1.0, dout=1.0 - 3*p_drop,
        nproc=nproc, 
        n=(int(shape[2]/nproc[0]), int(shape[1]/nproc[1]), int(shape[0]/nproc[2])), 
        N=shape, 
        analysis_interval=analysis_interval, visualization_interval=visualization_interval,
        tolerance=tolerance, out_format="vtk"
    )
    
    # Write executions to the individual sample script
    relative_target_dir = os.path.join(dataset_name, sample_name)
    with open(job_script_path, "a") as c_f:
        c_f.write(f"echo \"--- Launching simulation for {dataset_name}/{sample_name} (Gradient) ---\"\n")
        c_f.write(f"cd \"{relative_target_dir}/lbpm_grad_run\"\n") 
        c_f.write(f"{MPI_PATH} --oversubscribe -np {NTASKS} {LBPM_EXEC} lbpm.db\n")
        c_f.write("cd ../../../\n\n")  
        
        c_f.write(f"echo \"--- Launching simulation for {dataset_name}/{sample_name} (NN-Initiated) ---\"\n")
        c_f.write(f"cd \"{relative_target_dir}/lbpm_nn_run\"\n")
        c_f.write(f"{MPI_PATH} --oversubscribe -np {NTASKS} {LBPM_EXEC} lbpm.db\n")
        c_f.write("cd ../../../\n\n")  
        
        c_f.write(f"echo \"--> Simulation for {sample_name} finished.\"\n")

    # Append job submission to the master script
    with open(master_submit_path, "a") as m_f:
        m_f.write(f"job_id=$(sbatch --parsable --partition=$PARTITION $GRES_FLAG {job_script_name})\n")
        m_f.write(f"echo \"Submitted {job_script_name} to $PARTITION (Job: $job_id)\"\n")

# Finalize master script
with open(master_submit_path, "a") as m_f:
    m_f.write("\necho \"--> All individual rerun jobs submitted from root.\"\n")

os.chmod(master_submit_path, os.stat(master_submit_path).st_mode | stat.S_IEXEC)
print(f"\nSetup complete! Master submission script for reruns ready at: {master_submit_path}")