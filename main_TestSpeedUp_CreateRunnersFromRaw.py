import os
import glob
import numpy as np
import torch
import math
import stat
from scipy.ndimage import distance_transform_edt as edt
import random

# Custom Utilities
from Architectures.Unet import Extended_DannyKo
from Architectures.Models import SubModels_Composition
from Utilities import start_handler as sh
from Utilities import velocity_usage as vu

# ==============================================================================
# CONFIGURATION
# ==============================================================================
ROOT_DATASET_FOLDER = "../LBPMSimulations_BiggerCrops/DRP-247/Samples_256_256_256/"
shape = (256, 256, 256)

n_samples = None
shuffle = False

# Base Output Directory
RESULTS_DIR = "../TestSpeedUp_Simulations_BiggerCrops/DRP247_256_256_256/"
visualization_interval = 1000000000
tolerance = 1e-2

raw_file = "domain.raw"
device = "cpu"

# SLURM & Job Settings
jobs_running = 15
NTASKS = 1

LBPM_VERSION = "lbpm/cpu/lbpm_init_07f0eef"
PARTITION = "close_cpu"
GRES_STR = ""

MPI_PATH = "mpirun"
LBPM_EXEC = "lbpm_permeability_simulator"
analysis_interval = 200

os.makedirs(RESULTS_DIR, exist_ok=True)

# ==============================================================================
# MODEL INITIALIZATION (Done once)
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
# MAIN RUNNER SCRIPT INITIALIZATION
# ==============================================================================
master_submit_path = os.path.join(RESULTS_DIR, "submit_all_jobs.sh")
with open(master_submit_path, "w") as m_f:
    m_f.write("#!/bin/bash\n\n")
    m_f.write(f"# =========================================================\n")
    m_f.write(f"# GLOBAL RUN SETTINGS\n")
    m_f.write(f"# Script to submit all dataset jobs from the main directory\n")
    m_f.write(f"# =========================================================\n")
    m_f.write(f"export LBPM_VERSION=\"{LBPM_VERSION}\"\n")
    m_f.write(f"PARTITION=\"{PARTITION}\"\n")
    m_f.write(f"GRES_STR=\"{GRES_STR}\"\n\n")
    m_f.write("# Configuração dinâmica de GRES\n")
    m_f.write("GRES_FLAG=\"\"\n")
    m_f.write("if [ ! -z \"$GRES_STR\" ]; then\n")
    m_f.write("    GRES_FLAG=\"--gres=$GRES_STR\"\n")
    m_f.write("fi\n\n")
    m_f.write("echo \"=== Starting Bulk Submissions ===\"\n\n")

# ==============================================================================
# DATASET DISCOVERY & CHUNKING
# ==============================================================================
sample_paths = [f.path for f in os.scandir(ROOT_DATASET_FOLDER) if f.is_dir()]
sample_paths.sort()

if shuffle:
    random.shuffle(sample_paths)

if n_samples is not None:
    sample_paths = sample_paths[:n_samples]

total_samples = len(sample_paths)
if total_samples == 0:
    raise Exception(f"No sample subfolders found in {ROOT_DATASET_FOLDER}")

# Calculate chunk sizes based on total sample folders
jobs_running = min(jobs_running, total_samples)
chunk_size = math.ceil(total_samples / jobs_running)

print(f"Total Samples: {total_samples} | Dividing into {jobs_running} chunks (~{chunk_size} samples/job).")

# ==============================================================================
# GENERATE CHUNK SCRIPTS & PREPARE SIMULATIONS
# ==============================================================================
for chunk_idx in range(jobs_running):
    start_idx = chunk_idx * chunk_size
    end_idx = min(start_idx + chunk_size, total_samples)
    
    if start_idx >= total_samples:
        break
        
    chunk_str_id = f"{chunk_idx:03d}"
    chunk_script_name = f"run_chunk_{chunk_str_id}.sh"
    chunk_script_path = os.path.join(RESULTS_DIR, chunk_script_name)
    
    # Write Chunk SLURM Header
    with open(chunk_script_path, "w") as c_f:
        c_f.write("#!/bin/bash\n\n")
        c_f.write("# ---------------- SLURM Job Settings ----------------\n")
        c_f.write("#SBATCH --oversubscribe\n")
        c_f.write(f"#SBATCH --job-name=Perm_Chunk_{chunk_str_id}\n")
        c_f.write("#SBATCH -t 7-0:00\n")
        c_f.write(f"#SBATCH -o perm_chunk_{chunk_str_id}_%j.out\n")
        c_f.write(f"#SBATCH -e perm_chunk_{chunk_str_id}_%j.err\n")
        c_f.write(f"#SBATCH --ntasks={NTASKS}\n")
        c_f.write("#SBATCH --nodelist=node[008-020]\n")
        c_f.write("#SBATCH --cpus-per-task=1\n\n")
        c_f.write("# ---------------- Environment Setup ----------------\n")
        c_f.write("module load $LBPM_VERSION\n\n")
        c_f.write(f"echo \"=== Starting Chunk {chunk_str_id} (Samples {start_idx} to {end_idx - 1}) ===\"\n\n")

    # Process each sample assigned to this chunk
    for sample_idx in range(start_idx, end_idx):
        sample_path = sample_paths[sample_idx]
        dataset_name = os.path.basename(sample_path)
        current_results_dir = os.path.join(RESULTS_DIR, dataset_name)
        os.makedirs(current_results_dir, exist_ok=True)
        
        print(f"  -> Setting up sample: {dataset_name} (Chunk {chunk_str_id})")
        
        raw_files = glob.glob(os.path.join(sample_path, "*.raw"))
        if not raw_files:
            print(f"Warning: No .raw file found in {sample_path}. Skipping.")
            continue
        current_file = raw_files[0]
        
        grad_dir = os.path.join(current_results_dir, "lbpm_grad_run")
        nn_dir = os.path.join(current_results_dir, "lbpm_nn_run")
        os.makedirs(grad_dir, exist_ok=True)
        os.makedirs(nn_dir, exist_ok=True)
        
        # Read geometry & binarize
        x_numpy = np.fromfile(current_file, dtype=np.uint8).reshape(shape)
        geometry_bool = (x_numpy > 0)
        geometry_uint8 = geometry_bool.astype(np.uint8)
        
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

        sh.write_start_raw(filename=os.path.join(grad_dir, "Start.00000"), ux=ux_null, uy=uy_null, uz=uz_null, pr=pr_grad)
        sh.write_lbpm_db(
            path=grad_dir, db_name="lbpm.db", domain_filename=f"../{raw_file}",
            Start=True, tau=1.5, bc=3, din=1.0, dout=1.0 - 3*p_drop,
            nproc=(1, 1, NTASKS), n=(shape[2]//NTASKS, shape[1]//NTASKS, shape[0]//NTASKS), N=shape, 
            analysis_interval=analysis_interval, visualization_interval=visualization_interval,
            tolerance=tolerance, out_format="vtk"
        )
        
        # 2. Neural Network Setup
        geometry_edt = edt(geometry_uint8).astype("float32")
        geometry_edt = torch.from_numpy(geometry_edt).unsqueeze(0).unsqueeze(0)
        
        pred = model.predict(geometry_edt)
        pred = vu.tensor_denorm(out=pred, inp=geometry_edt)
        
        uz_nn = pred[0,0].numpy().astype(np.float64)
        uy_nn = pred[0,1].numpy().astype(np.float64)
        ux_nn = pred[0,2].numpy().astype(np.float64)
        pr_nn = pred[0,3].numpy().astype(np.float64)

        sh.write_start_raw(filename=os.path.join(nn_dir, "Start.00000"), ux=ux_nn, uy=uy_nn, uz=uz_nn, pr=pr_nn)
        sh.write_lbpm_db(
            path=nn_dir, db_name="lbpm.db", domain_filename=f"../{raw_file}",
            Start=True, tau=1.5, bc=3, din=1.0, dout=1.0 - 3*p_drop,
            nproc=(1, 1, NTASKS), n=(shape[2]//NTASKS, shape[1]//NTASKS, shape[0]//NTASKS), N=shape, 
            analysis_interval=analysis_interval, visualization_interval=visualization_interval,
            tolerance=tolerance, out_format="vtk"
        )
        
        # Append Execution Commands to Chunk Script using original path stepping
        with open(chunk_script_path, "a") as c_f:
            # Execution 1: Gradient-Started Run
            c_f.write(f"echo \"--- Launching simulation for {dataset_name} (Gradient-Initiated Run) ---\"\n")
            c_f.write(f"cd \"{dataset_name}/lbpm_grad_run\"\n") 
            c_f.write("echo \"Current Simulation: \" ${PWD##*/}\n")
            c_f.write(f"{MPI_PATH} --oversubscribe -np {NTASKS} {LBPM_EXEC} lbpm.db\n")
            c_f.write("cd ../../\n\n")  
            
            # Execution 2: NN-Started Run
            c_f.write(f"echo \"--- Launching simulation for {dataset_name} (NN-Initiated Run) ---\"\n")
            c_f.write(f"cd \"{dataset_name}/lbpm_nn_run\"\n")
            c_f.write("echo \"Current Simulation: \" ${PWD##*/}\n")
            c_f.write(f"{MPI_PATH} --oversubscribe -np {NTASKS} {LBPM_EXEC} lbpm.db\n")
            c_f.write("cd ../../\n\n")  

    # Close out chunk script
    with open(chunk_script_path, "a") as c_f:
        c_f.write(f"echo \"--> All simulations in chunk {chunk_str_id} finished.\"\n")

    # Add chunk to Master Script
    with open(master_submit_path, "a") as m_f:
        m_f.write(f"job_id=$(sbatch --parsable --partition=$PARTITION $GRES_FLAG {chunk_script_name})\n")
        m_f.write(f"echo \"Submitted {chunk_script_name} to $PARTITION (Job: $job_id)\"\n")

# Finalize Master Script
with open(master_submit_path, "a") as m_f:
    m_f.write("\necho \"--> All jobs submitted from root.\"\n")

os.chmod(master_submit_path, os.stat(master_submit_path).st_mode | stat.S_IEXEC)
print(f"Setup complete! Master submission script ready at: {master_submit_path}")