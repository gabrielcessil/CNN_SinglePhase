import os
import numpy as np
import torch
import math
import stat
import porespy as ps
from scipy.ndimage import distance_transform_edt as edt

# Custom Utilities
from Architectures.Unet import Extended_DannyKo
from Architectures.Models import SubModels_Composition
from Utilities import start_handler as sh
from Utilities import velocity_usage as vu

# ==============================================================================
# CONFIGURATION
# ==============================================================================
# Base Output Directory
RESULTS_DIR = "./TestSpeedUp_Simulations_1e4/Generated_Blob/"
os.makedirs(RESULTS_DIR, exist_ok=True)

raw_file = "domain.raw"
shape = (256, 256, 256)
device = "cpu"

# Generation Parameters
TARGET_POROSITY = 0.5
BLOBINESS       = 1.5

# SLURM & Job Settings
NTASKS          = 1

LBPM_VERSION    = "lbpm/cpu/lbpm_init_07f0eef"
PARTITION       = "close_cpu"
GRES_STR        = ""

MPI_PATH        = "mpirun"
LBPM_EXEC       = "lbpm_permeability_simulator"

analysis_interval       = 200
visualization_interval  = 1000000000
tolerance               = 1e-4

# ==============================================================================
# MODEL INITIALIZATION
# ==============================================================================
print("Loading Models...")
danny_model = Extended_DannyKo()

model = SubModels_Composition(
    main_model=danny_model, 
    z_name="./Trained_Models/NN_Trainning_13_July_2026_06-02PM_Job26267/model_LowerValidationLoss.pth",
    x_name="./Trained_Models/NN_Trainning_15_July_2026_03-59PM_Job26381/model_LowerValidationLoss.pth", 
    p_name="./Trained_Models/NN_Trainning_21_July_2026_05-22PM_Job26505/model_LowerValidationLoss.pth", 
    device=device, 
    is_eval=True
)

# ==============================================================================
# VOLUME GENERATION (Gaussian Filtered Blobs)
# ==============================================================================
print(f"\n{'='*80}")
print(f"Generating single sample: {shape} with {TARGET_POROSITY*100}% porosity...")
print(f"{'='*80}")

# Generate volume (True = Void space)
vol = ps.generators.blobs(shape=shape, porosity=TARGET_POROSITY, blobiness=BLOBINESS)
vol[:, :, 0]   = 0
vol[:, :, -1]  = 0
vol[:, 0, :]   = 0
vol[:, -1, :]  = 0
geometry_bool = vol
geometry_uint8 = geometry_bool.astype(np.uint8)

sample_name = f"Sample_GaussianBlob_p{TARGET_POROSITY}_b{BLOBINESS}"
sample_dir  = os.path.join(RESULTS_DIR, sample_name)
grad_dir    = os.path.join(sample_dir, "lbpm_grad_run")
nn_dir      = os.path.join(sample_dir, "lbpm_nn_run")

os.makedirs(grad_dir, exist_ok=True)
os.makedirs(nn_dir, exist_ok=True)

# Save domain.raw to the base sample folder
source_raw = os.path.join(sample_dir, raw_file)
geometry_uint8.tofile(source_raw)

# Calculate Pressure Drop
p_drop = vu.pressure_calculation(geometry_bool, tau=1.5, Re=0.1, Dens=1.0)
print(f"Calculated Pressure Drop: {p_drop}")

# ==================================================================
# 1. GRADIENT INITIALIZATION SETUP
# ==================================================================
print("Preparing Gradient initial conditions...")
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

sh.write_start_raw(
    filename=os.path.join(grad_dir, "Start.00000"),
    ux=ux_null, uy=uy_null, uz=uz_null, pr=pr_grad
)

sh.write_lbpm_db(
    path=grad_dir,
    db_name="lbpm.db",
    domain_filename=f"../{raw_file}",
    Start=True, tau=1.5, bc=3, din=1.0, dout=1.0 - 3*p_drop,
    nproc=(1, 1, NTASKS), n=(shape[2]//NTASKS, shape[1]//NTASKS, shape[0]//NTASKS), N=shape, 
    analysis_interval=analysis_interval, visualization_interval=visualization_interval,
    tolerance=tolerance, out_format="vtk"
)

# ==================================================================
# 2. NEURAL NETWORK INITIALIZATION SETUP
# ==================================================================
print("Preparing Neural Network predictions...")
geometry_edt = edt(geometry_uint8).astype("float32")
geometry_edt = torch.from_numpy(geometry_edt).unsqueeze(0).unsqueeze(0)

with torch.no_grad():
    pred = model.predict(geometry_edt)
    
pred = vu.tensor_denorm(out=pred, inp=geometry_edt)

uz_nn = pred[0,0].numpy().astype(np.float64)
uy_nn = pred[0,1].numpy().astype(np.float64)
ux_nn = pred[0,2].numpy().astype(np.float64)
pr_nn = pred[0,3].numpy().astype(np.float64)

sh.write_start_raw(
    filename=os.path.join(nn_dir, "Start.00000"),
    ux=ux_nn, uy=uy_nn, uz=uz_nn, pr=pr_nn
)

sh.write_lbpm_db(
    path=nn_dir,
    db_name="lbpm.db",
    domain_filename=f"../{raw_file}",
    Start=True, tau=1.5, bc=3, din=1.0, dout=1.0 - 3*p_drop,
    nproc=(1, 1, NTASKS), n=(shape[2]//NTASKS, shape[1]//NTASKS, shape[0]//NTASKS), N=shape, 
    analysis_interval=analysis_interval, visualization_interval=visualization_interval,
    tolerance=tolerance, out_format="vtk"
)

# ==========================================================================
# SLURM SCRIPT GENERATION FOR THE SINGLE RUN
# ==========================================================================
print("Writing SLURM job script...")
runner_script_path = os.path.join(RESULTS_DIR, "run_single_blob.sh")

with open(runner_script_path, "w") as c_f:
    c_f.write("#!/bin/bash\n\n")
    c_f.write("# ---------------- SLURM Job Settings ----------------\n")
    c_f.write(f"#SBATCH --partition=\"{PARTITION}\"\n")
    if GRES_STR:
        c_f.write(f"#SBATCH --gres={GRES_STR}\n")
    c_f.write("#SBATCH --oversubscribe\n")
    c_f.write(f"#SBATCH --job-name=Perm_Blob\n")
    c_f.write("#SBATCH -t 7-0:00\n")
    c_f.write(f"#SBATCH -o perm_blob_%j.out\n")
    c_f.write(f"#SBATCH -e perm_blob_%j.err\n")
    c_f.write(f"#SBATCH --ntasks={NTASKS}\n")
    c_f.write(f"#SBATCH --cpus-per-task={2*NTASKS}\n")
    c_f.write(f"#SBATCH --mem=4G\n\n")
    
    c_f.write("# ---------------- Environment Setup ----------------\n")
    c_f.write(f"module load {LBPM_VERSION}\n\n")
    
    # Execution 1: Gradient-Started Run
    c_f.write(f"echo \"--- Launching simulation for {sample_name} (Gradient-Initiated Run) ---\"\n")
    c_f.write(f"cd {sample_name}/lbpm_grad_run\n")
    c_f.write("echo \"Current Simulation: \" ${PWD##*/}\n")
    c_f.write(f"{MPI_PATH} --oversubscribe -np {NTASKS} {LBPM_EXEC} lbpm.db\n")
    c_f.write("cd ../../\n\n") 
    
    # Execution 2: NN-Started Run
    c_f.write(f"echo \"--- Launching simulation for {sample_name} (NN-Initiated Run) ---\"\n")
    c_f.write(f"cd {sample_name}/lbpm_nn_run\n")
    c_f.write("echo \"Current Simulation: \" ${PWD##*/}\n")
    c_f.write(f"{MPI_PATH} --oversubscribe -np {NTASKS} {LBPM_EXEC} lbpm.db\n")
    c_f.write("cd ../../\n\n")
    
    c_f.write("echo \"--> All simulations finished.\"\n")

# Make runner script executable
os.chmod(runner_script_path, os.stat(runner_script_path).st_mode | stat.S_IEXEC)
print(f"Done! Setup complete in: {RESULTS_DIR}")
print(f"Run the job with: sbatch {runner_script_path}")