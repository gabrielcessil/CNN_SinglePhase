import os
import glob
import numpy as np
import torch
from scipy.ndimage import distance_transform_edt as edt
import stat
import math
# Custom Utilities
from Architectures.Unet import Extended_DannyKo
from Architectures.Models import SubModels_Composition
from Utilities import start_handler as sh
from Utilities import velocity_usage as vu

# ==============================================================================
# CONFIGURATION
# ==============================================================================
datasets = ["../GradSimulations/DEBUG_DONE_Valid_Silveira_CylinGrain_120_120_120/"]
raw_file = "domain.raw"
shape = (120, 120, 120)
device = "cpu"


# SLURM & Job Settings
CHUNK_SIZE      = 10
NTASKS          = 4
LBPM_VERSION    = "lbpm/gpu/lbpm_fork_965bd0d"
PARTITION       = "all_gpu"
GRES_STR        = "gpu:a100:4"
MPI_PATH        = "mpirun"
LBPM_EXEC       = "lbpm_permeability_simulator"

# ==============================================================================
# MODEL INITIALIZATION
# ==============================================================================
danny_model = Extended_DannyKo()

concat_model = SubModels_Composition(
    main_model=danny_model, 
    z_name="./Trained_Models/NN_Trainning_13_July_2026_06-02PM_Job26267/model_LowerValidationLoss.pth",
    x_name="./Trained_Models/NN_Trainning_15_July_2026_03-59PM_Job26381/model_LowerValidationLoss.pth", 
    p_name="./Trained_Models/NN_Trainning_21_July_2026_05-22PM_Job26505/model_LowerValidationLoss.pth", 
    device=device, 
    is_eval=True
)

# ==============================================================================
# MAIN SETUP & SLURM SCRIPT GENERATION
# ==============================================================================
for dataset_path in datasets:
    dataset_abspath = os.path.abspath(dataset_path)
    sample_folders = sorted(glob.glob(os.path.join(dataset_abspath, "Sample_*")))
    total_samples = len(sample_folders)
    
    if total_samples == 0:
        continue
        
    num_chunks = math.ceil(total_samples / CHUNK_SIZE)
    print(f"\nProcessing {total_samples} samples into {num_chunks} chunks for {dataset_path}...")
    
    # 1. Initialize the Master Chained Submission Script[cite: 1]
    master_submit_path = os.path.join(dataset_abspath, "submit_all_chained.sh")
    with open(master_submit_path, "w") as m_f:
        m_f.write("#!/bin/bash\n\n")
        m_f.write(f"#SBATCH --partition=\"{PARTITION}\"\n")
        m_f.write("# =========================================================\n")
        m_f.write("# GLOBAL RUN SETTINGS\n")
        m_f.write("# Altere aqui para atualizar todos os jobs da corrente\n")
        m_f.write("# =========================================================\n")
        m_f.write(f"export LBPM_VERSION=\"{LBPM_VERSION}\"\n")
        m_f.write(f"PARTITION=\"{PARTITION}\"\n")
        m_f.write(f"GRES_STR=\"{GRES_STR}\"\n\n")
        m_f.write("# Configuração dinâmica de GRES\n")
        m_f.write("GRES_FLAG=\"\"\n")
        m_f.write("if [ ! -z \"$GRES_STR\" ]; then\n")
        m_f.write("    GRES_FLAG=\"--gres=$GRES_STR\"\n")
        m_f.write("fi\n\n")

    # 2. Process Samples in Chunks
    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * CHUNK_SIZE
        end_idx = min(start_idx + CHUNK_SIZE, total_samples)
        chunk_samples = sample_folders[start_idx:end_idx]
        
        chunk_str_id = f"{chunk_idx:03d}"
        chunk_script_name = f"run_lbpm_chunk_{chunk_str_id}.sh"
        chunk_script_path = os.path.join(dataset_abspath, chunk_script_name)
        
        # Initialize Chunk Script Headers[cite: 2]
        with open(chunk_script_path, "w") as c_f:
            c_f.write("#!/bin/bash\n\n")
            c_f.write("# ---------------- SLURM Job Settings ----------------\n")
            c_f.write("#SBATCH --oversubscribe\n")
            c_f.write(f"#SBATCH --job-name=Perm_chunk_{chunk_str_id}\n")
            c_f.write("#SBATCH -t 7-0:00\n")
            c_f.write(f"#SBATCH -o perm_chunk_{chunk_str_id}_%j.out\n")
            c_f.write(f"#SBATCH -e perm_chunk_{chunk_str_id}_%j.err\n")
            c_f.write(f"#SBATCH --ntasks={NTASKS}\n\n")
            
            c_f.write("# ---------------- Environment Setup ----------------\n")
            c_f.write("module load $LBPM_VERSION\n\n")
            
            c_f.write(f"echo \"=== Chunk {chunk_str_id} | Processing {os.path.basename(chunk_samples[0])} to {os.path.basename(chunk_samples[-1])} ({len(chunk_samples)} samples) ===\"\n\n")

        # Process each sample in the current chunk
        for sample_abspath in chunk_samples:
            sample_name = os.path.basename(sample_abspath)
            print(f"  -> Setting up: {sample_name} (Chunk {chunk_str_id})")
            
            run_dir = os.path.join(sample_abspath, "lbpm_run")
            started_dir = os.path.join(sample_abspath, "lbpm_started_run")
            
            os.makedirs(run_dir, exist_ok=True)
            os.makedirs(started_dir, exist_ok=True)
            
            # Predict Start Fields
            source_raw = os.path.join(sample_abspath, raw_file)
            geometry = (np.fromfile(source_raw, dtype=np.uint8).reshape(shape) > 0)
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
            
            p_drop = vu.pressure_calculation(geometry, tau=1.5, Re=0.1, Dens=1.0)
            
            # Write DB for standard run
            sh.write_lbpm_db(
                path=run_dir,
                db_name="lbpm.db",
                domain_filename=f"../{raw_file}",
                Start=False, tau=1.5, bc=3, din=1.0, dout=1.0 - 3*p_drop,
                nproc=(1, 1, NTASKS), n=shape, N=shape, 
                analysis_interval=50, visualization_interval=10000000,
                tolerance=1e-6, out_format="silo"
            )
            
            # Write DB for NN-started run
            sh.write_lbpm_db(
                path=started_dir,
                db_name="lbpm.db",
                domain_filename=f"../{raw_file}",
                Start=True, tau=1.5, bc=3, din=1.0, dout=1.0 - 3*p_drop,
                nproc=(1, 1, NTASKS), n=shape, N=shape, 
                analysis_interval=50, visualization_interval=10000000,
                tolerance=1e-6, out_format="silo"
            )
            
            # Append execution commands to the Chunk Script[cite: 2]
            with open(chunk_script_path, "a") as c_f:
                # Execution 1: Standard Run
                c_f.write(f"echo \"--- Launching simulation for {sample_abspath} (Standard Run) ---\"\n")
                c_f.write(f"cd {run_dir}\n")
                c_f.write("echo \"Current Simulation: \" ${PWD##*/}\n")
                c_f.write(f"{MPI_PATH} --oversubscribe -np {NTASKS} {LBPM_EXEC} lbpm.db\n\n")
                
                # Execution 2: NN-Started Run
                c_f.write(f"echo \"--- Launching simulation for {sample_abspath} (NN-Initiated Run) ---\"\n")
                c_f.write(f"cd {started_dir}\n")
                c_f.write("echo \"Current Simulation: \" ${PWD##*/}\n")
                c_f.write(f"{MPI_PATH} --oversubscribe -np {NTASKS} {LBPM_EXEC} lbpm.db --init 1\n\n")
                
        # Close out the chunk script[cite: 2]
        with open(chunk_script_path, "a") as c_f:
            c_f.write("echo \"--> All simulations in this chunk finished.\"\n")
        
        # Add submission logic to Master Script[cite: 1]
        with open(master_submit_path, "a") as m_f:
            if chunk_idx == 0:
                m_f.write(f"j{chunk_idx}=$(sbatch --parsable --partition=$PARTITION $GRES_FLAG {chunk_script_name})\n")
                m_f.write(f"echo \"Submitted {chunk_script_name} to $PARTITION (Job: $j{chunk_idx})\"\n\n")
            else:
                prev_idx = chunk_idx - 1
                m_f.write(f"j{chunk_idx}=$(sbatch --parsable --partition=$PARTITION $GRES_FLAG --dependency=afterok:$j{prev_idx} {chunk_script_name})\n")
                m_f.write(f"echo \"Submitted {chunk_script_name} to $PARTITION (Job: $j{chunk_idx})\"\n\n")
                
    # Close out the Master Script[cite: 1]
    with open(master_submit_path, "a") as m_f:
        m_f.write("echo \"--> All chained jobs submitted.\"\n")
        
    # Make master script executable
    os.chmod(master_submit_path, os.stat(master_submit_path).st_mode | stat.S_IEXEC)
    print(f"\nCompleted! Master chain script created at: {master_submit_path}")