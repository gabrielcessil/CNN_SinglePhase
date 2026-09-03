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
from Utilities import dataset_reader as dr 

# ============================================================================== 
# CONFIGURATION 
# ============================================================================== 
# Input Datasets 
dataset_paths = [ 
    "../NN_Datasets_Grad_Dist_40_5_55/Test_Oliveira_Berea_SAug_DNorm.h5" 
] 
n_samples = 1 
shuffle   = False 
# Base Output Directory 
RESULTS_DIR = "./TestSpeedUp_Simulations_1e1/" 
os.makedirs(RESULTS_DIR, exist_ok=True) 

raw_file = "domain.raw" 
shape = (120, 120, 120) 
device = "cpu" 

# SLURM & Job Settings 
jobs_running    = 10 
CHUNK_SIZE      = max(n_samples//jobs_running,1) 
NTASKS          = 1 

#LBPM_VERSION    = "lbpm/gpu/lbpm_fork_965bd0d" 
#PARTITION       = "all_gpu" 
#GRES_STR        = "gpu:k40m:1" 

LBPM_VERSION    = "lbpm/cpu/lbpm_init_07f0eef" 
PARTITION       = "close_cpu" 
GRES_STR        = "" 

MPI_PATH        = "mpirun" 
LBPM_EXEC       = "lbpm_permeability_simulator" 

analysis_interval       = 200 
visualization_interval  = 1000000000 
tolerance               = 1e-1 

# ============================================================================== 
# MODEL INITIALIZATION (Done once for all datasets) 
# ============================================================================== 
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
# DATASET ITERATION & SETUP 
# ============================================================================== 
for dataset_path in dataset_paths: 
     
    # Extract dataset name without the .h5 extension to create a folder 
    dataset_name = os.path.splitext(os.path.basename(dataset_path))[0] 
    current_results_dir = os.path.join(RESULTS_DIR, dataset_name) 
    os.makedirs(current_results_dir, exist_ok=True) 
     
    print(f"\n{'='*80}") 
    print(f"Loading Dataset: {dataset_path} ...") 
    print(f"{'='*80}") 
     
    dataset = dr.LazyDatasetTorch( 
        h5_path=dataset_path, 
        list_ids=None, 
        x_dtype=torch.float32, 
        y_dtype=torch.float32 
    ) 

    total_samples = len(dataset) 
     
    if n_samples is not None and total_samples > n_samples: 
        if shuffle: random.shuffle(dataset.list_ids) 
        dataset.list_ids = dataset.list_ids[:n_samples] 
        total_samples = len(dataset) 

    num_chunks = math.ceil(total_samples / CHUNK_SIZE) 
    print(f"Processing {total_samples} samples into {num_chunks} chunks.") 
    print(f"Outputs routed to: {current_results_dir}\n") 

    # ========================================================================== 
    # MAIN SETUP & SLURM SCRIPT GENERATION FOR CURRENT DATASET 
    # ========================================================================== 

    # 1. Initialize the Master Chained Submission Script 
    master_submit_path = os.path.join(current_results_dir, "submit_all_chained.sh") 
    with open(master_submit_path, "w") as m_f: 
        m_f.write("#!/bin/bash\n\n") 
        m_f.write(f"#SBATCH --partition=\"{PARTITION}\"\n") 
        m_f.write("# =========================================================\n") 
        m_f.write(f"# GLOBAL RUN SETTINGS ({dataset_name})\n") 
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
         
        chunk_str_id = f"{chunk_idx:03d}" 
        chunk_script_name = f"run_lbpm_chunk_{chunk_str_id}.sh" 
        chunk_script_path = os.path.join(current_results_dir, chunk_script_name) 
         
        # Initialize Chunk Script Headers 
        with open(chunk_script_path, "w") as c_f: 
            c_f.write("#!/bin/bash\n\n") 
            c_f.write("# ---------------- SLURM Job Settings ----------------\n") 
            c_f.write("#SBATCH --oversubscribe\n") 
            c_f.write(f"#SBATCH --job-name=Perm_{dataset_name[:10]}_{chunk_str_id}\n") 
            c_f.write("#SBATCH -t 7-0:00\n") 
            c_f.write(f"#SBATCH -o perm_chunk_{chunk_str_id}_%j.out\n") 
            c_f.write(f"#SBATCH -e perm_chunk_{chunk_str_id}_%j.err\n") 
            c_f.write(f"#SBATCH --ntasks={NTASKS}\n") 
            c_f.write("#SBATCH --nodelist=node[008-020]\n") 
            c_f.write("#SBATCH --cpus-per-task=1\n") 
            c_f.write("#SBATCH --mem=4G\n") 
             
             
            c_f.write("# ---------------- Environment Setup ----------------\n") 
            c_f.write("module load $LBPM_VERSION\n\n") 
             
            c_f.write(f"echo \"=== Chunk {chunk_str_id} | Processing samples {start_idx} to {end_idx - 1} ===\"\n\n") 

        # Process each sample index in the current chunk 
        for sample_idx in range(start_idx, end_idx): 
            actual_id = int(dataset.list_ids[sample_idx]) 
            sample_name = f"Sample_{actual_id:04d}" 
            print(f"  -> Setting up: {sample_name} (Chunk {chunk_str_id})") 
             
            # Create output directories for this specific sample 
            sample_dir  = os.path.join(current_results_dir, sample_name) 
            grad_dir    = os.path.join(sample_dir, "lbpm_grad_run") 
            nn_dir      = os.path.join(sample_dir, "lbpm_nn_run") 
             
            os.makedirs(grad_dir, exist_ok=True) 
            os.makedirs(nn_dir, exist_ok=True) 
             
            # ================================================================== 
            # DATA EXTRACTION & BINARIZATION 
            # ================================================================== 
            x_tensor, _ = dataset[sample_idx] 
            x_numpy = x_tensor.squeeze().numpy() 
             
            geometry_bool = (x_numpy > 0) 
            geometry_uint8 = geometry_bool.astype(np.uint8) 
             
            # Save domain.raw to the base sample folder 
            source_raw = os.path.join(sample_dir, raw_file) 
            geometry_uint8.tofile(source_raw) 
             
            # Calculate Pressure Drop 
            p_drop = vu.pressure_calculation(geometry_bool, tau=1.5, Re=0.1, Dens=1.0) 
             
            # ================================================================== 
            # 1. GRADIENT INITIALIZATION SETUP 
            # ================================================================== 
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
            geometry_edt = edt(geometry_uint8).astype("float32") 
            geometry_edt = torch.from_numpy(geometry_edt).unsqueeze(0).unsqueeze(0) 
             
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
             
            # ================================================================== 
            # APPEND EXECUTIONS TO CHUNK SCRIPT (RELATIVE PATHS) 
            # ================================================================== 
            with open(chunk_script_path, "a") as c_f: 
                # Execution 1: Gradient-Started Run 
                c_f.write(f"echo \"--- Launching simulation for {sample_name} (Gradient-Initiated Run) ---\"\n") 
                c_f.write(f"cd {sample_name}/lbpm_grad_run\n") 
                c_f.write("echo \"Current Simulation: \" ${PWD##*/}\n") 
                c_f.write(f"{MPI_PATH} --oversubscribe -np {NTASKS} {LBPM_EXEC} lbpm.db\n") 
                c_f.write("cd ../../\n\n")  # Step back out to the dataset root folder 
                 
                # Execution 2: NN-Started Run 
                c_f.write(f"echo \"--- Launching simulation for {sample_name} (NN-Initiated Run) ---\"\n") 
                c_f.write(f"cd {sample_name}/lbpm_nn_run\n") 
                c_f.write("echo \"Current Simulation: \" ${PWD##*/}\n") 
                c_f.write(f"{MPI_PATH} --oversubscribe -np {NTASKS} {LBPM_EXEC} lbpm.db\n") 
                c_f.write("cd ../../\n\n")  # Step back out to the dataset root folder 
                 
        # Close out the chunk script 
        with open(chunk_script_path, "a") as c_f: 
            c_f.write("echo \"--> All simulations in this chunk finished.\"\n") 
         
        # Add submission logic to Master Script 
        with open(master_submit_path, "a") as m_f: 
            if chunk_idx == 0: 
                m_f.write(f"j{chunk_idx}=$(sbatch --parsable --partition=$PARTITION $GRES_FLAG {chunk_script_name})\n") 
                m_f.write(f"echo \"Submitted {chunk_script_name} to $PARTITION (Job: $j{chunk_idx})\"\n\n") 
            else: 
                prev_idx = chunk_idx - 1 

                #m_f.write(f"j{chunk_idx}=$(sbatch --parsable --partition=$PARTITION $GRES_FLAG --dependency=afterok:$j{prev_idx} {chunk_script_name})\n") 
                m_f.write(f"j{chunk_idx}=$(sbatch --parsable --partition=$PARTITION $GRES_FLAG {chunk_script_name})\n") 
                m_f.write(f"echo \"Submitted {chunk_script_name} to $PARTITION (Job: $j{chunk_idx})\"\n\n") 
                 
    # Close out the Master Script for the current dataset 
    with open(master_submit_path, "a") as m_f: 
        m_f.write("echo \"--> All chained jobs submitted for this dataset.\"\n") 
         
    # Make master script executable 
    os.chmod(master_submit_path, os.stat(master_submit_path).st_mode | stat.S_IEXEC) 
    print(f"Completed dataset! Master chain script created at: {master_submit_path}")