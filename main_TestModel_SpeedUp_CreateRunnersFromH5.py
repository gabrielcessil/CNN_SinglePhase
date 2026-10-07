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
    
    "../NN_Datasets_Grad_Dist_40_5_55/Test_Oliveira_Leopard_SAug_DNorm.h5",
    "../NN_Datasets_Grad_Dist_40_5_55/Test_Oliveira_CastleGate_SAug_DNorm.h5",
    "../NN_Datasets_Grad_Dist_40_5_55/Test_Oliveira_BereaSinterGray_SAug_DNorm.h5",
    "../NN_Datasets_Grad_Dist_40_5_55/Test_Oliveira_BereaUpperGray_SAug_DNorm.h5", 
    "../NN_Datasets_Grad_Dist_40_5_55/Test_Oliveira_BereaBuff_SAug_DNorm.h5",
    "../NN_Datasets_Grad_Dist_40_5_55/Test_Oliveira_Berea_SAug_DNorm.h5",
    "../NN_Datasets_Grad_Dist_40_5_55/Test_Oliveira_Bentheimer_SAug_DNorm.h5",
    "../NN_Datasets_Grad_Dist_40_5_55/Test_Silveira_SphPore_SAug_DNorm.h5",
    "../NN_Datasets_Grad_Dist_40_5_55/Test_Silveira_SphGrain_SAug_DNorm.h5",
]

n_samples       = 30 
shuffle         = True 
# Base Output Directory 
RESULTS_DIR     = "../TestSpeedUp_Simulations_120/" 
os.makedirs(RESULTS_DIR, exist_ok=True) 

raw_file        = "domain.raw" 
shape           = (120, 120, 120) 
nproc           = (1,1,1)
device          = "cpu" 

# SLURM & Job Settings 
jobs_running    = 15 
CHUNK_SIZE      = max(n_samples//jobs_running,1) 
NTASKS          = nproc[0]*nproc[1]*nproc[2]
mem             = 20 #30 * 8 * shape[0]**3 / (1024**3) # GB
cpus_per_task   = 6 #2

LBPM_VERSION    = "lbpm/gpu/lbpm_fork_parallelinitdebug_7c32db3" 
PARTITION       = "all_gpu" 
GRES_STR        = "gpu:a100:1"

#LBPM_VERSION    = "lbpm/cpu/lbpm_init_07f0eef" 
#PARTITION       = "close_cpu" 
#GRES_STR        = "" 

MPI_PATH        = "mpirun" 
LBPM_EXEC       = "lbpm_permeability_simulator" 

visualization_interval  = 1000000000
tolerance               = -1
analysis_interval       = 200
timestep_max            = 500000

# ============================================================================== 
# MODEL INITIALIZATION (Done once for all datasets) 
# ============================================================================== 
danny_model = Extended_DannyKo() 
# Z- component
model_full_z_name = "./Trained_Models/NN_Trainning_26_August_2026_03-45PM_Job27376/model_LowerValidationLoss.pth"
# X- component
model_full_x_name = "./Trained_Models/NN_Trainning_26_August_2026_06-21PM_Job27380/model_LowerValidationLoss.pth"
# P- component
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
            if PARTITION=="close_cpu": c_f.write("#SBATCH --nodelist=node[008-020]\n") 
            c_f.write(f"#SBATCH --cpus-per-task={cpus_per_task}\n") 
            c_f.write(f"#SBATCH --mem={mem}G\n")  
             
             
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
                dirpath=grad_dir, 
                ux=ux_null, uy=uy_null, uz=uz_null, pr=pr_grad,
                nproc=nproc
            ) 
             
            sh.write_lbpm_db( 
                path=grad_dir, 
                db_name="lbpm.db", 
                domain_filename=f"../{raw_file}", 
                Start=True, tau=1.5, bc=3, din=1.0, dout=1.0 - 3*p_drop, 
                nproc=nproc, 
                n=(int(shape[2]/nproc[0]), int(shape[1]/nproc[1]), int(shape[0]/nproc[2])), 
                N=shape, 
                analysis_interval=analysis_interval, visualization_interval=visualization_interval, 
                tolerance=tolerance, out_format="vtk",timestep_max=timestep_max
            ) 
             
            # ================================================================== 
            # 2. NEURAL NETWORK INITIALIZATION SETUP 
            # ================================================================== 
            # Pad geometry before distance transform
            geometry_padded = sh.pad_geometry(geometry_uint8)
            geometry_edt = edt(geometry_padded > 0).astype("float32") 
            geometry_edt = torch.from_numpy(geometry_edt).unsqueeze(0).unsqueeze(0) 
             
            pred_padded = model.predict(geometry_edt) 
            pred_padded = vu.tensor_denorm(out=pred_padded, inp=geometry_edt) 
             
            # Unpad prediction back to original domain size
            pred = sh.unpad_geometry(pred_padded, geometry_bool.shape)
            
            uz_nn = pred[0,0].detach().cpu().numpy().astype(np.float64) 
            uy_nn = pred[0,1].detach().cpu().numpy().astype(np.float64) 
            ux_nn = pred[0,2].detach().cpu().numpy().astype(np.float64) 
            pr_nn = pred[0,3].detach().cpu().numpy().astype(np.float64)

            sh.write_start_raw( 
                dirpath=nn_dir, 
                ux=ux_nn, uy=uy_nn, uz=uz_nn, pr=pr_nn,
                nproc=nproc
            ) 
             
            sh.write_lbpm_db( 
                path=nn_dir, 
                db_name="lbpm.db", 
                domain_filename=f"../{raw_file}", 
                Start=True, tau=1.5, bc=3, din=1.0, dout=1.0 - 3*p_drop, 
                nproc=nproc, 
                n=(int(shape[2]/nproc[0]), int(shape[1]/nproc[1]), int(shape[0]/nproc[2])), 
                N=shape, 
                analysis_interval=analysis_interval, visualization_interval=visualization_interval, 
                tolerance=tolerance, out_format="vtk" ,timestep_max=timestep_max
            ) 
             
            # ================================================================== 
            # APPEND EXECUTIONS TO CHUNK SCRIPT (RELATIVE PATHS) 
            # ================================================================== 
            with open(chunk_script_path, "a") as c_f: 
                # Execution 1: Gradient-Started Run 
                c_f.write(f"echo \"--- Launching simulation for {sample_name} (Gradient-Initiated Run) ---\"\n") 
                c_f.write(f"cd {sample_name}/lbpm_grad_run\n") 
                c_f.write("echo \"Current Simulation: \" ${PWD##*/}\n") 
                c_f.write(f"{MPI_PATH} {LBPM_EXEC} lbpm.db\n") 
                c_f.write("cd ../../\n\n")  # Step back out to the dataset root folder 
                 
                # Execution 2: NN-Started Run 
                c_f.write(f"echo \"--- Launching simulation for {sample_name} (NN-Initiated Run) ---\"\n") 
                c_f.write(f"cd {sample_name}/lbpm_nn_run\n") 
                c_f.write("echo \"Current Simulation: \" ${PWD##*/}\n") 
                c_f.write(f"{MPI_PATH} {LBPM_EXEC} lbpm.db\n") 
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