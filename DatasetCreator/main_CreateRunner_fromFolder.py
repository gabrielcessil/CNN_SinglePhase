import os
import glob
import numpy as np
import utils
from pathlib import Path
import zlib


BASE_DIRECTORIES = [
    #"/home/gabriel/remote/hal/dissertacao/Simulations/Train_Danny_SphPore_120_120_120/",
    #"/home/gabriel/remote/hal/dissertacao/Simulations/Test_Danny_SphPore_120_120_120/",
    "/home/gabriel/Desktop/Dissertacao/GradSimulations/Train_Danny_SphPore_120_120_120/",
    ]

# --- Hardware Parameters ---
chunk_size          = 10   # Set for 1h of simulations (5 samples, 20 min per sample)
#gres                = "gpu:a100" #"gpu:k40m"#"gpu:a100"
#partition           = "all_gpu"
gres                = None #"gpu:k40m"#"gpu:a100"
partition           = "close_cpu"
n_proc              = 4
cpu                 = 12 
gpu                 = 64
use_low_prio        = False
include_allocation  = False 
lbpm_version        = "lbpm/cpu/lbpm_fork_2016010"

# --- Domains Parameters ---
RAW_FILENAME    = "mod_domain.raw"
VOL_SHAPE       = (120, 120, 120)
VOL_DTYPE       = np.uint8

# --- Simulation Parameters
Re   = 0.1
tau  = 1.5
Dens = 1.0
tolerance = 1e-4


folder_paths = []
for BASE_DIR in BASE_DIRECTORIES:
    print("Creating runners for ", BASE_DIR)
    
    raw_files = utils.find_raw_in_folder(BASE_DIR, RAW_FILENAME)

    # Create .db based on geometry
    for file_name in raw_files:
        vol = np.fromfile(file_name, dtype=np.uint8).reshape(VOL_SHAPE).astype(np.uint8)

        dP = utils.pressure_calculation(           
                vol,
                tau     = tau,
                Re      = Re,
                Dens    = Dens
            )
        
        timestep_max = utils.timestep_calculation(    
                matriz_binaria  =vol,
                tau             =tau,
                Re              =Re,
                Dens            =Dens,
                safety_factor   =10.0
                )
        
        # --- Save 3D domain as .raw ---
        folder_path = os.path.dirname(file_name)
        folder_paths.append(folder_path)
        
        utils.write_lbpm_db(
                path      = folder_path, 
                tau       = tau,
                bc        = 3,
                din       = 1.0+dP*3,
                dout      = 1.0,
                nproc     = (1, 1, n_proc),
                n         = (vol.shape[2], vol.shape[1], int(vol.shape[0]/n_proc)),
                N         = (vol.shape[2], vol.shape[1], vol.shape[0]),
                tolerance = 1e-4,
                domain_filename           = RAW_FILENAME,
                analysis_interval         = 1000, 
                visualization_interval    =timestep_max, 
                timestep_max              =timestep_max,
                subphase_analysis_interval=timestep_max,
                restart_interval          =timestep_max
                )
        
        utils.write_data_ini(
            path              = folder_path, 
            number_of_steps   = timestep_max,
            filename          = "domain.raw",
            size_x            = vol.shape[2],
            size_y            = vol.shape[1],
            size_z            = vol.shape[0],
            analysis_interval = 1000,
            tau               = tau,
            tolerance         = tolerance*100, # Percentual tolerance
            axis              = 2,  
            dp                = -dP,
            mDa               = True
        )
    
    total_created = len(folder_paths)
    print(f"Found {total_created} valid samples.")
    
    # Create .sh based on number of files
    utils.generate_slurm_run_scripts_chunks(
        folder_paths    = folder_paths,
        n_proc          = n_proc,      
        gres            = gres,       
        output_root     = BASE_DIR,   
        samples_per_job = 10, 
        cpu             = cpu,         
        gpu             = gpu,
        partition       = partition,                        
        dispatcher_name = f"Run_LBM_{0}_{total_created}.sh",
        lbpm_version    = lbpm_version,
        include_allocation  = include_allocation       
    )

    utils.generate_slurm_run_scripts_chunks_GRADLBM(
        folder_paths        = folder_paths,
        n_proc              = 1,
        output_root         = BASE_DIR,
        samples_per_job     = 20,
        partition           = "close_cpu",
        nodelist            = "node[008-020]",
        cpu_per_sim         = 1, 
        mem_gb_per_sim      = 6,
        dispatcher_name     = f"Run_GRAD_{0}_{total_created}.sh",
        lbm_folder          = "/home/gabriel.silveira/GRAD_LBM/",
        ini_name            = "grad.ini",
        chain_launchers     = False,
    )   