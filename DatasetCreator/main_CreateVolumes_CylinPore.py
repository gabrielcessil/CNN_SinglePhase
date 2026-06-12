import numpy as np
import porespy as ps
import os
import utils
import zlib

def make_tubes_volume(MEAN_RADIUS, tubes_fill, solid_tubes, SHAPE, seed=0):
    
    phi_max     = 75
    theta_max   = 75
    length      = 80
    maxiter     = 10


    params_generic = {
        "shape":        SHAPE,
        "r":            MEAN_RADIUS,
        "phi_max":      phi_max,
        "theta_max":    theta_max,
        "length":       length,
        "maxiter":      maxiter,
    }
    vol = ps.generators.cylinders(**params_generic, porosity=1-tubes_fill)
    
    # Sphere are void
    if not solid_tubes: vol = 1-vol
    
    return vol
    


# --- Simulation Parameters ---
chunk_size      = 10   # Set for 1h of simulations (5 samples, 20 min per sample)
gres            = "gpu:a100" #"gpu:k40m"#"gpu:a100"
partition       = "all_gpu"
#gres            = None #"gpu:k40m"#"gpu:a100"
#partition       = "close_cpu"
n_proc          = 4
cpu             = 12 
gpu             = 64
use_low_prio        = False
include_allocation  = False 
lbpm_version        = "lbpm/gpu/lbpm_fork_965bd0d"
# --- Domains Parameters ---
DIM             = 120
SHAPE           = [DIM, DIM, DIM] # Shape must be a List for the function signature you provided
AXIS_OF_FLOW    = 0 
include_walls   = True
remove_isolated = False

dataset_type    = 'train' # 'train', 'valid', 'test'


##########################
# CREATE SPHERICAL PORES #
##########################

if dataset_type =='test':
    output_root = "../../Simulations/Test_CylinPore_120_120_120"
    N_SAMPLES       = 2

if dataset_type =='valid':
    output_root = "../../Simulations/Valid_CylinPore_120_120_120"
    N_SAMPLES       = 1 
    
if dataset_type =='train':
    output_root = "../../Simulations/Train_CylinPore_120_120_120"
    N_SAMPLES       = 10 
    
os.makedirs(output_root, exist_ok=True)

volumes         = []
folder_paths    = []

# CYLIN GRAIN
config_pairs = [
    
    (0.3, 14),
    (0.3, 16),
    (0.3, 18),
    
    (0.4, 14),
    (0.4, 16),
    (0.4, 18),
    
    (0.5, 14),
    (0.5, 16),
    (0.5, 18),
    
    (0.6, 14),
    (0.6, 16),
    (0.6, 18),
    
    (0.7, 12),
    (0.7, 14),
    (0.7, 16),
]

solid_spheres       = False
total_created       = 0
for SPHERES_FILL, MEAN_RADIUS in config_pairs:
    config_created = 0
    for n in range(N_SAMPLES*50):
        if config_created >= N_SAMPLES: break
        print(f"Attempt to create sample {total_created}")
        # Create volumes
        sample_id   = f"{dataset_type}_fill{SPHERES_FILL}_r{MEAN_RADIUS}_idx{config_created}_{n}"
        seed_n      = zlib.crc32(sample_id.encode("utf-8"))
        print(f"-->Filling {SPHERES_FILL*100}% with Cylinders, Mean Radius {MEAN_RADIUS} ({n}). Seed: ", seed_n)
        vol         = make_tubes_volume(MEAN_RADIUS, SPHERES_FILL, solid_spheres, SHAPE, seed=seed_n).astype(np.uint8)
        
        # Transform sample for simulation:
        if include_walls: vol = utils.add_enclusure_walls(vol)
            
        if remove_isolated: filt_vol = utils.remove_isolated_pores(vol)
        else: filt_vol = vol
        
        
        
        # Check porosity
        actual_porosity = np.sum(vol) / vol.size
        print(f"-->Actual Porosity: {actual_porosity*100:.2f}%")
        
        # Sanity checks
        if not utils.is_percolating(vol, axis=0):
            print(f"-->Sample do not percolate and got removed.")
            
        elif not utils.check_local_thickness(vol, min_radius=5, max_radius=17, target_percentage=70):
            print(f"-->Sample has geometry out of scope and got removed.")
        else:        
            print(f"-->Sample {total_created} got included.")
            
            folder_base = f"Sample_{total_created:05d}"
            folder_path = utils.create_simulation_pressure_condition(vol, 
                                                                     output_root, 
                                                                     folder_base, 
                                                                     n_proc=n_proc, 
                                                                     include_walls=include_walls)
            folder_paths.append(folder_path)
            total_created +=1
            config_created+=1
        print("-" * 30)

        

# Create .sh based on number of files
utils.generate_slurm_run_scripts_chunks(
    folder_paths    = folder_paths,
    n_proc          = n_proc,      
    gres            = gres,       
    output_root     = output_root,   
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
    output_root         = output_root,
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
                                  