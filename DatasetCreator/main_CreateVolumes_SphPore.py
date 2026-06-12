import numpy as np
import porespy as ps
import os
import scipy.stats as sps
import utils
import zlib

def make_spheres_volume(MEAN_RADIUS, SPHERES_FILL, solid_spheres, SHAPE, seed=0):
    
    MIN_RADIUS          = min(MEAN_RADIUS/6,4)
    # Standard deviation estimation
    StdDev              = MEAN_RADIUS/3
    # Define the normal distribution object (Mean=5, StdDev=3)
    radius_distribution = sps.norm(loc=MEAN_RADIUS, scale=StdDev)
    
    # Call the function using the specific signature you requested
    vol = ps.generators.polydisperse_spheres(
        shape   =SHAPE,
        porosity=1-SPHERES_FILL,
        dist    =radius_distribution,   # Pass the statistical distribution object
        r_min   =MIN_RADIUS,            # Ensure the smallest generated sphere is at least 1 voxel
        seed    =seed                   # for reproducibility
    )
    # Sphere are void
    if not solid_spheres: vol = 1-vol
    
    return vol
    


# --- Simulation Parameters ---
gres                = "gpu:a100" #"gpu:k40m"#"gpu:a100"
partition           = "all_gpu"
#gres            = None #"gpu:k40m"#"gpu:a100"
#partition       = "close_cpu"
n_proc              = 4
cpu                 = 12 
gpu                 = 64
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
    output_root = "../../Simulations/Test_SphPore_120_120_120"
    N_SAMPLES       = 2

if dataset_type =='valid':
    output_root = "../../Simulations/Valid_SphPore_120_120_120"
    N_SAMPLES       = 1 
    
if dataset_type =='train':
    output_root = "../../Simulations/Train_SphPore_120_120_120"
    N_SAMPLES       = 10 
    
config_pairs = [
    
    (0.5, 8),
    (0.5, 10),
    (0.5, 12),
    (0.5, 14),
    
    (0.6, 8),
    (0.6, 10),
    (0.6, 12),
    (0.6, 14),
    
    (0.7, 6),
    (0.7, 8),
    (0.7, 10),
    (0.7, 12),
    
    (0.8, 6),
    (0.8, 8),
    (0.8, 10),
    (0.8, 12),

]

os.makedirs(output_root, exist_ok=True)
volumes             = []
solid_spheres       = False
folder_paths = []
total_created = 0
for SPHERES_FILL, MEAN_RADIUS in config_pairs: 
    config_created = 0
    for n in range(N_SAMPLES*50):
        if config_created >= N_SAMPLES: break
        print(f"Attempt to create sample {total_created}")
        
        # Create volumes
        sample_id   = f"{dataset_type}_fill{SPHERES_FILL}_r{MEAN_RADIUS}_idx{config_created}_{n}"
        seed_n      = zlib.crc32(sample_id.encode("utf-8"))
        print(f"-->Filling {SPHERES_FILL*100}% with Sphere, Mean Radius {MEAN_RADIUS} ({n}). Seed: ", seed_n)
        vol         = make_spheres_volume(MEAN_RADIUS, SPHERES_FILL, solid_spheres, SHAPE, seed=seed_n).astype(np.uint8)
        
        # Transform sample for simulation:
        if include_walls: vol = utils.add_enclusure_walls(vol)
            
        if remove_isolated: filt_vol = utils.remove_isolated_pores(vol)
        else: filt_vol = vol
        
        
        # Check porosity
        actual_porosity = np.sum(filt_vol) / filt_vol.size
        print(f"-->Actual Porosity: {actual_porosity*100:.2f}%")
        
        # Sanity checks
        if not utils.is_percolating(filt_vol, axis=0):
            print(f"-->Sample do not percolate and got removed.")
        elif not utils.check_local_thickness(filt_vol, min_radius=5, max_radius=17, target_percentage=70):
            print(f"-->Sample has geometry out of scope and got removed.")
        else:        
            print(f"-->Sample {total_created} got included.")
            folder_base = f"Sample_{total_created:05d}"
            folder_path = utils.create_simulation_pressure_condition(   filt_vol, 
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