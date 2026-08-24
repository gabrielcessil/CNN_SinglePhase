import re
import subprocess
from pathlib import Path
import numpy as np
import pyvista as pv
from typing import List, Tuple
import matplotlib.pyplot as plt
import shutil

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'Computer Modern Roman']

# ==============================================================================
# LBM HELPER FUNCTIONS
# ==============================================================================

def find_all_vis_silo_files(base_path: str) -> List[str]:
    p = Path(base_path)
    if not p.exists():
        raise FileNotFoundError(f"Base path does not exist: {base_path}")

    vis_pattern = re.compile(r'^vis(\d+)$')
    vis_folders = []
    
    for folder in p.iterdir():
        if folder.is_dir():
            match = vis_pattern.match(folder.name)
            if match:
                vis_folders.append((int(match.group(1)), folder))

    if not vis_folders:
        print(f"Warning: No 'vis<number>' folders found in {base_path}")
        return []

    vis_folders.sort(key=lambda x: x[0])
    
    pvti_files_found = []
    converter_path = "/home/gabriel/Desktop/LBPM_Install/converter_silo_vti/silo2vti"

    for num, folder in vis_folders:
        silo_file = folder / "summary.silo"
        pvti_file = folder / "summary.pvti"

        if silo_file.exists():            
            try:
                subprocess.run(
                    [converter_path, "summary.silo", "summary.pvti"],
                    cwd=str(folder), check=True, capture_output=True, text=True
                )
            except subprocess.CalledProcessError as e:
                print(f"Warning: Conversion failed for vis{num}: {e.stderr}")
                continue 
            
            if pvti_file.exists():
                pvti_files_found.append(str(pvti_file))
        else:
            print(f"Warning: No summary.silo found in {folder.name}, skipping.")

    return pvti_files_found


def cleanup_vis_folders(base_path: str):
    p = Path(base_path)
    if not p.exists():
        return
        
    print(f"Cleaning up old results in {base_path}...")
    vis_pattern = re.compile(r'^vis(\d+)$')
    for folder in p.iterdir():
        if folder.is_dir() and vis_pattern.match(folder.name):
            shutil.rmtree(folder)
            
def write_start_raw(filename: str, ux: np.ndarray, uy: np.ndarray, uz: np.ndarray, pr: np.ndarray):
    Nz, Ny, Nx = ux.shape
    N = Nz * Ny * Nx
    print(f"   -> Writing DENSE start file: {Nx}x{Ny}x{Nz} ({N} voxels)")
    dense_grid = np.stack((ux, uy, uz, pr), axis=-1) 
    buffer = dense_grid.astype(np.float64) 
    with open(filename+".raw", "wb") as f:
        buffer.tofile(f)

def write_domain_raw(path: str, domain_array: np.ndarray, filename: str = "domain.raw") -> str:
    p = Path(path) if path else Path(".")
    p.mkdir(parents=True, exist_ok=True)
    out_path = p / filename
    np.asarray(domain_array, dtype=np.uint8).tofile(out_path)
    print(f"   -> domain.raw written to: {out_path}")
    return str(out_path)

def write_lbpm_db(
    path: str,
    *,
    db_name:    str = "simulation.db",   # used if `path` is a directory
    bc:         int = 0,
    din:        float = 1.0,
    dout:       float = 1.0,
    fz:         float = 0.0,
    fx:         float = 0.0,
    fy:         float = 0.0,
    tau:        float = 1.5,
    timestep_max: int = 50000,
    tolerance: float = 1e-4,
    # Domain
    domain_filename:str = "domain.raw",
    read_type:      str = "8bit",
    nproc:          Tuple[int, int, int] = (1, 1, 1),
    n:              Tuple[int, int, int] = (256, 256, 256),
    N:              Tuple[int, int, int] = (256, 256, 256),
    offset:         Tuple[int, int, int] = (0, 0, 0),
    voxel_length:   float = 1.0,
    read_values:    Tuple[int, int] = (0, 1),
    write_values:   Tuple[int, int] = (0, 1),
    inlet_layers:   Tuple[int, int, int] = (0, 0, 0),
    outlet_layers:  Tuple[int, int, int] = (0, 0, 0),
    # Visualization
    write_silo:     bool = True,
    save_8bit_raw:  bool = True,
    save_phase_field: bool = True,
    save_pressure:  bool = True,
    save_velocity:  bool = True,
    # Analysis
    analysis_interval:          int = 100,
    subphase_analysis_interval: int = 100_000_000,
    n_threads:                  int = 0,
    visualization_interval:     int = 100_000_000,
    restart_interval:           int = 100_000_000,
    restart_file:               str = "Restart",
) -> str:
    def tsv3(v): return f"{v[0]}, {v[1]}, {v[2]}"
    def tsv2(v): return f"{v[0]}, {v[1]}"
    def b(v):    return "true" if v else "false"
    def ffmt(x): return f"{x:.6g}"

    text = f"""MRT {{
   tau         = {ffmt(tau)}
   din         = {din}   // inlet density (controls pressure)
   dout        = {dout}  // outlet density (controls pressure)
   F           = {ffmt(fx)}, {ffmt(fy)}, {ffmt(fz)}   // Fx, Fy, Fz
   timestepMax = {timestep_max}
   tolerance   = {ffmt(tolerance)}
}}
Domain {{
   Filename = "{domain_filename}"
   ReadType = "{read_type}"      // data type

   nproc = {tsv3(nproc)}
   n     = {tsv3(n)}
   N     = {tsv3(N)}

   offset         = {tsv3(offset)} // offset to read sub-domain
   voxel_length   = {ffmt(voxel_length)}     // voxel length (in microns)
   ReadValues     = {tsv2(read_values)}    // labels within the original image
   WriteValues    = {tsv2(write_values)}    // associated labels to be used by LBPM (0:solid, 1..N:fluids)
   BC             = {bc}       // boundary condition type (0 for periodic)
   InletLayers    = {tsv3(inlet_layers)}   // specify layers along the inlet
   OutletLayers   = {tsv3(outlet_layers)}  // specify layers along the outlet
}}
Visualization {{
   format            = "vtk"
   write_silo        = {b(write_silo)}     // SILO databases with assigned variables
   save_8bit_raw     = {b(save_8bit_raw)}  // labeled 8-bit binary files with phase assignments
   save_phase_field  = {b(save_phase_field)}  // phase field within SILO database
   save_pressure     = {b(save_pressure)}    // pressure field within SILO database
   save_velocity     = {b(save_velocity)}    // velocity field within SILO database
}}
Analysis {{
   analysis_interval             = {analysis_interval}        // logging interval for timelog.csv
   subphase_analysis_interval    = {subphase_analysis_interval}  // logging interval for subphase.csv
   N_threads                     = {n_threads}                // number of analysis threads (GPU version only)
   visualization_interval        = {visualization_interval}   // interval to write visualization files
   restart_interval              = {restart_interval}         // interval to write restart file
   restart_file                  = "{restart_file}"           // base name of restart file
}}
"""
    p = Path(path)
    if p.suffix == "" or p.is_dir():
        p.mkdir(parents=True, exist_ok=True)
        p = p / db_name
    else:
        p.parent.mkdir(parents=True, exist_ok=True)

    p.write_text(text, encoding="utf-8")
    return text

def write_and_submit_slurm(output_path: str, sim_mode: int, job_name: str):
    """Writes the SLURM batch script and submits it."""
    slurm_script = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --partition=gpu

cd {output_path}
/home/gabriel/Desktop/LBPM_Install/mpi/bin/mpirun -np 1 /home/gabriel/Desktop/LBPM_Install/LBPM_dir/tests/lbpm_single_phase start.db --init {sim_mode}
"""
    script_path = Path(output_path) / "submit.sh"
    script_path.write_text(slurm_script)
    
    try:
        subprocess.run(["sbatch", "submit.sh"], cwd=output_path, check=True)
        print(f"   -> Job submitted: {job_name}")
    except subprocess.CalledProcessError as e:
        print(f"   -> Failed to submit job for {job_name}: {e}")
    except FileNotFoundError:
        print("   -> 'sbatch' command not found. Are you on the cluster login node?")


def post_process_experiment(N, domain, ux_a, uy_a, uz_a, pr_a, analyzed_position, output_path):
    filenames = find_all_vis_silo_files(output_path)
    
    ux_r_t = []
    pr_r_t = []
    kin_t = []
    err_L2_t = []
    
    fluid_mask = domain == 1
    n_fluid = np.sum(fluid_mask)

    for filename in filenames:
        mesh = pv.read(filename)
        uz_n = mesh['Velocity_z'].reshape((N, N, N))
        uy_n = mesh['Velocity_y'].reshape((N, N, N)) 
        ux_n = mesh['Velocity_x'].reshape((N, N, N))
        pr_n = mesh['Pressure'].reshape((N, N, N))
        
        # Pointwise history
        ux_r_t.append(ux_n[analyzed_position])
        pr_r_t.append(pr_n[analyzed_position])
        
        # Total Kinetic energy over fluid domain
        kinetic_sum = np.sum(ux_n[fluid_mask]**2 + uy_n[fluid_mask]**2 + uz_n[fluid_mask]**2)
        kin_t.append(kinetic_sum / n_fluid)
        
        # L2 Error against analytical solution (fluid nodes only)
        err_ux = ux_n[fluid_mask] - ux_a[fluid_mask]
        err_uy = uy_n[fluid_mask] - uy_a[fluid_mask]
        err_uz = uz_n[fluid_mask] - uz_a[fluid_mask]
        l2_err = np.sqrt(np.mean(err_ux**2 + err_uy**2 + err_uz**2))
        err_L2_t.append(l2_err)
        
    return np.array(ux_r_t), np.array(pr_r_t), np.array(kin_t), np.array(err_L2_t)
    
# ==============================================================================
# WORKFLOW CONTROLS & CONFIGURATIONS
# ==============================================================================

# ---------------------------------------------------------
# Set SUBMIT_JOBS to True to write inputs and call sbatch.
# Set POST_PROCESS to True ONLY after jobs have finished.
# ---------------------------------------------------------
SUBMIT_JOBS  = True
POST_PROCESS = False

N_values     = [16, 32, 64] # Test sizes
U0           = 0.01      
P0           = 1.0/3.0   
tau          = 0.65      
save_interval = 2
n_timesteps  = 100

# ==============================================================================
# MAIN WORKFLOW LOOP
# ==============================================================================

# Data storage for post-processing
results = {}

for N in N_values:
    print(f"\n{'='*50}")
    print(f"PROCESSING RESOLUTION: N = {N}")
    print(f"{'='*50}")

    base_dir     = f"./Stokes_Sphere_Simulations/N_{N}"
    path_cte_eq  = f"{base_dir}/cte_eq"
    path_ini_eq  = f"{base_dir}/ini_eq"
    path_cte_neq = f"{base_dir}/cte_neq"
    path_ini_neq = f"{base_dir}/ini_neq"

    # Scale the sphere proportional to N (Radius = N/8)
    R = N * (8.0 / 64.0)
    cx, cy, cz = N//2, N//2, N//2
    analyzed_position = (cz, cy, cx + int(R) + 3) # Z, Y, X

    # --------------------------------------------------------------------------
    # 1. ANALYTICAL SOLUTIONS & DOMAIN
    # --------------------------------------------------------------------------
    z = np.arange(N)
    y = np.arange(N)
    x = np.arange(N)
    ZZ, YY, XX = np.meshgrid(z, y, x, indexing='ij')

    xc, yc, zc = XX - cx, YY - cy, ZZ - cz
    r  = np.sqrt(xc**2 + yc**2 + zc**2)
    r_safe = np.where(r == 0, 1e-10, r)

    domain = np.ones((N, N, N), dtype=np.uint8)
    domain[r <= R] = 0 

    term4 = (3 * R) / (4 * r_safe**3) - (3 * R**3) / (4 * r_safe**5)
    ux_a = U0 * (1 - (3 * R) / (4 * r_safe) - (R**3) / (4 * r_safe**3) - (xc**2) * term4)
    uy_a = -U0 * xc * yc * term4
    uz_a = -U0 * xc * zc * term4
    nu = (tau - 0.5) / 3.0
    pr_a = P0 - (3 * nu * R * U0 * xc) / (2 * r_safe**3)

    # Apply solid boundaries
    ux_a[domain == 0], uy_a[domain == 0], uz_a[domain == 0] = 0.0, 0.0, 0.0
    pr_a[domain == 0] = P0
    
    ux_a_pt = ux_a[analyzed_position]
    pr_a_pt = pr_a[analyzed_position]

    # Initial uniform fields
    pr_cte = np.ones_like(ux_a) * P0
    ux_cte = np.ones_like(ux_a) * U0
    uy_cte, uz_cte = np.zeros_like(ux_a), np.zeros_like(ux_a)

    # --------------------------------------------------------------------------
    # 2. JOB SUBMISSION PHASE
    # --------------------------------------------------------------------------
    if SUBMIT_JOBS:
        cleanup_vis_folders(path_cte_eq)
        cleanup_vis_folders(path_ini_eq)
        cleanup_vis_folders(path_cte_neq)
        cleanup_vis_folders(path_ini_neq)

        cases = [
            ("Eq_CTE",  path_cte_eq,  uz_cte, uy_cte, ux_cte, pr_cte, 1),
            ("Neq_CTE", path_cte_neq, uz_cte, uy_cte, ux_cte, pr_cte, 2),
            ("Eq_INI",  path_ini_eq,  uz_a,   uy_a,   ux_a,   pr_a,   1),
            ("Neq_INI", path_ini_neq, uz_a,   uy_a,   ux_a,   pr_a,   2),
        ]

        for case_name, out_path, uz_ini, uy_ini, ux_ini, pr_ini, sim_mode in cases:
            Path(out_path).mkdir(parents=True, exist_ok=True)
            write_domain_raw(out_path, domain)
            write_start_raw(str(Path(out_path) / "Start.00000"), ux_ini, uy_ini, uz_ini, pr_ini)
            
            # Write configuration database
            write_lbpm_db(
                path=out_path,
                db_name="start.db",
                tau=tau,
                bc=0,
                timestep_max=n_timesteps,
                nproc=(1, 1, 1),
                n=(N, N, N),
                N=(N, N, N),
                visualization_interval=save_interval,
                analysis_interval=save_interval
            )
            
            # Write and submit Slurm script
            job_name = f"LBM_N{N}_{case_name}"
            write_and_submit_slurm(out_path, sim_mode, job_name)

    # --------------------------------------------------------------------------
    # 3. POST-PROCESSING PHASE
    # --------------------------------------------------------------------------
    if POST_PROCESS:
        kin_a_tot = np.sum(ux_a[domain==1]**2 + uy_a[domain==1]**2 + uz_a[domain==1]**2) / np.sum(domain==1)
        
        paths = {
            'Eq_CTE': path_cte_eq,
            'Neq_CTE': path_cte_neq,
            'Eq_INI': path_ini_eq,
            'Neq_INI': path_ini_neq
        }
        
        results[N] = {'kin_a_tot': kin_a_tot, 'ux_a_pt': ux_a_pt, 'pr_a_pt': pr_a_pt}
        
        for case_name, out_path in paths.items():
            print(f"Processing {case_name}...")
            ux_t, pr_t, kin_t, err_L2_t = post_process_experiment(
                N, domain, ux_a, uy_a, uz_a, pr_a, analyzed_position, out_path)
            
            # Normalize kinetic energy
            if len(kin_t) > 0:
                kin_t = kin_t / kin_a_tot
                
            results[N][case_name] = {
                'ux_t': ux_t, 'pr_t': pr_t, 'kin_t': kin_t, 'err_t': err_L2_t
            }

# ==============================================================================
# PLOTS (ONLY RUNS IF POST_PROCESS IS TRUE)
# ==============================================================================
if POST_PROCESS and N_values:
    # Example plotting just for the LAST N processed (or you can loop through them)
    N_plot = N_values[-1]
    res = results[N_plot]
    
    if len(res['Eq_CTE']['kin_t']) > 0:
        timesteps = np.arange(len(res['Eq_CTE']['kin_t'])) * save_interval
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 12), dpi=300)

        # ax1: Velocity X at point
        ax1.axhline(res['ux_a_pt'], color='blue', linestyle=':', label=r'Analytical solution')
        ax1.plot(timesteps, res['Eq_CTE']['ux_t'], '--', color='black', label=r'F.eq. / Uniform Flow')
        ax1.plot(timesteps, res['Eq_INI']['ux_t'], '-', color='black', label=r'F.eq. / Analytical Init')
        ax1.plot(timesteps, res['Neq_CTE']['ux_t'],'--', color='#16A085', label=r'F.eq.+ F.neq. / Uniform Flow')
        ax1.plot(timesteps, res['Neq_INI']['ux_t'],'-', color='#16A085', label=r'F.eq.+ F.neq. / Analytical Init')
        ax1.set_ylabel(f'Velocity X at point') 

        # ax2: Normalized Kinetic Energy
        ax2.axhline(1.0, color='blue', linestyle=':', label=r'Analytical solution')
        ax2.plot(timesteps, res['Eq_CTE']['kin_t'], '--', color='black')
        ax2.plot(timesteps, res['Eq_INI']['kin_t'], '-', color='black')
        ax2.plot(timesteps, res['Neq_CTE']['kin_t'],'--', color='#16A085')
        ax2.plot(timesteps, res['Neq_INI']['kin_t'],'-', color='#16A085')
        ax2.set_ylabel(r'Normalized Kinetic Energy $K(t) / K_{analytical}$') 

        # ax3: L2 Error
        ax3.plot(timesteps, res['Eq_CTE']['err_t'], '--', color='black')
        ax3.plot(timesteps, res['Eq_INI']['err_t'], '-', color='black')
        ax3.plot(timesteps, res['Neq_CTE']['err_t'],'--', color='#16A085')
        ax3.plot(timesteps, res['Neq_INI']['err_t'],'-', color='#16A085')
        ax3.set_ylabel(r'RMSE $L_2$ Velocity Error')
        ax3.set_yscale('log')

        # ax4: Pressure at point
        ax4.axhline(res['pr_a_pt'], color='blue', linestyle=':', label=r'Analytical solution')
        ax4.plot(timesteps, res['Eq_CTE']['pr_t'],  '--', color='black')
        ax4.plot(timesteps, res['Eq_INI']['pr_t'],  '-', color='black')
        ax4.plot(timesteps, res['Neq_CTE']['pr_t'], '--', color='#16A085')
        ax4.plot(timesteps, res['Neq_INI']['pr_t'], '-', color='#16A085')
        ax4.set_ylabel(f'Pressure at point') 

        FONT_LABEL  = 14
        FONT_TICKS  = 12
        FONT_LEGEND = 12

        for ax in [ax1, ax2, ax3, ax4]:
            ax.set_box_aspect(1)
            if ax == ax1:
                ax.legend(frameon=True, fontsize=FONT_LEGEND, loc='best')
            
            ax.xaxis.label.set_size(FONT_LABEL)
            ax.yaxis.label.set_size(FONT_LABEL)
            ax.tick_params(axis='both', which='major', labelsize=FONT_TICKS)
            ax.grid(True, alpha=0.3)
            ax.set_xlabel('Time')

        plt.subplots_adjust(wspace=0.3, hspace=0.3) 
        plt.savefig(f'StokesSphereBenchmark_N{N_plot}.pdf', bbox_inches='tight')
        plt.show()
    else:
        print("No post-processing data found. Check if the jobs have finished successfully.")