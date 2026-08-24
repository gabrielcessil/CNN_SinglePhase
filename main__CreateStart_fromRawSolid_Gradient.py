import os
import numpy as np
from Utilities import start_handler as sh
from Utilities import velocity_usage as vu

# ==============================================================================
# CONFIGURATION
# ==============================================================================
data = {
      "./Example_SphPore/"           : (120, 120, 120),
      "./Example_Bentheimer/"        : (120, 120, 120)
}

raw_file = "domain.raw"

# ==============================================================================
# MAIN EXECUTION
# ============================================================================== 

for path, shape in data.items():
    print(f"Processing gradient initialization for {path}{raw_file}")
    
    # 1. Load geometry (True for fluid, False for solid)
    source_raw = os.path.join(path, raw_file)
    geometry = (np.fromfile(source_raw, dtype=np.uint8).reshape(shape) > 0)
    
    # 2. Calculate Pressure Drop
    tau    = 1.5
    Re     = 0.1
    Dens   = 1.0
    p_drop = vu.pressure_calculation(geometry, tau=tau, Re=Re, Dens=Dens)
    print(f"p_drop: {p_drop}")
    
    # 3. Generate Null Velocity and Linear Pressure Fields
    print(f"   -> Creating Start.00000 (Null Velocity, Linear Pressure)")
    
    # Velocities are 0 everywhere (implicitly satisfies the solid mask)
    uz_null = np.zeros(shape, dtype=np.float64)
    uy_null = np.zeros(shape, dtype=np.float64)
    ux_null = np.zeros(shape, dtype=np.float64)
    
    # Fix the gradient scaling to match din and dout
    pr_grad = np.zeros(shape, dtype=np.float64)
    z_steps = np.linspace(1.0/3.0, (1.0/3.0) - p_drop, shape[0])    
    
    for i in range(shape[0]):
        pr_grad[i, :, :] = z_steps[i]
        
    uz_null[~geometry] = 0.0
    uy_null[~geometry] = 0.0
    ux_null[~geometry] = 0.0
    pr_grad[~geometry] = 0.0
        
    # 4. Setup Output Directory
    out_folder = os.path.join(path, "Gradient_Init")
    os.makedirs(out_folder, exist_ok=True)
    
    # 5. Write Start File
    sh.write_start_raw(
        filename=os.path.join(out_folder, "Start.00000"),
        ux=ux_null, uy=uy_null, uz=uz_null, pr=pr_grad
    )
    
    # 6. Write Database File
    print(f"   -> Creating .db file")
    sh.write_lbpm_db(
        db_name = os.path.join(out_folder, "start_pressure.db"),
        path    = "",
        tau     = tau,
        bc      = 3,
        din     = 1.0,
        dout    = 1.0 - 3 * p_drop,
        nproc   = (1, 1, 1),
        n       = shape,
        N       = shape,
        analysis_interval = 50,
        tolerance         = 1e-8,
        out_format        = "silo",
        Start             = True
    )
    print(f"dout: {1.0 - 3 * p_drop}")
    
    # 7. Copy raw solid file into the new directory
    geometry.astype(np.uint8).tofile(os.path.join(out_folder, raw_file))
    
    print("   -> Done!\n")