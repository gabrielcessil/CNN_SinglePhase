import numpy as np
from pathlib import Path
from typing import Tuple, Union
from scipy.ndimage import distance_transform_edt
import pyvista as pv
import matplotlib.pyplot as plt
import re
import subprocess
import os
import math
import torch.nn.functional as F
import torch

from Utilities import velocity_usage as vu

def write_start_raw(
    dirpath: str,
    ux: np.ndarray,
    uy: np.ndarray,
    uz: np.ndarray,
    pr: np.ndarray,
    nproc: Tuple[int, int, int] = (1, 1, 1)
):
    """
    Writes Start.00000.raw as a DENSE 3D buffer.
    Input arrays must be (Nz, Ny, Nx) including Halos.
    Layout: [Ux(0,0,0), Uy(0,0,0), Uz(0,0,0), Pr(0,0,0), Ux(0,0,1)...]
    """
    
    if dirpath: os.makedirs(dirpath, exist_ok=True)
    
    nprocx, nprocy, nprocz  = nproc
    Nz, Ny, Nx              = ux.shape
    N                       = Nz * Ny * Nx
    nranks                  = nprocx * nprocy * nprocz
    local_nx                = Nx // nprocx
    local_ny                = Ny // nprocy
    local_nz                = Nz // nprocz
    
    for rank in range(nranks):
        rank_x = rank % nprocx
        rank_y = (rank // nprocx) % nprocy
        rank_z = rank // (nprocx * nprocy)
        
        x0 = rank_x * local_nx
        x1 = (rank_x + 1) * local_nx
        y0 = rank_y * local_ny
        y1 = (rank_y + 1) * local_ny
        z0 = rank_z * local_nz
        z1 = (rank_z + 1) * local_nz
        
        ux_local = ux[z0:z1, y0:y1, x0:x1]
        uy_local = uy[z0:z1, y0:y1, x0:x1]
        uz_local = uz[z0:z1, y0:y1, x0:x1]
        pr_local = pr[z0:z1, y0:y1, x0:x1]
                
        dense_grid = np.stack((ux_local, uy_local, uz_local, pr_local), axis=-1) 
        buffer     = dense_grid.astype(np.float64) 
        
        filename = os.path.join(dirpath, f"Start.{rank:05d}.raw")        
        with open(filename, "wb") as f:
            buffer.tofile(f)


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
    timestep_max: int = 100000000,
    tolerance: float = 1e-6,
    Start: bool = True,
    # Domain
    domain_filename:str = "domain.raw",
    read_type:      str = "8bit",
    nproc:          Tuple[int, int, int] = (1, 1, 4),
    n:              Tuple[int, int, int] = (256, 256, 128),
    N:              Tuple[int, int, int] = (256, 256, 512),
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
    analysis_interval:          int = 1000,
    subphase_analysis_interval: int = 5000,
    n_threads:                  int = 0,
    visualization_interval:     int = 5000,
    restart_interval:           int = 100_000_000,
    restart_file:               str = "Restart",
    out_format:                 str = "vtk"
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
   Start       = {b(Start)}
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
   format            = "{out_format}"
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
    # If `path` is a directory or lacks a suffix, write inside it
    if p.suffix == "" or p.is_dir():
        p.mkdir(parents=True, exist_ok=True)
        p = p / db_name
    else:
        p.parent.mkdir(parents=True, exist_ok=True)

    p.write_text(text, encoding="utf-8")
    return text



# Shape must be (Z,Y,X). No channel or batch dimension
def pad_geometry(np_array, shape=None):

    # Input is always a NumPy array with shape (Z, Y, X)
    current_z, current_y, current_x = np_array.shape

    # Calculate desired shape
    if shape is None:
        target_z = 2 ** math.ceil(math.log2(current_z))
        target_y = 2 ** math.ceil(math.log2(current_y))
        target_x = 2 ** math.ceil(math.log2(current_x))
    else:
        target_z, target_y, target_x = shape

    # Calculate padding at the end of each dimension
    pad_z = target_z - current_z
    pad_y = target_y - current_y
    pad_x = target_x - current_x

    tensor = torch.from_numpy(np_array)
    tensor = tensor.unsqueeze(0).unsqueeze(0) # Add batch and dimensions channels

    # Z: reflect padding on the back
    if pad_z > 0:
        tensor = F.pad(
            tensor,
            (0, 0, 0, 0, 0, pad_z),
            mode='reflect'
        )

    # Y/X: zero padding on bottom/right
    if pad_y > 0 or pad_x > 0:
        tensor = F.pad(
            tensor,
            (0, pad_x, 0, pad_y, 0, 0),
            mode='constant',
            value=0.0
        )

    # Convert back to NumPy
    return tensor.squeeze(0).squeeze(0).numpy()


def unpad_geometry(tensor, original_shape):
    return tensor[..., :original_shape[0], :original_shape[1], :original_shape[2]]