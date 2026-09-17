import numpy as np
from scipy.ndimage import distance_transform_edt as edt
import torch
import os

from Architectures.Unet import Extended_DannyKo
from Architectures.Models import SubModels_Composition

from Utilities import start_handler as sh
from Utilities import velocity_usage as vu


# ==============================================================================
# CONFIGURATION
# ==============================================================================

paths = [
    "./DEBUG_PARALLEL_INIT/2_2_2/"
]

raw_file = "domain.raw"
shape = (120, 120, 120)       # Geometry shape (Z,Y,X)
device = "cpu"
nproc = (2, 2, 2)             # LBPM parallelization


# ==============================================================================
# LOADING PRE-TRAINED MODELS
# ==============================================================================

print("Initializing Neural Network Models...")

danny_model = Extended_DannyKo()

# Z-component
model_full_z_name = (
    "./Trained_Models/"
    "NN_Trainning_26_August_2026_03-45PM_Job27376/"
    "model_LowerValidationLoss.pth"
)

# X-component
model_full_x_name = (
    "./Trained_Models/"
    "NN_Trainning_26_August_2026_06-21PM_Job27380/"
    "model_LowerValidationLoss.pth"
)

# P-component
model_full_p_name = (
    "./Trained_Models/"
    "NN_Trainning_26_August_2026_03-47PM_Job27377/"
    "model_LowerValidationLoss.pth"
)

model = SubModels_Composition(
    main_model=danny_model,
    z_name=model_full_z_name,
    x_name=model_full_x_name,
    p_name=model_full_p_name,
    device=device,
    is_eval=True
)


# ==============================================================================
# PARALLELIZATION SETUP
# ==============================================================================

nprocx, nprocy, nprocz = nproc

nranks = nprocx * nprocy * nprocz

Nx, Ny, Nz = shape

if Nx % nprocx != 0:
    raise ValueError(
        f"X dimension {Nx} is not divisible by nprocx={nprocx}"
    )

if Ny % nprocy != 0:
    raise ValueError(
        f"Y dimension {Ny} is not divisible by nprocy={nprocy}"
    )

if Nz % nprocz != 0:
    raise ValueError(
        f"Z dimension {Nz} is not divisible by nprocz={nprocz}"
    )

local_nx = Nx // nprocx
local_ny = Ny // nprocy
local_nz = Nz // nprocz

print()
print("Parallelization:")
print(f"   -> nproc : {nproc}")
print(f"   -> nranks: {nranks}")
print(f"   -> local dimensions (X,Y,Z): "
      f"({local_nx},{local_ny},{local_nz})")
print()


# ==============================================================================
# MAIN
# ==============================================================================

for path in paths:

    print("=" * 78)
    print(f"Processing: {path}")
    print("=" * 78)

    # --------------------------------------------------------------------------
    # 1. READ GEOMETRY
    # --------------------------------------------------------------------------

    geometry_uint8 = np.fromfile(
        path + raw_file,
        dtype=np.uint8
    ).reshape(shape)

    # Explicit binary geometry
    geometry_bool = geometry_uint8 > 0

    print(f"   -> Original geometry shape: {geometry_bool.shape}")
    print(f"   -> Fluid voxels: {np.count_nonzero(geometry_bool)}")
    print(f"   -> Solid voxels: {np.count_nonzero(~geometry_bool)}")


    # --------------------------------------------------------------------------
    # 2. PAD GEOMETRY
    # --------------------------------------------------------------------------

    geometry_padded = sh.pad_geometry(geometry_uint8)

    print(f"   -> Padded geometry shape:   {geometry_padded.shape}")

    # Debug: save exactly the geometry given to the NN
    #padded_raw = os.path.join(
    #    path,
    #    "geometry_padded.raw"
    #)
    #
    #geometry_padded.astype(np.uint8).tofile(padded_raw)
    #print(f"   -> Saved padded geometry:   {padded_raw}")


    # --------------------------------------------------------------------------
    # 3. DISTANCE TRANSFORM
    # --------------------------------------------------------------------------

    geometry_edt = edt(
        geometry_padded > 0
    ).astype("float32")

    print(f"   -> EDT shape:               {geometry_edt.shape}")

    # Convert:
    #
    # NumPy:  (Z,Y,X)
    # Tensor: (B,C,Z,Y,X)

    geometry_edt = torch.from_numpy(
        geometry_edt
    ).unsqueeze(0).unsqueeze(0)

    print(f"   -> EDT tensor shape:        {geometry_edt.shape}")


    # --------------------------------------------------------------------------
    # 4. NEURAL NETWORK PREDICTION
    # --------------------------------------------------------------------------

    print()
    print(f"Creating prediction for {path}{raw_file}:")

    pred_padded = model.predict(
        geometry_edt
    )

    print(f"   -> pred_padded:             {pred_padded.shape}")


    # --------------------------------------------------------------------------
    # 5. DENORMALIZE
    # --------------------------------------------------------------------------

    pred_padded = vu.tensor_denorm(
        out=pred_padded,
        inp=geometry_edt
    )


    # --------------------------------------------------------------------------
    # 6. REMOVE PADDING
    # --------------------------------------------------------------------------

    pred = sh.unpad_geometry(
        pred_padded,
        geometry_bool.shape
    )

    print(f"   -> pred:                    {pred.shape}")


    # --------------------------------------------------------------------------
    # 7. CONVERT PREDICTION TO NUMPY
    # --------------------------------------------------------------------------

    uz = pred[0, 0].detach().cpu().numpy()
    uy = pred[0, 1].detach().cpu().numpy()
    ux = pred[0, 2].detach().cpu().numpy()
    pr = pred[0, 3].detach().cpu().numpy()


    # --------------------------------------------------------------------------
    # 8. SANITY CHECKS
    # --------------------------------------------------------------------------

    print()
    print("Sanity checks:")

    # Shape matching
    if not (
        uz.shape == shape and
        uy.shape == shape and
        ux.shape == shape and
        pr.shape == shape
    ):
        raise Exception(
            "Prediction doesn't match specified .raw shape."
        )

    print("   -> Shape matching:          OK")


    # NaN / Inf
    if (
        np.isnan(pred.detach().cpu().numpy()).any()
        or
        np.isinf(pred.detach().cpu().numpy()).any()
    ):
        raise ValueError(
            "Model predicted NaN or Inf values!"
        )

    print("   -> NaN/Inf check:            OK")


    # --------------------------------------------------------------------------
    # 9. SOLID VELOCITY CHECK
    # --------------------------------------------------------------------------

    solid_mask = ~geometry_bool

    solid_vel_mag = np.sqrt(
        ux[solid_mask] ** 2 +
        uy[solid_mask] ** 2 +
        uz[solid_mask] ** 2
    )

    if solid_vel_mag.size > 0:

        print(
            f"   -> Solid velocity | "
            f"max: {solid_vel_mag.max():.6e} | "
            f"mean: {solid_vel_mag.mean():.6e}"
        )

    else:

        print("   -> Solid velocity: no solid voxels")


    # --------------------------------------------------------------------------
    # 10. LBM STABILITY CHECK
    # --------------------------------------------------------------------------

    velocity_magnitude = np.sqrt(
        ux ** 2 +
        uy ** 2 +
        uz ** 2
    )

    max_v = np.max(velocity_magnitude)

    print(
        f"   -> Maximum velocity:        {max_v:.6e}"
    )

    if max_v > 0.7:
        raise ValueError(
            f"Model predicted a max velocity of {max_v:.4f}. "
            f"LBPM may be unstable due to Mach limit."
        )


    # --------------------------------------------------------------------------
    # 11. WRITE START FILE
    # --------------------------------------------------------------------------

    print()
    print("   -> Creating Start file")

    sh.write_start_raw(
        dirpath=path,
        ux=ux,
        uy=uy,
        uz=uz,
        pr=pr,
        nproc=nproc
    )


    # --------------------------------------------------------------------------
    # 12. CALCULATE PRESSURE DROP
    # --------------------------------------------------------------------------

    print("   -> Calculating pressure drop")

    tau = 1.5
    Re = 0.1
    Dens = 1.0

    p_drop = vu.pressure_calculation(
        geometry_bool,
        tau=tau,
        Re=Re,
        Dens=Dens
    )

    print(
        f"   -> Pressure drop:           {p_drop:.6e}"
    )


    # --------------------------------------------------------------------------
    # 13. WRITE LBPM DATABASE
    # --------------------------------------------------------------------------

    print("   -> Creating .db file")

    sh.write_lbpm_db(
        db_name=path + "start_pressure.db",
        path="",
        tau=tau,
        bc=3,
        din=1.0,
        dout=1.0 - 3 * p_drop,
        nproc=nproc,
        n=(
            int(shape[2] / nproc[0]),
            int(shape[1] / nproc[1]),
            int(shape[0] / nproc[2])
        ),
        N=shape,
        analysis_interval=1000,
        tolerance=1e-6,
        out_format="vtk",
        Start=True
    )


    # --------------------------------------------------------------------------
    # 14. REWRITE ORIGINAL DOMAIN.RAW AS UINT8
    # --------------------------------------------------------------------------

    geometry_bool.astype(np.uint8).tofile(
        path + raw_file
    )


    # --------------------------------------------------------------------------
    # 15. PERMEABILITY FROM NN PREDICTION
    # --------------------------------------------------------------------------

    pred_perm = vu.permeability_calculation(
        pred,
        geometry_edt,
        denorm=False
    )

    perm_val = float(pred_perm)


    # --------------------------------------------------------------------------
    # 16. STATISTICS
    # --------------------------------------------------------------------------

    print()
    print("Prediction statistics:")

    print(
        f"   -> Perm | {perm_val:.6e}"
    )

    print(
        f"   -> Uz   | "
        f"max: {uz.max():>13.6e} | "
        f"mean: {uz.mean():>13.6e} | "
        f"min: {uz.min():>13.6e}"
    )

    print(
        f"   -> Uy   | "
        f"max: {uy.max():>13.6e} | "
        f"mean: {uy.mean():>13.6e} | "
        f"min: {uy.min():>13.6e}"
    )

    print(
        f"   -> Ux   | "
        f"max: {ux.max():>13.6e} | "
        f"mean: {ux.mean():>13.6e} | "
        f"min: {ux.min():>13.6e}"
    )

    print(
        f"   -> Pr   | "
        f"max: {pr.max():>13.6e} | "
        f"mean: {pr.mean():>13.6e} | "
        f"min: {pr.min():>13.6e}"
    )

    print()