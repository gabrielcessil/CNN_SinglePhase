import os
import glob
import re
import pandas as pd
import numpy as np
import torch
import pyvista as pv

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

from Utilities.error_metrics import (
    Flux_Comparison,
    Bias_Comparison,
    Magnitude_Comparison,
    Divergent_Residual,
    Tortuosity_Comparison,
    Angular_Comparison,
    Correlation_Comparison
)

# ==============================================================================
# CONFIGURATION & SPEED-UP TABLE LOADING
# ==============================================================================
RESULTS_DIR = "../TestSpeedUp_Simulations_120/Test_Silveira_SphGrain_SAug_DNorm/"
SHAPE = (120, 120, 120)
NPROC = (1, 1, 1)
SPEEDUP_TABLE_PATH  = "./Tables/SpeedUp/SpeedUp_120.csv"


#RESULTS_DIR = "../TestSpeedUp_Simulations_BiggerCrops/IC_Doddington_256_256_256/"
#SHAPE = (256, 256, 256)
#NPROC = (1, 1, 4)

OUTPUT_TABLE        = "./Tables/SpeedUp_Error_Metrics.csv"
PLOTS_DIR           = "./SpeedUp_Correlation_Plots/"

os.makedirs(os.path.dirname(OUTPUT_TABLE), exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# ------------------------------------------------------------------------------
# Load Speed-Up Ratios from Rate Analysis Table
# ------------------------------------------------------------------------------
speedup_lookup = {}

if os.path.exists(SPEEDUP_TABLE_PATH):
    df_speedup_ref = pd.read_csv(SPEEDUP_TABLE_PATH)
    # Map Sample name (e.g. 'Sample_00000') -> Speedup_Ratio
    speedup_lookup = dict(zip(df_speedup_ref["Sample"], df_speedup_ref["Speedup_Ratio"]))
    print(f"Loaded {len(speedup_lookup)} speed-up records from {SPEEDUP_TABLE_PATH}")
else:
    print(f"Warning: Speed-up table not found at {SPEEDUP_TABLE_PATH}.")
    print("Run `main_TestSpeedUp_RateAnalyze.py` first to generate this file.\n")


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================
def get_simulation_timesteps(run_dir):
    """
    Attempts to extract convergence timesteps from timelog.csv, .log files, or VTK outputs.
    Returns np.nan instead of raising an exception if logs are missing.
    """
    if not os.path.exists(run_dir):
        return np.nan

    log_candidates = (
        [os.path.join(run_dir, "timelog.csv")] +
        glob.glob(os.path.join(run_dir, "*.csv")) +
        glob.glob(os.path.join(run_dir, "*.log")) +
        glob.glob(os.path.join(run_dir, "*.txt"))
    )

    for filepath in log_candidates:
        if os.path.exists(filepath):
            try:
                df = pd.read_csv(filepath, sep=r'\s+|,', engine='python', on_bad_lines='skip')
                if df.empty:
                    continue
                cols = [str(c).lower().strip() for c in df.columns]

                for col_name in ['timestep', 'step', 'time_step', 'iter', 'n', 'time']:
                    if col_name in cols:
                        idx = cols.index(col_name)
                        val = pd.to_numeric(df.iloc[:, idx], errors='coerce').dropna()
                        if not val.empty:
                            return float(val.iloc[-1])

                val = pd.to_numeric(df.iloc[:, 0], errors='coerce').dropna()
                if not val.empty:
                    return float(val.iloc[-1])
            except Exception:
                continue

    vtk_files = glob.glob(os.path.join(run_dir, "**", "*.vtk"), recursive=True)
    if vtk_files:
        steps = []
        for f in vtk_files:
            nums = re.findall(r'\d+', os.path.basename(f))
            if nums:
                steps.append(int(nums[-1]))
        if steps:
            return float(max(steps))

    return np.nan


def load_start_raw(nn_dir, shape=SHAPE, nproc=NPROC, dtype=np.float64):
    """
    Loads and stitches MPI-decomposed Start.XXXXX.raw files (float64) into a full 3D volume.
    Strictly aligns spatial subdomains with the provided nproc = (nz, ny, nx) layout.
    """
    nx, ny, nz = nproc
    total_ranks = nz * ny * nx

    sz = shape[0] // nz
    sy = shape[1] // ny
    sx = shape[2] // nx
    sub_voxels = sz * sy * sx

    pattern_5d = os.path.join(nn_dir, "Start.[0-9][0-9][0-9][0-9][0-9].raw")
    pattern_1d = os.path.join(nn_dir, "Start.[0-9]*.raw")
    files = sorted(glob.glob(pattern_5d)) or sorted(glob.glob(pattern_1d))

    if len(files) != total_ranks:
        raise FileNotFoundError(
            f"Expected {total_ranks} rank files for nproc={nproc}, but found {len(files)} in {nn_dir}"
        )

    bytes_per_elem = np.dtype(dtype).itemsize
    file_bytes = os.path.getsize(files[0])
    total_elems = file_bytes // bytes_per_elem

    if total_elems % sub_voxels != 0:
        raise ValueError(
            f"File size ({file_bytes} bytes = {total_elems} float64 elements) is not evenly divisible "
            f"by subdomain voxel count ({sub_voxels} = {sz}x{sy}x{sx}) for nproc={nproc}."
        )

    num_channels = total_elems // sub_voxels
    full_grid = np.zeros((*shape, num_channels), dtype=dtype)

    for iz in range(nz):
        for iy in range(ny):
            for ix in range(nx):
                rank = iz * (ny * nx) + iy * nx + ix

                filename = os.path.join(nn_dir, f"Start.{rank:05d}.raw")
                if not os.path.exists(filename):
                    filename = os.path.join(nn_dir, f"Start.{rank}.raw")

                if not os.path.exists(filename):
                    raise FileNotFoundError(f"Missing file for MPI rank {rank}: {filename}")

                raw_data = np.fromfile(filename, dtype=dtype)
                sub_grid = raw_data.reshape((sz, sy, sx, num_channels))

                z_start, z_end = iz * sz, (iz + 1) * sz
                y_start, y_end = iy * sy, (iy + 1) * sy
                x_start, x_end = ix * sx, (ix + 1) * sx

                full_grid[z_start:z_end, y_start:y_end, x_start:x_end, :] = sub_grid

    if num_channels == 4:
        uz = full_grid[..., 0]
        uy = full_grid[..., 1]
        ux = full_grid[..., 2]
        pr = full_grid[..., 3]
    elif num_channels == 2:
        uz = full_grid[..., 0]
        uy = np.zeros(shape, dtype=dtype)
        ux = np.zeros(shape, dtype=dtype)
        pr = full_grid[..., 1]
    elif num_channels == 1:
        uz = full_grid[..., 0]
        uy = np.zeros(shape, dtype=dtype)
        ux = np.zeros(shape, dtype=dtype)
        pr = np.zeros(shape, dtype=dtype)
    else:
        raise ValueError(f"Unsupported channel count ({num_channels}) in Start.*.raw files.")

    return uz, uy, ux, pr

def load_vtk_final_state(grad_dir, shape=SHAPE):
    """
    Loads the final converged simulation state from LBPM vis*/summary.pvti files.
    Handles separate scalar velocity arrays (Velocity_x, Velocity_y, Velocity_z).
    """
    pvti_files = glob.glob(os.path.join(grad_dir, "vis*", "summary.pvti"))
    
    if not pvti_files:
        pvti_files = glob.glob(os.path.join(grad_dir, "**", "*.pvti"), recursive=True)
        if not pvti_files:
            pvti_files = glob.glob(os.path.join(grad_dir, "**", "*.vtk"), recursive=True)

    if not pvti_files:
        raise FileNotFoundError(f"No summary.pvti or VTK files found in {grad_dir}")

    def extract_step(filepath):
        nums = re.findall(r'\d+', filepath)
        return int(nums[-1]) if nums else 0

    pvti_files.sort(key=extract_step)
    latest_file = pvti_files[-1]

    mesh = pv.read(latest_file)
    arr_names_lower = {k.lower(): k for k in mesh.array_names}

    if "velocity_x" in arr_names_lower and "velocity_y" in arr_names_lower and "velocity_z" in arr_names_lower:
        ux = mesh[arr_names_lower["velocity_x"]].reshape(shape)
        uy = mesh[arr_names_lower["velocity_y"]].reshape(shape)
        uz = mesh[arr_names_lower["velocity_z"]].reshape(shape)
    else:
        vel_key = next((k for k in mesh.array_names if "velocity" in k.lower() or "u" in k.lower()), None)
        if vel_key is None:
            raise KeyError(f"Velocity fields not found in {latest_file}. Available arrays: {mesh.array_names}")

        vel_data = mesh[vel_key]
        if vel_data.ndim == 2 and vel_data.shape[1] == 3:
            ux = vel_data[:, 0].reshape(shape)
            uy = vel_data[:, 1].reshape(shape)
            uz = vel_data[:, 2].reshape(shape)
        else:
            raise ValueError(f"Unexpected array shape for {vel_key} in {latest_file}: {vel_data.shape}")

    pr_key = next((k for k in mesh.array_names if "pressure" in k.lower() or k.lower() == "p"), None)
    pr = mesh[pr_key].reshape(shape) if pr_key is not None else np.zeros(shape)

    return uz, uy, ux, pr

# ==============================================================================
# MAIN PROCESSING LOOP
# ==============================================================================
sample_dirs = [f.path for f in os.scandir(RESULTS_DIR) if f.is_dir()]
sample_dirs.sort()

results = []

print(f"Found {len(sample_dirs)} simulation directories. Starting evaluation...\n")

for sample_path in sample_dirs:
    dataset_name = os.path.basename(sample_path)
    grad_dir = os.path.join(sample_path, "lbpm_grad_run")
    nn_dir   = os.path.join(sample_path, "lbpm_nn_run")
    raw_geom = os.path.join(sample_path, "domain.raw")

    if not (os.path.exists(grad_dir) and os.path.exists(nn_dir) and os.path.exists(raw_geom)):
        print(f"Skipping {dataset_name}: Missing simulation subfolders or domain.raw.")
        continue

    try:
        speed_up = speedup_lookup.get(dataset_name, np.nan)

        geom_bytes = np.fromfile(raw_geom, dtype=np.uint8).reshape(SHAPE)
        geometry_bool = (geom_bytes > 0)
        batch_inputs = torch.tensor(geometry_bool, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        uz_pred, uy_pred, ux_pred, pr_pred = load_start_raw(nn_dir, shape=SHAPE, nproc=NPROC)
        pred_stack = np.stack([uz_pred, uy_pred, ux_pred, pr_pred], axis=0)
        batch_outputs = torch.tensor(pred_stack, dtype=torch.float32).unsqueeze(0)

        uz_true, uy_true, ux_true, pr_true = load_vtk_final_state(grad_dir, shape=SHAPE)
        target_stack = np.stack([uz_true, uy_true, ux_true, pr_true], axis=0)
        batch_targets = torch.tensor(target_stack, dtype=torch.float32).unsqueeze(0)

        e_flux = Flux_Comparison(batch_inputs, batch_outputs, batch_targets)[0]
        e_bias = Bias_Comparison(batch_inputs, batch_outputs, batch_targets)[0]
        e_mag  = Magnitude_Comparison(batch_inputs, batch_outputs, batch_targets)[0]
        e_div  = Divergent_Residual(batch_inputs, batch_outputs)[0]
        e_tort = Tortuosity_Comparison(batch_inputs, batch_outputs, batch_targets)[0]
        e_ang  = Angular_Comparison(batch_inputs, batch_outputs, batch_targets)[0]
        r_corr = Correlation_Comparison(batch_inputs, batch_outputs, batch_targets)[0]

        results.append({
            "Dataset": dataset_name,
            "Speed_Up": speed_up,
            "Flux_Error": e_flux,
            "Bias_Error": e_bias,
            "Magnitude_Error": e_mag,
            "Divergence_Residual": e_div,
            "Tortuosity_Error": e_tort,
            "Angular_Error": e_ang,
            "Prediction_Correlation": r_corr
        })

        sp_str = f"{speed_up:.2f}x" if not np.isnan(speed_up) else "N/A"
        print(f"Processed {dataset_name} | Speed-Up (Loaded): {sp_str} | Correlation: {r_corr:.4f}")

    except Exception as e:
        print(f"Error processing {dataset_name}: {e}")

# Save CSV Table
df_results = pd.DataFrame(results)

if df_results.empty:
    print("\nError: No samples were successfully evaluated. Verify domain.raw and raw prediction files.")
else:
    df_results.to_csv(OUTPUT_TABLE, index=False)
    print(f"\nMetrics for {len(df_results)} samples saved to {OUTPUT_TABLE}")

    # ==============================================================================
    # CORRELATION & SCATTER PLOTS (COLOR-CODED BY SPEED-UP)
    # ==============================================================================
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 11,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.linewidth": 1.0,
    })

    error_metrics_to_plot = [
        "Flux_Error", "Bias_Error", "Magnitude_Error",
        "Divergence_Residual", "Tortuosity_Error", "Angular_Error",
        "Prediction_Correlation"
    ]

    print("\n--- Pearson Correlation with Speed-Up ---")
    for metric in error_metrics_to_plot:
        valid_df = df_results[[metric, "Speed_Up"]].dropna()
        if len(valid_df) < 2:
            print(f"Skipping plot for {metric}: Insufficient valid (Speed_Up, Metric) pairs.")
            continue

        corr_val = valid_df[metric].corr(valid_df["Speed_Up"])
        print(f"{metric:<25} vs Speed-Up: r = {corr_val:.4f}")

        fig, ax = plt.subplots(figsize=(6.5, 5), dpi=300)

        # Scatter plot color-mapped by Speed_Up
        sc = ax.scatter(
            valid_df[metric],
            valid_df["Speed_Up"],
            c=valid_df["Speed_Up"],
            cmap="viridis",
            s=65,
            edgecolors="black",
            linewidths=0.6,
            alpha=0.85,
            zorder=3
        )

        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label("Speed-Up ($S$)", fontsize=11)

        ax.set_title(f"{metric} vs Speed-Up", fontweight="bold")
        ax.set_ylabel("Speed-Up ($S = N_{\\text{grad}} / N_{\\text{nn}}$)", fontweight="bold")
        ax.set_xlabel(metric, fontweight="bold")
        ax.grid(True, alpha=0.3, ls="--", zorder=1)

        ax.annotate(
            f"$r = {corr_val:.2f}$",
            xy=(0.05, 0.93),
            xycoords='axes fraction',
            ha='left',
            va='top',
            fontsize=11,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.85)
        )

        plt.tight_layout()
        plot_filename = os.path.join(PLOTS_DIR, f"Scatter_{metric}_vs_SpeedUp.png")
        plt.savefig(plot_filename, dpi=300, bbox_inches="tight")
        plt.close()

    # ------------------------------------------------------------------------------
    # SPECIFIC PLOT: BIAS ERROR VS PREDICTION CORRELATION (COLORED BY SPEED-UP)
    # ------------------------------------------------------------------------------
    valid_bias_corr = df_results[["Prediction_Correlation", "Bias_Error", "Speed_Up"]].dropna()

    if len(valid_bias_corr) >= 2:
        corr_bias_pred = valid_bias_corr["Prediction_Correlation"].corr(valid_bias_corr["Bias_Error"])
        print(f"\nPrediction_Correlation vs Bias_Error: r = {corr_bias_pred:.4f}")

        fig, ax = plt.subplots(figsize=(6.5, 5), dpi=300)

        sc = ax.scatter(
            valid_bias_corr["Prediction_Correlation"],
            valid_bias_corr["Bias_Error"],
            c=valid_bias_corr["Speed_Up"],
            cmap="viridis",
            s=65,
            edgecolors="black",
            linewidths=0.6,
            alpha=0.85,
            zorder=3
        )

        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label("Speed-Up ($S$)", fontsize=11)

        ax.set_title("Bias Error vs Prediction Correlation", fontweight="bold")
        ax.set_xlabel("Prediction Correlation ($R$)", fontweight="bold")
        ax.set_ylabel("Bias Error", fontweight="bold")
        ax.grid(True, alpha=0.3, ls="--", zorder=1)

        ax.annotate(
            f"$r = {corr_bias_pred:.2f}$",
            xy=(0.05, 0.93),
            xycoords='axes fraction',
            ha='left',
            va='top',
            fontsize=11,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.85)
        )

        plt.tight_layout()
        bias_corr_plot = os.path.join(PLOTS_DIR, "Scatter_Bias_Error_vs_Prediction_Correlation.png")
        plt.savefig(bias_corr_plot, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved Bias Error vs Prediction Correlation scatter plot to {bias_corr_plot}")

    print(f"\nAll scatter plots successfully generated in {PLOTS_DIR}")