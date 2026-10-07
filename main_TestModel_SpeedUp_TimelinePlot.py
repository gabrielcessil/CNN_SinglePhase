import os
import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import torch
from scipy.ndimage import distance_transform_edt as edt

import Utilities.velocity_usage as vu

# ==============================================================================
# 1. CONFIGURATION & HELPERS
# ==============================================================================

# Feel free to uncomment the dictionary you are currently analyzing
file = "SpeedUp_crossDataset"
main_folders = {
    "Premature":          "../TestSpeedUp_Simulations_PrematureConvergence/",
}

# ---------------------------------------------------------
# GLOBAL ACADEMIC STYLING CONFIGURATION
# ---------------------------------------------------------
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 11,
        "axes.labelsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "axes.linewidth": 1.0,
    }
)

# ==============================================================================
# SET YOUR SIMULATION ANALYSIS INTERVAL HERE
# ==============================================================================
analysis_interval       = 200  # Matches exactly where the time plots should begin visually
PLOT_INITIAL_NN_POINT   = True
PLOT_NULL_CASE          = False # Toggle showing the Constant Pressure (lbpm_null) case


def get_permeability_data(folder_path):
    """
    Reads the Permeability.csv file from the simulation folder and 
    returns the entire DataFrame (useful for temporal dynamics).
    """
    p = Path(folder_path)
    if not p.exists():
        return None
        
    perm_file = p / "Permeability.csv"
    
    if perm_file.exists():
        try:
            # The CSV data is whitespace-separated
            df_perm = pd.read_csv(perm_file, sep=r'\s+')
            if 'absperm(mDa)' in df_perm.columns:
                return df_perm
        except Exception as e:
            print(f"Warning: Could not read {perm_file} due to {e}")
            
    return None

def compute_metrics(k_array, k_ref, is_ref_run=False):
    """
    Computes convergence metrics. e_K is now calculated against a fixed reference
    (k_ref). The last point is only masked out if this is the reference run itself.
    """
    e_K         = 100 * np.abs(k_array - k_ref) / (np.abs(k_ref) + 1e-15)    
    k_prev      = np.roll(k_array, 1)
    k_prev[0]   = np.nan
    k_step      = 100 * np.abs(k_array - k_prev) / (np.abs(k_array) + 1e-15)
    
    # Only mask out the final point of e_K for the reference run  
    # because it compares exactly to itself (0.0). For other runs, the final point
    # is a valid, non-zero offset against the reference truth.
    if is_ref_run:
        e_K[-1] = np.nan 
    
    return e_K, k_step

def sci_notation_formatter(x, pos):
    """
    Forces LaTeX-style scientific notation for all ticks. 
    Outputs formats like 10^2, 10^-5, or 5 x 10^1.
    """
    if x == 0 or np.isclose(x, 0, atol=1e-20):
        return "$0$"
    
    sign = "-" if x < 0 else ""
    x_abs = abs(x)
    power = np.log10(x_abs)
    
    # Check if the number is an exact power of 10 (e.g., 100, 0.001)
    if np.isclose(power, np.round(power), atol=1e-3):
        return f"{sign}$10^{{{int(np.round(power))}}}$"
    else:
        # Format intermediate ticks like 50 -> 5 x 10^1
        power_int = int(np.floor(power))
        coeff = x_abs / (10.0 ** power_int)
        coeff = np.round(coeff, 2)  # Clean up floating-point precision issues
        
        if coeff == 1.0:
            return f"{sign}$10^{{{power_int}}}$"
        elif coeff == 10.0:  # Rare rounding edge case
            return f"{sign}$10^{{{power_int + 1}}}$"
        
        return f"{sign}${coeff:g} \\times 10^{{{power_int}}}$"

def enforce_at_least_two_yticks(ax):
    """
    Checks if a subplot has fewer than 2 y-ticks within its limits.
    If so, and it is on a log scale, it expands the limits outward to 
    the nearest powers of 10 to ensure perfectly round numbers (10^2, 10^3).
    """
    ymin, ymax = sorted(ax.get_ylim())
    
    # Handle perfectly flat data
    if np.isclose(ymin, ymax):
        offset = max(abs(ymin) * 0.1, 1e-9)
        ax.set_ylim(ymin - offset, ymax + offset)
        ymin, ymax = sorted(ax.get_ylim())

    # Count how many ticks are currently visible in the data range
    valid_ticks = [t for t in ax.get_yticks() if ymin <= t <= ymax]
    scale = ax.yaxis.get_scale()
    
    if len(valid_ticks) < 2:
        if scale == 'log':
            # Expand limits outward to the nearest exact decades (powers of 10)
            new_ymin = 10.0 ** np.floor(np.log10(ymin)) if ymin > 0 else ymin
            new_ymax = 10.0 ** np.ceil(np.log10(ymax)) if ymax > 0 else ymax
            
            # Fallback if limits are identical (e.g. exact same power of 10)
            if np.isclose(new_ymin, new_ymax) and new_ymin > 0:
                new_ymin /= 10.0
                new_ymax *= 10.0

            ax.set_ylim(new_ymin, new_ymax)
        else:
            # Fallback for linear scales
            locator = ticker.MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10])
            ax.yaxis.set_major_locator(locator)
            
    # Apply standard scientific format to Y axis
    if scale == 'log':
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(sci_notation_formatter))
    else:
        formatter = ticker.ScalarFormatter()
        formatter.set_scientific(True)
        formatter.set_powerlimits((-3, 4))
        ax.yaxis.set_major_formatter(formatter)

def enforce_one_tick_margin(ax, data_min, data_max):
    """
    Adjusts the x-axis limits to start one tick before the data minimum
    and end one tick after the data maximum, utilizing scientific notation.
    """
    if pd.isna(data_min) or pd.isna(data_max):
        return
        
    if data_min >= data_max:
        data_min *= 0.5
        data_max = data_min * 4.0 if data_min > 0 else 1.0

    locator = ax.xaxis.get_major_locator()
    # Pull potential ticks across a broad expanded range
    ticks = locator.tick_values(data_min * 0.001, data_max * 1000)
    ticks = sorted([t for t in ticks if t > 0]) # Keep strictly positive ticks
    
    ticks_before = [t for t in ticks if t < data_min]
    ticks_after = [t for t in ticks if t > data_max]
    
    # Extract the nearest tick directly outside the data bounds
    left_lim = ticks_before[-1] if ticks_before else data_min * 0.8
    right_lim = ticks_after[0] if ticks_after else data_max * 1.2
    
    ax.set_xlim(left_lim, right_lim)
    
    # Apply standard scientific format to X axis
    scale = ax.xaxis.get_scale()
    if scale == 'log':
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(sci_notation_formatter))
    else:
        formatter = ticker.ScalarFormatter()
        formatter.set_scientific(True)
        formatter.set_powerlimits((-3, 4))
        ax.xaxis.set_major_formatter(formatter)

# ==============================================================================
# 2. PLOTTING INDIVIDUAL SAMPLE DYNAMICS
# ==============================================================================

output_dir = Path("./Plots/Dynamics")
output_dir.mkdir(parents=True, exist_ok=True)

print(f"\n--- Starting Subplot Generation for Individual Samples ---")
x_start = analysis_interval

for dataset_name, dataset_path in main_folders.items():
    dataset_p = Path(dataset_path)
    
    for sample_folder in dataset_p.glob("*Sample*"):
        if not sample_folder.is_dir():
            continue
            
        sample_name = sample_folder.name
        
        # Directories for the 3 distinct cases
        run_dir     = sample_folder / "lbpm_grad_run"
        started_dir = sample_folder / "lbpm_nn_run"
        null_dir    = sample_folder / "lbpm_null"
        
        # Extract data
        df_grad = get_permeability_data(run_dir)
        df_nn   = get_permeability_data(started_dir)
        df_null = get_permeability_data(null_dir)
        
        # Skip if no simulation data exists for this sample
        if df_grad is None and df_nn is None and df_null is None:
            continue
            
        print(f"Processing: {dataset_name} | {sample_name}...")
        
        # Setup Figure & Axes (2x2 Layout)
        fig, axs = plt.subplots(2, 2, figsize=(14, 10), dpi=200)
        fig.suptitle(f"Permeability Convergence Dynamics\n{dataset_name} - {sample_name}", fontweight='bold', fontsize=15)
        
        # Styles for the three sets of data
        style_grad = {'color': '#1f77b4', 'label': 'Pressure Gradient', 'alpha': 0.8, 'linewidth': 2, 'marker': '.', 'markersize': 5}
        style_nn   = {'color': '#d62728', 'label': 'Predicted Fields', 'alpha': 0.8, 'linewidth': 2, 'linestyle': '--', 'marker': '.', 'markersize': 5}
        style_null = {'color': '#2ca02c', 'label': 'Constant Pressure', 'alpha': 0.8, 'linewidth': 2, 'linestyle': ':', 'marker': '.', 'markersize': 5}
        
        # Determine the global reference final permeability primarily from lbpm_grad_run
        if df_grad is not None and len(df_grad) > 0:
            global_k_ref = df_grad['absperm(mDa)'].values[-1]
        elif df_null is not None and len(df_null) > 0:
            global_k_ref = df_null['absperm(mDa)'].values[-1] # Fallback
        else:
            global_k_ref = df_nn['absperm(mDa)'].values[-1]   # Fallback

        # ======================================================================
        # CALCULATE INITIAL PERMEABILITY (t=1) FOR NN RUN
        # ======================================================================
        k_init_nn = None
        
        if PLOT_INITIAL_NN_POINT:
            domain_file = sample_folder / "domain.raw"
            start_file_nn = started_dir / "Start.00000.raw"
            
            if domain_file.exists() and start_file_nn.exists():
                try:
                    # 1. Read geometry and compute EDT
                    geom_raw = np.fromfile(domain_file, dtype=np.uint8)
                    L = int(np.round(len(geom_raw)**(1/3))) # Dynamically infer cubic size
                    geom_3d = geom_raw.reshape((L, L, L))
                    
                    inp_edt = edt(geom_3d).astype(np.float32)
                    inp_tensor = torch.from_numpy(inp_edt).unsqueeze(0).unsqueeze(0)
                    
                    # 2. Read Start.00000.raw (ux, uy, uz, pr stacked in 4 columns)
                    start_data = np.fromfile(start_file_nn, dtype=np.float64).reshape(-1, 4)
                    uz_3d = start_data[:, 2].reshape((L, L, L)).astype(np.float32) # u_z is index 2
                    
                    # 3. Form input tensor for permeability_calculation
                    out_tensor = torch.zeros((1, 4, L, L, L), dtype=torch.float32)
                    out_tensor[0, 0] = torch.from_numpy(uz_3d) # Assign u_z to 0th channel
                    
                    # 4. Compute Permeability using provided function
                    k_init_tensor = vu.permeability_calculation(
                        out=out_tensor,
                        inp=inp_tensor,
                        tau=1.5,
                        Re=0.1,
                        dens=1.0,
                        denorm=False # Explicitly as requested
                    )
                    k_init_nn = k_init_tensor[0].item()
                    print(f"  -> Successfully calculated NN Initial Permeability (t=1): {k_init_nn:.4f} mDa")
                except Exception as e:
                    print(f"  -> Failed to calculate NN Initial Permeability: {e}")

        # Data bound trackers for automatic tick adjustment
        time_min, time_max = float('inf'), float('-inf')
        kstep_min, kstep_max = float('inf'), float('-inf')

        # We now pass the trackers in as arguments and return the updated versions
        def plot_to_axes(df, style_dict, t_min, t_max, k_min, k_max, is_ref_run=False, plot_errors=True):
            if df is None or len(df) == 0:
                return t_min, t_max, k_min, k_max
                
            k_array = df['absperm(mDa)'].values
            
            # -------------------------------------------------------------
            # DETERMINE CORRECT TIMESTEPS (Scaled by Analysis_Interval)
            # -------------------------------------------------------------
            if 'Step' in df.columns:
                steps = df['Step'].values
                if np.max(steps) <= len(steps):
                    steps = steps * analysis_interval
            elif 'Time(s)' in df.columns:
                steps = df['Time(s)'].values
            else:
                steps = np.arange(1, len(k_array) + 1) * analysis_interval
            
            # Update time boundaries
            if len(steps) > 0:
                t_min = min(t_min, np.nanmin(steps))
                t_max = max(t_max, np.nanmax(steps))
                
            e_K, k_step = compute_metrics(k_array, global_k_ref, is_ref_run)
            
            # 1. Permeability over time
            axs[0, 0].plot(steps, k_array, **style_dict)
            
            if plot_errors:
                # 3. k_step vs time (plotted for runs tracking errors)
                axs[1, 0].plot(steps, k_step, **style_dict)
                
                # Update k_step boundaries exclusively for lines we actually plot
                valid_kstep = k_step[1:][~np.isnan(k_step[1:])]
                if len(valid_kstep) > 0:
                    k_min = min(k_min, np.nanmin(valid_kstep))
                    k_max = max(k_max, np.nanmax(valid_kstep))
                
                # 2. e_K vs time (Plotted for all tracking runs, including the reference)
                axs[0, 1].plot(steps, e_K, **style_dict)
                
                # 4. e_K vs k_step (e_K on X axis, k_step on Y axis)
                axs[1, 1].scatter(e_K[1:], k_step[1:], color=style_dict['color'], 
                                  label=style_dict['label'], alpha=0.5, s=25, edgecolor='k', linewidth=0.5)
                              
            return t_min, t_max, k_min, k_max

        # Plot data for all runs
        if PLOT_NULL_CASE:
            time_min, time_max, kstep_min, kstep_max = plot_to_axes(df_null, style_null, time_min, time_max, kstep_min, kstep_max, is_ref_run=False, plot_errors=False)
            
        time_min, time_max, kstep_min, kstep_max = plot_to_axes(df_grad, style_grad, time_min, time_max, kstep_min, kstep_max, is_ref_run=True, plot_errors=True)
        time_min, time_max, kstep_min, kstep_max = plot_to_axes(df_nn, style_nn, time_min, time_max, kstep_min, kstep_max, is_ref_run=False, plot_errors=True)

        # Fallbacks in case data was completely empty (avoids math errors)
        if time_min == float('inf') or np.isnan(time_min):
            time_min, time_max = x_start, x_start * 10
        if kstep_min == float('inf') or np.isnan(kstep_min):
            kstep_min, kstep_max = 1e-5, 100
        
        # Determine the forced exact left limit for the Time X-axis 
        # (Using * 0.999 guarantees that if analysis_interval is 100, the tick starts at 10)
        forced_time_start_x = 10.0 ** np.floor(np.log10(analysis_interval * 0.999))
        
        # ---------------------------------------------------------
        # Set Titles & Scales before universal Grid/Tick Enforcer
        # ---------------------------------------------------------
        axs[0, 0].set_title("1. Permeability over Time", fontweight='bold')
        axs[0, 0].set_xlabel("Time Step")
        axs[0, 0].set_ylabel("Permeability (mDa)")
        axs[0, 0].set_xscale('log')
        axs[0, 0].set_yscale('symlog')
        enforce_one_tick_margin(axs[0, 0], time_min, time_max) 
        axs[0, 0].set_xlim(left=forced_time_start_x) # Hard-force to exact scale 
        
        axs[0, 1].set_title(r"2. Error relative to Final Pressure Gradient [%]", fontweight='bold')
        axs[0, 1].set_xlabel("Time Step")
        axs[0, 1].set_ylabel(r"$e_K = 100 \times |k(t) - k_{grad\_final}| / k_{grad\_final} $ (%)")
        axs[0, 1].set_xscale('log')
        axs[0, 1].set_yscale('log')
        enforce_one_tick_margin(axs[0, 1], time_min, time_max)
        axs[0, 1].set_xlim(left=forced_time_start_x)
        
        axs[1, 0].set_title(r"3. Step Error over Time", fontweight='bold')
        axs[1, 0].set_xlabel("Time Step")
        axs[1, 0].set_ylabel(r"$k_{step} = 100 \times |k(t) - k(t-1)| / k(t)$ (%)")
        axs[1, 0].set_xscale('log')
        axs[1, 0].set_yscale('log')
        enforce_one_tick_margin(axs[1, 0], time_min, time_max)
        axs[1, 0].set_xlim(left=forced_time_start_x)
        
        axs[1, 1].set_title(r"4. Convergence criteria ($e_K$ vs $k_{step}$)", fontweight='bold')
        axs[1, 1].set_xlabel(r"$e_K$ (%)")
        axs[1, 1].set_ylabel(r"$k_{step}$ (%)")
        axs[1, 1].set_xscale('log')
        axs[1, 1].set_yscale('log')
        enforce_one_tick_margin(axs[1, 1], kstep_min, kstep_max)

        # ======================================================================
        # ENFORCE CONSISTENT TICKS AND GRIDS ON ALL SUBPLOTS
        # ======================================================================
        for ax in axs.flat:
            enforce_at_least_two_yticks(ax)
            
            # Safely handle Y minor locators to prevent "Black Bar" tick overlap
            ymin, ymax = sorted(ax.get_ylim())
            if ymin <= 0:
                ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
            else:
                decades_y = np.log10(ymax / ymin)
                if decades_y > 8: # If span is massive, hide minor ticks
                    ax.yaxis.set_minor_locator(ticker.NullLocator())
                elif decades_y >= 1:
                    ax.yaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=np.arange(2, 10)))
                else:
                    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
                    
            # Safely handle X minor locators
            xmin, xmax = sorted(ax.get_xlim())
            if xmin <= 0:
                ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
            else:
                decades_x = np.log10(xmax / xmin)
                if decades_x > 8:
                    ax.xaxis.set_minor_locator(ticker.NullLocator())
                elif decades_x >= 1:
                    ax.xaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=np.arange(2, 10)))
                else:
                    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
                    
            # Unified grid styling for every subplot
            ax.grid(True, which='major', linestyle='--', alpha=0.7)
            
            # Only draw minor grids if minor ticks are actually rendered
            if (ymin > 0 and decades_y <= 8) or (xmin > 0 and decades_x <= 8):
                ax.grid(True, which='minor', linestyle=':', alpha=0.4, linewidth=0.8)
        
        # ---------------------------------------------------------
        # Add the dot for NN Initial Permeability exactly on the Y-axis spine
        # ---------------------------------------------------------
        if PLOT_INITIAL_NN_POINT and k_init_nn is not None:
            left_spine_x = axs[0, 0].get_xlim()[0]
            
            # 1. Place the square marker directly on the spine 
            axs[0, 0].scatter([left_spine_x], [k_init_nn], color='black', marker='s', s=90, zorder=10, 
                              edgecolors='black', clip_on=False, label='Predicted Permeability')
            
            # 2. Emulate a standard tick label beside it
            axs[0, 0].annotate(f"{k_init_nn:.3g}", 
                               xy=(left_spine_x, k_init_nn), 
                               xytext=(-8, 0), # 8 points to the left of the spine
                               textcoords="offset points", 
                               ha="right", va="center", 
                               color='black', # Changed to match the black square
                               fontweight='bold',
                               fontsize=10,
                               clip_on=False)
        
        # ---------------------------------------------------------
        # Deduplicate legends
        # ---------------------------------------------------------
        handles, labels = axs[0, 0].get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        if by_label:
            axs[0, 0].legend(by_label.values(), by_label.keys(), loc="best", framealpha=0.9, edgecolor='black')
            
        handles, labels = axs[1, 1].get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        if by_label:
            axs[1, 1].legend(by_label.values(), by_label.keys(), loc="best", framealpha=0.9, edgecolor='black')

        # Final layout adjustments
        plt.tight_layout(rect=[0, 0, 1, 0.95]) # Leaves space for the suptitle
        
        # Clean naming string to prevent path traversal issues
        safe_dataset = re.sub(r'[^a-zA-Z0-9_\-]', '_', dataset_name)
        safe_sample  = re.sub(r'[^a-zA-Z0-9_\-]', '_', sample_name)
        
        # Save figure
        fig_name = f"Dynamics_{safe_dataset}_{safe_sample}.png"
        fig_path = output_dir / fig_name
        
        plt.savefig(fig_path, bbox_inches='tight')
        plt.close(fig)

print(f"\n--- Done! All dynamic plots are saved in '{output_dir.absolute()}' ---")