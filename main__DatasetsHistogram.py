import torch
import numpy as np
from Utilities import dataset_reader as dr
from Utilities import velocity_usage as vu

import matplotlib.pyplot as plt
import os
from matplotlib import patheffects # Add this import at the top of your script
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

from   scipy.ndimage import distance_transform_edt


def Plot_Velocity_Front_Comparison(datasets, sample_idx=0, slice_idx=60, component=0, save_mode=False, save_tag=""):
    """
    Plots a slice of the velocity field side-by-side (Front View) with Flux Annotation.
    """
    num_plots = len(datasets)
    fig, axes = plt.subplots(1, num_plots, figsize=(6 * num_plots, 6), constrained_layout=True)
    
    if num_plots == 1: axes = [axes]
        
    comp_name = {0: "Uz", 1: "Uy", 2: "Ux"}.get(component, f"Ch {component}")
    folder = "Velocity_Front_Comparison_" + save_tag
    if save_mode and not os.path.exists(folder): 
        os.makedirs(folder)

    for i, (ds_name, ds) in enumerate(datasets.items()):
        _, targets = ds[sample_idx]
        targets = targets.numpy()
        
        vel_slice = targets[component, slice_idx, :, :]
        vel_masked = np.ma.masked_where(vel_slice == 0, vel_slice)
        
        cmap = plt.colormaps["plasma"].copy()
        cmap.set_bad("white")  
        im = axes[i].imshow(vel_masked, cmap=cmap)
        
        # --- HIGH-VISIBILITY FRONTAL ANNOTATION ---
        # Using \otimes to represent flow into the screen for Uz
        flux_text = r"$z$ $\otimes$"
        ann = axes[i].annotate(r'$z$ $\otimes$', 
                         xy=(0.5, 0.2),      # Same base position
                         xytext=(0.5, 0.08),  # No arrow length needed here
                         color='#00FF00', 
                         fontsize=18, 
                         fontweight='bold',
                         ha='center', 
                         va='bottom',
                         xycoords='axes fraction',
                         textcoords='axes fraction')

        ann.set_path_effects([
            patheffects.withStroke(linewidth=4, foreground='black')
        ])
        # ------------------------------------------
        
        clean_name = ds_name.replace('.h5', '')
        axes[i].set_title(f"{clean_name}\n({comp_name} - Front View)")
        axes[i].axis('off')
        plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)

    if save_mode:
        plt.savefig(f"{folder}/Sample_{sample_idx}_{comp_name}.png", dpi=300, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def Plot_Velocity_Side_Comparison(datasets, sample_idx=0, slice_idx=60, component=0, save_mode=False, save_tag=""):
    """
    Plots a slice of the velocity field side-by-side for all datasets (Side View).
    """
    num_plots = len(datasets)
    fig, axes = plt.subplots(1, num_plots, figsize=(6 * num_plots, 6), constrained_layout=True)
    
    if num_plots == 1: axes = [axes]
        
    comp_name = {0: "Uz", 1: "Uy", 2: "Ux"}.get(component, f"Ch {component}")
    folder = "Velocity_Side_Comparison_" + save_tag
    if save_mode and not os.path.exists(folder): 
        os.makedirs(folder)

    for i, (ds_name, ds) in enumerate(datasets.items()):
        _, targets = ds[sample_idx]
        targets = targets.numpy()
        
        # Slice along the X-axis (Width)
        vel_slice = targets[component, :, :, slice_idx]
        
        vel_masked = np.ma.masked_where(vel_slice == 0, vel_slice)
        cmap = plt.colormaps["plasma"].copy()
        cmap.set_bad("white")  
        im = axes[i].imshow(vel_masked, cmap=cmap)
        ann = axes[i].annotate('$z$', 
                         xy=(0.5, 0.2),         # Arrow tip
                         xytext=(0.5, 0.08),     # Text position (closer to tail)
                         arrowprops=dict(
                             facecolor='#00FF00', 
                             edgecolor='black', 
                             linewidth=1.5,     # Border for the arrow
                             shrink=0.05, 
                             width=5, 
                             headwidth=15
                         ),
                         color='#00FF00', 
                         fontsize=18, 
                         fontweight='bold',
                         ha='center', 
                         va='center',
                         xycoords='axes fraction',
                         textcoords='axes fraction')

        # Add a black border/outline to the text
        ann.set_path_effects([
            patheffects.withStroke(linewidth=3, foreground='black')
        ])
        
        clean_name = ds_name.replace('.h5', '')
        axes[i].set_title(f"{clean_name}\n({comp_name} - Side View)")
        axes[i].axis('off')
        plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)

    if save_mode:
        plt.savefig(f"{folder}/Sample_{sample_idx}_{comp_name}.png", dpi=300, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()

def Compare_Histograms(
        datasets,
        sample_indices=None,
        bins=200,
        save_folder="Normalization_comparisons", 
        base_filename="global_histogram"
    ):
    # 1. Create folder if it doesn't exist
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
        print(f"Created folder: {save_folder}")

    # Dictionary to store all processed data per dataset
    collected_data = {
        "Uz": {}, "Uy": {}, "Ux": {}, "P": {}, "Mag": {}
    }

    # 2. Extract and format data (Raw / Normalized Data)
    for ds_name, ds in datasets.items():
        short_name = ds_name.replace(".h5", "")
        uz_all, uy_all, ux_all, pr_all, mag_all = [], [], [], [], []
        
        current_indices = range(len(ds)) if sample_indices is None else sample_indices

        for sample_idx in current_indices:
            _, targets = ds[sample_idx]
            targets = targets.numpy()
            
            uz = targets[0]
            uy = targets[1]
            ux = targets[2]
            
            # Safely check for pressure channel to avoid IndexError
            has_pressure = targets.shape[0] > 3
            if has_pressure:
                p = targets[3]
                pr_flat = p.flatten()
            else:
                pr_flat = np.array([])
                
            mag = np.sqrt(uz**2 + uy**2 + ux**2)
            
            uz_flat = uz.flatten()
            uy_flat = uy.flatten()
            ux_flat = ux.flatten()
            mag_flat = mag.flatten()
            
            fluid_mask = mag_flat > 1e-20

            uz_all.append(uz_flat[fluid_mask])
            uy_all.append(uy_flat[fluid_mask])
            ux_all.append(ux_flat[fluid_mask])
            mag_all.append(mag_flat[fluid_mask])
            if has_pressure:
                pr_all.append(pr_flat[fluid_mask])

        # Concatenate all samples for the current dataset
        collected_data["Uz"][short_name] = np.concatenate(uz_all)
        collected_data["Uy"][short_name] = np.concatenate(uy_all)
        collected_data["Ux"][short_name] = np.concatenate(ux_all)
        collected_data["Mag"][short_name] = np.concatenate(mag_all)
        if len(pr_all) > 0:
            collected_data["P"][short_name]  = np.concatenate(pr_all)

        # Print global statistics
        print(f"\nDataset: {short_name}")
        print(f"   Uz  -> Mean: {collected_data['Uz'][short_name].mean():.5f} | Std: {collected_data['Uz'][short_name].std():.5f}")
        print(f"   Mag -> Mean: {collected_data['Mag'][short_name].mean():.5f} | Std: {collected_data['Mag'][short_name].std():.5f}")

    # 3. Plotting configurations 
    plot_configs = {
        "Uz":  {"xlabel": " $u_z$",         "xlim": (-0.05,1.0), "ylim": (1e-1, None)},
        "Uy":  {"xlabel": " $u_y$",       "xlim": (-0.5, 0.5),  "ylim": (1e-1, None)},
        "Ux":  {"xlabel": " $u_x$",       "xlim": (-0.5, 0.5),  "ylim": (1e-1, None)},
        "P":   {"xlabel": " $p$",           "xlim": None,        "ylim": (1e-1, None)},
        "Mag": {"xlabel": " $|\\vec{u}|$",  "xlim": (-0.05, 0.8),"ylim": (1e-1, None)}
    }

    # 4. Generate Scientific Step-Plots
    colors = plt.cm.tab10.colors # Distinct, professional colors

    for var, config in plot_configs.items():
        if len(collected_data[var]) == 0: continue 
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for idx, ds_name in enumerate(collected_data[var].keys()):
            data_arr = collected_data[var][ds_name]
            color = colors[idx % len(colors)]
            
            # Plot the histogram
            ax.hist(data_arr, bins=bins, density=True, histtype='step', log=True,
                    linewidth=2.5, alpha=0.9, color=color, label=ds_name)
            
            if var != "P" and var != "Ux" and var != "Uy":
                mean_val = data_arr.mean()
                ax.axvline(mean_val, color=color, linestyle='--', linewidth=1.5, alpha=0.8)
                
                
                y_text_position = 0.98 - (idx * 0.02)
                
                ax.text(mean_val, y_text_position, f"{mean_val:.4f} ", color=color, 
                        transform=ax.get_xaxis_transform(), rotation=90, 
                        va='top', ha='right', fontsize=12, fontweight='bold')

        # Academic Formatting
        ax.set_xlabel(config["xlabel"], fontsize=20)
        ax.set_ylabel("Densidade de Probabilidade", fontsize=20)
        
        # Grid aesthetics: soft and unobtrusive
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='both', which='major', labelsize=14)
        
        # --- ADIÇÃO: Elemento fantasma para indicar a "Média" na legenda ---
        ax.plot([], [], color='gray', linestyle='--', linewidth=1.5, label='Médias')
        
        # Place legend OUTSIDE the plot to avoid obscuring data peaks
        ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=14, frameon=False)
        
        # Apply limits
        if config["xlim"] is not None:
            ax.set_xlim(config["xlim"])
        if "ylim" in config and config["ylim"] is not None:
            ax.set_ylim(config["ylim"])

        # Save logic (bbox_inches='tight' prevents external legend from being cut off)
        filename = f"{base_filename}_{var}.png"
        full_save_path = os.path.join(save_folder, filename)
        plt.savefig(full_save_path, dpi=300, bbox_inches="tight")
        print(f"{var} Histogram saved to: {full_save_path}")
        plt.close(fig)
        
#######################################################
#************ INPUTS                      *************#
#######################################################
dataset_folder = "../NN_Datasets/"

datasets        = {
    
    "Ko et. al":            "../NN_Datasets_Grad/Train_Danny_SphPore_SAug_DNorm.h5",
    
    "Spherical Pores":      "../NN_Datasets_Grad/Train_Silveira_SphPore_SAug_DNorm.h5",
    "Spherical Grains":     "../NN_Datasets_Grad/Train_Silveira_SphGrain_SAug_DNorm.h5",
    ##"Cylindrical Pores":    "../NN_Datasets_Grad/Train_Silveira_CylinPore_SAug_DNorm.h5",
    ##"Cylindrical Grains":   "../NN_Datasets_Grad/Train_Silveira_CylinGrain_SAug_DNorm.h5",
     
    "Bentheimer":           "../NN_Datasets_Grad/Train_Oliveira_Bentheimer_SAug_DNorm.h5",
    ##"Berea Buff":           "../NN_Datasets_Grad/Train_Oliveira_BereaBuff_SAug_DNorm.h5",
    ##"Leopard":              "../NN_Datasets_Grad/Train_Oliveira_Leopard_SAug_DNorm.h5",
    ##"Castle Gate":          "../NN_Datasets_Grad/Train_Oliveira_CastleGate_SAug_DNorm.h5",
    "Berea Upper Gray":     "../NN_Datasets_Grad/Train_Oliveira_BereaUpperGray_SAug_DNorm.h5",
    ##"Berea Sinter Gray":    "../NN_Datasets_Grad/Train_Oliveira_BereaSinterGray_SAug_DNorm.h5",
    ##"Berea":                "../NN_Datasets_Grad/Train_Oliveira_Berea_SAug_DNorm.h5",
    
    
    }



#######################################################
#************ INITIALIZE DATASETS         ************#
#######################################################
datasets_data   = {}

print("Initializing datasets...")
for ds_name, ds_path in datasets.items():
    dataset_full_name = dataset_folder + ds_path
    datasets_data[ds_name] = dr.LazyDatasetTorch(
        h5_path=dataset_full_name, 
        list_ids=None, 
        x_dtype=torch.float32,
        y_dtype=torch.float32
    )
#######################################################
#****** SAMPLE-BY-SAMPLE VEL. Field ANALYSIS   *******#
#######################################################
"""
samples_to_plot = [0, 1, 2] 
component = 3
for sample_idx in samples_to_plot:
    print(f"Generating Side-by-Side Plots for Sample {sample_idx}...")
    
    # Compare Uz (component=0) Front View
    Plot_Velocity_Front_Comparison(
        datasets=datasets_data, 
        sample_idx=sample_idx, 
        slice_idx=60, 
        save_mode=True, 
        component=component,
    )
    
    # Compare Uz (component=0) Side View
    Plot_Velocity_Side_Comparison(
        datasets=datasets_data, 
        sample_idx=sample_idx, 
        slice_idx=60, 
        save_mode=True, 
        component=component,
    )
"""
#######################################################
#****** SAMPLE-BY-SAMPLE HISTOGRAM ANALYSIS   ********#
#######################################################
Compare_Histograms(
    datasets=datasets_data, 
    sample_indices=None,
    bins=1000,
    save_folder="../NN_Datasets_Grad/",
    base_filename=f"Component_Histogram"
)
