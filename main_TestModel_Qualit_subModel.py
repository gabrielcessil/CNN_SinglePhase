import torch
import numpy              as np
import matplotlib.pyplot  as plt
import tensorflow         as tf
from scipy.stats          import gaussian_kde
from matplotlib.ticker    import LogLocator, LogFormatterSciNotation
from torch.utils.data     import DataLoader
from matplotlib.ticker    import ScalarFormatter

from Architectures.Unet   import Extended_DannyKo
from Architectures.MSnet  import Extended_JavierSantos
from Architectures.Models import SubModels_Composition
from Utilities            import dataset_reader as dr

  
#######################################################
#************ UTILS:                       ***********#
#######################################################

def print_n_params(model, pytorch=True):
    if pytorch:
        trainable       = sum(p.numel() for p in model.parameters() if p.requires_grad)
        non_trainable   = sum(p.numel() for p in model.parameters() if not p.requires_grad)

    else:
        trainable       = sum(tf.keras.backend.count_params(w) for w in model.trainable_weights)
        non_trainable   = sum(tf.keras.backend.count_params(w) for w in model.non_trainable_weights)

    print("Trainable params:     ", trainable)
    print("Non-trainable params: ", non_trainable)
    print("Total params:         ", trainable + non_trainable)

def get_masked_slices(inp, tar, slice_idx, axis='front'):
    """Extracts and masks 2D slices from 3D volumes based on orientation."""
    if axis == 'front':
        # XY Plane (slice along Z)
        i_slc = inp[slice_idx, :, :].cpu().numpy()
        t_slc = tar[slice_idx, :, :].cpu().numpy()
    elif axis == 'side':
        # YZ Plane (slice along Y)
        i_slc = inp[:, :, slice_idx].cpu().numpy()
        t_slc = tar[:, :, slice_idx].cpu().numpy()
    
    mask = (i_slc == 0)
    return np.ma.array(t_slc, mask=mask)

def apply_scientific_colorbar(cbar, orientation='horizontal'):
    """Force scientific notation on colorbar ticks."""

    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((0, 0))

    # Set formatter through the Colorbar object itself
    cbar.formatter = formatter

    # Update ticks after assigning the formatter
    cbar.update_ticks()

    # Ensure scientific notation offset is visible
    if orientation == 'horizontal':
        cbar.ax.xaxis.get_offset_text().set_visible(True)
    else:
        cbar.ax.yaxis.get_offset_text().set_visible(True)
  
#######################################################
#************ COMPARISONS:                 ***********#
####################################################### 

import os
def Plot_Front_Comparison(models, datapath, component, sample_idx=0, slice_idx=60, save_mode=False, save_tag=""):
    """Saves Target and Models to 'Plot_Front_Comparison/' folder."""
    
    dataset    = dr.LazyDatasetTorch(h5_path=datapath, 
                                    list_ids=None, 
                                    x_dtype=torch.float32,
                                    y_dtype=torch.float32)
    
    inp, tar    = dataset[sample_idx]
    inp, tar    = inp.unsqueeze(0).to(dtype=torch.float32), tar.unsqueeze(0).to(dtype=torch.float32)
    
    # Prepare target to plot
    tar_z           = tar.squeeze(0)    # Remove batch dim,
    tar_z           = tar_z[component]  # Get component channel: z=0, y=1, x=2, p=3
    tar_z_masked    = get_masked_slices(inp.squeeze(0).squeeze(0), tar_z, slice_idx, axis='front') # Put zeros on solid
    
    # Prepare color range
    vmin, vmax      = np.percentile(tar_z_masked.compressed(), [1, 99])
    
    folder = "./Plots/Plot_Front_Comparison"
    if save_mode and not os.path.exists(folder): os.makedirs(folder)

    if save_mode:
        # Save Target
        plt.figure(figsize=(6, 6))
        plt.imshow(tar_z_masked, cmap='plasma', vmin=vmin, vmax=vmax)
        plt.axis('off')
        plt.colorbar(orientation='horizontal', fraction=0.046, pad=0.04)
        plt.savefig(f"{folder}/{save_tag}_{sample_idx}_Target.png", dpi=300, bbox_inches='tight')
        plt.close()
    else:
        num_plots = len(models) + 1
        # Increased height from 5 to 6 to fit horizontal colorbars nicely
        fig, axes = plt.subplots(1, num_plots, figsize=(5 * num_plots, 6), constrained_layout=True)
        im0 = axes[0].imshow(tar_z_masked, cmap='plasma', vmin=vmin, vmax=vmax)
        axes[0].set_title("Target (Front View)")
        axes[0].axis('off')
        plt.colorbar(im0, ax=axes[0], orientation='horizontal', fraction=0.046, pad=0.04)

    for i, (name, model) in enumerate(models.items(), 1):
        with torch.no_grad():
            out = model.predict(inp) if hasattr(model, 'predict') else model(inp)
            
        out_z       = out.squeeze(0)[0]          # Remove batch dim, get first channel
        o_z_masked  = get_masked_slices(inp.squeeze(0).squeeze(0), out_z, slice_idx, axis='front') # Put zeros on solid
        vmin, vmax      = np.percentile(o_z_masked.compressed(), [1, 99])
        if save_mode:
            plt.figure(figsize=(6, 6))
            plt.imshow(o_z_masked, cmap='plasma', vmin=vmin, vmax=vmax)
            plt.axis('off')
            plt.colorbar(orientation='horizontal', fraction=0.046, pad=0.04)
            plt.savefig(f"{folder}/{save_tag}_{sample_idx}_{name.replace(' ', '_')}.png", dpi=300, bbox_inches='tight')
            plt.close()
        else:
            vmin, vmax      = np.percentile(o_z_masked.compressed(), [1, 99])
            im = axes[i].imshow(o_z_masked, cmap='plasma', vmin=vmin, vmax=vmax)
            axes[i].set_title(f"{name} (Front)")
            axes[i].axis('off')
            plt.colorbar(im, ax=axes[i], orientation='horizontal', fraction=0.046, pad=0.04)
            
    if not save_mode: plt.show()


def Plot_Side_Comparison(models, datapath, component, sample_idx=0, slice_idx=60, save_mode=False, save_tag=""):
    """Saves Target and Models to 'Plot_Side_Comparison/' folder."""
    
    dataset    = dr.LazyDatasetTorch(h5_path=datapath, 
                                    list_ids=None, 
                                    x_dtype=torch.float32,
                                    y_dtype=torch.float32)
    
    inp, tar    = dataset[sample_idx] # Shape (C,Z,Y,X)
    inp, tar    = inp.unsqueeze(0).to(dtype=torch.float32), tar.unsqueeze(0).to(dtype=torch.float32) # Add channel for prediction
    
    # Prepare target to plot
    tar_z         = tar.squeeze(0)    # Remove batch dim,
    tar_z         = tar_z[component]  # Get component channel: z=0, y=1, x=2, p=3
    tar_z_masked    = get_masked_slices(inp.squeeze(0).squeeze(0), tar_z, slice_idx, axis='side') # Put zeros on solid
    
    # Prepare color range
    vmin, vmax      = tar_z_masked.mean()-tar_z_masked.std()*3, tar_z_masked.mean()+tar_z_masked.std()*3
    vmin, vmax      = max(vmin, tar_z_masked.min()), min(vmax, tar_z_masked.max())
    vmin, vmax      = round(vmin,3), round(vmax,3)
    folder = "./Plots/Plot_Side_Comparison"
    if save_mode and not os.path.exists(folder): os.makedirs(folder)

    if save_mode:
        # Save Target
        plt.figure(figsize=(6, 6))
        im_t = plt.imshow(tar_z_masked, cmap='plasma', vmin=vmin, vmax=vmax)
        plt.axis('off')
        cbar_t = plt.colorbar(im_t, orientation='horizontal', fraction=0.046, pad=0.04)
        cbar_t.set_ticks(np.linspace(vmin, vmax, 5))
        apply_scientific_colorbar(cbar_t)
        plt.savefig(f"{folder}/{save_tag}_{sample_idx}_Target.png", dpi=300, bbox_inches='tight')
        plt.close()
    else:
        num_plots = len(models) + 1
        # Increased height from 5 to 6
        fig, axes = plt.subplots(1, num_plots, figsize=(5 * num_plots, 6), constrained_layout=True)
        im0 = axes[0].imshow(tar_z_masked, cmap='plasma', vmin=vmin, vmax=vmax)
        axes[0].axis('off')
        axes[0].set_title("Target (Side)")
        cbar0 = plt.colorbar(im0, ax=axes[0], orientation='horizontal', fraction=0.046, pad=0.04)
        cbar0.set_ticks(np.linspace(vmin, vmax, 5))
        apply_scientific_colorbar(cbar0)

    for i, (name, model) in enumerate(models.items(), 1):
        with torch.no_grad():
            out = model.predict(inp) if hasattr(model, 'predict') else model(inp)
        
        out_z       = out.squeeze(0)[0]    
        o_z_masked  = get_masked_slices(inp.squeeze(0).squeeze(0), out_z, slice_idx, axis='side') # Put zeros on solid
        vmin, vmax  = o_z_masked.mean()-o_z_masked.std()*3, o_z_masked.mean()+o_z_masked.std()*3
        vmin, vmax  = max(vmin, o_z_masked.min()), min(vmax, o_z_masked.max())
        vmin, vmax  = round(vmin,3), round(vmax,3)
        
        if save_mode:
            plt.figure(figsize=(6, 6))
            im_m = plt.imshow(o_z_masked, cmap='plasma', vmin=vmin, vmax=vmax)
            plt.axis('off')
            cbar_m = plt.colorbar(im_m, orientation='horizontal', fraction=0.046, pad=0.04)
            cbar_m.set_ticks(np.linspace(vmin, vmax, 5))
            apply_scientific_colorbar(cbar_m)
            plt.savefig(f"{folder}/{save_tag}_{sample_idx}_{name.replace(' ', '_')}.png", dpi=300, bbox_inches='tight')
            plt.close()
        else:
            im = axes[i].imshow(o_z_masked, cmap='plasma', vmin=vmin, vmax=vmax)
            axes[i].set_title(f"{name} (Side)")
            axes[i].axis('off')
            cbar = plt.colorbar(im, ax=axes[i], orientation='horizontal', fraction=0.046, pad=0.04)
            cbar.set_ticks(np.linspace(vmin, vmax, 5))
            apply_scientific_colorbar(cbar)
            
    if not save_mode: plt.show()



def Plot_Side_Error(models, datapath, component, sample_idx=0, slice_idx=60, save_mode=False, save_tag=""):
    
    dataset    = dr.LazyDatasetTorch(h5_path=datapath, 
                                    list_ids=None, 
                                    x_dtype=torch.float32,
                                    y_dtype=torch.float32)
    
    inp, tar    = dataset[sample_idx] # Shape (C,Z,Y,X)
    inp, tar    = inp.unsqueeze(0).to(dtype=torch.float32), tar.unsqueeze(0).to(dtype=torch.float32) # Add channel for prediction
    
    # Prepare target to plot
    tar_z           = tar.squeeze(0)    # Remove batch dim,
    tar_z           = tar_z[component]  # Get component channel: z=0, y=1, x=2, p=3
    tar_z_masked    = get_masked_slices(inp.squeeze(0).squeeze(0), tar_z, slice_idx, axis='side') # Put zeros on solid
    
    # Prepare color range
    
    folder = "./Plots/Plot_Side_Error"
    if save_mode and not os.path.exists(folder): os.makedirs(folder)


    for i, (name, model) in enumerate(models.items(), 1):
        with torch.no_grad():
            out = model.predict(inp) if hasattr(model, 'predict') else model(inp)
        
        out_z       = out.squeeze(0)[0]   
        o_z_masked  = get_masked_slices(inp.squeeze(0).squeeze(0), out_z, slice_idx, axis='side') # Put zeros on solid
        
        
        e = (out_z-tar_z)/out_z.abs().mean()
        e_masked =  get_masked_slices(inp.squeeze(0).squeeze(0), e, slice_idx, axis='side')
        limit = np.percentile(np.abs(e_masked.compressed()), 99)
        vmin, vmax = -limit, limit
        if save_mode:
            plt.figure(figsize=(6, 6))
            plt.imshow(e_masked, cmap='coolwarm', vmin=vmin, vmax=vmax)
            plt.axis('off')
            plt.colorbar(orientation='horizontal', fraction=0.046, pad=0.04)
            plt.savefig(f"{folder}/{save_tag}_{sample_idx}_{name.replace(' ', '_')}_error.png", dpi=300, bbox_inches='tight')
            plt.close()
        
            
    if not save_mode: plt.show()

    

#######################################################
#************ MAIN:                        ***********#
#######################################################

z_direction_only    = True
device              = 'cpu'
batch_size          = 1
save_mode           = True
sample_idexes       = [76, 68, 23]
datasets = {
    # ID
    "Spherical Pores":      "../NN_Datasets_Grad_Dist_40_5_55/Test_Silveira_SphPore_SAug_DNorm.h5", #76
    "Spherical Grains":     "../NN_Datasets_Grad_Dist_40_5_55/Test_Silveira_SphGrain_SAug_DNorm.h5", #68
    #"Cylindrical Pores":    "../NN_Datasets_Grad_Dist_40_5_55/Test_Silveira_CylinPore_SAug_DNorm.h5",
    #"Cylindrical Grains":   "../NN_Datasets_Grad_Dist_40_5_55/Test_Silveira_CylinGrain_SAug_DNorm.h5",
    "Bentheimer":           "../NN_Datasets_Grad_Dist_40_5_55/Test_Oliveira_Bentheimer_SAug_DNorm.h5", # 23
    #"Berea Buff":           "../NN_Datasets_Grad_Dist_40_5_55/Test_Oliveira_BereaBuff_SAug_DNorm.h5",
    #"Leopard":              "../NN_Datasets_Grad_Dist_40_5_55/Test_Oliveira_Leopard_SAug_DNorm.h5",
    #"Castle Gate":          "../NN_Datasets_Grad_Dist_40_5_55/Test_Oliveira_CastleGate_SAug_DNorm.h5",
    #"Berea Upper Gray":     "../NN_Datasets_Grad_Dist_40_5_55/Test_Oliveira_BereaUpperGray_SAug_DNorm.h5",
    #"Berea Sinter Gray":    "../NN_Datasets_Grad_Dist_40_5_55/Test_Oliveira_BereaSinterGray_SAug_DNorm.h5",
    #"Berea":                "../NN_Datasets_Grad_Dist_40_5_55/Test_Oliveira_Berea_SAug_DNorm.h5",
    
    # Previous
    #"Sph Pore (Ko et al)":  "../NN_Datasets_Grad_Dist_40_5_55/Test_Danny_SphPore_DAug_DNorm.h5",
    
    # OOD (Type)
    #"Ketton 120³":          "../NN_Datasets_LBPM_SpeedUp/Test_ImpCol_Ketton_120_120_120_SAug_DNorm.h5",
    #"Estaillades 120³":     "../NN_Datasets_LBPM_SpeedUp/Test_ImpCol_Estaillades_120_120_120_SAug_DNorm.h5",
    #"Doddington 120³":      "../NN_Datasets_LBPM_SpeedUp/Test_ImpCol_Doddington_120_120_120_SAug_DNorm.h5",
    
    # OOD (Size and Type)
    #"Ketton 256³":          "../NN_Datasets_LBPM_SpeedUp/Test_ImpCol_Ketton_256_256_256_SAug_DNorm.h5", #16
    #"Estaillades 256³":     "../NN_Datasets_LBPM_SpeedUp/Test_ImpCol_Estaillades_256_256_256_SAug_DNorm.h5",
    #"Doddington 256³":      "../NN_Datasets_LBPM_SpeedUp/Test_ImpCol_Doddington_256_256_256_SAug_DNorm.h5" #4
}
shape               = (120,120,120)


#datasets = {
#    "Ketton 256³":          "../NN_Datasets_LBPM_SpeedUp/Test_ImpCol_Ketton_256_256_256_SAug_DNorm.h5",
#    }
#shape               = (256,256,256)

component           = 0 # Uz=0, Uy=1, Ux=2, P=3


# DEFINE MODELS
models_dict          = {}

# Z-Component models
if component==0:
    #==========================================================================
    """
    danny_model_base3 = Extended_DannyKo()
    model_z_3 = danny_model_base3.z_model
    model_z_3.load_state_dict(torch.load("./Trained_Models/NN_Trainning_9_September_2026_02-04PM_Job28308/model_LowerValidationLoss.pth", map_location=torch.device(device), weights_only=True))
    model_z_3.bin_input = True
    model_z_3.eval()
    models_dict["Ko et al (D0 N4)"] = model_z_3
    
    danny_model_base3 = Extended_DannyKo()
    model_z_3 = danny_model_base3.z_model
    model_z_3.load_state_dict(torch.load("./Trained_Models/NN_Trainning_9_September_2026_01-49PM_Job28304/model_LowerValidationLoss.pth", map_location=torch.device(device), weights_only=True))
    model_z_3.bin_input = True
    model_z_3.eval()
    models_dict["Ko et al (D1 N4)"] = model_z_3
    
    danny_model_base3 = Extended_DannyKo()
    model_z_3 = danny_model_base3.z_model
    model_z_3.load_state_dict(torch.load("./Trained_Models/NN_Trainning_9_September_2026_01-51PM_Job28305/model_LowerValidationLoss.pth", map_location=torch.device(device), weights_only=True))
    model_z_3.bin_input = True
    model_z_3.eval()
    models_dict["Ko et al (D2 N4)"] = model_z_3
    
    danny_model_base3 = Extended_DannyKo()
    model_z_3 = danny_model_base3.z_model
    model_z_3.load_state_dict(torch.load("./Trained_Models/NN_Trainning_26_August_2026_03-45PM_Job27376/model_LowerValidationLoss.pth", map_location=torch.device(device), weights_only=True))
    model_z_3.bin_input = True
    model_z_3.eval()
    models_dict["Ko et al (D3 N4)"] = model_z_3
    """
    #==========================================================================
    
    
    #==========================================================================
    # Increasing Diversity - Javier Santos

    javier_model_base3 = Extended_JavierSantos()
    model_z_3 = javier_model_base3.z_model
    model_z_3.load_state_dict(torch.load("./Trained_Models/NN_Trainning_9_September_2026_01-53PM_Job28306/model_LowerValidationLoss.pth", map_location=torch.device(device), weights_only=True))
    model_z_3.bin_input = False
    models_dict["Javier Santos (D1 N4)"] = model_z_3
    
    javier_model_base3 = Extended_JavierSantos()
    model_z_3 = javier_model_base3.z_model
    model_z_3.load_state_dict(torch.load("./Trained_Models/NN_Trainning_10_September_2026_12-33PM_Job28377/model_LowerValidationLoss.pth", map_location=torch.device(device), weights_only=True))
    model_z_3.bin_input = False
    models_dict["Javier Santos (D3 N4)"] = model_z_3
    #==========================================================================

# X-Component models
elif component==2:
    danny_model_base3 = Extended_DannyKo()
    model_x_3 = danny_model_base3.x_model
    model_x_3.load_state_dict(torch.load("./Trained_Models/NN_Trainning_26_August_2026_06-21PM_Job27380/model_LowerValidationLoss.pth", map_location=torch.device(device), weights_only=True))
    model_x_3.bin_input = True
    model_x_3.eval()
    models_dict["Ko et al (D3) X"] = model_x_3

# P-Component models
elif component==3:
    danny_model_base3 = Extended_DannyKo()
    model_p_3 = danny_model_base3.p_model
    model_p_3.load_state_dict(torch.load("./Trained_Models/NN_Trainning_26_August_2026_03-47PM_Job27377/model_LowerValidationLoss.pth", map_location=torch.device(device), weights_only=True))
    model_p_3.bin_input = True
    model_p_3.eval()
    models_dict["Ko et al (D3) P"] = model_p_3

    
    
elif component==5:
   print("Initializing Neural Network Models...")
   danny_model = Extended_DannyKo() 
   model_full_z_name = "./Trained_Models/NN_Trainning_26_August_2026_03-45PM_Job27376/model_LowerValidationLoss.pth"
   model_full_x_name = "./Trained_Models/NN_Trainning_26_August_2026_06-21PM_Job27380/model_LowerValidationLoss.pth"
   model_full_p_name = "./Trained_Models/NN_Trainning_26_August_2026_03-47PM_Job27377/model_LowerValidationLoss.pth"

   model = SubModels_Composition( 
       main_model=danny_model,  
       z_name=model_full_z_name, 
       x_name=model_full_x_name,  
       p_name=model_full_p_name,  
       device=device,  
       is_eval=True 
   ) 
   models_dict["Ko et al (D3)"] = model
    
else: raise Exception(f"Specified component {component} not implemented. Please verify.")

    

# --- Execution Block ---

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",

    "font.size": 30,          # Default font size
    "axes.titlesize": 18,     # Plot titles
    "axes.labelsize": 20,     # Axis labels
    "xtick.labelsize": 20,    # X-axis tick labels
    "ytick.labelsize": 22,    # Y-axis tick labels
    "legend.fontsize": 14,    # Legend
    "figure.titlesize": 20,   # Figure title

    "axes.linewidth": 1.0,
})

for dataname, datapath in datasets.items():
    for sample_idx in sample_idexes:
        #Plot_Front_Comparison(models_dict, datapath, component, sample_idx= sample_idx, slice_idx=shape[0]//2, save_mode=save_mode, save_tag = save_tag)
        Plot_Side_Comparison (models_dict, datapath, component, sample_idx= sample_idx, slice_idx=shape[2]//2, save_mode=save_mode, save_tag=dataname)
        #Plot_Side_Error(models_dict, datapath, component, sample_idx= sample_idx, slice_idx=shape[2]//2, save_mode=save_mode, save_tag=dataname)
        