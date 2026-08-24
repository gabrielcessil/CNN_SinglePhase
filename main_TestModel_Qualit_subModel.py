import torch
import numpy              as np
import matplotlib.pyplot  as plt
import tensorflow         as tf
from scipy.stats          import gaussian_kde
from matplotlib.ticker    import LogLocator, LogFormatterSciNotation
from torch.utils.data     import DataLoader

from Architectures.Unet   import Extended_DannyKo
from Architectures.MSnet  import JavierSantos_Extended
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
        # XZ Plane (slice along Y)
        i_slc = inp[:, :, slice_idx].cpu().numpy()
        t_slc = tar[:, :, slice_idx].cpu().numpy()
    
    mask = (i_slc == 0)
    return np.ma.array(t_slc, mask=mask)


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
    
    folder = "Plot_Front_Comparison"
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
    tar_z           = tar.squeeze(0)    # Remove batch dim,
    tar_z           = tar_z[component]  # Get component channel: z=0, y=1, x=2, p=3
    tar_z_masked    = get_masked_slices(inp.squeeze(0).squeeze(0), tar_z, slice_idx, axis='side') # Put zeros on solid
    
    # Prepare color range
    vmin, vmax      = np.percentile(tar_z_masked.compressed(), [1, 99])
    
    folder = "Plot_Side_Comparison"
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
        # Increased height from 5 to 6
        fig, axes = plt.subplots(1, num_plots, figsize=(5 * num_plots, 6), constrained_layout=True)
        im0 = axes[0].imshow(tar_z_masked, cmap='plasma', vmin=vmin, vmax=vmax)
        axes[0].axis('off')
        axes[0].set_title("Target (Side)")
        plt.colorbar(im0, ax=axes[0], orientation='horizontal', fraction=0.046, pad=0.04)

    for i, (name, model) in enumerate(models.items(), 1):
        with torch.no_grad():
            out = model.predict(inp) if hasattr(model, 'predict') else model(inp)
        
        out_z       = out.squeeze(0)[0]   
        o_z_masked  = get_masked_slices(inp.squeeze(0).squeeze(0), out_z, slice_idx, axis='side') # Put zeros on solid
        vmin, vmax      = np.percentile(o_z_masked.compressed(), [1, 99])
        if save_mode:
            plt.figure(figsize=(6, 6))
            plt.imshow(o_z_masked, cmap='plasma', vmin=vmin, vmax=vmax)
            plt.axis('off')
            plt.colorbar(orientation='horizontal', fraction=0.046, pad=0.04)
            plt.savefig(f"{folder}/{save_tag}_{sample_idx}_{name.replace(' ', '_')}.png", dpi=300, bbox_inches='tight')
            plt.close()
        else:
            im = axes[i].imshow(o_z_masked, cmap='plasma', vmin=vmin, vmax=vmax)
            axes[i].set_title(f"{name} (Side)")
            axes[i].axis('off')
            plt.colorbar(im, ax=axes[i], orientation='horizontal', fraction=0.046, pad=0.04)
            
    if not save_mode: plt.show()

def Plot_Error_Comparison(models, datapath, sample_idx=0, slice_idx=60, axis='front', save_mode=False, save_tag=""):
    """Saves Absolute Error maps to folder. (No target here as error is relative)."""
    
    dataset    = dr.LazyDatasetTorch(h5_path=datapath, 
                                    list_ids=None, 
                                    x_dtype=torch.float32,
                                    y_dtype=torch.float32)
    
    inp, tar    = dataset[sample_idx]
    inp, tar    = inp.unsqueeze(0).to(dtype=torch.float32), tar.unsqueeze(0).to(dtype=torch.float32)
    
    # Prepare target to plot
    tar_z           = tar.squeeze(0)[0] # Remove batch dim, get first channel
    tar_z_masked    = get_masked_slices(inp.squeeze(0).squeeze(0), tar_z, slice_idx, axis='front') # Put zeros on solid
    
    
    folder = f"Plot_Error_Comparison_{axis}"
    if save_mode and not os.path.exists(folder): os.makedirs(folder)

    if not save_mode:
        fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 5), constrained_layout=True)

    for i, (name, model) in enumerate(models.items()):
        with torch.no_grad():
            out = model.predict(inp) if hasattr(model, 'predict') else model(inp)
        out_z         = out.squeeze(0)[0]          # Remove batch dim, get first channel
        o_z_masked    = get_masked_slices(inp.squeeze(0).squeeze(0), out_z, slice_idx, axis='front') # Put zeros on solid
        
        error_map = np.abs(tar_z_masked - o_z_masked)

        if save_mode:
            plt.figure(figsize=(6, 6))
            plt.imshow(error_map, cmap='Reds')
            plt.title(f"{name} Error ({axis})")
            plt.axis('off')
            plt.colorbar(fraction=0.046, pad=0.04)
            plt.savefig(f"{folder}/{save_tag}_{sample_idx}_{name.replace(' ', '_')}.png", dpi=300, bbox_inches='tight')
            plt.close()
        else:
            im = axes[i].imshow(error_map, cmap='Reds')
            axes[i].set_title(f"{name} Error")
            axes[i].axis('off')
            plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
    if not save_mode: plt.show()
    

#######################################################
#************ MAIN:                        ***********#
#######################################################

z_direction_only    = True
device              = 'cpu'
batch_size          = 1
save_mode           = True
sample_idexes       = [11,12,13,14,15,16,17,18,19,20]#[1,2,3,4,5,6,7,8,9,10]
datasets        = {
    
    #"Ko et. al":            "../NN_Datasets_Grad/Test_Danny_SphPore_DAug_DNorm.h5",
    
    "Spherical Pores":      "../NN_Datasets_Grad/Test_Silveira_SphPore_SAug_DNorm.h5",
    "Spherical Grains":     "../NN_Datasets_Grad/Test_Silveira_SphGrain_SAug_DNorm.h5",
    "Cylindrical Pores":    "../NN_Datasets_Grad/Test_Silveira_CylinPore_SAug_DNorm.h5",
    "Cylindrical Grains":   "../NN_Datasets_Grad/Test_Silveira_CylinGrain_SAug_DNorm.h5",
     
    "Bentheimer":           "../NN_Datasets_Grad/Test_Oliveira_Bentheimer_SAug_DNorm.h5",
    "Berea Buff":           "../NN_Datasets_Grad/Test_Oliveira_BereaBuff_SAug_DNorm.h5",
    "Leopard":              "../NN_Datasets_Grad/Test_Oliveira_Leopard_SAug_DNorm.h5",
    "Castle Gate":          "../NN_Datasets_Grad/Test_Oliveira_CastleGate_SAug_DNorm.h5",
    "Berea Upper Gray":     "../NN_Datasets_Grad/Test_Oliveira_BereaUpperGray_SAug_DNorm.h5",
    "Berea Sinter Gray":    "../NN_Datasets_Grad/Test_Oliveira_BereaSinterGray_SAug_DNorm.h5",
    "Berea":                "../NN_Datasets_Grad/Test_Oliveira_Berea_SAug_DNorm.h5",
    
    }

shape               = (120,120,120)
component           = 0 # Uz=0, Uy=1, Ux=2, P=3
models          = {}
# DEFINE DATASETS
datasets        = {
    
    #"Ko et. al":            "../NN_Datasets_Grad/Test_Danny_SphPore_DAug_DNorm.h5",
    
    "Spherical Pores":      "../NN_Datasets_Grad/Test_Silveira_SphPore_SAug_DNorm.h5",
    "Spherical Grains":     "../NN_Datasets_Grad/Test_Silveira_SphGrain_SAug_DNorm.h5",
    "Cylindrical Pores":    "../NN_Datasets_Grad/Test_Silveira_CylinPore_SAug_DNorm.h5",
    "Cylindrical Grains":   "../NN_Datasets_Grad/Test_Silveira_CylinGrain_SAug_DNorm.h5",
     
    "Bentheimer":           "../NN_Datasets_Grad/Test_Oliveira_Bentheimer_SAug_DNorm.h5",
    "Berea Buff":           "../NN_Datasets_Grad/Test_Oliveira_BereaBuff_SAug_DNorm.h5",
    "Leopard":              "../NN_Datasets_Grad/Test_Oliveira_Leopard_SAug_DNorm.h5",
    "Castle Gate":          "../NN_Datasets_Grad/Test_Oliveira_CastleGate_SAug_DNorm.h5",
    "Berea Upper Gray":     "../NN_Datasets_Grad/Test_Oliveira_BereaUpperGray_SAug_DNorm.h5",
    "Berea Sinter Gray":    "../NN_Datasets_Grad/Test_Oliveira_BereaSinterGray_SAug_DNorm.h5",
    "Berea":                "../NN_Datasets_Grad/Test_Oliveira_Berea_SAug_DNorm.h5",
    
    }


# DEFINE MODELS
models          = {}

# Z-Component models
if component==0:
    danny_model         = Extended_DannyKo()
    danny_model_z       = danny_model.z_model
    model_full_name = "./Trained_Models/NN_Trainning_6_July_2026_01-12PM_Job26188/model_LowerValidationLoss.pth"
    danny_model_z.load_state_dict(torch.load(model_full_name, map_location=torch.device('cpu'), weights_only=True))
    danny_model_z.bin_input = True
    danny_model_z.eval()
    models["Ko et. al (Etapa 0)"]     = danny_model_z
    
    danny_model         = Extended_DannyKo()
    danny_model_z       = danny_model.z_model
    model_full_name = "./Trained_Models/NN_Trainning_6_July_2026_01-31PM_Job26190/model_LowerValidationLoss.pth"
    danny_model_z.load_state_dict(torch.load(model_full_name, map_location=torch.device('cpu'), weights_only=True))
    danny_model_z.bin_input = True
    danny_model_z.eval()
    models["Ko et. al (Etapa 1)"]     = danny_model_z
    
    danny_model         = Extended_DannyKo()
    danny_model_z       = danny_model.z_model
    model_full_name = "./Trained_Models/NN_Trainning_6_July_2026_01-28PM_Job26189/model_LowerValidationLoss.pth"
    danny_model_z.load_state_dict(torch.load(model_full_name, map_location=torch.device('cpu'), weights_only=True))
    danny_model_z.bin_input = True
    danny_model_z.eval()
    models["Ko et. al (Etapa 2)"]     = danny_model_z
    
    danny_model         = Extended_DannyKo()
    danny_model_z       = danny_model.z_model
    model_full_name = "./Trained_Models/NN_Trainning_13_July_2026_06-02PM_Job26267/model_LowerValidationLoss.pth"
    danny_model_z.load_state_dict(torch.load(model_full_name, map_location=torch.device('cpu'), weights_only=True))
    danny_model_z.bin_input = True
    danny_model_z.eval()
    models["Ko et. al (Etapa 3)"]     = danny_model_z

# X-Component models
elif component==2:
    danny_model         = Extended_DannyKo()
    danny_model_x       = danny_model.x_model
    model_full_name = "./Trained_Models/NN_Trainning_15_July_2026_03-59PM_Job26381/model_LowerValidationLoss.pth"
    danny_model_x.load_state_dict(torch.load(model_full_name, map_location=torch.device('cpu'), weights_only=True))
    danny_model_x.bin_input = True
    danny_model_x.eval()
    models["Ko et. al (Etapa 3)"]     = danny_model_x

# P-Component models
elif component==3:
    danny_model         = Extended_DannyKo()
    danny_model_p       = danny_model.p_model 
    model_full_name = "./Trained_Models/NN_Trainning_21_July_2026_05-22PM_Job26505/model_LowerValidationLoss.pth" # substituir apos treinar tudo
    danny_model_p.load_state_dict(torch.load(model_full_name, map_location=torch.device('cpu'), weights_only=True))
    danny_model_p.bin_input = True
    danny_model_p.eval()
    models["Ko et. al (Etapa 3)"]     = danny_model_p

    
    
elif component==5:
    # Base model
    danny_model         = Extended_DannyKo()
    
    # Z- component
    model_full_z_name = "./Trained_Models/NN_Trainning_13_July_2026_06-02PM_Job26267/model_LowerValidationLoss.pth"
    # X- component
    model_full_x_name = "./Trained_Models/NN_Trainning_15_July_2026_03-59PM_Job26381/model_LowerValidationLoss.pth"
    # P- component
    model_full_p_name = "./Trained_Models/NN_Trainning_21_July_2026_05-22PM_Job26505/model_LowerValidationLoss.pth"

    # Concatenation model (no main model)
    concat_model      = SubModels_Composition(main_model=danny_model, 
                                              z_name=model_full_z_name,
                                              x_name=model_full_x_name, 
                                              p_name=model_full_p_name, 
                                              device=device, 
                                              is_eval=True)
    
    models["Ko et. al (Etapas 3)"]     = concat_model
    
else: raise Exception(f"Specified component {component} not implemented. Please verify.")

    

# --- Execution Block ---

# Set the font to Times New Roman
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'Computer Modern Roman', 'Liberation Serif', 'Bitstream Vera Serif']


for dataname, datapath in datasets.items():
    for sample_idx in sample_idexes:
        #Plot_Front_Comparison(models, datapath, component, sample_idx= sample_idx, slice_idx=shape[0]//2, save_mode=save_mode, save_tag = save_tag)
        Plot_Side_Comparison (models, datapath, component, sample_idx= sample_idx, slice_idx=shape[2]//2, save_mode=save_mode, save_tag=dataname)
        #Plot_Error_Comparison(models, datapath, slice_idx=60, save_mode=save_mode, save_tag = save_tag, axis='side')
