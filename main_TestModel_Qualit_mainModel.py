import os
import torch
import numpy              as np
import matplotlib.pyplot  as plt
import tensorflow         as tf
from scipy.stats          import gaussian_kde
from matplotlib.ticker    import LogLocator, LogFormatterSciNotation
from torch.utils.data     import DataLoader

from Architectures.Models import SubModels_Composition
from Architectures.Unet   import Extended_DannyKo
from Architectures.PINN_Model import MY_PIMODEL, MY_PIMODEL_2, MY_PIMODEL_3, MY_PIMODEL_4
from Architectures.MSnet  import Extended_JavierSantos

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

    print("Trainable params:      ", trainable)
    print("Non-trainable params: ", non_trainable)
    print("Total params:          ", trainable + non_trainable)

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
#************ COMPARISONS (MAGNITUDE):     ***********#
####################################################### 

def Plot_Front_Comparison(models, datapath, sample_idx=0, slice_idx=60, save_mode=False, save_tag=""):
    """Saves Target and Models Magnitude to 'Plot_Front_Comparison/' folder."""
    
    dataset    = dr.LazyDatasetTorch(h5_path=datapath, 
                                    list_ids=None, 
                                    x_dtype=torch.float32,
                                    y_dtype=torch.float32)
    
    inp, tar    = dataset[sample_idx]
    inp, tar    = inp.unsqueeze(0).to(dtype=torch.float32), tar.unsqueeze(0).to(dtype=torch.float32)
    
    # Prepare target to plot (Magnitude)
    tar_sq          = tar.squeeze(0)    
    tar_mag         = torch.sqrt(tar_sq[0]**2 + tar_sq[1]**2 + tar_sq[2]**2)
    tar_mag_masked  = get_masked_slices(inp.squeeze(0).squeeze(0), tar_mag, slice_idx, axis='front') 
    
    vmin, vmax      = np.percentile(tar_mag_masked.compressed(), [1, 99])
    
    folder = "./Plots/Plot_Front_Comparison"
    if save_mode and not os.path.exists(folder): os.makedirs(folder)

    if save_mode:
        plt.figure(figsize=(6, 6))
        plt.imshow(tar_mag_masked, cmap='plasma', vmin=vmin, vmax=vmax)
        plt.title("Target Magnitude (Front)")
        plt.axis('off')
        plt.colorbar(orientation='horizontal', fraction=0.046, pad=0.04)
        plt.savefig(f"{folder}/{save_tag}_{sample_idx}_Target.png", dpi=300, bbox_inches='tight')
        plt.close()
    else:
        num_plots = len(models) + 1
        fig, axes = plt.subplots(1, num_plots, figsize=(5 * num_plots, 6), constrained_layout=True)
        im0 = axes[0].imshow(tar_mag_masked, cmap='plasma', vmin=vmin, vmax=vmax)
        axes[0].set_title("Target Magnitude (Front)")
        axes[0].axis('off')
        plt.colorbar(im0, ax=axes[0], orientation='horizontal', fraction=0.046, pad=0.04)

    for i, (name, model) in enumerate(models.items(), 1):
        with torch.no_grad():
            out = model.predict(inp) if hasattr(model, 'predict') else model(inp)
            
        out_sq      = out.squeeze(0)
        out_mag     = torch.sqrt(out_sq[0]**2 + out_sq[1]**2 + out_sq[2]**2)
        o_mag_masked= get_masked_slices(inp.squeeze(0).squeeze(0), out_mag, slice_idx, axis='front') 
        
        vmin, vmax  = np.percentile(o_mag_masked.compressed(), [1, 99])
        
        if save_mode:
            plt.figure(figsize=(6, 6))
            plt.imshow(o_mag_masked, cmap='plasma', vmin=vmin, vmax=vmax)
            plt.title(f"{name} Magnitude (Front)")
            plt.axis('off')
            plt.colorbar(orientation='horizontal', fraction=0.046, pad=0.04)
            plt.savefig(f"{folder}/{save_tag}_{sample_idx}_{name.replace(' ', '_')}.png", dpi=300, bbox_inches='tight')
            plt.close()
        else:
            im = axes[i].imshow(o_mag_masked, cmap='plasma', vmin=vmin, vmax=vmax)
            axes[i].set_title(f"{name} Mag (Front)")
            axes[i].axis('off')
            plt.colorbar(im, ax=axes[i], orientation='horizontal', fraction=0.046, pad=0.04)
            
    if not save_mode: plt.show()


def Plot_Side_Comparison(models, datapath, sample_idx=0, slice_idx=60, save_mode=False, save_tag=""):
    """Saves Target and Models Magnitude to 'Plot_Side_Comparison/' folder."""
    
    dataset    = dr.LazyDatasetTorch(h5_path=datapath, 
                                    list_ids=None, 
                                    x_dtype=torch.float32,
                                    y_dtype=torch.float32)
    
    inp, tar    = dataset[sample_idx]
    inp, tar    = inp.unsqueeze(0).to(dtype=torch.float32), tar.unsqueeze(0).to(dtype=torch.float32)
    
    tar_sq          = tar.squeeze(0)   
    tar_mag         = torch.sqrt(tar_sq[0]**2 + tar_sq[1]**2 + tar_sq[2]**2)
    tar_mag_masked  = get_masked_slices(inp.squeeze(0).squeeze(0), tar_mag, slice_idx, axis='side') 
    
    vmin, vmax      = np.percentile(tar_mag_masked.compressed(), [1, 99])
    
    folder = "./Plots/Plot_Side_Comparison"
    if save_mode and not os.path.exists(folder): os.makedirs(folder)

    if save_mode:
        plt.figure(figsize=(6, 6))
        plt.imshow(tar_mag_masked, cmap='plasma', vmin=vmin, vmax=vmax)
        plt.title("Target Magnitude (Side)")
        plt.axis('off')
        plt.colorbar(orientation='horizontal', fraction=0.046, pad=0.04)
        plt.savefig(f"{folder}/{save_tag}_{sample_idx}_Target.png", dpi=300, bbox_inches='tight')
        plt.close()
    else:
        num_plots = len(models) + 1
        fig, axes = plt.subplots(1, num_plots, figsize=(5 * num_plots, 6), constrained_layout=True)
        im0 = axes[0].imshow(tar_mag_masked, cmap='plasma', vmin=vmin, vmax=vmax)
        axes[0].axis('off')
        axes[0].set_title("Target Magnitude (Side)")
        plt.colorbar(im0, ax=axes[0], orientation='horizontal', fraction=0.046, pad=0.04)

    for i, (name, model) in enumerate(models.items(), 1):
        with torch.no_grad():
            out = model.predict(inp) if hasattr(model, 'predict') else model(inp)
        
        out_sq      = out.squeeze(0)
        out_mag     = torch.sqrt(out_sq[0]**2 + out_sq[1]**2 + out_sq[2]**2)
        o_mag_masked= get_masked_slices(inp.squeeze(0).squeeze(0), out_mag, slice_idx, axis='side')
        
        vmin, vmax  = np.percentile(o_mag_masked.compressed(), [1, 99])
        
        if save_mode:
            plt.figure(figsize=(6, 6))
            plt.imshow(o_mag_masked, cmap='plasma', vmin=vmin, vmax=vmax)
            plt.title(f"{name} Magnitude (Side)")
            plt.axis('off')
            plt.colorbar(orientation='horizontal', fraction=0.046, pad=0.04)
            plt.savefig(f"{folder}/{save_tag}_{sample_idx}_{name.replace(' ', '_')}.png", dpi=300, bbox_inches='tight')
            plt.close()
        else:
            im = axes[i].imshow(o_mag_masked, cmap='plasma', vmin=vmin, vmax=vmax)
            axes[i].set_title(f"{name} Mag (Side)")
            axes[i].axis('off')
            plt.colorbar(im, ax=axes[i], orientation='horizontal', fraction=0.046, pad=0.04)
            
    if not save_mode: plt.show()


def Plot_Error_Comparison(models, datapath, sample_idx=0, slice_idx=60, axis='front', save_mode=False, save_tag=""):
    """Saves Absolute Error maps of the Magnitude to folder."""
    
    dataset    = dr.LazyDatasetTorch(h5_path=datapath, 
                                    list_ids=None, 
                                    x_dtype=torch.float32,
                                    y_dtype=torch.float32)
    
    inp, tar    = dataset[sample_idx]
    inp, tar    = inp.unsqueeze(0).to(dtype=torch.float32), tar.unsqueeze(0).to(dtype=torch.float32)
    
    tar_sq          = tar.squeeze(0)
    tar_mag         = torch.sqrt(tar_sq[0]**2 + tar_sq[1]**2 + tar_sq[2]**2)
    tar_mag_masked  = get_masked_slices(inp.squeeze(0).squeeze(0), tar_mag, slice_idx, axis=axis) 
    
    folder = f"./Plots/Plot_Error_Comparison_{axis}"
    if save_mode and not os.path.exists(folder): os.makedirs(folder)

    if not save_mode:
        fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 5), constrained_layout=True)

    for i, (name, model) in enumerate(models.items()):
        with torch.no_grad():
            out = model.predict(inp) if hasattr(model, 'predict') else model(inp)
            
        out_sq        = out.squeeze(0)
        out_mag       = torch.sqrt(out_sq[0]**2 + out_sq[1]**2 + out_sq[2]**2)
        o_mag_masked  = get_masked_slices(inp.squeeze(0).squeeze(0), out_mag, slice_idx, axis=axis) 
        
        error_map = np.abs(tar_mag_masked - o_mag_masked)

        if save_mode:
            plt.figure(figsize=(6, 6))
            plt.imshow(error_map, cmap='Reds')
            plt.title(f"{name} Mag Error ({axis})")
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
#************ DIVERGENCE:                  ***********#
####################################################### 

def Plot_Divergence_Comparison(models, datapath, sample_idx=0, slice_idx=60, axis='side', save_mode=False, save_tag=""):
    """Computes and plots the divergence of the velocity field."""
    
    dataset    = dr.LazyDatasetTorch(h5_path=datapath, 
                                    list_ids=None, 
                                    x_dtype=torch.float32,
                                    y_dtype=torch.float32)
    
    inp, tar    = dataset[sample_idx]
    inp, tar    = inp.unsqueeze(0).to(dtype=torch.float32), tar.unsqueeze(0).to(dtype=torch.float32)
    
    tar_sq = tar.squeeze(0) 
    
    # 1. Create a 3D mask for the entire void space
    # Assuming solid is 0 and void is != 0 in the input geometry
    void_mask_3d = (inp.squeeze(0).squeeze(0) != 0)
    
    # Divergence: dUz/dz + dUy/dy + dUx/dx
    # dim=0 is Z, dim=1 is Y, dim=2 is X
    dUz_dz = torch.gradient(tar_sq[0], dim=0)[0]
    dUy_dy = torch.gradient(tar_sq[1], dim=1)[0]
    dUx_dx = torch.gradient(tar_sq[2], dim=2)[0]
    tar_div = dUz_dz + dUy_dy + dUx_dx
    
    # 2. Compute Mean Absolute Divergence over the entire 3D void space
    tar_mean_abs_div = tar_div[void_mask_3d].abs().mean().item()
    
    tar_div_masked = get_masked_slices(inp.squeeze(0).squeeze(0), tar_div, slice_idx, axis=axis) 
    
    # Symmetric color range around 0 for divergence
    vmax = np.percentile(np.abs(tar_div_masked.compressed()), 99)
    vmin = -vmax
    
    folder = f"./Plots/Plot_Divergence_Comparison_{axis}"
    if save_mode and not os.path.exists(folder): os.makedirs(folder)

    if save_mode:
        plt.figure(figsize=(6, 6))
        ax = plt.gca()
        plt.imshow(tar_div_masked, cmap='coolwarm', vmin=vmin, vmax=vmax)
        plt.title(f"Target Divergence ({axis})")
        plt.axis('off')
        
        # Add annotation box
        ax.text(0.05, 0.95, f'Mean |∇·U|:\n{tar_mean_abs_div:.2e}', transform=ax.transAxes,
                fontsize=11, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))
                
        plt.colorbar(orientation='horizontal', fraction=0.046, pad=0.04)
        plt.savefig(f"{folder}/{save_tag}_{sample_idx}_Target.png", dpi=300, bbox_inches='tight')
        plt.close()
    else:
        num_plots = len(models) + 1
        fig, axes = plt.subplots(1, num_plots, figsize=(5 * num_plots, 6), constrained_layout=True)
        im0 = axes[0].imshow(tar_div_masked, cmap='coolwarm', vmin=vmin, vmax=vmax)
        axes[0].axis('off')
        axes[0].set_title(f"Target Div ({axis})")
        
        axes[0].text(0.05, 0.95, f'Mean |∇·U|:\n{tar_mean_abs_div:.2e}', transform=axes[0].transAxes,
                     fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))
                     
        plt.colorbar(im0, ax=axes[0], orientation='horizontal', fraction=0.046, pad=0.04)

    for i, (name, model) in enumerate(models.items(), 1):
        with torch.no_grad():
            out = model.predict(inp) if hasattr(model, 'predict') else model(inp)
        
        out_sq = out.squeeze(0)
        dUz_dz = torch.gradient(out_sq[0], dim=0)[0]
        dUy_dy = torch.gradient(out_sq[1], dim=1)[0]
        dUx_dx = torch.gradient(out_sq[2], dim=2)[0]
        out_div = dUz_dz + dUy_dy + dUx_dx
        
        # 3. Compute Mean Absolute Divergence for the prediction
        out_mean_abs_div = out_div[void_mask_3d].abs().mean().item()
        
        o_div_masked = get_masked_slices(inp.squeeze(0).squeeze(0), out_div, slice_idx, axis=axis)
        
        # Calculate dynamic bounds per-plot to view numerical noise, or fix to target bounds
        c_vmax = np.percentile(np.abs(o_div_masked.compressed()), 99)
        c_vmin = -c_vmax

        if save_mode:
            plt.figure(figsize=(6, 6))
            ax = plt.gca()
            plt.imshow(o_div_masked, cmap='coolwarm', vmin=c_vmin, vmax=c_vmax)
            plt.title(f"{name} Divergence ({axis})")
            plt.axis('off')
            
            # Add annotation box
            ax.text(0.05, 0.95, f'Mean |∇·U|:\n{out_mean_abs_div:.2e}', transform=ax.transAxes,
                    fontsize=11, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))
                    
            plt.colorbar(orientation='horizontal', fraction=0.046, pad=0.04)
            plt.savefig(f"{folder}/{save_tag}_{sample_idx}_{name.replace(' ', '_')}.png", dpi=300, bbox_inches='tight')
            plt.close()
        else:
            im = axes[i].imshow(o_div_masked, cmap='coolwarm', vmin=c_vmin, vmax=c_vmax)
            axes[i].set_title(f"{name} Div ({axis})")
            axes[i].axis('off')
            
            axes[i].text(0.05, 0.95, f'Mean |∇·U|:\n{out_mean_abs_div:.2e}', transform=axes[i].transAxes,
                         fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))
                         
            plt.colorbar(im, ax=axes[i], orientation='horizontal', fraction=0.046, pad=0.04)
            
    if not save_mode: plt.show()


#######################################################
#************ MAIN:                        ***********#
#######################################################

device              = 'cpu'
batch_size          = 1
save_mode           = True
sample_idexes       = [11, 45]

datasets        = {
    #"Spherical Pores":      "../NN_Datasets_Grad_Dist_40_5_55/Test_Silveira_SphPore_SAug_DNorm.h5",
    #"Spherical Grains":     "../NN_Datasets_Grad_Dist_40_5_55/Test_Silveira_SphGrain_SAug_DNorm.h5",
    #"Cylindrical Pores":    "../NN_Datasets_Grad_Dist_40_5_55/Test_Silveira_CylinPore_SAug_DNorm.h5",
    #"Cylindrical Grains":   "../NN_Datasets_Grad_Dist_40_5_55/Test_Silveira_CylinGrain_SAug_DNorm.h5",
    "Bentheimer":           "../NN_Datasets_Grad_Dist_40_5_55/Test_Oliveira_Bentheimer_SAug_DNorm.h5",
    #"Berea Buff":           "../NN_Datasets_Grad_Dist_40_5_55/Test_Oliveira_BereaBuff_SAug_DNorm.h5",
    "Leopard":              "../NN_Datasets_Grad_Dist_40_5_55/Test_Oliveira_Leopard_SAug_DNorm.h5",
    #"Castle Gate":          "../NN_Datasets_Grad_Dist_40_5_55/Test_Oliveira_CastleGate_SAug_DNorm.h5",
    #"Berea Upper Gray":     "../NN_Datasets_Grad_Dist_40_5_55/Test_Oliveira_BereaUpperGray_SAug_DNorm.h5",
    "Berea Sinter Gray":    "../NN_Datasets_Grad_Dist_40_5_55/Test_Oliveira_BereaSinterGray_SAug_DNorm.h5",
    #"Berea":                "../NN_Datasets_Grad_Dist_40_5_55/Test_Oliveira_Berea_SAug_DNorm.h5",
}

shape               = (120,120,120)

# --- 1. DEFINE FULL MODELS ---
models          = {}

print("Loading Models predicting all components...")

# Load Full Danny Ko Model
danny_f_model       = Extended_DannyKo()
danny_f_name        = "../NN_Results/NN_Trainning_17_September_2026_02-32PM_Job28985/model_LowerValidationLoss.pth"
danny_f_model.load_state_dict(torch.load(danny_f_name, map_location=torch.device(device), weights_only=True))
danny_f_model.eval()
models["Danny"] = danny_f_model

# Load Full MY_PIMODEL_1
pinn_model          = MY_PIMODEL()
pinn_name        = "../NN_Results/NN_Trainning_17_September_2026_03-19PM_Job28987/model_LowerValidationLoss.pth"
pinn_model.load_state_dict(torch.load(pinn_name, map_location=torch.device(device), weights_only=True))
pinn_model.eval()
models["Silveira 1"] = pinn_model

pinn_model          = MY_PIMODEL()
pinn_name        = "../NN_Results/NN_Trainning_21_September_2026_07-01PM_Job29841/model_LowerValidationLoss.pth"
pinn_model.load_state_dict(torch.load(pinn_name, map_location=torch.device(device), weights_only=True))
pinn_model.eval()
models["Silveira 1(2)"] = pinn_model

# Load Full MY_PIMODEL_2 
pinn_model          = MY_PIMODEL_2()
pinn_name        = "../NN_Results/NN_Trainning_17_September_2026_03-42PM_Job28989/model_LowerValidationLoss.pth"
pinn_model.load_state_dict(torch.load(pinn_name, map_location=torch.device(device), weights_only=True))
pinn_model.eval()
models["Silveira 2"] = pinn_model

# Load Full MY_PIMODEL_3
pinn_model          = MY_PIMODEL_3()
pinn_name        = "../NN_Results/NN_Trainning_19_September_2026_09-23AM_Job29583/model_LowerValidationLoss.pth" #ok
pinn_model.load_state_dict(torch.load(pinn_name, map_location=torch.device(device), weights_only=True))
pinn_model.eval()
models["Silveira 3"] = pinn_model

# Load Full MY_PIMODEL_4
pinn_model          = MY_PIMODEL_4()
pinn_name        = "../NN_Results/NN_Trainning_18_September_2026_07-20PM_Job29580/model_LowerValidationLoss.pth"
pinn_model.load_state_dict(torch.load(pinn_name, map_location=torch.device(device), weights_only=True))
pinn_model.eval()
models["Silveira 4"] = pinn_model

# --- 2. EXECUTION BLOCK ---

# Set the font to Times New Roman
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'Computer Modern Roman', 'Liberation Serif', 'Bitstream Vera Serif']

for dataname, datapath in datasets.items():
    print(f"Generating plots for dataset: {dataname}")
    for sample_idx in sample_idexes:
        # Plot Magnitude
        #Plot_Front_Comparison(models, datapath, sample_idx=sample_idx, slice_idx=shape[0]//2, save_mode=save_mode, save_tag=dataname)
        #Plot_Side_Comparison(models, datapath, sample_idx=sample_idx, slice_idx=shape[2]//2, save_mode=save_mode, save_tag=dataname)
        #Plot_Error_Comparison(models, datapath, sample_idx=sample_idx, slice_idx=shape[2]//2, axis='side', save_mode=save_mode, save_tag=dataname)
        
        # Plot Divergence
        Plot_Divergence_Comparison(models, datapath, sample_idx=sample_idx, slice_idx=shape[2]//2, axis='side', save_mode=save_mode, save_tag=dataname)
        #Plot_Divergence_Comparison(models, datapath, sample_idx=sample_idx, slice_idx=shape[0]//2, axis='front', save_mode=save_mode, save_tag=dataname)