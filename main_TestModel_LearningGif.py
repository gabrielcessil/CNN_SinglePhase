import torch
import numpy as np
from pathlib import Path
import imageio.v2 as imageio  # stable v2 API
import os
import re
from glob import glob
import matplotlib.colors as mcolors
import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

# Assuming these are available in your env
from Utilities import dataset_reader as dr
from Utilities import nn_trainner as nnt

from Architectures.Unet   import Extended_DannyKo
from Architectures.MSnet  import JavierSantos_Extended
from Architectures.Models import SubModels_Composition

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'Computer Modern Roman', 'Liberation Serif', 'Bitstream Vera Serif']

#######################################################
#***************** HELPER FUNCTIONS ******************#
#######################################################

def Set_solids_to_value(array, bin_array, value=0, solid_value=0):
    arr = array.copy()
    arr[bin_array==solid_value] = value
    return arr

def Plot_Continuous_Domain_2D(
    values,
    filename,
    title="",
    remove_value=None,              
    colormap="viridis",
    vmin=None,
    vmax=None,
    clip_percentiles=None,          
    show_colorbar=True,
    special_colors=None,            
    dpi=300
):
    folder = os.path.dirname(filename)
    if folder and not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)

    values = np.asarray(values)
    if values.ndim == 3 and values.shape[0] == 1:
        values = values[0]
    if values.ndim != 2:
        raise ValueError(f"`values` must be 2D or (1,H,W). Got {values.shape}")

    mask = np.isnan(values)
    if remove_value is not None:
        if np.isscalar(remove_value):
            mask |= (values == remove_value)
        else:
            mask |= np.isin(values, list(remove_value))

    data = np.ma.masked_array(values, mask=mask)

    finite_vals = data.compressed()
    if finite_vals.size == 0:
        raise ValueError("All values masked or NaN; nothing to plot.")
    if (vmin is None or vmax is None):
        if clip_percentiles is not None:
            lowp, highp = clip_percentiles
            vmin_auto, vmax_auto = np.percentile(finite_vals, [lowp, highp])
        else:
            vmin_auto, vmax_auto = np.min(finite_vals), np.max(finite_vals)
        vmin = vmin if vmin is not None else vmin_auto
        vmax = vmax if vmax is not None else vmax_auto
        if vmin == vmax:
            vmin, vmax = vmin - 1e-8, vmax + 1e-8

    cmap = mpl.colormaps[colormap].copy()
    if hasattr(cmap, "set_bad"):
        cmap.set_bad((0, 0, 0, 0))  

    # --- 1. FORCE ABSOLUTE COLOR LIMITS ---
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(data, cmap=cmap, norm=norm, interpolation="none")

    if special_colors:
        for val, color in special_colors.items():
            mask_special = (values == val)
            if np.any(mask_special):
                overlay = np.zeros((*values.shape, 4))
                overlay[mask_special] = mcolors.to_rgba(color)
                ax.imshow(overlay, interpolation="none")

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel("")

    if show_colorbar:
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize="large")
        
        # --- 2. FORCE STATIC TICKS ---
        ticks = np.linspace(vmin, vmax, 5)
        cbar.set_ticks(ticks)
        cbar.ax.set_yticklabels([f"{t:.3f}" for t in ticks]) # Lock label lengths

    fig.tight_layout()
    out_path = f"{filename}.png"
    plt.savefig(out_path, bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    return out_path


def Plot_Loss_History_Frame(train_loss, val_loss, current_epoch, best_epoch, filename, dpi=300):
    fig, ax = plt.subplots(figsize=(10, 8))
    epochs = np.arange(len(train_loss))

    ax.plot(epochs, train_loss, label='Train', color='#005b96', linewidth=2.5)
    ax.plot(epochs, val_loss, label='Validation', color='#000000', linewidth=2.5)

    if best_epoch < len(val_loss):
        ax.scatter([best_epoch], [val_loss[best_epoch]], 
                   c='#2ca02c', edgecolors='black', linewidths=1.5, s=450, zorder=5, 
                   label=f'Best Model (Ep {best_epoch})', marker='*')

    idx = min(current_epoch, len(val_loss) - 1)
    ax.scatter([idx], [val_loss[idx]], 
               c='#d62728', edgecolors='black', linewidths=1.5, s=150, zorder=6, label='Current Epoch')

    ax.set_xlabel("Epochs", fontsize=20)
    ax.set_ylabel("Loss (Log Scale)", fontsize=20)
    
    ax.set_yscale('log')
    ax.tick_params(labelsize=16)
    ax.legend(frameon=False, fontsize=16)
    
    ax.grid(True, which='major', linestyle='--', alpha=0.5)
    ax.grid(True, which='minor', linestyle=':', alpha=0.2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    white_cmap = mcolors.ListedColormap(['white'])
    sm = plt.cm.ScalarMappable(cmap=white_cmap, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(['0.000', '0.000'])
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(colors='white')

    fig.tight_layout()
    plt.savefig(filename, bbox_inches='tight', dpi=dpi, facecolor='white') 
    plt.close(fig)

def get_files_dict(directory):
    files_dict = {}
    if not os.path.exists(directory):
        raise FileNotFoundError(f"The directory {directory} does not exist. Check your model_base_path.")
        
    for f in os.listdir(directory):
        if f.startswith("train_checkpoint_"):
            full_path = os.path.join(directory, f)
            if os.path.isfile(full_path):
                name_no_ext = os.path.splitext(f)[0]
                try:
                    number_part = name_no_ext.split("train_checkpoint_")[-1]
                    number = int(number_part)
                    files_dict[number] = f
                except ValueError:
                    continue
                    
    if not files_dict:
        raise ValueError(f"No valid checkpoint files found in {directory}. Make sure they are named 'train_checkpoint_X'.")
        
    return dict(sorted(files_dict.items()))


def get_masked_slices(inp, tar, out, axis='side'):
    if axis == 'front':
        slice_idx=EXAMPLES_SHAPE[0]//2
        i_slc = inp[0,    0, slice_idx, :, : ].cpu().numpy()
        t_slc = tar[0,    0, slice_idx, :, : ].cpu().numpy()
        o_slc = out[0,    0, slice_idx, :, : ].cpu().numpy()
        
    elif axis == 'side':
        slice_idx=EXAMPLES_SHAPE[1]//2
        i_slc = inp[0,    0, :, slice_idx, :].cpu().numpy()
        t_slc = tar[0,    0, :, slice_idx, :].cpu().numpy()
        o_slc = out[0,    0, :, slice_idx, :].cpu().numpy()
    
    mask = (i_slc == 0)
    return np.ma.array(t_slc, mask=mask), np.ma.array(o_slc, mask=mask)

def mean_normalize(inp, x): 
    B, C, Z, Y, Xdim = x.shape
    mag     = torch.linalg.vector_norm(x, dim=1)  
    mask    = (inp > 0)  
    mask    = mask[:, 0] 

    means = []
    for b in range(B):
        vals    = mag[b][mask[b]]
        m       = vals.mean()
        means.append(m.unsqueeze(0))
    means = torch.stack(means, dim=0).view(B, 1, 1, 1, 1)
    return x / (means + 1e-12)

def get_serif_font(size):
    font_names = [
        "times.ttf", "Times_New_Roman.ttf", "timesbd.ttf", 
        "LiberationSerif-Regular.ttf", "DejaVuSerif.ttf", "FreeSerif.ttf"
    ]
    for fn in font_names:
        try:
            return ImageFont.truetype(fn, size)
        except IOError:
            continue
    return ImageFont.load_default()

def draw_panel_title(draw, text, x_center, y_center, font):
    if hasattr(draw, "textbbox"):
        x0, y0, x1, y1 = draw.textbbox((0, 0), text, font=font)
        tw, th = x1 - x0, y1 - y0
    else:
        tw, th = draw.textsize(text, font=font)
    
    x = x_center - tw // 2
    y = y_center - th // 2
    draw.text((x, y), text, font=font, fill=(0, 0, 0, 255))


#######################################################
#******************** INPUTS *************************#
#######################################################

NN_DATASETS_DIR     = Path("./")
DATASET_NAME        = "../NN_Datasets_Grad/Test_Oliveira_Bentheimer_SAug_DNorm.h5"
EXAMPLES_SHAPE      = (120, 120, 120)
sample_idx          = 20

# LOAD MODEL
model_base_path = "../NN_Results/NN_Trainning_13_July_2026_06-02PM_Job26267/" 
model_aux       = Extended_DannyKo()
model           = model_aux.z_model


#######################################################
#***************** SETUP        **********************#
#######################################################

FRAMES_DIR = Path(model_base_path+"frames/")
FRAMES_DIR.mkdir(parents=True, exist_ok=True)

dataset_full_path   = NN_DATASETS_DIR / DATASET_NAME

dataset    = dr.LazyDatasetTorch(h5_path=dataset_full_path, 
                                list_ids=None, 
                                x_dtype=torch.float32,
                                y_dtype=torch.float32)

DEVICE = torch.device("cpu")


#######################################################
#************ EXTRACT STATIC LIMITS ******************#
#######################################################
print("Calculating fixed colormap limits (10% margin)...")
net_in_static, net_t_static = dataset[sample_idx]
net_in_static = net_in_static.unsqueeze(0).to(dtype=torch.float32)
net_t_static = net_t_static.unsqueeze(0).to(dtype=torch.float32)
net_t_static = mean_normalize(net_in_static, net_t_static)

t_masked_static, _ = get_masked_slices(net_in_static, net_t_static, net_t_static, axis='side')
vmin_base, vmax_base = np.percentile(t_masked_static.compressed(), [1, 99])

# Apply 10% Margin
val_range = vmax_base - vmin_base
fixed_vmin = vmin_base - 0.1 * val_range
fixed_vmax = vmax_base + 0.1 * val_range
print(f"Limits locked to: vmin={fixed_vmin:.4f}, vmax={fixed_vmax:.4f}")

#######################################################
#************ EVALUATE MULTIPLE CHECKPOINTS **********#
#######################################################

files = get_files_dict(model_base_path)

print("Extracting loss history from the latest checkpoint...")
max_epoch = max(files.keys())
_, latest_ckpt = nnt.load_model_from_checkpoint(model, model_base_path, epoch=max_epoch, device='cpu')

train_costs_h = latest_ckpt['train_costs_h']
val_costs_h   = latest_ckpt['val_costs_h']

loss_key = list(train_costs_h[0].keys())[0] 
train_loss_arr = [ep[loss_key] for ep in train_costs_h]
val_loss_arr   = [ep[loss_key] for ep in val_costs_h]
best_epoch     = np.argmin(val_loss_arr)

print("Computing Output from each epoch ...")
for epoch, checkpoint_name in files.items():
    print("Plotting frame from epoch ", epoch)
    model,_ = nnt.load_model_from_checkpoint(model, model_base_path, epoch=epoch, device='cpu')
    model.bin_input = True
    
    net_input, net_target  = dataset[sample_idx]
    net_input, net_target = net_input.unsqueeze(0).to(dtype=torch.float32), net_target.unsqueeze(0).to(dtype=torch.float32)
    net_output = model.predict(net_input)
    
    net_target = mean_normalize(net_input, net_target)
    net_output = mean_normalize(net_input, net_output)
    
    net_input  = net_input[0,0, :,:,:].numpy()
    net_target = net_target[0,0, :,:,:].numpy()
    net_output = net_output[0,0, :,:,:].numpy()
    
    net_error = np.abs(net_target - net_output)

    dimx, dimy, dimz = net_target.shape
    
    solid_mask                      = net_input==0
    net_input[solid_mask]           = -1
    net_target[solid_mask]          = -1
    net_output[solid_mask]          = -1
    net_error[solid_mask]           = -1

    # --- 3. ADDED remove_value=-1 TO ALL PLOT CALLS ---
    
    # Save Prediction Panel
    fname_no_ext = FRAMES_DIR / f"frame_{epoch:03d}"  
    Plot_Continuous_Domain_2D(
        values=net_output[:, dimy // 2, :],
        filename=str(fname_no_ext),
        colormap="plasma",
        show_colorbar=True,
        vmax=fixed_vmax,
        vmin=fixed_vmin,
        remove_value=-1, 
        special_colors={-1: (1, 1, 1, 1)},
    )
    
    # Save Absolute Error Panel
    err_fname_no_ext = FRAMES_DIR / f"error_{epoch:03d}"
    Plot_Continuous_Domain_2D(
        values=net_error[:, dimy // 2, :],
        filename=str(err_fname_no_ext),
        colormap="inferno", 
        show_colorbar=True,
        vmax=fixed_vmax,  
        vmin=0.0,
        remove_value=-1,
        special_colors={-1: (1, 1, 1, 1)},
    )

    # Save Loss History Plot Panel
    loss_fname = FRAMES_DIR / f"loss_{epoch:03d}.png"
    Plot_Loss_History_Frame(
        train_loss=train_loss_arr, 
        val_loss=val_loss_arr, 
        current_epoch=epoch, 
        best_epoch=best_epoch, 
        filename=str(loss_fname)
    )

# Save a static PNG of the *final* net_target (Target Plot)
left_fname_no_ext = FRAMES_DIR / "static_input_final"
Plot_Continuous_Domain_2D(
    values=net_target[:, dimy // 2, :], 
    filename=str(left_fname_no_ext),
    colormap="plasma",
    show_colorbar=True,
    vmax=fixed_vmax,  
    vmin=fixed_vmin,
    remove_value=-1,
    special_colors={-1: (1, 1, 1, 1)},
)

#######################################################
#************** STITCHING 2x2 PANELS ******************#
#######################################################

print("Stitching images into 2x2 grid...")
target_png = left_fname_no_ext.with_suffix(".png")
target_img = Image.open(target_png).convert("RGBA")

# Extract dimensions
target_h = target_img.height
target_w = target_img.width

def resize_to_match(img, target_h):
    if img.height != target_h:
        new_w = int(img.width * (target_h / img.height))
        return img.resize((new_w, target_h), Image.BICUBIC)
    return img

header_h = 100 
total_width = target_w * 2
total_height = (target_h + header_h) * 2

title_font = get_serif_font(max(24, target_h // 16))
epoch_font = get_serif_font(max(20, target_h // 18))

combined_frames = []
sorted_epochs = sorted(list(files.keys()))

for epoch in sorted_epochs:
    loss_img = Image.open(FRAMES_DIR / f"loss_{epoch:03d}.png").convert("RGBA")
    pred_img = Image.open(FRAMES_DIR / f"frame_{epoch:03d}.png").convert("RGBA")
    err_img  = Image.open(FRAMES_DIR / f"error_{epoch:03d}.png").convert("RGBA")

    loss_img = resize_to_match(loss_img, target_h)
    pred_img = resize_to_match(pred_img, target_h)
    err_img  = resize_to_match(err_img, target_h)

    canvas = Image.new("RGB", (total_width, total_height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    
    # --- SMART ALIGNMENT LOGIC ---
    err_x = (target_w - err_img.width) // 2
    err_right = err_x + err_img.width
    loss_x = err_right - loss_img.width
    
    target_x = target_w + (target_w - target_img.width) // 2
    pred_x = target_w + (target_w - pred_img.width) // 2
    
    placements = [
        (loss_img,   loss_x,   0, "Loss History"),
        (target_img, target_x, 0, "Ground Truth Target"),
        (err_img,    err_x,    1, "Absolute Error"),
        (pred_img,   pred_x,   1, "Neural Network Output")
    ]
    
    for img, img_x, row, title in placements:
        cell_y = row * (target_h + header_h)
        img_y = cell_y + header_h
        
        mask = img if img.mode == 'RGBA' else None
        canvas.paste(img, (img_x, img_y), mask)
        
        title_x = img_x + img.width // 2
        title_y = cell_y + (header_h // 3) 
        draw_panel_title(draw, title, title_x, title_y, title_font)
        
    txt  = f"Epoch: {epoch}"
    draw.text((30, 25), txt, font=epoch_font, fill=(0, 0, 0, 255))    
    
    combined_frames.append(np.array(canvas))


#######################################################
#****************** EXPORT VIDEOS ********************#
#######################################################

out_gif = FRAMES_DIR / "comparison_2x2.gif"
imageio.mimsave(out_gif, combined_frames, duration=0.4, loop=0)
print(f"Saved 2x2 comparison GIF to: {out_gif}")

out_mp4 = FRAMES_DIR / "comparison_2x2.mp4"
fps = 1 / 0.8
try:
    imageio.mimsave(
        out_mp4,
        combined_frames,
        format='FFMPEG',
        fps=fps,
        codec="libx264",
        quality=8
    )
    print(f"Saved 2x2 comparison video to: {out_mp4}")
except Exception as e:
    print(f"\nWarning: Could not save MP4. Error: {e}")