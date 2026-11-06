import os
import torch
import matplotlib.pyplot as plt

# Import Plot_LossHistory from your training script/module (change 'train_utils' to your actual filename)
from Utilities.nn_trainner import  Plot_LossHistory

def load_checkpoint_and_plot_losses(results_folder: str, output_path: str = "loss_history_from_checkpoint.png"):
    """
    Scans the results folder for the latest training checkpoint, extracts 
    the training and validation loss history arrays, and plots them.
    """
    results_folder = results_folder.rstrip("/") + "/"
    
    if not os.path.exists(results_folder):
        raise FileNotFoundError(f"Directory not found: {results_folder}")

    # Find latest checkpoint using the established prefix convention
    latest_checkpoint_path = get_latest_checkpoint(results_folder, prefix="train_checkpoint_")
    
    if latest_checkpoint_path is None:
        raise FileNotFoundError(f"No checkpoints found in {results_folder}")
        
    print(f"Loading checkpoint from: {latest_checkpoint_path}")
    
    # Load checkpoint data
    checkpoint = torch.load(latest_checkpoint_path, map_location="cpu", weights_only=False)
    
    print("\nCheckpoint contents:")
    for key, value in checkpoint.items():
        print(f"{key}: {value if not isinstance(value, (dict, list)) else type(value)}")
    
    if 'train_costs_h' not in checkpoint or 'val_costs_h' not in checkpoint:
        raise KeyError("Checkpoint does not contain 'train_costs_h' or 'val_costs_h' keys.")
        
    train_costs_h = checkpoint['train_costs_h']
    val_costs_h = checkpoint['val_costs_h']
    
    print(f"Successfully loaded loss histories up to Epoch {checkpoint.get('epoch', 'unknown')}.")
    
    # Generate and save the plot using the imported Plot_LossHistory function
    Plot_LossHistory(train_costs_h, 
                     val_costs_h, 
                     checkpoint_path=latest_checkpoint_path,
                     output_path=output_path,
                     save_mode=True
                     )
    print(f"Loss history plot successfully saved to: {output_path}")


def get_latest_checkpoint(folder_path, prefix="train_checkpoint_"):
    """
    Scans a folder for files matching the checkpoint prefix and returns 
    the path to the one with the highest epoch number.
    """
    if not os.path.exists(folder_path):
        return None

    try:
        files = os.listdir(folder_path)
    except FileNotFoundError:
        return None

    latest_file = None
    max_value   = -1
    for filename in files:
        if filename.startswith(prefix):
            try:
                name_no_ext = os.path.splitext(filename)[0]
                number_part = name_no_ext.split(prefix)[-1]
                val = int(number_part)
                if val > max_value:
                    max_value = val
                    latest_file = filename
            except ValueError:
                continue 

    if latest_file:
        return os.path.join(folder_path, latest_file)
    
    return None


# --- Example Usage ---
target_results_folder = "../NN_Results/NN_Trainning_26_August_2026_03-45PM_Job27376/"
output_plot_path = os.path.join(target_results_folder, "LossHistory_Restored.png")

load_checkpoint_and_plot_losses(
    results_folder=target_results_folder, 
    output_path=output_plot_path,
)