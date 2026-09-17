import torch.nn as nn
import json
import torch
import os
import sys
import numpy as np
import argparse
from   torch.utils.data import DataLoader

from Utilities import loss_functions as lf
from Utilities import nn_trainner as nnt
from Utilities import model_handler as mh
from Utilities import dataset_reader as dr
from Architectures import Unet
from Architectures import MSnet
from Architectures import PINN_Model


#######################################################
#************ USER INPUTS (from command line):   *****#
#######################################################

# Read parsed input
parser = argparse.ArgumentParser(description="Neural Networks Training Inputs")
parser.add_argument('--config', type=str, default='config.json', help="Path to .json file with training configurations. (Default: config.json)")
parser.add_argument('--folder', type=str, default=None,          help="If passed, ignores --config and uses metadata.json inside this folder to restart training.")
args = parser.parse_args()
# If --folder was passed: use metadata.json from it
if args.folder is not None:  
    json_path = os.path.join("../NN_Results/"+args.folder, "metadata.json")
    print(f"[*] Resuming/Loading configs from results folder: {json_path}")
# If not, use the --config (Default: config.json) to train
else:
    json_path = args.config
    print(f"[*] Loading configs from standard config file: {json_path}")
    
# Finnaly, read the proper .json
with open(json_path, 'r') as file:
    config = json.load(file)
    


#######################################################
#************ USER INPUTS (from .json):    ***********#
#######################################################

# Model Aspects
model_name              = config["model_name"]
binary_input            = config["binary_input"]
# Data aspects
NN_dataset_folder       = config["NN_dataset_folder"]
dataset_train_name      = config["dataset_train_name"]
dataset_valid_name      = config["dataset_valid_name"]
t_range                 = config.get("train_range", None)
v_range                 = config.get("valid_range", None)
train_range             = tuple(t_range) if t_range is not None else None
valid_range             = tuple(v_range) if v_range is not None else None
train_fraction          = config.get("train_fraction", 1) if train_range is None else 1
valid_fraction          = config.get("valid_fraction", 1) if valid_range is None else 1

# Hardware aspects
num_workers             = config["num_workers"]
num_threads             = config.get("num_threads", None)

# Learning aspects
batch_size              = config["batch_size"]
N_epochs                = config["N_epochs"]
partial_epochs          = config["partial_epochs"]
patience                = config.get("patience", N_epochs//10)
tolerance               = config["tolerance"]
learning_rate           = config["learning_rate"]
backPropagation_loss    = config["backPropagation_loss"]
earlyStopping_loss      = config.get("earlyStopping_loss", backPropagation_loss)
optimizer               = config["optimizer"]
weight_init             = config["weight_init"]
seed                    = config.get("seed", 42)
train_comment           = config.get("train_comment", "No comments included.")
device_set              = config["device"]

# Set seed to random initializations
nnt.set_global_seed(seed) 

#######################################################
#************ HANDLE RESULTS FOLDER:       ***********#
#######################################################

# If no configuration folder was passed: create a new folder for results
if args.folder is not None:  
    NN_results_folder       = args.folder
    # Ignore and update .json (the metadata.json saved in the folder will have the correct path)
    config["NN_results_folder"] = NN_results_folder
else:
    # If a folder was specified in .json:
    NN_results_folder       = config["NN_results_folder"]
    # If the specified value was None: create a new folder for the data
    if NN_results_folder is None:
        NN_results_folder           = nnt.create_training_data_folder(base_dir="../NN_Results")
        # Ignore and update .json (the metadata.json saved in the folder will have the new path)
        config["NN_results_folder"] = NN_results_folder
        
# Redirect prints to results folder
nnt.set_logger_output_folder(NN_results_folder)


#######################################################
#************ HANDLE DEVICE CHOICE:       ***********#
#######################################################

if isinstance(device_set, int):
    torch.cuda.set_device(device_set)
    device = torch.device(f'cuda:{device_set}')
    print('Current device name:', torch.cuda.get_device_name(device))
elif device_set is None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
else:
    device = torch.device(device_set)
print('Current device:     ', device)

dtype                   = torch.float32



#######################################################
#************ TRACKED LOSS FUNCTIONS:      ***********#
#######################################################

loss_functions  = {
    # Optimization Loss Functions:          "Thresholded" = False, to evaluate the outputs 
    "PI-MSE":                  {"obj":  lf.MSE_Divergent(div_weight=3),              "Thresholded": False},
    # Perfomance analysis Loss Functions:   "Thresholded" = True, to evaluate in final prediction mode
    "Divergent":               {"obj":  lf.Divergent(),                              "Thresholded": True}, 
    "MSE in Void Space":       {"obj":  lf.Mask_LossFunction(nn.MSELoss()),          "Thresholded": True}, 
    "Bias Error":              {"obj":  lf.Mask_LossFunction(lf.MeanBiasError()),    "Thresholded": True},
    "Pearson Correlation":     {"obj":  lf.Mask_LossFunction(lf.PearsonCorr()),      "Thresholded": True},
}


#######################################################
#************ REGISTER METADATA **********************#
#######################################################    

print("\n\nConfigurations:")
print(json.dumps(config, indent=4))
print("\n")
metadata_file = nnt.save_metadata(
    config, 
    loss_functions, 
)
print(f"Metadata saved at: {metadata_file}")


#######################################################
#************ LOADING DATA          ******************#
#######################################################

print("Loading Training Data ... ")
# Prepares dataset names for MultiLazy class (that receives a list of '.h5' files)
if isinstance(dataset_train_name, list):
    dataset_train_full_name = [os.path.join(NN_dataset_folder, item) for item in dataset_train_name]
    train_ds                = dr.MultiLazyDatasetTorch(h5_paths = dataset_train_full_name,
                                                       x_dtype = torch.float32,
                                                       y_dtype = torch.float32,
                                                       fraction= train_fraction)
    if train_range is not None: raise Exception("Setting the index interval is not possible if a list of datasets is provided.")

# Prepares dataset name for single Lazy class (that receives one '.h5' file)
else:
    dataset_train_full_name = os.path.join(NN_dataset_folder, dataset_train_name)
    t_list_ids              = None if train_range is None else np.arange(train_range[0],train_range[1])
    train_ds                = dr.LazyDatasetTorch(h5_path  = dataset_train_full_name,
                                                  list_ids = t_list_ids,
                                                  x_dtype  = torch.float32,
                                                  y_dtype  = torch.float32,
                                                  fraction= train_fraction)
print(f"  - {len(train_ds)} samples considered.")


print("Loading Validation Data ... ")
# Prepares dataset names for MultiLazy class (that receives a list of '.h5' files)
if isinstance(dataset_valid_name, list):
    dataset_valid_full_name = [os.path.join(NN_dataset_folder, item) for item in dataset_valid_name]
    valid_ds                = dr.MultiLazyDatasetTorch(h5_paths = dataset_valid_full_name,
                                                       x_dtype = torch.float32,
                                                       y_dtype = torch.float32,
                                                       fraction= valid_fraction)
    if valid_range is not None: raise Exception("Setting the index interval is not possible if a list of datasets is provided.")
    
# Prepares dataset name for single Lazy class (that receives one '.h5' file)
else:
    dataset_valid_full_name = os.path.join(NN_dataset_folder, dataset_valid_name)
    v_list_ids              = None if valid_range is None else np.arange(valid_range[0],valid_range[1]), 
    valid_ds                = dr.LazyDatasetTorch(h5_path = dataset_valid_full_name, 
                                                  list_ids= v_list_ids,
                                                  x_dtype = torch.float32,
                                                  y_dtype = torch.float32,
                                                  fraction= valid_fraction)
print(f"  - {len(valid_ds)} samples considered.")



#######################################################
#******************** MODEL **************************#
#######################################################

print("Loading Model ... ")

            
if model_name=="javier_zyxp":
    model   = MSnet.JavierSantos_Extended()
    # Make loss function multiscale 
    for loss_name, items in loss_functions.items():
        if not items["Thresholded"]: 
            loss_functions[loss_name]["obj"] = MSnet.MultiScaleLoss(loss_functions[loss_name]["obj"], norm_mode='var')
    
    # Loading pre-trained sub-models
    model_full_name = "./Trained_Models/None.pth"
    model.z_model.load_state_dict(torch.load(model_full_name, map_location=torch.device(device_set), weights_only=True))
    
    model_full_name = "./Trained_Models/None.pth"
    model.y_model.load_state_dict(torch.load(model_full_name, map_location=torch.device(device_set), weights_only=True))
    
    model_full_name = "./Trained_Models/None.pth"
    model.x_model.load_state_dict(torch.load(model_full_name, map_location=torch.device(device_set), weights_only=True))
    
    model_full_name = "./Trained_Models/None.pth"
    model.p_model.load_state_dict(torch.load(model_full_name, map_location=torch.device(device_set), weights_only=True))
    
    # Freeze sub-models    
    nnt.freeze_on_training([model.z_model, model.y_model, model.x_model, model.p_model])

elif model_name=="danny_zyxp":
    
    model_full_z_name = "./Trained_Models/NN_Trainning_26_August_2026_03-45PM_Job27376/model_LowerValidationLoss.pth"
    model_full_x_name = "./Trained_Models/NN_Trainning_26_August_2026_06-21PM_Job27380/model_LowerValidationLoss.pth"
    model_full_p_name = "./Trained_Models/NN_Trainning_26_August_2026_03-47PM_Job27377/model_LowerValidationLoss.pth"
    
    model = Unet.Extended_DannyKo()
    # Loading pre-trained sub-models
    model.z_model.load_state_dict(torch.load(model_full_z_name, map_location=torch.device(device_set), weights_only=True))
    model.x_model.load_state_dict(torch.load(model_full_x_name, map_location=torch.device(device_set), weights_only=True))
    model.p_model.load_state_dict(torch.load(model_full_p_name, map_location=torch.device(device_set), weights_only=True))
    
    # Freeze sub-models
    nnt.freeze_on_training([model.z_model, model.y_model, model.x_model, model.p_model])
    
elif model_name=="silveira_zyxp_1":
    
    model_full_z_name = "./Trained_Models/NN_Trainning_26_August_2026_03-45PM_Job27376/model_LowerValidationLoss.pth"
    model_full_x_name = "./Trained_Models/NN_Trainning_26_August_2026_06-21PM_Job27380/model_LowerValidationLoss.pth"
    model_full_p_name = "./Trained_Models/NN_Trainning_26_August_2026_03-47PM_Job27377/model_LowerValidationLoss.pth"
    
    model = PINN_Model.MY_PIMODEL()
    model.z_model.load_state_dict(torch.load(model_full_z_name, map_location=torch.device(device_set), weights_only=True))
    model.x_model.load_state_dict(torch.load(model_full_x_name, map_location=torch.device(device_set), weights_only=True))
    model.p_model.load_state_dict(torch.load(model_full_p_name, map_location=torch.device(device_set), weights_only=True))
    # Freeze sub-models
    nnt.freeze_on_training([model.z_model, model.y_model, model.x_model, model.p_model])
    
elif model_name=="silveira_zyxp_2":
    
    model_full_z_name = "./Trained_Models/NN_Trainning_26_August_2026_03-45PM_Job27376/model_LowerValidationLoss.pth"
    model_full_x_name = "./Trained_Models/NN_Trainning_26_August_2026_06-21PM_Job27380/model_LowerValidationLoss.pth"
    model_full_p_name = "./Trained_Models/NN_Trainning_26_August_2026_03-47PM_Job27377/model_LowerValidationLoss.pth"
    
    model = PINN_Model.MY_PIMODEL_2()
    model.z_model.load_state_dict(torch.load(model_full_z_name, map_location=torch.device(device_set), weights_only=True))
    model.x_model.load_state_dict(torch.load(model_full_x_name, map_location=torch.device(device_set), weights_only=True))
    model.p_model.load_state_dict(torch.load(model_full_p_name, map_location=torch.device(device_set), weights_only=True))
    # Freeze sub-models
    nnt.freeze_on_training([model.z_model, model.y_model, model.x_model, model.p_model])
    
elif model_name=="silveira_zyxp_3":
    
    model_full_z_name = "./Trained_Models/NN_Trainning_26_August_2026_03-45PM_Job27376/model_LowerValidationLoss.pth"
    model_full_x_name = "./Trained_Models/NN_Trainning_26_August_2026_06-21PM_Job27380/model_LowerValidationLoss.pth"
    model_full_p_name = "./Trained_Models/NN_Trainning_26_August_2026_03-47PM_Job27377/model_LowerValidationLoss.pth"
    
    model = PINN_Model.MY_PIMODEL_3()
    model.z_model.load_state_dict(torch.load(model_full_z_name, map_location=torch.device(device_set), weights_only=True))
    model.x_model.load_state_dict(torch.load(model_full_x_name, map_location=torch.device(device_set), weights_only=True))
    model.p_model.load_state_dict(torch.load(model_full_p_name, map_location=torch.device(device_set), weights_only=True))
    # Freeze sub-models
    nnt.freeze_on_training([model.z_model, model.y_model, model.x_model, model.p_model])
    
elif model_name=="silveira_zyxp_4":
    
    model_full_z_name = "./Trained_Models/NN_Trainning_26_August_2026_03-45PM_Job27376/model_LowerValidationLoss.pth"
    model_full_x_name = "./Trained_Models/NN_Trainning_26_August_2026_06-21PM_Job27380/model_LowerValidationLoss.pth"
    model_full_p_name = "./Trained_Models/NN_Trainning_26_August_2026_03-47PM_Job27377/model_LowerValidationLoss.pth"
    
    model = PINN_Model.MY_PIMODEL_4()
    model.z_model.load_state_dict(torch.load(model_full_z_name, map_location=torch.device(device_set), weights_only=True))
    model.x_model.load_state_dict(torch.load(model_full_x_name, map_location=torch.device(device_set), weights_only=True))
    model.p_model.load_state_dict(torch.load(model_full_p_name, map_location=torch.device(device_set), weights_only=True))
    # Freeze sub-models
    nnt.freeze_on_training([model.z_model, model.y_model, model.x_model, model.p_model])

else:
    raise Exception(f"Specified model {model_name} is not defined.")

# Define input type
model.bin_input = binary_input

# Weights initialization to TRAINABLE model
if   weight_init is None or weight_init in ('none'):        pass
elif weight_init.lower() in ('xavier'):           model.main_model.apply(nnt.init_weights_xavier)
elif weight_init.lower() in ('he'):               model.main_model.apply(nnt.init_weights_he)
elif weight_init.lower() in ('zero', 'zeros'):    model.main_model.apply(nnt.init_weights_zeros)
elif weight_init.lower() in ('normal'):           model.main_model.apply(nnt.init_weights_normal)
else: raise(f"Weights initialization mode {weight_init} not implemented.")

        


#######################################################
#************ OPTIMIZER    ***************************#
#######################################################

trainable_params  =  [p for p in model.parameters() if p.requires_grad]

if      optimizer == 'ADAM':    optimizer = torch.optim.Adam (trainable_params, lr=learning_rate)
elif    optimizer == 'ADAMW':   optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate)
elif    optimizer == 'SGD':     optimizer = torch.optim.SGD  (trainable_params, lr=learning_rate)
else:   raise Exception(f"Optimizer {optimizer} is not implemented.")

print('Model size: {:.3f}MB'     .format(mh.get_MB_storage_size(model)))
print('Model size: {} total parameters'.format(mh.get_total_params(model)))
print('Model size: {} trainable parameters'.format(mh.get_n_trainable_params(model)))
print('Model size: {} frozen parameters'.format(mh.get_n_non_trainable_params(model)))
print()

#######################################################
#************ CREATE DATALOADER         **************#
#######################################################

# Create dataloader
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
if num_threads is not None: torch.set_num_threads(num_threads)


#######################################################
#************ COMPUTATIONS ***************************#
#######################################################

print(f"Starting Train on {device}... \n")
nnt.partial_train(
    model, 
    train_loader,
    valid_loader,
    loss_functions,
    earlyStopping_loss,
    backPropagation_loss,
    optimizer,
    partial_epochs       = partial_epochs,
    N_epochs             = N_epochs,
    scheduler            = None,
    results_folder       = NN_results_folder,
    device               = device,
    patience             = patience,
    tolerance            = tolerance,
    dtype                = torch.float32
    )
print("Ending Train ... ")

#######################################################
#************ DELETE OBJECTS   ***********************#
#######################################################
mh.delete_model(model)
del train_loader
del valid_loader