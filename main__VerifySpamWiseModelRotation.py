import numpy              as np
import torch
import tensorflow         as tf
import matplotlib.pyplot  as plt
import pandas             as pd
from torch.utils.data     import DataLoader, Subset
import torch.nn as nn

from Architectures.Unet   import Extended_DannyKo
from Architectures.MSnet  import JavierSantos_Extended
from Architectures.Models import SubModels_Composition

from Utilities            import dataset_reader as dr
from Utilities            import error_metrics as em 
from Utilities            import model_handler as mh

from Architectures        import Functional



"""
This script aims to verify the use of rotation to predict spam-wise fields.
Using invariance property and rotation is training data, we have a single model
that predicts Ux, and using this model we are also capable of predicting Uy.
For this: a copy of the sample is rotated, the prediction is made, the prediction is rotated.
"""

datapath                =  "../NN_Datasets_Grad_Dist_40_5_55/Test_Silveira_SphGrain_SAug_DNorm.h5"

dataset                 = dr.LazyDatasetTorch(
                            h5_path=datapath, 
                            list_ids=None, 
                            x_dtype=torch.float32,
                            y_dtype=torch.float32)


danny_model             = Extended_DannyKo()
danny_model_x           = danny_model.x_model
model_full_name         = "./Trained_Models/NN_Trainning_26_August_2026_06-21PM_Job27380/model_LowerValidationLoss.pth"
danny_model_x.load_state_dict(torch.load(model_full_name, map_location=torch.device('cpu'), weights_only=True))
danny_model_x.bin_input = True
danny_model_x.eval()

danny_model_y = Functional.Ux2Uy(danny_model_x)

sample                      = 3
samp_input, samp_target     = dataset[sample]
samp_input                  = samp_input.unsqueeze(0)
samp_target                 = samp_target.unsqueeze(0)

# PRED UY
pred_uy     = danny_model_y.predict(samp_input)
# PRED UX
pred_ux     = danny_model_x.predict(samp_input)

# Calculate correlation
def coeff(batch_input, ten1, ten2):
    
    fluid_mask = (batch_input > 0)
    x_flat      = ten1[fluid_mask].flatten()
    y_flat      = ten2[fluid_mask].flatten()
    
    correlation_matrix      = np.corrcoef(x_flat, y_flat)
    correlation_coefficient = correlation_matrix[0, 1]
    
    return correlation_coefficient

# 
target_uy = samp_target[:, 1:2] 
target_ux = samp_target[:, 2:3] 


print("Corr Pred Uy vs Tar Uy: ",coeff(samp_input[0][0], pred_uy[0][0], target_uy[0][0]))
print("Corr Pred Uy vs Tar Ux: ",coeff(samp_input[0][0], pred_uy[0][0], target_ux[0][0]))
print("Corr Pred Ux vs Tar Ux: ",coeff(samp_input[0][0], pred_ux[0][0], target_ux[0][0]))
print("Corr Pred Ux vs Tar Uy: ",coeff(samp_input[0][0], pred_ux[0][0], target_uy[0][0]))
#"""
