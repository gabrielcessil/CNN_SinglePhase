import torch
from   torch.utils.data import DataLoader
import torch.nn as nn
from Utilities import dataset_reader as dr
from Utilities import loss_functions as lf
from Danny_Original.architecture import Danny_KerasModel

datapath   = "../NN_Datasets/ForceDriven/Test_Oliveira_Parker_120_120_120.h5" 
batch_size = 4


baseline_model = Danny_KerasModel()


dataset = dr.LazyDatasetTorch(
    h5_path=datapath, 
    list_ids=None, 
    x_dtype=torch.float32,
    y_dtype=torch.float32
)

loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)



with torch.no_grad():
    for batch_idx, (inp, tar) in enumerate(loader):
        
        inp = inp.to(dtype=torch.float32)
        tar = tar.to(dtype=torch.float32)
        out = baseline_model.predict(inp)
         
        # Gradient using torch.gradient()
        """
        out_grad = torch.gradient(out, dim = (2,3,4)) # Dim= (B,C, Z,Y,X)
        out_div = out_grad[0][:,0] + out_grad[1][:,1] + out_grad[2][:,2]
        """

        # Gradient using torch.gradient() channel by channel
        """
        out_div = torch.gradient(out, dim=2)[0][:, 0]
        out_div += torch.gradient(out, dim=3)[0][:, 1]
        out_div += torch.gradient(out, dim=4)[0][:, 2]
        print(out_div.abs().mean())
        """
        
        # Gradient using manual central differences
        out_div =  (out[:, 0, 2:  , 1:-1, 1:-1] - out[:, 0,  :-2, 1:-1, 1:-1]) / 2.0
        out_div += (out[:, 1, 1:-1, 2:  , 1:-1] - out[:, 1, 1:-1,  :-2, 1:-1]) / 2.0
        out_div += (out[:, 2, 1:-1, 1:-1, 2:  ] - out[:, 2, 1:-1, 1:-1,  :-2]) / 2.0
        
        
        tar_div =  (tar[:, 0, 2:  , 1:-1, 1:-1] - tar[:, 0,  :-2, 1:-1, 1:-1]) / 2.0
        tar_div += (tar[:, 1, 1:-1, 2:  , 1:-1] - tar[:, 1, 1:-1,  :-2, 1:-1]) / 2.0
        tar_div += (tar[:, 2, 1:-1, 1:-1, 2:  ] - tar[:, 2, 1:-1, 1:-1,  :-2]) / 2.0
        
        print(f"{out_div.abs().mean()} {tar_div.abs().mean()}")
        
        fn = lf.Divergent()
        
        print(fn(out, tar))
        
