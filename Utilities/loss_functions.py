import torch.nn as nn
from torchmetrics.classification import Accuracy
from torch.nn import BCELoss, functional
import torch
import numpy as np


#######################################################
#************ LOSS FUNCTION UTILITIES ****************#
#######################################################

# Apply threshold
class Binarize(nn.Module):
    def __init__(self, threshold=0.5):
        super().__init__()
        self.threshold = threshold

    def forward(self, x):
        return (torch.sigmoid(x) > self.threshold).float()
    
# Defines the specific pixels that the loss function might look into
# Mode: 
# - flatten if a mask operates and a flatten tensor is propagated (default)
# - overwrite if output is overwrited by the target in the mask locations, then the structure is kept
class Mask_LossFunction(nn.Module):
    def __init__(self, lossFunction, mask_law=None, mode="flatten"):
        super(Mask_LossFunction, self).__init__()
        
        self.lossFunction = lossFunction
        
        self.mode = mode
        
        if mask_law is None: 
            self.mask_law = self._default_mask_law
        else:
            self.mask_law = mask_law
            
    # Do not consider cells with 0 value  
    # The loss function used must be a mean across the tensor lenght, 
    # so that the quantity of solid cells do not affect the loss
    def _default_mask_law(self,output, target, threshold=0): 
        # Mask consider only target != 0, i.e, non-solid cells
        #mask = (target > threshold) | (target < -threshold)
        mask = torch.abs(target) > threshold
        return mask
    
    def forward(self, output, target):
        
        if output.size() != target.size():
            raise ValueError(f"CustomLoss forward: Tensors have different sizes ({output.size()} vs {target.size()})")
        mask = self.mask_law(output, target)
        
        if self.mode == 'flatten':
            return self.lossFunction(output[mask], target[mask])
        
        elif self.mode == 'overwrite':
            temp_output = output.clone()
            temp_output[~mask] = target[~mask]
            output = temp_output            
            return self.lossFunction(output, target)
        
        else: raise Exception(f"Mask_LossFunction mode {self.mode} not implemented. Must be one of flatten or overwrite")

        
    
    
    
    
#######################################################
#************ LOSS FUNCTIONS  ************************#
#######################################################

# MY COMPOSED FUNCTIONS
    
# Permeability Relative Percentual Error
class MeanBiasError(nn.Module):
    def __init__(self):
        super(MeanBiasError, self).__init__()
        
    def forward(self, output, target):
        if output.shape != target.shape:
            raise ValueError(f"Shape mismatch: {output.shape} vs {target.shape}")
            
        mean_error = 100*( (output.mean() - target.mean())/target.mean() ).abs()
        return mean_error
    

class PearsonCorr(nn.Module):
    def __init__(self, N_samples, eps=1e-16, reverse=False):
        super(PearsonCorr, self).__init__()
        self.eps=eps
        self.N_samples=N_samples 
        self.reverse = reverse
        
    def forward(self, output, target):
        if output.shape != target.shape:
            raise ValueError(f"Shape mismatch: {output.shape} vs {target.shape}")
            
        # 1. Flatten the tensors to treat all cells as a single distribution
        x = output.flatten()
        y = target.flatten()
        
        indx = torch.randperm(x.size(0))[:self.N_samples]
        x = x[indx]
        y = y[indx]
        
        # 2. Centering (Subtract the mean)
        x_mu = x - x.mean()
        y_mu = y - y.mean()
        
        numerator   = torch.sum(x_mu * y_mu)
        denominator = torch.sqrt(torch.sum(x_mu**2)) * torch.sqrt(torch.sum(y_mu**2))
        corr = (numerator / (denominator + self.eps))
        
        return corr if not self.reverse else 1-corr
    
class Divergent(nn.Module):
    def __init__(self):
        super(Divergent, self).__init__()
        
    def forward(self, output, target):
        if output.shape[0] != target.shape[0] or output.shape[2:] != target.shape[2:]:
            raise ValueError(f"Shape mismatch: {output.shape} vs {target.shape}")
            
        if output.shape[1] < 3 or target.shape[1] < 3:
            raise ValueError(f"Shape mismatch: output {output.shape} or target {target.shape} need to have at least 3 channels (z,y,x)")
            
        out_div =  (output[:, 0, 2:  , 1:-1, 1:-1] - output[:, 0,  :-2, 1:-1, 1:-1]) / 2.0
        out_div += (output[:, 1, 1:-1, 2:  , 1:-1] - output[:, 1, 1:-1,  :-2, 1:-1]) / 2.0
        out_div += (output[:, 2, 1:-1, 1:-1, 2:  ] - output[:, 2, 1:-1, 1:-1,  :-2]) / 2.0
        
        return (out_div).abs().mean()
    
    
class MSE_Divergent(nn.Module):
    def __init__(self, div_weight: np.uint = 1):
        super(MSE_Divergent, self).__init__()
        self.div    = Divergent()
        self.mse    = nn.MSELoss()
        self.alpha  = div_weight
        
    def forward(self, output, target):
        loss =  self.mse(output[:,0], target[:,0])
        loss += self.mse(output[:,1], target[:,1])
        loss += self.mse(output[:,2], target[:,2])
        loss += self.alpha * self.div(output, target)
        return loss
    
    
class KGE(nn.Module):
    def __init__(self):
        super(KGE, self).__init__()
        
        self.corr = PearsonCorr(2000)
      
    def forward(self, output, target):
        mean_pred = torch.mean(output)
        mean_true = torch.mean(target)
        bias      = 1.0 - (mean_pred / mean_true)
        inv_corr  = 1.0 - self.corr(output, target)
        
        return torch.sqrt(bias**2 + inv_corr**2)