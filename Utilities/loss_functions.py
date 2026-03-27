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
    

class MultiScaleLoss(nn.Module):
    # 'normalize_mode' can be : 
    #  - 'none' to return the raw sum, 
    #  - 'n_scales' to divide by the num. of scales and return the mean across scales
    #  - 'var' to divide by the variance of higher resolution's scale target
    def __init__(self, loss_fn, n_scales=4, norm_mode='none'):
        """
        Args:
            loss_fn: any PyTorch loss function (e.g., nn.MSELoss(), nn.L1Loss())
        """
        super(MultiScaleLoss, self).__init__()
        self.loss_fn    = loss_fn
        self.norm_mode  = norm_mode
        self.scales     = n_scales
        

    def forward(self, y_pred, y):
        """
        Args:
            y_pred (List[Tensor]): predictions at each scale
            y (List[Tensor] or Tensor): ground truths at each scale
        Returns:
            loss: total multiscale loss
        """
        # If the ground truth is a tensor: make it multi-scale
        if torch.is_tensor(y):
            y = self.get_coarsened_list(y)
        
        # Validate input types
        if not isinstance(y_pred, (list, tuple)):
            raise TypeError(f"Expected y_pred to be list or tuple, got {type(y_pred)}")
        if not isinstance(y, (list, tuple)):
            raise TypeError(f"Expected y to be list or tuple, got {type(y)}")
        if len(y_pred) != len(y):
            raise ValueError(f"Mismatch in number of scales: {len(y_pred)} predictions vs {len(y)} targets")
        
        total_loss  = y_pred[-1].new_tensor(0.0)
        y_vars      = torch.var(y[-1], dim=list(range(1, y[-1].ndim))) # Compute var over batch dimension, reducing all others        
        y_max       = torch.amax(y[-1].abs(), dim=list(range(1, y[-1].ndim)))
        y_avg       = torch.mean(y[-1].abs(), dim=list(range(1, y[-1].ndim)))
        
        # For each scale
        for scale, (y_hats, y_trues) in enumerate(zip(y_pred, y)): # Iterate over listed scales
            if y_hats.shape != y_trues.shape:
                raise ValueError(f"Shape mismatch at scale {scale}: {y_hats.shape} vs {y_trues.shape}")
            
            # For each sample
            for sample_idx, (y_hat, y_true) in enumerate(zip(y_hats, y_trues)):
                # Get the scaled image loss, then include it to the total
                
                if self.norm_mode=='var':       total_loss += self.loss_fn(y_hat, y_true)/(len(y_pred)*y_vars[sample_idx])
                elif self.norm_mode=='max':     total_loss += self.loss_fn(y_hat, y_true)/(len(y_pred)*y_max[sample_idx])
                elif self.norm_mode=='avg':     total_loss += self.loss_fn(y_hat, y_true)/(len(y_pred)*y_avg[sample_idx])
                else:                           total_loss += self.loss_fn(y_hat, y_true)/len(y_pred)
                
        return total_loss
    
    def get_coarsened_list(self, x):    
        ds_x = []
        ds_x.append(x)
        for i in range( self.scales-1 ): 
            ds_x.append( self.scale_tensor( ds_x[-1], scale_factor=1/2 ) )
        return ds_x[::-1] # returns the reversed list (small images first)
    
    def scale_tensor(self, x, scale_factor=1):
        
        # Downscale images
        if scale_factor<1:
            return nn.AvgPool3d(kernel_size = int(1/scale_factor))(x)
        
        # Upscale images
        elif scale_factor>1:
            for repeat in range (0, int(np.log2(scale_factor)) ):  # number of repeatsx2
                for ax in range(2,5): # (B,C,  H,W,D), repeat only the 3D axis, not batch and channel
                    x=x.repeat_interleave(repeats=2, axis=ax)
            return x
        
        # Do not change images
        elif scale_factor==1:
            return x
        
        else: raise ValueError(f"Scale factor not understood: {scale_factor}")
       
    
class MSE(nn.Module):
    def __init__(self):
        super(MSE, self).__init__()

    def forward(self, output, target):
        # Ensure the output and target tensors have the same shape.
        if output.size() != target.size():
            raise ValueError(f"Shape mismatch: {output.size()} vs {target.size()}")    
        errors = (target-output)
        
        loss = errors.abs()
        loss += 1 - torch.exp(- (2000.0 * errors)**2.0)
        loss += 1 - torch.exp(- (20.0 * errors)**2.0)
        loss += 1 - torch.exp(- (2.0 * errors)**2.0)
        loss += 1 - torch.exp(- (0.2 * errors)**2.0)

        return loss.mean()