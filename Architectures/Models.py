import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from Architectures.FunctionalBlocks import (
    BASE_MODEL,
    ConvBlock,
    PoolingBlock,
    UpSampleBlock,
    ChannelConcat_Block,
    InceptionBlock
)

    
class Inception(nn.Module):
    def __init__(self, input_size, in_channels, out_channels, features_per_block = 16):
        super().__init__() 
        
        
        
        head = InceptionBlock(
            input_size      = input_size, 
            in_channels     = in_channels, 
            b1_out_channels = max(features_per_block//16,1), # 64
            
            b2_mid_channels = max(features_per_block//4,1), # 96
            b2_out_channels = max(features_per_block//2,1), # 128
            
            b3_mid_channels = max(features_per_block//8,1), # 16
            b3_out_channels = max(features_per_block//16,1), #32
            
            b4_out_channels = max(features_per_block//16,1)  # 32
        )
        
        body = InceptionBlock(
            input_size      = head.output_size, 
            in_channels     = head.out_channels, 
            b1_out_channels = max(features_per_block//16,1), # 64
            
            b2_mid_channels = max(features_per_block//8,1), # 16
            b2_out_channels = max(features_per_block//16,1), #32
            
            b3_mid_channels = max(features_per_block//4,1), # 96
            b3_out_channels = max(features_per_block//2,1), # 128
            
            b4_out_channels = max(features_per_block//16,1)  # 32
        )
        
        bodies = [body]
        
        for i in range(6):
        
            body = InceptionBlock(
                input_size      = bodies[-1].output_size, 
                in_channels     = bodies[-1].out_channels, 
                b1_out_channels = max(features_per_block//16,1), # 64
                
                b2_mid_channels = max(features_per_block//8,1), # 16
                b2_out_channels = max(features_per_block//16,1), #32
                
                b3_mid_channels = max(features_per_block//4,1), # 96
                b3_out_channels = max(features_per_block//2,1), # 128
                
                b4_out_channels = max(features_per_block//16,1)  # 32
            )
            
            bodies.append(body)


        tail = ConvBlock(input_size     =bodies[-1].output_size, 
                         in_channels    =bodies[-1].out_channels,
                         out_channels   =out_channels, 
                         kernel_size    =1)
                
        self.model = nn.Sequential(head, *bodies, tail)
    
    def forward(self, x):
        return self.model(x)
    
    def predict(self, x):
        
        if self.bin_input: x = (x > 0).to(torch.float32)
        
        with torch.no_grad():
            out     = self.forward(x)

            # Mask Output, making solid always zero
            mask    = (x > 0).to(torch.float32) 
            mask    = mask.expand(-1, out.shape[1], -1, -1, -1)
            return out * mask
    


