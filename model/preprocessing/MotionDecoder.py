import torch
import torch.nn as nn
from typing import List, Optional
from torch import nn, Tensor
import numpy as np
from model.model_utils import  lengths_to_mask, PositionalEncoding
"""
Replication Project (CS6741) : 2023 Winter
Advisor : Alexander Sasha Rush

TEMOS : Generating diverse human motions from textual descriptions
(Petrovich et al.)

Written by Choong Hee (Frank) Kim

This code includes Motion Decoder part in TEMOS

Input : 
    1. latent_vector (z) sampled from text-motion joint space
    2. Index of motion sequence 

Output :
    Generated motion sequence 
"""

class MotionDecoder(nn.Module):
    def __init__(self, n_feature: int = 64,
                 latent_dim : int = 256,
                 ff_size : int = 1024,
                 num_layers : int=6,
                 num_heads : int =4,
                 dropout : float=0.1,
                 activation : str="gelu",**kwargs) -> None:
        
        super().__init__()
        
        output_feature = n_feature
        self.positional_encoding = PositionalEncoding(latent_dim, dropout)
        single_decoder_layer = nn.TransformerDecoderLayer(d_model=latent_dim,
                                                           nhead = num_heads,
                                                           dim_feedforward = ff_size,
                                                           dropout = dropout,
                                                           activation = activation)
        self.Decoder = nn.TransformerDecoder(single_decoder_layer, num_layers=num_layers)
        self.final_layer = nn.Linear(latent_dim,output_feature)
    def forward(self, z: Tensor, length : List[int]):
        mask = lengths_to_mask(length ,z.device)
        latent_dim = z.shape[1]
        bs, nframes = mask.shape
        
        z = z[None] # sequence of 1 element 
        
        # time queries (length to generate motion)
        time_queries = torch.zeros(nframes, bs, latent_dim, device = z.device)
        time_queries = self.positional_encoding(time_queries)
        
        output = self.Decoder(tgt = time_queries, memory = z, tgt_key_padding_mask=~mask)
        output = self.final_layer(output)
        
        # zero for padded area
        output[~mask.T] = 0
        feats = output.permute(1,0,2)
        return feats