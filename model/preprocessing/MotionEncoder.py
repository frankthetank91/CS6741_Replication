import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional, Union
from torch import nn, Tensor
from torch.distributions.distribution import Distribution
from model.model_utils import PositionalEncoding, lengths_to_mask


class MotionEncoder(nn.Module):
    def __init__(self, nfeats : int, vae : bool,
                 latent_dim: int=256, ff_size : int=1024,
                 num_layers: int=6, num_heads: int=4,
                 dropout: float =0.1,
                 activation : str = "gelu", **kwargs)-> None:
        super().__init__()
        
        input_features = nfeats
        
        self.skeleton_embedding = nn.Linear(input_features, latent_dim)
        
        if vae:
            self.mu_token = nn.Parameter(torch.randn(latent_dim))
            self.logvar_token = nn.Parameter(torch.randn(latent_dim))
        else:
            self.emb_token = nn.Parameter(torch.randn(latent_dim))
        
        self.positional_encoding = PositionalEncoding(latent_dim,dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model = latent_dim,
                                                   nhead = num_heads,
                                                   dim_feedforward= ff_size,
                                                   dropout=dropout,
                                                   activation=activation)
        self.SkeletonEncoder = nn.TransformerEncoder(encoder_layer,num_layers=num_layers)
        
    
    def forward(self,features: torch.Tensor, lengths : Optional[List[int]]=None)->Union[torch.Tensor,Distribution]: 
        if lengths is None:
            lengths = [len(feature) for feature in features]
        
        device = features.device
        
        bs, nframes, nfeats = features.shape # bs -> batch size
        mask = lengths_to_mask(lengths,device)
        
        x = features
        x = self.skeleton_embedding(x)
        x = x.permute(1,0,2) # [sequence , batch_size, ... ]-> pytorch transformer input
        
        if self.vae:
            mu_token = torch.tile(self.mu_token,(bs,)).reshape(bs,-1)
            logvar_token = torch.tile(self.logvar_token,(bs,)).reshape(bs,-1)
            
            x_seq = torch.cat((mu_token[None],logvar_token[None],x),0)
            
            token_mask = torch.ones((bs, 2), dtype=bool, device=x.device)
            aug_mask = torch.cat((token_mask, mask), 1)
        else:
            emb_token = torch.tile(self.emb_token, (bs,)).reshape(bs, -1)    
            x_seq = torch.cat((emb_token[None], x), 0)
            token_mask = torch.ones((bs, 1), dtype=bool, device=x.device)
            aug_mask = torch.cat((token_mask, mask), 1)
        
        x_seq = self.positional_encoding(x_seq)
        final = self.SkeletonEncoder(x_seq, src_key_padding_mask =~aug_mask)
        
        if self.vae:
            mu,logvar = final[0],final[1]
            std = logvar.exp().pow(0.5)
            dist = torch.distributions.Normal(mu,std)
            return dist # generate a latent vector z drawn from normal distribution
        else:
            return final[0]