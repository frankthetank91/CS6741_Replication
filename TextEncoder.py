from typing import List, Union
import torch.nn as nn
import os
import torch
from torch import Tensor
from torch.distributions.distribution import Distribution
from torch.distributions.distribution import Distribution
from model.model_utils import PositionalEncoding, lengths_to_mask

class DistillBertEncoderBase(nn.Module):
    def __init__(self, modelpath : str, finetune: bool = False) -> None:
        super().__init__()
        
        from transformers import AutoTokenizer, AutoModel
        
        self.tokenizer = AutoTokenizer.from_pretrained(modelpath)
        self.text_model = AutoModel.from_pretrained(modelpath)
        
        if not finetune:
            self.text_model.training = False # Directly use the pre-trained model
            for p in self.text_model.parameters():
                p.requires_grad = False
        
        self.text_encoded_dim = self.text_model.config.hidden_size
    
    def get_last_hidden_state(self,texts: List[str],return_mask: bool=False)->Union[torch.Tensor, tuple[torch.Tensor,torch.Tensor]]:
        
        encoded_inputs = self.tokenizer(texts, return_tensors="pt", padding = True)
        output = self.text_model(**encoded_inputs.to(self.text_model.device))
        if not return_mask:
            return output.last_hidden_state
        return output.last_hidden_state, encoded_inputs.attention_mask.to(dtype=bool) 

class DistillBert(DistillBertEncoderBase):
    def __init__(self, modelpath: str, 
                 finetune: bool = False,
                 vae : bool = True,
                 latent_dim : int = 256,
                 ff_size : int = 1024,
                 num_layers : int = 6, num_heads : int=4,
                 dropout : float = 0.1,
                 activation : str="gelu",**kwargs) -> None:
        super().__init__(modelpath=modelpath, finetune=finetune)
        
        encoded_dim = self.text_encoded_dim
        self.vae = vae

        self.projection = nn.Sequential(nn.Relu(),
                                        nn.Linear(encoded_dim,latent_dim))
        
        if self.vae:
            self.mu_token = nn.Parameter(torch.randn(latent_dim))
            self.logvar_token = nn.Parameter(torch.randn(latent_dim))
        else:
            self.emb_token = nn.Parameter(torch.randn(latent_dim))
        self.positional_encoding = PositionalEncoding(latent_dim,dropout)
        
        text_encoder_layer = nn.TransformerEncoderLayer(d_model = latent_dim,nhead=num_heads,dim_feedforward=ff_size,dropout=dropout,activation=activation)
        self.textTransformer  = nn.TransformerEncoder(text_encoder_layer,num_layers = num_layers)
    
    def forward(self,texts: List[str]) -> Union[torch.Tensor,Distribution]: 
        
        encoded_text, mask = self.get_last_hidden_state(texts,return_mask=True)
        x = self.projection(encoded_text)
        bs, nframes, _ = x.shape
        x = x.permute(1,0,2)
        
        if self.vae:
            mu_token = torch.tile(self.mu_token,(bs,)).reshape(bs,-1)
            logvar_token = torch.tile(self.logvar_token,(bs,)).reshape(bs,-1)
            
            x_seq = torch.cat((mu_token[None],logvar_token[None],x),0) # adding distribution tokens for all sequences
            
            token_mask = torch.ones((bs,2),dtype=bool,device=x.device)
            aug_mask = torch.cat((token_mask,mask),1)
            
        else:
            emb_token = torch.tile(self.emb_token,(bs,)).reshape(bs,-1)
            xseq = torch.ones((bs,1),dtype=bool,device=x.device)
            aug_mask = torch.cat((token_mask,mask),1)
            
        x_seq = self.positional_encoding(x_seq) 
        final = self.textTransformer(x_seq, src_key_padding_mask =~aug_mask)
        
        if self.vae:
            mu, logvar = final[0],final[1]
            std = logvar.exp().pow(0.5)
            dist = torch.distributions.Normal(mu,std)
            return dist 
        else:
            return final[0]