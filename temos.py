from typing import List, Optional
import torch
from torch import Tensor
from torch.distributions.distribution import Distribution

from model.MotionEncoder import MotionEncoder
from model.MotionDecoder import MotionDecoder
from model.TextEncoder import DistillBert

from model.model_utils import  remove_padding
import torch.nn as nn

from model.preprocessing.xyztransform import XYZTransform , XYZDatastruct
# data_struct & loss 

class TEMOSNet(nn.Module):
    def __init__(self, 
                 nfeats : int,
                 vae:bool,
                 latent_dim:int,
                 motion_encoder = MotionEncoder, 
                 motion_decoder = MotionDecoder, 
                 text_encoder = DistillBert
                 ):
    
        super().__init__()
        self.text_encoder = text_encoder
        self.vae = vae
        self.motion_encoder = motion_encoder(nfeats=nfeats,vae=self.vae)
        self.motion_decoder = motion_decoder(nfeats=nfeats,vae=self.vae)
        
        self.transforms = XYZTransform
        self.data_struct = XYZDatastruct

       
        
        self.sample_mean = False
        self.fact = None
        # self.__post__init()
    # text-to-motion (text encoder -> motion decoder)
    def forward(self,batch:dict) -> List[torch.Tensor]:
        
        datastruct_from_text = self.text_to_motion_forward(batch["text"],batch["length"])
        
        return remove_padding(datastruct_from_text.joints, batch["length"])

    def sample_from_distribution(self,distribution : Distribution, *,
                                 fact : Optional[bool]=None,
                                 sample_mean : Optional[bool]=False)-> torch.Tensor:
        
        fact = fact if fact is not None else self.fact
        sample_mean = sample_mean if sample_mean is not None else self.sample_mean
        
        if sample_mean:
            return distribution.loc
        if fact is None:
            return distribution.rsample()
        
        # rescaling the eps     
        eps = distribution.rsample()-distribution.loc
        latent_vector = distribution.loc+fact*eps
        return latent_vector

    def text_to_motion_forward(self,text_sentences : List[str],lengths : List[int],*,return_latent : bool=False):
        if self.vae:
            distribution = self.text_encoder(text_sentences)
            latent_vector = self.sample_from_distribution(distribution)
        else:
            distribution = None
            latent_vector = self.text_encoder(text_sentences)
        
        features = self.motion_decoder(latent_vector,lengths)
        data_struct = self.data_struct(features=features)
        
        if not return_latent:
            return data_struct
        return data_struct, latent_vector, distribution
    
    def motion_to_motion_forward(self,data_struct, lengths: Optional[List[int]]=None,return_latent : bool =False):
        data_struct.transforms = self.transforms
        
        if self.vae:
            distribution = self.motion_encoder(data_struct.features, lengths)
            latent_vector = self.sample_from_distribution(distribution)
        else:
            distribution = None
            latent_vector: Tensor = self.motion_encoder(data_struct.features, lengths)
        
        features = self.motion_decoder(latent_vector,lengths)
        data_struct = self.data_struct(features=features)
        
        if not return_latent:
            return data_struct
        return data_struct, latent_vector, distribution
    
    # def split_step(self,split:str, batch,batch_idx):
        
    #     ttm = self.text_to_motion_forward(batch["text"],batch["length"],return_latent=True)
    #     data_struct_text, latent_text, distribution_text = ttm
        
    #     mtm = self.motion_to_motion_forward(batch["data_struct"],batch["length"],return_latent=True)
    #     data_struct_motion, latent_motion, distribution_motion = mtm

    #     # ground_truth
    #     data_struct_gt = batch["data_struct"]
        
    #     if self.vae:
            
    #         mu_gt = torch.zeros_like(distribution_text.loc)
    #         scale_ref = torch.ones_like(distribution_text.scale)
    #         distribution_gt = torch.distributions.Normal(mu_gt,scale_ref)
    #     else:
    #         distribution_gt = None
        
    #     loss = self.losses[split].update(ds_text=data_struct_text,
    #                                      ds_motion=data_struct_motion,
    #                                      ds_ref=data_struct_gt,
    #                                      lat_text=latent_text,
    #                                      lat_motion=latent_motion,
    #                                      dis_text=distribution_text,
    #                                      dis_motion=distribution_motion,
    #                                      dis_ref=distribution_gt)
        if split =="val":
            self.metrics.update(data_struct_text.detach().joints,
                                data_struct_gt.detach().joints,
                                batch["length"])
        return loss