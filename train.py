import torch
import sys
from model.temos import TEMOSNet
from compute import ComputeMetrics
from model.preprocessing.kit import KITDataModule
"""
Main Train Loop for TEMOS project
Replication Project for CS6741 @ Cornell University
Choong Hee (Frank) Kim 2023. Apr.
Instructor : Sasha A. Rush
"""
from dataloader import make_dataloader


def main():
    kit_split = './dataset/kit-splits'
    dataset = './dataset/dataset_raw'
    
    train_dataloader = make_dataloader(kit_split=kit_split,dataset=dataset,split='train')
    val_dataloader = make_dataloader(kit_split=kit_split,dataset=dataset,split='val')
    test_dataloader = make_dataloader(kit_split=kit_split,dataset=dataset,split='test')

    
    
    
    model = TEMOSNet(nfeats = 64, vae = True, latent_dim = 256)
    # Import TEMOS model
    model.train()
    torch.set_grad_enabled(True)
    # Set up Optimizer
    lr = 1e-4 # learning rate
    optimizer = torch.optim.AdamW(model.parameters(),lr=lr)
    
    # Loss Function
    losses=[]
    EPOCH = 1000
    
    for epoch in range(EPOCH):
        batch_count = 0
        for batch in train_dataloader:
            
        # """
        # Each Batch has following information
        # "datastruct"
        # "length"
        # "text"
        # "keyid"
        # """
            
            text_to_motion = model.text_to_motion_forward(batch["text"],
                                                          batch["length"],
                                                          return_latent = True)
            datastruct_from_text, latent_from_text, distribution_from_text = text_to_motion

            motion_to_motion = model.motion_to_motion_forward(batch["datastruct"],
                                                              batch["length"],
                                                              return_latent = True)
            datastruct_from_motion, latent_from_motion, distribution_from_motion = motion_to_motion
            
            datastruct_ref = batch["datastruct"] # GT is used in VAE way of training
            mu_ref = torch.Tensor.zeros_like(distribution_from_text.loc)
            scale_ref = torch.Tensor.ones_like(distribution_from_text.scale)
            distribution_ref = torch.distributions.Normal(mu_ref,scale_ref)    
     
            recon_motion2motion = torch.nn.SmoothL1Loss(datastruct_from_motion.jfeats,datastruct_ref.jfeats)
            recon_text2motion = torch.nn.SmoothL1Loss(datastruct_from_text.jfeats,datastruct_ref.jfeats)
            L_recon = recon_motion2motion + recon_text2motion
            kl_text2motion = torch.distributions.kl_divergence(distribution_from_text,distribution_from_motion).mean()
            kl_motion2text = torch.distributions.kl_divergence(distribution_from_motion,distribution_from_text).mean()
            kl_text = torch.distributions.kl_divergence(distribution_from_text,distribution_ref).mean()
            kl_motion = torch.distributions.kl_divergence(distribution_from_motion,distribution_ref).mean()
            L_kl = kl_motion + kl_text + kl_motion2text + kl_text2motion
            L_emb = torch.nn.SmoothL1Loss(latent_from_text,latent_from_motion)
            loss = L_recon + 1e-5*L_kl + 1e-5*L_emb
            
            loss.backward() # backpropagation
            optimizer.step()
            optimizer.zero_grad()
            losses.append(loss)
        
        # Validating performance 
        if batch_count%100 == 0:
            with torch.no_grad():
                Val_loss = []
                for val_batch in validation_dataloader:
                    val_loss = ComputeMetrics.update(datastruct_from_text.detach().joints,datastruct_ref.detach().joints,val_batch["length"])
                    Val_loss.append(val_loss)
                Val_loss = torch.mean(torch.Tensor(Val_loss)) 
        batch_count +=1   
if __name__ == "__main__":
    main()
