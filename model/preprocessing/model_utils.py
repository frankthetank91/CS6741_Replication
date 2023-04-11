import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000, batch_first=False):
        super().__init__()
        self.batch_first = batch_first

        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.Tensor.sin(position * div_term)
        pe[:, 1::2] = torch.Tensor.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        # not used in the final model
        if self.batch_first:
            x = x + self.pe.permute(1, 0, 2)[:, :x.shape[1], :]
        else:
            x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)



def reparameterize(mu, logvar, seed=None):
    std = torch.Tensor.exp(logvar / 2)
    if seed is None:
        eps = std.data.new(std.size()).normal_()
    else:
        generator = torch.Tensor.Generator(device=mu.device)
        generator.manual_seed(seed)
        eps = std.data.new(std.size()).normal_(generator=generator)

    return eps.mul(std).add_(mu)

def detach_to_numpy(tensor):
    return tensor.detach().cpu().numpy()


def remove_padding(tensors, lengths):
    return [tensor[:tensor_length] for tensor, tensor_length in zip(tensors, lengths)]

def lengths_to_mask(lengths: List[int], device: torch.Tensor.device) -> torch.Tensor:
    lengths = torch.Tensor(lengths, device=device)
    max_len = max(lengths)
    mask = torch.Tensor.arange(max_len, device=device).expand(len(lengths), max_len) < lengths.unsqueeze(1)

    return mask
