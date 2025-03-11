import sys 
sys.path.append('../')

import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from models import register_model
from models.modules import clones, clip_loss
from models.modules import EncoderLayer, DecoderLayer, LayerNorm, MultiHeadedAttention, PositionwiseFeedForward
from models.base import BaseModel, SpectralEncoding, MolecularEncoding


class SpectralEncoder(nn.Module):
    def __init__(self, d_model=768, nhead=8, d_ff=2048, nlayer=6, dropout=0.1, d_proj=256, in_channel=1):
        super().__init__()

        self_attn = MultiHeadedAttention(nhead, d_model, dropout)
        feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        layer = EncoderLayer(d_model, self_attn, feed_forward, dropout)

        self.layers = clones(layer, nlayer)
        self.norm = LayerNorm(d_model)
        self.proj = nn.Sequential(nn.Linear(d_model, d_proj), nn.ReLU(), nn.Linear(d_proj, d_proj))
        
    def forward(self, input_embeds):
        layer_output = input_embeds
        for layer in self.layers:
            layer_output = layer(layer_output, mask=None)
        layer_output = self.norm(layer_output)

        cls_token = layer_output[:, 0]
        proj_out = self.proj(cls_token)
        return {'hidden_states': layer_output, 'proj_output': proj_out}


class MolecularEncoder(nn.Module):
    def __init__(self, d_model=768, nhead=8, d_ff=2048, nlayer=6, dropout=0.1, d_proj=256):
        super().__init__()

        self_attn = MultiHeadedAttention(nhead, d_model, dropout)
        feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        layer = EncoderLayer(d_model, self_attn, feed_forward, dropout)

        self.layers = clones(layer, nlayer)
        self.norm = LayerNorm(d_model)       
        self.proj = nn.Sequential(nn.Linear(d_model, d_proj), nn.ReLU(), nn.Linear(d_proj, d_proj))

    def forward(self, input_embeds, mask):
        layer_output = input_embeds
        for layer in self.layers:
            layer_output = layer(layer_output, mask=mask)
        layer_output = self.norm(layer_output)

        cls_token = layer_output[:, 0]
        proj_out = self.proj(cls_token)
        return {'hidden_states': layer_output, 'proj_output': proj_out}


class Vib2Mol_CL(BaseModel):
    def __init__(self, 
                 d_proj=256, 
                 spectral_channel=1, 
                 d_model=768, 
                 nhead=8, 
                 d_ff=2048, 
                 nlayer=6,
                 **kwargs):
        super().__init__()

        self.spectral_encoding = SpectralEncoding(d_model=d_model, spectral_channel=spectral_channel)
        self.molecular_encoding = MolecularEncoding(d_model=d_model, num_embeddings=500)
        self.spectral_encoder = SpectralEncoder(d_model=d_model, nhead=nhead, d_ff=d_ff, nlayer=nlayer, d_proj=d_proj, in_channel=spectral_channel)
        self.molecular_encoder = MolecularEncoder(d_model=d_model, nhead=nhead, d_ff=d_ff, nlayer=nlayer, d_proj=d_proj)

        self._init_weights()
    
    def forward(self, input, 
                return_loss=True,
                return_proj_output=False
                ):
        
        spectra_input = self.load_spectra(input)
        spectra_embeds = self.spectral_encoding(spectra_input)
        spectral_output = self.spectral_encoder(spectra_embeds)
        
        molecular_input_ids = input['smiles']['input_ids']
        molecular_attention_mask = input['smiles']['attention_mask']
        molecular_embeds = self.molecular_encoding(molecular_input_ids, use_cls_token=True)
        contrastive_mask = torch.cat([torch.ones(molecular_embeds.size(0), 1).to(molecular_attention_mask.device), molecular_attention_mask], dim=1)
        molecular_output = self.molecular_encoder(molecular_embeds, mask=contrastive_mask)
        
        result_dict = {}
        if return_loss:
            clip_loss = self.compute_clip_loss(molecular_output['proj_output'], spectral_output['proj_output'])
            loss = clip_loss

            result_dict['clip_loss'] = clip_loss
            result_dict['loss'] = loss

        if return_proj_output:
            result_dict['molecular_proj_output'] = molecular_output['proj_output']
            result_dict['spectral_proj_output'] = spectral_output['proj_output']
        return result_dict


@register_model
def vib2mol_cl(pretrained=False, **kwargs):
    model = Vib2Mol_CL(nlayer=6, **kwargs)
    return model


