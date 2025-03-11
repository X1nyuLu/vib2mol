import sys 
sys.path.append('../')

import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from models import register_model
from models.modules import clones
from models.modules import EncoderLayer, DecoderLayer, LayerNorm, MultiHeadedAttention, PositionwiseFeedForward
from models.base import BaseModel, SpectralEncoding, MolecularEncoding


class SpectralEncoder(nn.Module):
    def __init__(self, d_model=768, nhead=12, d_ff=3072, nlayer=6, dropout=0.1, in_channel=1):
        super().__init__()

        self_attn = MultiHeadedAttention(nhead, d_model, dropout)
        feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        layer = EncoderLayer(d_model, self_attn, feed_forward, dropout)

        self.layers = clones(layer, nlayer)
        self.norm = LayerNorm(d_model)
        # self.proj = nn.Sequential(nn.Linear(d_model, d_proj), nn.Tanh(), nn.Linear(d_proj, d_proj))
        
    def forward(self, input_embeds):
        layer_output = input_embeds
        for layer in self.layers:
            layer_output = layer(layer_output, mask=None)
        layer_output = self.norm(layer_output)
        return {'hidden_states': layer_output}
        # cls_token = layer_output[:, 0]
        # proj_out = self.proj(cls_token)
        # return {'hidden_states': layer_output, 'proj_output': proj_out}
    

class MolecularDecoder(nn.Module):
    def __init__(self, d_model=768, nhead=12, d_ff=3072, nlayer=6, dropout=0.1, vocab_size=181):
        super().__init__()

        self_attn = MultiHeadedAttention(nhead, d_model, dropout)
        src_attn = MultiHeadedAttention(nhead, d_model, dropout)
        feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        layer = DecoderLayer(d_model, self_attn, src_attn, feed_forward, dropout)

        self.layers = clones(layer, nlayer)
        self.norm = LayerNorm(d_model)
        self.proj = nn.Sequential(nn.Linear(d_model, vocab_size), nn.Tanh(), nn.Linear(vocab_size, vocab_size))
        
    def forward(self, input_embeds, memory, src_mask, tgt_mask):
        layer_output = input_embeds
        for layer in self.layers:
            layer_output = layer(layer_output, memory, src_mask, tgt_mask)
        layer_output = self.norm(layer_output)
        proj_out = self.proj(layer_output)
        return {'hidden_states': layer_output, 'proj_output': proj_out, 'mask': tgt_mask}


class Vib2Mol_MLM(BaseModel):
    def __init__(self, 
                 d_proj=256, 
                 vocab_size=181, 
                 spectral_channel=1, 
                 d_model=768, 
                 nhead=8, 
                 d_ff=2048, 
                 nlayer=6,
                 mask_prob=0.45,
                 **kwargs):
        super().__init__()
        self.mask_prob = mask_prob
        self.spectral_encoding = SpectralEncoding(d_model=d_model)
        self.molecular_encoding = MolecularEncoding(d_model=d_model)
        self.spectral_encoder = SpectralEncoder(d_model=d_model, nhead=nhead, d_ff=d_ff, nlayer=nlayer, in_channel=spectral_channel)
        self.molecular_decoder = MolecularDecoder(d_model=d_model, nhead=nhead, d_ff=d_ff, nlayer=nlayer, vocab_size=vocab_size)
        
        self._init_weights()

    
    def forward(self, input, 
                return_loss=True,
                return_proj_output=False):
        
        spectra_input = self.load_spectra(input)
        spectra_embeds = self.spectral_encoding(spectra_input)
        spectral_output = self.spectral_encoder(spectra_embeds)
        
        molecular_input_ids = input['smiles']['input_ids']
        molecular_attention_mask = input['smiles']['attention_mask']
        
        # Mask Language Modeling
        masked_input_ids, mlm_mask = self.generate_mlmmask(molecular_input_ids, mask_prob=self.mask_prob)
        masked_molecular_embeds = self.molecular_encoding(masked_input_ids, use_cls_token=False)
        mlm_tokens = self.molecular_decoder(masked_molecular_embeds, spectral_output['hidden_states'], 
                                            src_mask=None,
                                            tgt_mask=molecular_attention_mask)

        result_dict = {}
        if return_loss:
            loss = self.compute_mlm_loss(mlm_tokens['proj_output'][mlm_mask], molecular_input_ids[mlm_mask])
            result_dict['loss'] = loss
            result_dict['mlm_loss'] = loss
        if return_proj_output:
            result_dict['pred_logits'] = mlm_tokens['proj_output'][mlm_mask]
            result_dict['tgt_tokens'] = molecular_input_ids[mlm_mask]
        return result_dict

    def infer_mlm(self, input):
        # Mask Language Infer
        spectra_input = self.load_spectra(input)
        spectra_embeds = self.spectral_encoding(spectra_input)
        spectral_output = self.spectral_encoder(spectra_embeds)

        molecular_input_ids = input['smiles']['input_ids']
        molecular_attention_mask = input['smiles']['attention_mask']
        masked_molecular_embeds = self.molecular_encoding(molecular_input_ids, use_cls_token=False)
        mlm_tokens = self.molecular_decoder(masked_molecular_embeds, spectral_output['hidden_states'], 
                                            src_mask=None,
                                            tgt_mask=molecular_attention_mask)
        return mlm_tokens['proj_output']


@register_model
def vib2mol_mlm(pretrained=False, **kwargs):
    model = Vib2Mol_MLM(nlayer=6, **kwargs)
    return model


