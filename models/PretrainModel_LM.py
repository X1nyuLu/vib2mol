import sys 
sys.path.append('../')

import os
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from models import register_model
from models.modules import clones, make_std_mask, subsequent_mask
from models.modules import EncoderLayer, DecoderLayer, LayerNorm, MultiHeadedAttention, PositionwiseFeedForward
from models.base import BaseModel, SpectralEncoding, MolecularEncoding


class SpectralEncoder(nn.Module):
    def __init__(self, d_model=768, nhead=8, d_ff=2048, nlayer=6, dropout=0.1, in_channel=1):
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
    def __init__(self, d_model=768, nhead=8, d_ff=2048, nlayer=6, dropout=0.1, vocab_size=181):
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


class PretrainModel_LM(BaseModel):
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

        self.spectral_encoding = SpectralEncoding(d_model=d_model)
        self.molecular_encoding = MolecularEncoding(d_model=d_model)
        self.spectral_encoder = SpectralEncoder(d_model=d_model, nhead=nhead, d_ff=d_ff, nlayer=nlayer, in_channel=spectral_channel)
        self.molecular_decoder = MolecularDecoder(d_model=d_model, nhead=nhead, d_ff=d_ff, nlayer=nlayer, vocab_size=vocab_size)

        self._init_weights()
    
    def forward(self, input, 
                return_loss=True,
                return_proj_output=False
                ):

        spectra_input = self.load_spectra(input)
        spectra_embeds = self.spectral_encoding(spectra_input)
        spectral_output = self.spectral_encoder(spectra_embeds)
        
        molecular_input = input['smiles']
        label_ids = molecular_input['input_ids'][:, 1:]
        input_ids = molecular_input['input_ids'][:, :-1]    
        input_embeds = self.molecular_encoding(input_ids)    
        casual_mask = make_std_mask(input_ids, pad=1).type_as(molecular_input['attention_mask'])
        
        # Casual Language Modeling      
        causal_tokens = self.molecular_decoder(input_embeds, spectral_output['hidden_states'], 
                                               src_mask=None, # spectrum-structure
                                               tgt_mask=casual_mask # structure-structure
                                               )

        result_dict = {}
        if return_loss:
            loss = self.compute_lm_loss(causal_tokens['proj_output'], label_ids)
            result_dict['loss'] = loss
            result_dict['lm_loss'] = loss
        if return_proj_output:
            result_dict['pred_logits'] = causal_tokens['proj_output']
            result_dict['tgt_tokens'] = molecular_input['input_ids']
        return result_dict
    
    def infer_lm(self, 
              input, 
              max_len=256,
              return_metrics=False,
              target_ids=None,
              ):
        spectra_input = self.load_spectra(input)
        spectra_embeds = self.spectral_encoding(spectra_input)
        spectral_output = self.spectral_encoder(spectra_embeds)
        pred_ids = torch.zeros(spectra_input.size(0), 1, dtype=torch.long, device=spectra_input.device)
        
        for i in range(max_len-1):
            pred_emebds = self.molecular_encoding(pred_ids)  
            casual_mask = subsequent_mask(pred_ids.size(1)).type_as(pred_ids.data)        
            pred_output = self.molecular_decoder(pred_emebds, spectral_output['hidden_states'], 
                                                 src_mask=None, 
                                                 tgt_mask=casual_mask # 为什么在测试时，把casual_mask改为None就会导致性能的大幅下降
                                                 )
            prob = pred_output['proj_output'][:, -1]
            _, next_word = torch.max(prob, dim=1)
            pred_ids = torch.cat([pred_ids, (next_word).reshape(-1, 1)], dim=1)
        
        result_dict = {'pred_ids':pred_ids}
                
        if return_metrics:
            ntokens = (target_ids != 1).sum() # ignore all <pad> tokens
            target_ids = target_ids.contiguous().view(-1)
            pred_ids = pred_ids.contiguous().view(-1)
            accuracy = sum(pred_ids[target_ids != 1] == target_ids[target_ids != 1]) / ntokens
            result_dict['metrics'] = accuracy
        return result_dict


@register_model
def vib2mol_lm(pretrained=False, **kwargs):
    model = PretrainModel_LM(nlayer=6)
    return model


