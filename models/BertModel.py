import sys 
sys.path.append('../')

import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from models import register_model
from models.modules import clones
from models.modules import EncoderLayer, LayerNorm, MultiHeadedAttention
from models.modules import PositionwiseFeedForward, LearnableClassEmbedding, LearnablePositionalEncoding


class SpectralEncoding(nn.Module):
    def __init__(self, dim, patch_size, norm_layer, in_channel=1):
        super().__init__()
        self.encoding = nn.Conv1d(
            in_channel, dim, kernel_size=patch_size, stride=patch_size, bias=False)
        self.norm = norm_layer(dim) if norm_layer else nn.Identity()

    def forward(self, input):
        input_embeds = self.encoding(input).transpose(1, 2)  # B, C, L -> B, L, C
        return self.norm(input_embeds)


class MolecularEncoder(nn.Module):
    def __init__(self, d_model=768, nhead=12, d_ff=3072, nlayer=12, dropout=0.1, vocab_size=181):

        super().__init__()

        self_attn = MultiHeadedAttention(nhead, d_model, dropout)
        feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        layer = EncoderLayer(d_model, self_attn, feed_forward, dropout)

        self.layers = clones(layer, nlayer)
        self.norm = LayerNorm(d_model)       
        self.proj = nn.Sequential(nn.Linear(d_model, d_model), nn.Tanh(), nn.Linear(d_model, vocab_size))

    def forward(self, input):
        attention_mask = input['attention_mask']
        input_embeds = input['input_embeds']

        layer_output = input_embeds
        for layer in self.layers:
            layer_output = layer(layer_output, mask=attention_mask)
        layer_output = self.norm(layer_output)

        proj_out = self.proj(layer_output)
        return {'hidden_states': layer_output, 'proj_output': proj_out}



class PretrainModel(nn.Module):
    def __init__(self, 
                 spectral_channel=1, 
                 mask_prob=0.45, 
                 d_model=768,
                 nhead=12, 
                 d_ff=2048, 
                 nlayer=12, 
                 num_embeddings=500):
        super().__init__()
        self.mask_prob = mask_prob

        self.molecular_embedding = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=d_model, padding_idx=1)
        self.spectral_encoding = SpectralEncoding(dim=d_model, patch_size=8, norm_layer=LayerNorm, in_channel=spectral_channel)
        self.positional_encoding = LearnablePositionalEncoding(d_model=d_model, dropout=0.1)
        self.encoder = MolecularEncoder(d_model=d_model, nhead=nhead, d_ff=d_ff, nlayer=nlayer)
        
        self.logit_scale = nn.Parameter(torch.rand([]))
        self._init_weights()    
    
    def forward(self, input, 
                return_loss=True,
                return_proj_output=False
                ):
        
        if 'spectra' in input:
            spectra_input = input['spectra']
        elif 'raman' in input and 'ir' in input:
            spectra_input = torch.cat([input['raman'], input['ir']], dim=1)
        elif 'raman' in input and 'ir' not in input:
            spectra_input = input['raman']
        elif 'ir' in input and 'raman' not in input:
            spectra_input = input['ir']
        elif 'exp_ir' in input:
            spectra_input = input['exp_ir']
        elif 'exp_raman' in input:
            spectra_input = input['exp_raman']

        molecular_input = input['smiles']
        masked_input_ids, mlm_mask = self.generate_mlmmask(molecular_input['input_ids'], mask_prob=self.mask_prob)

        spectra_embeds = self.spectral_encoding(spectra_input)
        b, l, d = spectra_embeds.size()

        molecular_embeds = self.molecular_embedding(masked_input_ids)

        total_embeds = torch.cat([spectra_embeds, molecular_embeds], dim=1)
        attention_mask = torch.cat([torch.ones(b, l).to(spectra_embeds.device), molecular_input['attention_mask']], dim=1)

        total_input = self.positional_encoding(total_embeds)
        mlm_tokens = self.encoder({'input_embeds': total_input, 'attention_mask':attention_mask})


        result_dict = {}
        if return_loss:
            mlm_loss = self.compute_mlm_loss(mlm_tokens['proj_output'][:, l:][mlm_mask], molecular_input['input_ids'][mlm_mask])
            loss = mlm_loss
            result_dict['loss'] = loss
            result_dict['mlm_loss'] = loss
        if return_proj_output:
            result_dict['pred_logits'] = mlm_tokens['proj_output'][:, l:][mlm_mask]
            result_dict['tgt_tokens'] = molecular_input['input_ids'][mlm_mask]
        return result_dict

    def generate_mlmmask(self, input_ids, mask_prob=0.45):
        masked_ids = input_ids.clone()
        probability_matrix = torch.full(masked_ids.shape, mask_prob)         
        masked_indices = torch.bernoulli(probability_matrix).bool()                                  
        masked_indices[masked_ids == 1] = False # tokenizer.pad_token_id = 1
        masked_ids[masked_indices] = 4 # tokenizer.mask_token_id = 4
        
        return masked_ids, masked_indices

    def compute_mlm_loss(self, pred, target):
        loss = F.cross_entropy(pred, target, ignore_index=1)
        return loss

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0.0)

    def infer_mlm(self, input):
        # Mask Language Infer
        
        if 'spectra' in input:
            spectra_input = input['spectra']
        elif 'raman' in input and 'ir' in input:
            spectra_input = torch.cat([input['raman'], input['ir']], dim=1)
        elif 'raman' in input and 'ir' not in input:
            spectra_input = input['raman']
        elif 'ir' in input and 'raman' not in input:
            spectra_input = input['ir']
        
        
        masked_molecular_input = input['smiles']

        spectra_embeds = self.spectral_encoding(spectra_input)
        b, l, d = spectra_embeds.size()

        molecular_embeds = self.molecular_embedding(masked_molecular_input['input_ids'])

        total_embeds = torch.cat([spectra_embeds, molecular_embeds], dim=1)
        attention_mask = torch.cat([torch.ones(b, l).to(spectra_embeds.device), masked_molecular_input['attention_mask']], dim=1)

        total_input = self.positional_encoding(total_embeds)
        mlm_tokens = self.encoder({'input_embeds': total_input, 'attention_mask':attention_mask})
        return mlm_tokens['proj_output'][:, 128:]


@register_model
def bert_mlm(**kwargs):
    model = PretrainModel(nlayer=12, **kwargs)
    return model
