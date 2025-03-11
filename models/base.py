import sys 
sys.path.append('../')

import os
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.modules import clip_loss, subsequent_mask
from models.modules import LayerNorm, PositionalEncoding, LearnableClassEmbedding
from utils.base import seed_everything

seed_everything(624)
class SpectralEncoding(nn.Module):
    def __init__(self, d_model=768, patch_size=8, norm_layer=LayerNorm, dropout=0.1, spectral_channel=1):
        super().__init__()
        self.encoding = nn.Conv1d(spectral_channel, d_model, kernel_size=patch_size, stride=patch_size, bias=False)
        self.norm = norm_layer(d_model) if norm_layer else nn.Identity()
        self.class_encoding = LearnableClassEmbedding(d_model, dropout)
        self.positional_encoding = PositionalEncoding(d_model, dropout)        
        
    def forward(self, input_spectra):
        input_embeds = self.encoding(input_spectra).transpose(1, 2)  # B, C, L -> B, L, C
        input_embeds = self.norm(input_embeds)
        input_embeds = self.class_encoding(input_embeds)
        input_embeds = self.positional_encoding(input_embeds)
        return input_embeds
    
    
class MolecularEncoding(nn.Module):
    def __init__(self, d_model=768, num_embeddings=512, dropout=0.1):

        super().__init__()
        self.d_model = d_model
        self.molecular_embedding = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=d_model, padding_idx=1)
        self.class_encoding = LearnableClassEmbedding(d_model, dropout)
        self.positional_encoding = PositionalEncoding(d_model, dropout)
        self.mask_token = nn.Parameter(torch.randn(d_model))

    def forward(self, input_ids, use_cls_token=False, mask_token_id=4):
        input_embeds = self.molecular_embedding(input_ids)
        mask_positions = (input_ids == mask_token_id).unsqueeze(-1)  
        input_embeds = torch.where(mask_positions, self.mask_token, input_embeds)  
        # input_embeds = self.molecular_embedding(input_ids) * math.sqrt(self.d_model)
        if use_cls_token:
            input_embeds = self.class_encoding(input_embeds)            
        input_embeds = self.positional_encoding(input_embeds)
        return input_embeds


class BaseModel(nn.Module):
    def __init__(self, 
                 d_proj=256, 
                 vocab_size=181, 
                 spectral_channel=1, 
                 d_model=768, 
                 nhead=12, 
                 d_ff=3072, 
                 nlayer=6,
                 mask_prob=0.45,
                 in_channel=1):
        super().__init__()
        self.mask_prob = mask_prob
        self.logit_scale = nn.Parameter(torch.rand([]))
    
    def forward(self, input, 
                return_loss=True,
                return_proj_output=False):
        pass
    
    def infer(self, 
              input, 
              max_len=256,
              return_metrics=False,
              target_ids=None):
        pass
    
    def load_spectra(self, input):
        if 'spectra' in input:
            spectra_input = input['spectra']
        elif 'ir' in input and 'raman' in input:
            spectra_input = torch.cat([input['raman'], input['ir']], dim=1)
        elif 'exp_ir' in input and 'raman' in input:
            spectra_input = torch.cat([input['raman'], input['exp_ir']], dim=1)
        elif 'ir' in input and 'raman' not in input:
            spectra_input = input['ir']
        elif 'exp_ir' in input and 'raman' not in input:
            spectra_input = input['exp_ir']
        elif 'raman' in input and 'ir' not in input and 'exp_ir' not in input:
            spectra_input = input['raman']
        return spectra_input
    
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
    
    def compute_lm_loss(self, pred, target):
        loss = F.cross_entropy(pred.contiguous().view(-1, pred.size(-1)), target.contiguous().view(-1), ignore_index=1)
        return loss
    
    def compute_clip_loss(self, molecular_output, spectral_output):
        molecular_output = F.normalize(molecular_output, p=2, dim=1)
        spectral_output = F.normalize(spectral_output, p=2, dim=1)

        logit_scale = self.logit_scale.exp()
        logits_per_smiles = torch.matmul(
            molecular_output, spectral_output.t()) * logit_scale
        logits_per_spectrum = logits_per_smiles.T
        loss = clip_loss(logits_per_spectrum)
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
                
    def get_spectral_embeddings(self, input):
        spectra_input = self.load_spectra(input)
        spectra_embeds = self.spectral_encoding(spectra_input)
        spectral_output = self.spectral_encoder(spectra_embeds)
        return spectral_output
    
    def get_molecular_embeddings(self, input, use_cls_token=False):
        if 'smiles' in input:
            molecular_input_ids = input['smiles']['input_ids']
            molecular_attention_mask = input['smiles']['attention_mask']
        elif 'sequence' in input:
            molecular_input_ids = input['sequence']['input_ids']
            molecular_attention_mask = input['sequence']['attention_mask']

        molecular_embeds = self.molecular_encoding(molecular_input_ids, use_cls_token=use_cls_token)
        if use_cls_token:
            molecular_attention_mask = torch.cat([torch.ones(molecular_embeds.size(0), 1).to(molecular_attention_mask.device), molecular_attention_mask], dim=1)
        molecular_output = self.molecular_encoder(molecular_embeds, molecular_attention_mask)
        return molecular_output
    
    def infer_mlm(self, input):
        # Mask Language Infer
        spectra_input = self.load_spectra(input)
        spectra_embeds = self.spectral_encoding(spectra_input)
        spectral_output = self.spectral_encoder(spectra_embeds)

        molecular_input_ids = input['smiles']['input_ids']
        molecular_attention_mask = input['smiles']['attention_mask']
        masked_molecular_embeds = self.molecular_encoding(molecular_input_ids, use_cls_token=False)
        masked_molecular_output = self.molecular_encoder(masked_molecular_embeds, molecular_attention_mask)
        mlm_tokens = self.molecular_decoder(masked_molecular_output['hidden_states'], spectral_output['hidden_states'], 
                                            src_mask=None,
                                            tgt_mask=molecular_attention_mask)
        return mlm_tokens['proj_output']
    
    
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
                                                 src_mask=None, # spectrum-structure
                                                 tgt_mask=casual_mask # structure-structure
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
    

    def beam_infer_lm(self, 
                      input, 
                      max_len=256,
                      beam_size=3,  
                      temperature=15,
                      ):

        spectra_input = self.load_spectra(input)
        batch_size = spectra_input.size(0)  
        spectra_embeds = self.spectral_encoding(spectra_input)
        spectral_output = self.spectral_encoder(spectra_embeds)
        
        # init start token
        start_token = torch.zeros(batch_size, 1, dtype=torch.long, device=spectra_input.device)  
        pred_seqs = start_token.unsqueeze(1).expand(batch_size, beam_size, 1)  # (batch_size, beam_size, seq_len)
        beam_scores = torch.zeros(batch_size, beam_size, device=spectra_input.device)  # (batch_size, beam_size)

        final_outputs = [[] for _ in range(batch_size)]  

        for i in range(max_len - 1):
            # reshape to (batch_size * beam_size, seq_len)
            flat_pred_seqs = pred_seqs.reshape(batch_size * beam_size, -1)
            pred_embeds = self.molecular_encoding(flat_pred_seqs)
            casual_mask = subsequent_mask(flat_pred_seqs.size(1)).type_as(flat_pred_seqs.data)
            
            # molecular decoder
            pred_output = self.molecular_decoder(
                pred_embeds,
                spectral_output['hidden_states'].repeat_interleave(beam_size, dim=0),  
                src_mask=None,  # spectrum-structure
                tgt_mask=casual_mask  # structure-structure
            )

            # get current logp
            prob = pred_output['proj_output'][:, -1]  
            log_prob = torch.log_softmax(prob / temperature, dim=-1)  # log-probability
            vocab_size = log_prob.size(-1)
            
            # add noise
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(log_prob)))  
            log_prob = log_prob + gumbel_noise * 0.1 

            log_prob = log_prob.view(batch_size, beam_size, vocab_size)  # (batch_size, beam_size, vocab_size)

            expanded_scores = beam_scores.unsqueeze(-1) + log_prob  # (batch_size, beam_size, vocab_size)

            topk_scores, topk_indices = torch.topk(expanded_scores.view(batch_size, -1), beam_size, dim=-1)  # (batch_size, beam_size)
            beam_indices = topk_indices // vocab_size  
            word_indices = topk_indices % vocab_size  

            # update beams
            pred_seqs = torch.cat([
                pred_seqs.gather(1, beam_indices.unsqueeze(-1).expand(-1, -1, pred_seqs.size(-1))),  
                word_indices.unsqueeze(-1)  
            ], dim=-1)
            beam_scores = topk_scores

            # detect </s> token
            for b in range(batch_size):
                for j in range(beam_size):
                    if pred_seqs[b, j, -1].item() == 2:  
                        final_outputs[b].append((beam_scores[b, j].item(), pred_seqs[b, j].clone()))
                        beam_scores[b, j] = -1e9  

            if all(len(outputs) >= beam_size for outputs in final_outputs):
                break

        for b in range(batch_size):
            final_outputs[b] = sorted(final_outputs[b], key=lambda x: x[0], reverse=True)[:beam_size]
        result_dict = {'pred_ids': [[output[1] for output in outputs] for outputs in final_outputs],
                       'score':[[output[0] for output in outputs] for outputs in final_outputs]} 

        return result_dict