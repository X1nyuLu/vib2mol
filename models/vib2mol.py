
import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from models import register_model
from models.modules import clones, cl_loss, make_std_mask, subsequent_mask
from models.modules import EncoderLayer, DecoderLayer, LayerNorm, MultiHeadedAttention, PositionwiseFeedForward
from models.base import BaseModel, SpectralEncoding, MolecularEncoding, FormulaEncoding
    
    
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
        
        self.uni_encoder_layers = nn.ModuleList()
        self.multi_encoder_layers = nn.ModuleList()
        
        for _ in range(nlayer):
            self_attn = MultiHeadedAttention(nhead, d_model, dropout)
            src_attn = MultiHeadedAttention(nhead, d_model, dropout)
            feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
            
            uni_encoder_layer = EncoderLayer(d_model, self_attn, feed_forward, dropout)
            multi_encoder_layer = DecoderLayer(d_model, self_attn, src_attn, feed_forward, dropout)
            
            self.uni_encoder_layers.append(uni_encoder_layer)
            self.multi_encoder_layers.append(multi_encoder_layer)
        
        self.uni_enc_norm = LayerNorm(d_model)
        self.multi_enc_norm = LayerNorm(d_model)

        self.uni_proj = nn.Sequential(nn.Linear(d_model, d_proj), nn.ReLU(), nn.Linear(d_proj, d_proj))
        self.multi_proj = nn.Sequential(nn.Linear(d_model, d_proj), nn.ReLU(), nn.Linear(d_proj, 2))

    def forward(self, input_embeds, memory=None, mask=None, src_mask=None, tgt_mask=None, multimodal=False):
        layer_output = input_embeds

        if multimodal:
            for layer in self.multi_encoder_layers:
                layer_output = layer(layer_output, memory, src_mask, tgt_mask)
            layer_output = self.multi_enc_norm(layer_output)
            cls_token = layer_output[:, 0]
            proj_out = self.multi_proj(cls_token)

        else:
            for layer in self.uni_encoder_layers:
                layer_output = layer(layer_output, mask=mask)
            layer_output = self.uni_enc_norm(layer_output)
            cls_token = layer_output[:, 0]
            proj_out = self.uni_proj(cls_token)

        return {'hidden_states': layer_output, 'proj_output': proj_out}


class MultiModalEncoder(nn.Module):
    def __init__(self, d_model=768, nhead=8, d_ff=2048, nlayer=3, dropout=0.1):
        super().__init__()

        self_attn = MultiHeadedAttention(nhead, d_model, dropout)
        feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        layer = EncoderLayer(d_model, self_attn, feed_forward, dropout)

        self.layers = clones(layer, nlayer)
        self.norm = LayerNorm(d_model)       

    def forward(self, input_embeds, mask):
        layer_output = input_embeds
        for layer in self.layers:
            layer_output = layer(layer_output, mask=mask)
        layer_output = self.norm(layer_output)

        return {'hidden_states': layer_output}
    
    
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
    

class Vib2Mol(BaseModel):
    def __init__(self, 
                 d_proj=256, 
                 spectral_channel=1, 
                 d_model=768, 
                 nhead=8, 
                 d_ff=2048, 
                 encoder_nlayer=6,
                 decoder_nlayer=6,
                 multimodal_nlayer=3,
                 mask_prob=0.45,
                 phase=1,
                 **kwargs):
        
        super().__init__()
        self.phase = phase
        self.mask_prob = mask_prob
        self.spectral_encoding = SpectralEncoding(d_model=d_model, spectral_channel=spectral_channel)
        self.molecular_encoding = MolecularEncoding(d_model=d_model, num_embeddings=500)
        self.formula_encoding = FormulaEncoding(d_model=d_model, num_embeddings=500)
        
        self.spectral_encoder = SpectralEncoder(d_model=d_model, nhead=nhead, d_ff=d_ff, nlayer=encoder_nlayer, d_proj=d_proj, in_channel=spectral_channel)
        self.molecular_encoder = MolecularEncoder(d_model=d_model, nhead=nhead, d_ff=d_ff, nlayer=encoder_nlayer, d_proj=d_proj)
        self.molecular_decoder = MolecularDecoder(d_model=d_model, nhead=nhead, d_ff=d_ff, nlayer=decoder_nlayer)

        self.multimodal_encoder = MultiModalEncoder(d_model=d_model, nhead=nhead, d_ff=d_ff, nlayer=multimodal_nlayer)
        
        self._init_weights()

    def forward(self, input, 
                return_loss=True,
                return_proj_output=False
                ):
        
        spectral_input = self.load_spectra(input)
        spectral_embeds = self.spectral_encoding(spectral_input)
        spectral_output = self.spectral_encoder(spectral_embeds)
        
        molecular_input_ids = input['smiles']['input_ids']
        molecular_attention_mask = input['smiles']['attention_mask']
        
        result_dict = {}
        loss = torch.tensor(0, device=spectral_input.device, dtype=spectral_input.dtype)
        
        if self.phase == 1:
            # Contrastive Learning
            molecular_embeds = self.molecular_encoding(molecular_input_ids, use_cls_token=True)
            molecular_attention_mask_with_cls = torch.cat([torch.ones(molecular_embeds.size(0), 1).to(molecular_attention_mask.device), 
                                                           molecular_attention_mask], dim=1)
            
            molecular_output = self.molecular_encoder(molecular_embeds, mask=molecular_attention_mask_with_cls)

            spectral_contra_token = spectral_output['proj_output']
            molecular_contra_token = molecular_output['proj_output']
            
            cl_loss, sim_m2s, sim_s2m = self.compute_cl_loss(molecular_contra_token, spectral_contra_token, return_sim=True)
            result_dict['cl_loss'] = cl_loss
            loss += cl_loss
            
            # spectrum-structure matching
            
            with torch.no_grad():                
                bs = spectral_contra_token.size(0)
                
                mask = torch.eye(bs, dtype=torch.bool, device=spectral_contra_token.device)

                weights_m2s = F.softmax(sim_m2s, dim=1)
                weights_m2s.masked_fill_(mask, 0)            
                weights_s2m = F.softmax(sim_s2m, dim=1)
                weights_s2m.masked_fill_(mask, 0)     
                
                # select negtive idx
                molecular_neg_idx = []
                spectral_neg_idx = []
                
                for b in range(bs):
                    molecular_neg_idx.append(torch.multinomial(weights_s2m[b], 1).item())
                    spectral_neg_idx.append(torch.multinomial(weights_m2s[b], 1).item())
                    
                molecular_neg_idx = torch.tensor(molecular_neg_idx, device=spectral_contra_token.device)
                spectral_neg_idx = torch.tensor(spectral_neg_idx, device=spectral_contra_token.device)
                    
            # concatenate positive and negative samples
            molecular_all_embeds = torch.cat([molecular_embeds, molecular_embeds[molecular_neg_idx]], dim=0)
            molecular_all_masks = torch.cat([molecular_attention_mask_with_cls, molecular_attention_mask_with_cls[molecular_neg_idx]], dim=0)
            
            spectral_all_outputs = torch.cat([spectral_output['hidden_states'][spectral_neg_idx], spectral_output['hidden_states']], dim=0)
            spectral_all_masks = None
            
            multimodal_neg_outputs = self.molecular_encoder(molecular_all_embeds, 
                                                        memory=spectral_all_outputs,  
                                                        src_mask=spectral_all_masks,
                                                        tgt_mask=molecular_all_masks,
                                                        multimodal=True,
                                                        )
            
            multimodal_pos_outputs = self.molecular_encoder(molecular_embeds, 
                                                        memory=spectral_output['hidden_states'],  
                                                        src_mask=None,
                                                        tgt_mask=molecular_attention_mask_with_cls,
                                                        multimodal=True,
                                                        )
            
            # create labels for matching
            
            matching_outputs = torch.cat([multimodal_pos_outputs['proj_output'], 
                                            multimodal_neg_outputs['proj_output']], dim=0)

            matching_labels = torch.cat([torch.ones(bs, dtype=torch.long), torch.zeros(2*bs, dtype=torch.long)], dim=0).to(spectral_contra_token.device)

            # calculate matching loss
            matching_loss = F.cross_entropy(matching_outputs, matching_labels)  
            result_dict['matching_loss'] = matching_loss
            loss += matching_loss

            # calculate matching accuracy
            _ , matching_pred = torch.max(matching_outputs, dim=-1)
            accuracy = torch.eq(matching_pred, matching_labels).sum() / len(matching_labels)
            result_dict['matching_accuracy'] = accuracy

            if return_loss:
                result_dict['loss'] = loss

            if return_proj_output:
                result_dict['molecular_proj_output'] = molecular_contra_token
                result_dict['spectral_proj_output'] = spectral_contra_token
                
            return result_dict
        
        elif self.phase == 2:
            src_embeds = spectral_output['hidden_states']
            src_mask = None

            if 'formula' in input:
                formula_input_ids = input['formula']['input_ids']
                formula_attention_mask = input['formula']['attention_mask']
                formula_embeds = self.formula_encoding(formula_input_ids)
                
                src_embeds = torch.cat([spectral_output['hidden_states'], formula_embeds], dim=1)
                src_mask = torch.cat([torch.ones(spectral_embeds.size(0), spectral_embeds.size(1)).type_as(molecular_attention_mask), 
                                      formula_attention_mask], dim=1)
                
                src_embeds = self.multimodal_encoder(src_embeds, src_mask)
                src_embeds = src_embeds['hidden_states']

            # Mask Language Modeling
            masked_input_ids, mlm_mask = self.generate_mlmmask(molecular_input_ids, mask_prob=self.mask_prob)
            masked_molecular_embeds = self.molecular_encoding(masked_input_ids, use_cls_token=False)
            masked_molecular_output = self.molecular_encoder(masked_molecular_embeds, molecular_attention_mask)
            mlm_tokens = self.molecular_decoder(masked_molecular_output['hidden_states'], 
                                                src_embeds,
                                                src_mask=src_mask,
                                                tgt_mask=molecular_attention_mask
                                                )
            
            # Casual Language Modeling
            casual_label_ids = molecular_input_ids[:, 1:]
            casual_input_ids = molecular_input_ids[:, :-1]    
            casual_molecular_embeds = self.molecular_encoding(casual_input_ids, use_cls_token=False)
                
            casual_mask = make_std_mask(casual_input_ids, pad=1).type_as(molecular_attention_mask)
            causal_tokens = self.molecular_decoder(casual_molecular_embeds, 
                                                src_embeds,
                                                src_mask=src_mask, # spectrum-structure
                                                tgt_mask=casual_mask # structure-structure
                                                )
            if return_loss:
                mlm_loss = self.compute_mlm_loss(mlm_tokens['proj_output'][mlm_mask], molecular_input_ids[mlm_mask])
                lm_loss = self.compute_lm_loss(causal_tokens['proj_output'], casual_label_ids)

                loss += mlm_loss
                loss += lm_loss

                result_dict['mlm_loss'] = mlm_loss
                result_dict['lm_loss'] = lm_loss
                result_dict['loss'] = loss
        else:
            raise 'phase error'

        if return_proj_output:
            if self.phase == 1:
                result_dict['molecular_proj_output'] = molecular_output['proj_output']
            
            elif self.phase == 2:
                molecular_embeds = self.molecular_encoding(molecular_input_ids, use_cls_token=True)
                molecular_attention_mask_with_cls = torch.cat([torch.ones(molecular_embeds.size(0), 1).to(molecular_attention_mask.device), molecular_attention_mask], dim=1)
                molecular_output = self.molecular_encoder(molecular_embeds, mask=molecular_attention_mask_with_cls)
                result_dict['molecular_proj_output'] = molecular_output['proj_output']
                
            result_dict['spectral_proj_output'] = spectral_output['proj_output']
        return result_dict

    def compute_cl_loss(self, molecular_output, spectral_output, return_sim=False):
        molecular_output = F.normalize(molecular_output, p=2, dim=1)
        spectral_output = F.normalize(spectral_output, p=2, dim=1)

        logit_scale = self.logit_scale.exp()
        logits_per_smiles = torch.matmul(
            molecular_output, spectral_output.t()) * logit_scale
        logits_per_spectrum = logits_per_smiles.T
        loss = cl_loss(logits_per_spectrum)
        
        if return_sim:
            return loss, logits_per_smiles, logits_per_spectrum
        else:
            return loss
        
    def matching(self, inputs):
        
        spectral_input = self.load_spectra(inputs)
        spectral_embeds = self.spectral_encoding(spectral_input)
        spectral_output = self.spectral_encoder(spectral_embeds)
        
        molecular_input_ids = inputs['smiles']['input_ids']
        molecular_attention_mask = inputs['smiles']['attention_mask']
        molecular_embeds = self.molecular_encoding(molecular_input_ids, use_cls_token=True)
        molecular_attention_mask_with_cls = torch.cat([torch.ones(molecular_attention_mask.size(0), 1, dtype=molecular_attention_mask.dtype, device=molecular_attention_mask.device), 
                                                       molecular_attention_mask], dim=1)
        
        multimodal_outputs = self.molecular_encoder(molecular_embeds, 
                                                     memory=spectral_output['hidden_states'],  
                                                     src_mask=None,
                                                     tgt_mask=molecular_attention_mask_with_cls,
                                                     multimodal=True,
                                                     )
        matching_outputs = multimodal_outputs['proj_output']
        return matching_outputs
    
    def infer_mlm(self, input):
        # Mask Language Infer
        spectral_input = self.load_spectra(input)
        spectral_embeds = self.spectral_encoding(spectral_input)
        spectral_output = self.spectral_encoder(spectral_embeds)
        
        src_embeds = spectral_output['hidden_states']
        src_mask = None

        if 'formula' in input:
            formula_input_ids = input['formula']['input_ids']
            formula_attention_mask = input['formula']['attention_mask']
            formula_embeds = self.formula_encoding(formula_input_ids)
            
            src_embeds = torch.cat([spectral_output['hidden_states'], formula_embeds], dim=1)
            src_mask = torch.cat([torch.ones(spectral_embeds.size(0), spectral_embeds.size(1)).type_as(src_embeds), 
                                    formula_attention_mask], dim=1)
            
            src_embeds = self.multimodal_encoder(src_embeds, src_mask)
            src_embeds = src_embeds['hidden_states']
            

        molecular_input_ids = input['smiles']['input_ids']
        molecular_attention_mask = input['smiles']['attention_mask']
        
        masked_molecular_embeds = self.molecular_encoding(molecular_input_ids, use_cls_token=False)
        masked_molecular_output = self.molecular_encoder(masked_molecular_embeds, molecular_attention_mask)
        mlm_tokens = self.molecular_decoder(masked_molecular_output['hidden_states'], 
                                            src_embeds,
                                            src_mask=src_mask,
                                            tgt_mask=molecular_attention_mask
                                            )
                         
        return mlm_tokens['proj_output']

    def infer_lm(self, 
              input, 
              max_len=256,
              return_metrics=False,
              target_ids=None,
              ):
        spectral_input = self.load_spectra(input)
        spectral_embeds = self.spectral_encoding(spectral_input)
        spectral_output = self.spectral_encoder(spectral_embeds)
        
        src_embeds = spectral_output['hidden_states']
        src_mask = None
            
        pred_ids = torch.zeros(spectral_input.size(0), 1, dtype=torch.long, device=spectral_input.device)
        
        if 'formula' in input:
            formula_input_ids = input['formula']['input_ids']
            formula_attention_mask = input['formula']['attention_mask']
            formula_embeds = self.formula_encoding(formula_input_ids)
            
            src_embeds = torch.cat([spectral_output['hidden_states'], formula_embeds], dim=1)
            src_mask = torch.cat([torch.ones(spectral_embeds.size(0), spectral_embeds.size(1)).type_as(src_embeds), 
                                    formula_attention_mask], dim=1)
            
            src_embeds = self.multimodal_encoder(src_embeds, src_mask)
            src_embeds = src_embeds['hidden_states']
            
        for i in range(max_len-1):
            pred_emebds = self.molecular_encoding(pred_ids)
            casual_mask = subsequent_mask(pred_ids.size(1)).type_as(pred_ids.data)

            pred_output = self.molecular_decoder(pred_emebds, src_embeds, 
                                                 src_mask=src_mask, # spectrum-structure
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
                      temperature=1.0, # Adjusted to 1.0 as typical for log_softmax without explicit temperature scaling for sampling
                      ):
        # --- Initial Encoding (equivalent to the 'encode' part in the reference) ---
        spectral_input = self.load_spectra(input)
        batch_size = spectral_input.size(0)

        # Encode spectral input
        spectral_embeds = self.spectral_encoding(spectral_input)
        spectral_output = self.spectral_encoder(spectral_embeds)
        
        # Mocking the output format of spectral_encoder to match reference's memory
        src_embeds_initial = spectral_output['hidden_states'] # First call to model.encode
        src_mask_initial = torch.ones(spectral_embeds.size(0), spectral_embeds.size(1)).type_as(src_embeds_initial)


        # Handle formula input if present
        if isinstance(input, dict) and 'formula' in input:
            formula_input_ids = input['formula']['input_ids']
            formula_attention_mask = input['formula']['attention_mask']
            formula_embeds = self.formula_encoding(formula_input_ids)

            # Combine spectral and formula embeddings for multimodal encoding
            # This logic is now implicitly handled within self.model.encode if formula is passed
            src_embeds = torch.cat([src_embeds_initial, formula_embeds], dim=1)
            src_mask = torch.cat([src_mask_initial, formula_attention_mask], dim=1)
            
            # This step conceptually becomes part of the model.encode logic
            memory = self.multimodal_encoder(src_embeds, src_mask)['hidden_states']
        else:
            memory = src_embeds_initial
            src_mask = src_mask_initial # Only spectral mask


        # --- Beam Search Initialization (similar to reference's first step) ---
        # Initialize target sequence with BOS token
        ys = torch.zeros(batch_size, 1, dtype=torch.long, device=spectral_input.device)
        ys_embeds = self.molecular_encoding(ys)
        # Perform initial decoding step to get first probabilities
        # Repeat memory and src_mask for all beams for the first step (beam_size=1 effectively)
        # No repeat_interleave here yet because we are getting the *initial* topk for beam_size paths
        out = self.molecular_decoder(
            ys_embeds, 
            memory,
            src_mask=src_mask,
            tgt_mask=subsequent_mask(ys.size(1)).type_as(spectral_input.data),
        )

        prob = out['proj_output'][:, -1] # Get logits for the last token
        log_prob = F.log_softmax(prob / temperature, dim=-1) # Apply log_softmax and temperature

        # Select top-k initial tokens to form the first set of beams
        # For initialization, we assume prob is for a single sequence per batch item
        topk_scores, topk_indices = log_prob.topk(k=beam_size, dim=-1) # (batch_size, beam_size)

        # Initialize beams: (current_sequence, current_score)
        # Expand ys for each beam in the batch
        pred_seqs = ys.unsqueeze(1).repeat(1, beam_size, 1) # (batch_size, beam_size, 1)
        pred_seqs = torch.cat([pred_seqs, topk_indices.unsqueeze(-1)], dim=-1) # (batch_size, beam_size, 2)
        beam_scores = topk_scores # (batch_size, beam_size)

        final_outputs = [[] for _ in range(batch_size)] # Store (score, sequence) for each batch item

        # --- Beam Search Loop ---
        for i in range(max_len - 1): # Max_len - 1 because we already generated 1 token
            flat_pred_seqs = pred_seqs.reshape(batch_size * beam_size, -1) # (batch*beam, seq_len)
            flat_pred_embeds = self.molecular_encoding(flat_pred_seqs)
            # Prepare memory and src_mask for repeated beams
            # memory and src_mask need to be repeated beam_size times for decoding each beam
            expanded_memory = memory.repeat_interleave(beam_size, dim=0)
            expanded_src_mask = src_mask.repeat_interleave(beam_size, dim=0)

            casual_mask = subsequent_mask(flat_pred_seqs.size(1)).type_as(flat_pred_seqs.data)

            # Molecular decoder step (now handled by self.model.decode)
            pred_output = self.molecular_decoder(
                flat_pred_embeds,
                expanded_memory,
                tgt_mask=casual_mask,
                src_mask=expanded_src_mask
            )

            # Get current log probabilities for the next token
            prob = pred_output['proj_output'][:, -1]
            log_prob = F.log_softmax(prob / temperature, dim=-1)
            vocab_size = log_prob.size(-1)

            # Reshape log_prob to (batch_size, beam_size, vocab_size)
            log_prob = log_prob.view(batch_size, beam_size, vocab_size)

            # Add current beam scores to the new log probabilities
            expanded_scores = beam_scores.unsqueeze(-1) + log_prob

            # Select top-k candidates across all beams for the next step
            # Flatten to (batch_size, beam_size * vocab_size) to find overall top-k
            topk_scores, topk_indices = expanded_scores.view(batch_size, -1).topk(beam_size, dim=-1)

            # Determine which beam and which word each top-k candidate came from
            beam_indices = topk_indices // vocab_size
            word_indices = topk_indices % vocab_size

            # Update sequences and scores for the next iteration
            # Use gather to select the correct parent sequences
            pred_seqs_temp = pred_seqs.gather(1, beam_indices.unsqueeze(-1).expand(-1, -1, pred_seqs.size(-1)))
            pred_seqs = torch.cat([pred_seqs_temp, word_indices.unsqueeze(-1)], dim=-1)
            beam_scores = topk_scores

            # Detect </s> token (EOS_idx) for early termination and collect finished sequences
            for b in range(batch_size):
                for j in range(beam_size):
                    if pred_seqs[b, j, -1].item() == 2:
                        # Append the finished sequence and its score
                        final_outputs[b].append((beam_scores[b, j].item(), pred_seqs[b, j].clone()))
                        # Invalidate this beam so it's not chosen again
                        beam_scores[b, j] = -1e9 # A very small number to effectively remove it from topk consideration

            # Check if all beams for all batches have terminated
            # This loop condition is simplified from the original reference
            # The reference just runs for max_len-2 steps.
            # If you want early stopping based on all beams finishing, uncomment/adapt below:
            if all(len(outputs) >= beam_size for outputs in final_outputs):
                break

        # Collect any remaining unfinished sequences if the loop completes without all terminating
        for b in range(batch_size):
            # Add any beams that haven't reached EOS but are still in the top-k
            for j in range(beam_size):
                if pred_seqs[b, j, -1].item() != 2 and beam_scores[b, j].item() != -1e9:
                    final_outputs[b].append((beam_scores[b, j].item(), pred_seqs[b, j].clone()))

            # Sort and select the top 'beam_size' (or 'n_best' if you introduce that parameter)
            final_outputs[b] = sorted(final_outputs[b], key=lambda x: x[0], reverse=True)[:beam_size]

        # Prepare the result dictionary
        result_dict = {
            'pred_ids': [[output[1] for output in outputs] for outputs in final_outputs],
            'score': [[output[0] for output in outputs] for outputs in final_outputs],
        }

        return result_dict
    
@register_model
def vib2mol(pretrained=False, **kwargs):
    model = Vib2Mol(encoder_nlayer=6, decoder_nlayer=6, **kwargs)
    return model


if __name__ == "__main__":
    from thop import profile
    net = vib2mol(phase=1)
    data = {'smiles':{'input_ids': torch.randint(0, 100, (10, 64)),
                      'attention_mask': torch.ones(10, 64)}, 
            'spectra':torch.randn(10, 1, 1024)}
    output = net(data)
    print(output.items())
    macs, params = profile(net, inputs=({'smiles':{'input_ids': torch.randint(0, 100, (2, 64)), 
                                                   'attention_mask': torch.ones(2, 64)}, 
                                        'spectra':torch.randn(2, 1, 1024)},))
    
    print(f'params: {params/1e6} M, macs: {macs/1e9} G') # params: 113.7216 M, macs: 768.65339392 G

    



