# import sys
# sys.path.append('/inspire/hdd/project/chemicalreaction/luxinyu-240207020178/vib2mol')
import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from models import register_model
from models.modules import clones, cl_loss, make_std_mask, subsequent_mask
from models.modules import EncoderLayer, DecoderLayer, LayerNorm, MultiHeadedAttention, PositionwiseFeedForward
from models.base import BaseModel, SpectralEncoding, MolecularEncoding, FormulaEncoding
from utils.base import PHASE_ALIGN, PHASE_GENERATE, normalize_phase


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

    def forward_step(self, input_embed, memory, src_mask, cache=None):
        cache = cache or [None] * len(self.layers)
        output, new_cache = input_embed, []
        for layer, layer_cache in zip(self.layers, cache):
            output, layer_cache = layer.forward_step(output, memory, src_mask, layer_cache)
            new_cache.append(layer_cache)
        output = self.norm(output)
        return self.proj(output), new_cache


class Vib2Mol(BaseModel):
    def __init__(self,
                 d_proj=256,
                 spectral_channel=2,
                 d_model=768,
                 nhead=8,
                 d_ff=2048,
                 encoder_nlayer=6,
                 decoder_nlayer=6,
                 multimodal_nlayer=3,
                 mask_prob=0.45,
                 phase=PHASE_ALIGN,
                 **kwargs):

        super().__init__(gpu_align=kwargs.get('gpu_align', False))
        self.phase = normalize_phase(phase)
        self.mask_prob = mask_prob
        self.spectral_encoding = SpectralEncoding(d_model=d_model, spectral_channel=spectral_channel)
        self.molecular_encoding = MolecularEncoding(d_model=d_model, num_embeddings=500)
        self.formula_encoding = FormulaEncoding(d_model=d_model, num_embeddings=500)

        self.spectral_encoder = SpectralEncoder(d_model=d_model, nhead=nhead, d_ff=d_ff, nlayer=encoder_nlayer, d_proj=d_proj, in_channel=spectral_channel)
        self.molecular_encoder = MolecularEncoder(d_model=d_model, nhead=nhead, d_ff=d_ff, nlayer=encoder_nlayer, d_proj=d_proj)
        self.molecular_decoder = MolecularDecoder(d_model=d_model, nhead=nhead, d_ff=d_ff, nlayer=decoder_nlayer)

        self.multimodal_encoder = MultiModalEncoder(d_model=d_model, nhead=nhead, d_ff=d_ff, nlayer=multimodal_nlayer)

        self.spectral_mask_token = nn.Parameter(torch.rand([]))
        self._init_weights()

    def get_spectral_embeddings(self, input, raman_only=False, ir_only=False):

        spectral_input = self.load_spectra(input)
        if not self.training and raman_only:
            masked_channel = 1
            spectral_input[:, masked_channel, :] = self.spectral_mask_token
        elif not self.training and ir_only:
            masked_channel = 0
            spectral_input[:, masked_channel, :] = self.spectral_mask_token

        spectra_embeds = self.spectral_encoding(spectral_input)
        spectral_output = self.spectral_encoder(spectra_embeds)
        return spectral_output

    def forward(self, input,
                return_loss=True,
                return_proj_output=False,
                spectral_mask_prob=0.5,
                ):

        spectral_input = self.load_spectra(input) # B, 2, L
        if self.training and spectral_mask_prob > 0:
            if torch.rand(1).item() < spectral_mask_prob:
                masked_channel = 0 if torch.rand(1).item() < 0.5 else 1
                spectral_input[:, masked_channel, :] = self.spectral_mask_token

        spectral_embeds = self.spectral_encoding(spectral_input)
        spectral_output = self.spectral_encoder(spectral_embeds)

        molecular_input_ids = input['smiles']['input_ids']
        molecular_attention_mask = input['smiles']['attention_mask']

        result_dict = {}
        loss = torch.tensor(0, device=spectral_input.device, dtype=spectral_input.dtype)

        if self.phase == PHASE_ALIGN:
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

        elif self.phase == PHASE_GENERATE:
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
            if self.phase == PHASE_ALIGN:
                result_dict['molecular_proj_output'] = molecular_output['proj_output']

            elif self.phase == PHASE_GENERATE:
                molecular_embeds = self.molecular_encoding(molecular_input_ids, use_cls_token=True)
                molecular_attention_mask_with_cls = torch.cat([torch.ones(molecular_embeds.size(0), 1).to(molecular_attention_mask.device), molecular_attention_mask], dim=1)
                molecular_output = self.molecular_encoder(molecular_embeds, mask=molecular_attention_mask_with_cls)
                result_dict['molecular_proj_output'] = molecular_output['proj_output']

            result_dict['spectral_proj_output'] = spectral_output['proj_output']
        return result_dict

    def matching(self, inputs, raman_only=False, ir_only=False):

        spectral_input = self.load_spectra(inputs)
        if not self.training and raman_only:
            masked_channel = 1
            spectral_input[:, masked_channel, :] = self.spectral_mask_token
        elif not self.training and ir_only:
            masked_channel = 0
            spectral_input[:, masked_channel, :] = self.spectral_mask_token


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
              raman_only=False, ir_only=False
              ):

        spectral_input = self.load_spectra(input)
        if not self.training and raman_only:
            masked_channel = 1
            spectral_input[:, masked_channel, :] = self.spectral_mask_token
        elif not self.training and ir_only:
            masked_channel = 0
            spectral_input[:, masked_channel, :] = self.spectral_mask_token

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

        if src_mask is None:
            src_mask = torch.ones(
                src_embeds.size(0), src_embeds.size(1),
                device=src_embeds.device, dtype=src_embeds.dtype,
            )
        return self._greedy_decode_cached(
            src_embeds, src_mask, spectral_input.size(0), max_len,
            return_metrics=return_metrics, target_ids=target_ids,
        )


    def beam_infer_lm(self,
                                input,
                                max_len=256,
                                beam_size=3,
                                temperature=1.0, # Adjusted to 1.0 as typical for log_softmax without explicit temperature scaling for sampling
                                raman_only=False, ir_only=False
                                ):



        # --- Initial Encoding (equivalent to the 'encode' part in the reference) ---
        spectral_input = self.load_spectra(input)
        if not self.training and raman_only:
            masked_channel = 1
            spectral_input[:, masked_channel, :] = self.spectral_mask_token
        elif not self.training and ir_only:
            masked_channel = 0
            spectral_input[:, masked_channel, :] = self.spectral_mask_token

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

        return self._beam_decode_cached(
            memory, src_mask, batch_size, max_len, beam_size, temperature
        )

@register_model
def vib2mol_mmm(pretrained=False, **kwargs):
    return Vib2Mol(encoder_nlayer=6, decoder_nlayer=6, **kwargs)


@register_model
def vib2mol_matching_shared_mask(pretrained=False, **kwargs):
    """Backward-compatible alias for the multi-modality masking model."""
    return vib2mol_mmm(pretrained=pretrained, **kwargs)
