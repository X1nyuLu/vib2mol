import os
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from models.modules import cl_loss, subsequent_mask
from models.modules import LayerNorm, PositionalEncoding, LearnableClassEmbedding
from utils.base import seed_everything

seed_everything(624)
class SpectralEncoding(nn.Module):
    def __init__(self, d_model=512, patch_size=8, norm_layer=LayerNorm, dropout=0.1, spectral_channel=1):
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
    def __init__(self, d_model=512, num_embeddings=512, dropout=0.1):

        super().__init__()
        self.d_model = d_model
        self.molecular_embedding = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=d_model, padding_idx=1)
        self.class_encoding = LearnableClassEmbedding(d_model, dropout)
        self.positional_encoding = PositionalEncoding(d_model, dropout)
        self.mask_token = nn.Parameter(torch.randn(d_model))

    def forward(
        self, input_ids, use_cls_token=False, mask_token_id=4,
        position_offset=0,
    ):
        input_embeds = self.molecular_embedding(input_ids)
        mask_positions = (input_ids == mask_token_id).unsqueeze(-1)
        input_embeds = torch.where(mask_positions, self.mask_token, input_embeds)
        # input_embeds = self.molecular_embedding(input_ids) * math.sqrt(self.d_model)
        if use_cls_token:
            input_embeds = self.class_encoding(input_embeds)
        input_embeds = self.positional_encoding(
            input_embeds, position_offset=position_offset
        )
        return input_embeds


class FormulaEncoding(nn.Module):
    def __init__(self, d_model=512, num_embeddings=512, dropout=0.1):

        super().__init__()
        self.d_model = d_model
        self.formula_embedding = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=d_model, padding_idx=1)
        self.class_encoding = LearnableClassEmbedding(d_model, dropout)
        self.positional_encoding = PositionalEncoding(d_model, dropout)

    def forward(self, input_ids):
        input_embeds = self.formula_embedding(input_ids)
        input_embeds = self.positional_encoding(input_embeds)
        return input_embeds


class BaseModel(nn.Module):
    def __init__(self,
                 d_proj=256,
                 vocab_size=181,
                 spectral_channel=1,
                 d_model=512,
                 nhead=12,
                 d_ff=3072,
                 nlayer=6,
                 mask_prob=0.45,
                 in_channel=1,
                 gpu_align=False,
                 masking_strategy='random'):
        super().__init__()
        self.mask_prob = mask_prob
        self.gpu_align = gpu_align
        self.masking_strategy = masking_strategy
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
            spectral_input = input['spectra']
        elif 'ir' in input and 'raman' in input:
            spectral_input = torch.cat([input['raman'], input['ir']], dim=1)
        elif 'exp_ir' in input and 'raman' in input:
            spectral_input = torch.cat([input['raman'], input['exp_ir']], dim=1)
        elif 'ir' in input and 'raman' not in input:
            spectral_input = input['ir']
        elif 'exp_ir' in input and 'raman' not in input:
            spectral_input = input['exp_ir']
        elif 'raman' in input and 'ir' not in input and 'exp_ir' not in input:
            spectral_input = input['raman']
        return spectral_input

    def load_spectra_masked(self, input, spectral_mask_prob=0.0, raman_only=False, ir_only=False):
        spectral_input = self.load_spectra(input)

        if self.training and spectral_mask_prob > 0:
            if torch.rand(1).item() < spectral_mask_prob:
                masked_channel = 0 if torch.rand(1).item() < 0.5 else 1
                spectral_input[:, masked_channel, :] = self.spectral_mask_value

        if not self.training and raman_only:
            masked_channel = 1
            spectral_input[:, masked_channel, :] = self.spectral_mask_value
        elif not self.training and ir_only:
            masked_channel = 0
            spectral_input[:, masked_channel, :] = self.spectral_mask_value

        return spectral_input

    def generate_mlmmask(self, input_ids, mask_prob=0.45):
        masked_ids = input_ids.clone()
        eligible = ~torch.isin(
            masked_ids, torch.tensor([0, 1, 2], device=masked_ids.device)
        )
        if self.masking_strategy == 'random':
            probability_matrix = torch.full(
                masked_ids.shape, mask_prob, device=masked_ids.device
            )
            masked_indices = torch.bernoulli(probability_matrix).bool()
            masked_indices &= eligible
        elif self.masking_strategy == 'span':
            masked_indices = torch.zeros_like(masked_ids, dtype=torch.bool)
            for row in range(masked_ids.size(0)):
                positions = eligible[row].nonzero(as_tuple=False).flatten()
                token_count = positions.numel()
                if token_count == 0 or mask_prob == 0:
                    continue
                span_length = min(
                    token_count,
                    max(1, int(token_count * mask_prob + 0.5)),
                )
                start = torch.randint(
                    token_count - span_length + 1, (1,),
                    device=masked_ids.device,
                ).item()
                masked_indices[row, positions[start:start + span_length]] = True
        else:
            raise ValueError(
                f'Unknown masking strategy: {self.masking_strategy!r}'
            )
        masked_ids[masked_indices] = 4 # tokenizer.mask_token_id = 4
        return masked_ids, masked_indices

    def compute_mlm_loss(self, pred, target):
        loss = F.cross_entropy(pred, target, ignore_index=1)
        return loss

    def compute_lm_loss(self, pred, target):
        loss = F.cross_entropy(pred.contiguous().view(-1, pred.size(-1)), target.contiguous().view(-1), ignore_index=1)
        return loss

    @staticmethod
    def _expand_kv_cache(cache, beam_size):
        return [
            {kind: {name: tensor.repeat_interleave(beam_size, dim=0)
                    for name, tensor in values.items()}
             for kind, values in layer.items()}
            for layer in cache
        ]

    @staticmethod
    def _reorder_kv_cache(cache, indices):
        return [
            {kind: {name: tensor.index_select(0, indices)
                    for name, tensor in values.items()}
             for kind, values in layer.items()}
            for layer in cache
        ]

    def _beam_decode_cached(
        self, memory, src_mask, batch_size, max_len, beam_size, temperature
    ):
        device = memory.device
        sequences = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
        embeds = self.molecular_encoding(sequences)[:, -1:]
        logits, cache = self.molecular_decoder.forward_step(
            embeds, memory, src_mask, cache=None
        )
        log_probs = F.log_softmax(logits[:, -1] / temperature, dim=-1)
        beam_scores, next_tokens = log_probs.topk(beam_size, dim=-1)
        sequences = sequences.unsqueeze(1).repeat(1, beam_size, 1)
        sequences = torch.cat([sequences, next_tokens.unsqueeze(-1)], dim=-1)

        memory = memory.repeat_interleave(beam_size, dim=0)
        src_mask = src_mask.repeat_interleave(beam_size, dim=0)
        cache = self._expand_kv_cache(cache, beam_size)
        finished = [[] for _ in range(batch_size)]

        for _ in range(max_len - 2):
            latest_tokens = sequences[:, :, -1].reshape(
                batch_size * beam_size, 1
            )
            embeds = self.molecular_encoding(
                latest_tokens, position_offset=sequences.size(-1) - 1
            )
            logits, next_cache = self.molecular_decoder.forward_step(
                embeds, memory, src_mask, cache
            )
            log_probs = F.log_softmax(logits[:, -1] / temperature, dim=-1)
            vocab_size = log_probs.size(-1)
            scores = beam_scores.unsqueeze(-1) + log_probs.view(
                batch_size, beam_size, vocab_size
            )
            beam_scores, indices = scores.view(batch_size, -1).topk(
                beam_size, dim=-1
            )
            parent_indices = indices // vocab_size
            next_tokens = indices % vocab_size
            sequences = sequences.gather(
                1, parent_indices.unsqueeze(-1).expand(-1, -1, sequences.size(-1))
            )
            sequences = torch.cat([sequences, next_tokens.unsqueeze(-1)], dim=-1)

            offsets = torch.arange(batch_size, device=device).unsqueeze(1) * beam_size
            flat_parents = (parent_indices + offsets).reshape(-1)
            cache = self._reorder_kv_cache(next_cache, flat_parents)
            memory = memory.index_select(0, flat_parents)
            src_mask = src_mask.index_select(0, flat_parents)

            for batch_index in range(batch_size):
                for beam_index in range(beam_size):
                    if sequences[batch_index, beam_index, -1].item() == 2:
                        finished[batch_index].append((
                            beam_scores[batch_index, beam_index].item(),
                            sequences[batch_index, beam_index].clone(),
                        ))
                        beam_scores[batch_index, beam_index] = -1e9
            if all(len(outputs) >= beam_size for outputs in finished):
                break

        for batch_index in range(batch_size):
            for beam_index in range(beam_size):
                if (sequences[batch_index, beam_index, -1].item() != 2 and
                        beam_scores[batch_index, beam_index].item() != -1e9):
                    finished[batch_index].append((
                        beam_scores[batch_index, beam_index].item(),
                        sequences[batch_index, beam_index].clone(),
                    ))
            finished[batch_index] = sorted(
                finished[batch_index], key=lambda item: item[0], reverse=True
            )[:beam_size]
        return {
            'pred_ids': [[item[1] for item in outputs] for outputs in finished],
            'score': [[item[0] for item in outputs] for outputs in finished],
        }

    def _greedy_decode_cached(
        self, memory, src_mask, batch_size, max_len,
        return_metrics=False, target_ids=None,
    ):
        device = memory.device
        pred_ids = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
        cache = None
        for position in range(max_len - 1):
            latest_token = pred_ids[:, -1:]
            embeds = self.molecular_encoding(
                latest_token, position_offset=position
            )
            logits, cache = self.molecular_decoder.forward_step(
                embeds, memory, src_mask, cache
            )
            next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
            pred_ids = torch.cat([pred_ids, next_token], dim=1)

        result = {'pred_ids': pred_ids}
        if return_metrics:
            valid = target_ids != 1
            result['metrics'] = (pred_ids[valid] == target_ids[valid]).sum() / valid.sum()
        return result

    def compute_cl_loss(self, molecular_output, spectral_output, return_sim=False):
        molecular_output = F.normalize(molecular_output, p=2, dim=1)
        spectral_output = F.normalize(spectral_output, p=2, dim=1)

        logit_scale = self.logit_scale.exp()
        local_logits_per_smiles = torch.matmul(
            molecular_output, spectral_output.t()) * logit_scale
        local_logits_per_spectrum = local_logits_per_smiles.T

        world_size = 1
        global_molecular_output = molecular_output
        global_spectral_output = spectral_output
        # DistributedSampler guarantees equal local training batch shapes.
        # Evaluation shards can be uneven, so validation metrics are reduced
        # by the trainer instead of gathering features here.
        if (self.gpu_align and self.training
                and dist.is_available() and dist.is_initialized()):
            world_size = dist.get_world_size()
            if world_size > 1:
                rank = dist.get_rank()
                global_molecular_output = self._gather_feature_with_local_grad(
                    molecular_output, rank, world_size
                )
                global_spectral_output = self._gather_feature_with_local_grad(
                    spectral_output, rank, world_size
                )

        global_similarity = torch.matmul(
            global_molecular_output, global_spectral_output.t()
        )
        global_logits = global_similarity * logit_scale
        # all_gather leaves remote features detached. Compensate for DDP's
        # gradient averaging, following Vib2Conf's GPU-aligned CL objective.
        loss = cl_loss(global_logits.T)
        if world_size > 1:
            loss = loss * world_size
            # The scalar temperature receives the complete global gradient on
            # every rank, unlike encoder features. Cancel its extra scaling.
            scale_only_logits = global_similarity.detach() * logit_scale
            loss = loss + (1 - world_size) * cl_loss(scale_only_logits.T)

        if return_sim:
            # Matching heads index local hidden states, so their hard-negative
            # sampler must continue to receive a local [B, B] matrix.
            return loss, local_logits_per_smiles, local_logits_per_spectrum
        else:
            return loss

    @staticmethod
    def _gather_feature_with_local_grad(feature, rank, world_size):
        gathered = [torch.zeros_like(feature) for _ in range(world_size)]
        dist.all_gather(gathered, feature.contiguous())
        gathered[rank] = feature
        return torch.cat(gathered, dim=0)

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

    def get_spectral_embeddings(self, input, **kwargs):
        spectral_input = self.load_spectra(input)
        spectra_embeds = self.spectral_encoding(spectral_input)
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
        molecular_output = self.molecular_encoder(molecular_embeds, mask=molecular_attention_mask)
        return molecular_output

    def infer_mlm(self, input):
        # Mask Language Infer
        spectral_input = self.load_spectra(input)
        spectra_embeds = self.spectral_encoding(spectral_input)
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
        spectral_input = self.load_spectra(input)
        spectra_embeds = self.spectral_encoding(spectral_input)
        spectral_output = self.spectral_encoder(spectra_embeds)
        pred_ids = torch.zeros(spectral_input.size(0), 1, dtype=torch.long, device=spectral_input.device)

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

        spectral_input = self.load_spectra(input)
        batch_size = spectral_input.size(0)
        spectra_embeds = self.spectral_encoding(spectral_input)
        spectral_output = self.spectral_encoder(spectra_embeds)

        # init start token
        start_token = torch.zeros(batch_size, 1, dtype=torch.long, device=spectral_input.device)
        pred_seqs = start_token.unsqueeze(1).expand(batch_size, beam_size, 1)  # (batch_size, beam_size, seq_len)
        beam_scores = torch.zeros(batch_size, beam_size, device=spectral_input.device)  # (batch_size, beam_size)

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
                       'score':[[output[0] for output in outputs] for outputs in final_outputs],}  # 提取序列

        return result_dict
