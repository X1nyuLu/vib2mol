'''
basic functions
'''


import os
import random
import logging

from tqdm import tqdm
import numpy as np
from sklearn import metrics
import torch


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def load_state(net, state_dict, ddp=False, strict=False):
    # Remove 'module.' prefix if not using DDP
    if not ddp:
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    # Check keys and load weights
    net_keys = net.state_dict().keys()
    state_dict_keys = state_dict.keys()
    for key in net_keys:
        if key in state_dict_keys:
            if net.state_dict()[key].shape == state_dict[key].shape:
                net.state_dict()[key].copy_(state_dict[key])
            else:
                print(f'Shape mismatch for key: {key} (expected {net.state_dict()[key].shape}, got {state_dict[key].shape})')
                if strict:
                    raise RuntimeError(f'Shape mismatch for key: {key}')
        else:
            print(f'Key error: {key} does not exist in the checkpoint')
            if strict:
                raise RuntimeError(f'Key error: {key} not found in checkpoint')

    return net


def compute_recall(similarity_matrix, k, verbose=False):
    num_queries = similarity_matrix.size(0)
    _, topk_indices = similarity_matrix.topk(k, dim=1, largest=True, sorted=True)
    correct = 0
    for i in range(num_queries):
        if i in topk_indices[i]:
            correct += 1
    recall_at_k = correct / num_queries
    
    if verbose:
        print(f'recall@{k}:{recall_at_k:.5f}')
    else:
        return recall_at_k

class AverageMeter:
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum/self.count


class EarlyStop:
    def __init__(self, patience=10, mode='max', delta=0.0001):
        self.patientce = patience
        self.counter = 0
        self.mode = mode
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        if self.mode == 'min':
            self.val_score = np.inf
        else:
            self.val_score = -np.inf

    def __call__(self, epoch_score, model, model_path):
        if self.mode == 'min':
            score = -1. * epoch_score
        else:
            score = np.copy(epoch_score)
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(epoch_score, model, model_path)
        elif score < self.best_score+self.delta:
            self.counter += 1
            if self.counter >= self.patientce:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(epoch_score, model, model_path)
            self.counter = 0

    def save_checkpoint(self, epoch_score, model, model_path):
        torch.save(model.state_dict(), model_path)
        self.val_score = epoch_score


class BaseEngine:
    def __init__(self, train_loader=None, eval_loader=None, test_loader=None,
                 optimizer=None, scheduler=None,
                 model=None, device='cpu', device_rank=0, ddp=False, **kwargs):

        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.test_loader = test_loader

        self.optimizer = optimizer
        self.scheduler = scheduler

        self.model = model
        self.device = device
        self.device_rank = device_rank
        self.ddp = ddp

    def _put_on_device(self, data):
        if type(data) == dict:
            for k, v in data.items():
                if type(v) == torch.Tensor:
                    data[k] = v.to(self.device)
                elif 'smiles' in k:
                    data[k]['input_ids'] = data[k]['input_ids'].to(self.device)
                    data[k]['attention_mask'] = data[k]['attention_mask'].to(self.device)
        else:
            data = data.to(self.device)
        return data
    
    def train_epoch(self, epoch):
        train_losses = AverageMeter()
        self.model.train()

        bar = tqdm(self.train_loader) if self.device_rank == 0 else self.train_loader
        for batch in bar:
            self.optimizer.zero_grad()
            
            data = batch['data']
            data = self._put_on_device(data)
            
            output = self.model(data)          
            loss = output['loss']
            loss.backward()
            self.optimizer.step()

            train_losses.update(loss.item(), batch['batch_size'])
            if self.device_rank == 0:
                bar.set_description(
                    f'Epoch{epoch:4d}, train loss:{train_losses.avg:6f}')

        if self.scheduler:
            self.scheduler.step()

        if self.device_rank == 0:
            logging.info(f'Epoch{epoch:4d}, train loss:{train_losses.avg:6f}')
        return train_losses.avg

    
    @torch.no_grad()
    def eval_epoch(self, epoch):
        '''
        rewrite this method to evaluate each epoch
        '''        
        pass

    def infer(self):
        self.model.eval()
        
        all_smiles_embeddings = []
        all_spectra_embeddings = []

        bar = tqdm(self.test_loader) if self.device_rank == 0 else self.test_loader

        with torch.no_grad():
            for batch in bar:
                data = batch['data']
                
                data = self._put_on_device(data)
                spectra_output = self.model.get_spectral_embeddings(data)
                molecular_output = self.model.get_molecular_embeddings(data, use_cls_token=True)      
                
                all_spectra_embeddings.append(spectra_output['proj_output'].detach().cpu())
                all_smiles_embeddings.append(molecular_output['proj_output'].detach().cpu())

            all_spectra_embeddings = torch.cat(all_spectra_embeddings, dim=0)
            all_smiles_embeddings = torch.cat(all_smiles_embeddings, dim=0)
        return {'molecular_proj_output': all_smiles_embeddings, 'spectral_proj_output': all_spectra_embeddings}