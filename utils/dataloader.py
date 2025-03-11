import os
import random
import logging

from functools import lru_cache
from tqdm import tqdm

import pickle
import lmdb

import numpy as np
from sklearn import metrics

import torch

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer


class lmdbDataset(Dataset):
    def __init__(self, lmdb_path, target_keys, device):
        self.lmdb_path = lmdb_path
        self.target_keys = target_keys
        self.device = device

        assert os.path.isfile(
            self.lmdb_path), "{} not found".format(self.lmdb_path)

        self.env = self._connect_db()
        with self.env.begin() as txn:
            self._keys = list(txn.cursor().iternext(values=False))

    def _connect_db(self):
        env = lmdb.open(
            self.lmdb_path,
            subdir=False, readonly=True,
            lock=False, readahead=False,
            meminit=False, max_readers=256
        )
        return env

    def __len__(self):
        return len(self._keys)

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        if not hasattr(self, "env"):
            self._connect_db(self.lmdb_path, save_to_self=True)
        key = self._keys[idx]
        pickled_data = self.env.begin().get(key)
        data = pickle.loads(pickled_data)
        output = {}
        for k in self.target_keys:
            if k == 'kekule_smiles':
                output['smiles'] = data[k]
            elif 'kekule_smiles' in k:
                output[k] = data[k]
            elif 'sequence' in k:
                output['sequence'] = data[k]
            elif k in ('spectra', 'ir', 'nmr', 'mass', 'raman', 'uv', 'exp_ir'):
                output[k] = torch.as_tensor(data[k])
            elif 'raman' in k:
                output[k] = torch.as_tensor(data[k])
            elif k == 'Yield':
                output[k] = torch.as_tensor(data[k])
        return output

    
class Dataloader:
    def __init__(self, 
                 lmdb_path, 
                 data_dir='',
                 target_keys=None, 
                 collate_fn=None,
                 device='cpu'):

        self.lmdb_path = lmdb_path
        self.target_keys = target_keys
        self.data_dir = data_dir
        self.collate_fn = collate_fn
        self.device = device
        
    def generate_dataset(self, verbose=False):

        if verbose: 
            print(f'[train set] = {self.lmdb_path} | [target keys] = {self.target_keys}')
        self.dataset = lmdbDataset(f'{self.data_dir}/{self.lmdb_path}/{self.lmdb_path}_{self.mode}.lmdb', 
                                   target_keys=self.target_keys, 
                                   device=self.device)
        
    def generate_dataloader(self,
                            mode='train',
                            batch_size=16, 
                            num_workers=0, 
                            ddp=False):
        
        self.mode = mode
        self.generate_dataset()
        shuffle = True if mode == 'train' else False
        
        if ddp:
            data_sampler = DistributedSampler(self.dataset, shuffle=shuffle)
            dataloader = DataLoader(self.dataset, batch_size=batch_size, collate_fn=self.collate_fn, sampler=data_sampler)
            
            if self.mode == 'train':
                return dataloader, data_sampler
            else: 
                return dataloader
        else:
            dataloader = DataLoader(self.dataset, batch_size=batch_size, collate_fn=self.collate_fn,
                                num_workers=num_workers, shuffle=shuffle)
            return dataloader
