import os
import random
import logging
from copy import copy

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
from torch.utils.data import Sampler
from utils.lmdb_utils import read_lmdb_length


class DistributedEvalSampler(Sampler):
    """Shard evaluation data without padding or duplicating samples."""

    def __init__(self, dataset):
        import torch.distributed as dist
        self.dataset = dataset
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()

    def __iter__(self):
        return iter(range(self.rank, len(self.dataset), self.world_size))

    def __len__(self):
        return (len(self.dataset) - self.rank + self.world_size - 1) // self.world_size
from transformers import AutoTokenizer



class lmdbDataset(Dataset):
    def __init__(self, lmdb_path, target_keys, device):
        self.lmdb_path = lmdb_path
        self.target_keys = target_keys
        self.device = device

        assert os.path.isfile(
            self.lmdb_path), "{} not found".format(self.lmdb_path)

        env = self._connect_db()
        with env.begin() as txn:
            self.num_samples = read_lmdb_length(txn)
        env.close()
        self.env = None

    def __getstate__(self):
        """Exclude the unpickleable LMDB handle from spawned workers."""
        state = self.__dict__.copy()
        state['env'] = None
        return state

    def __del__(self):
        env = getattr(self, 'env', None)
        if env is not None:
            env.close()

    def _connect_db(self):
        env = lmdb.open(
            self.lmdb_path,
            subdir=False, readonly=True,
            lock=False, readahead=False,
            meminit=False, max_readers=256
        )
        return env

    def __len__(self):
        return self.num_samples

    @lru_cache(maxsize=64)
    def __getitem__(self, idx):
        if self.env is None:
            self.env = self._connect_db()
        key = str(idx).encode('ascii')
        with self.env.begin() as txn:
            pickled_data = txn.get(key)
        if pickled_data is None:
            raise IndexError(f'Index {idx} not found in {self.lmdb_path}')
        data = pickle.loads(pickled_data)
        
        output = {}
        for k in self.target_keys:
            if k in data:  
                if 'kekule_smiles' in k:
                    output['smiles' if k == 'kekule_smiles' else k] = data[k]
                elif 'norm_smiles' in k:
                    output['smiles' if k == 'norm_smiles' else k] = data[k]
                elif 'sequence' in k:
                    output['sequence'] = data[k]
                elif 'formula' in k:
                    output['formula'] = data[k]
                elif 'raman' in k or 'ir' in k: 
                    output[k] = torch.as_tensor(data[k])
                elif k == 'Yield':
                    output[k] = torch.as_tensor(data[k])
                else:
                    pass 
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
            data_sampler = (
                DistributedSampler(self.dataset, shuffle=True)
                if mode == 'train'
                else DistributedEvalSampler(self.dataset)
            )
            dataloader = DataLoader(
                self.dataset, batch_size=batch_size,
                collate_fn=copy(self.collate_fn), sampler=data_sampler,
                num_workers=num_workers, pin_memory=True,
                persistent_workers=(num_workers > 0),
            )
            
            if self.mode == 'train':
                return dataloader, data_sampler
            else: 
                return dataloader
        else:
            dataloader = DataLoader(
                self.dataset, batch_size=batch_size,
                collate_fn=copy(self.collate_fn), num_workers=num_workers,
                shuffle=shuffle, pin_memory=True,
                persistent_workers=(num_workers > 0),
            )
            return dataloader
