from tqdm import tqdm
import torch

from sklearn import metrics

from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR

from utils.base import AverageMeter, EarlyStop, BaseEngine
from utils.dataloader import lmdbDataset, Dataloader
from utils.collators import BaseCollator


class BaseTrainer:
    def __init__(self,
                 model, model_save_path=None, device='cpu', ddp=False, rank=-1, config=None,
                 lmdb_path=None, tokenizer_path=None, task=None, data_dir=None,
                 losses=None, **kwargs):
        
        self.model = model
        self.model_save_path = model_save_path
        self.device = device
        self.ddp = ddp
        self.rank = rank
        self.config = config
        self.lmdb_path = lmdb_path
        self.tokenizer_path = tokenizer_path
        self.task = task
        self.data_dir = data_dir
        self.collator = BaseCollator
        
    def init_dataset(self, Collator=None):
        if Collator is not None:
            self.collator = Collator
        if self.rank == 0:
            self.writer = SummaryWriter(self.model_save_path.replace('checkpoints', 'runs'))

        split_delimiter = '-' if '-' in self.task else '_'
        target_keys = self.task.split(split_delimiter)
        spectral_types = [item for item in target_keys if item in ('ms', 'nmr', 'ir', 'raman', 'exp_ir', 'exp_raman')]
        if len(spectral_types) == 0:
            spectral_types = ['spectra']
        dataloader = Dataloader(lmdb_path=self.lmdb_path, 
                                data_dir=self.data_dir, 
                                target_keys=target_keys, 
                                collate_fn=self.collator(spectral_types=spectral_types, tokenizer_path=self.tokenizer_path), 
                                device=self.device)
        
        if self.ddp:
            self.train_loader, self.train_sampler = dataloader.generate_dataloader(mode='train',
                                                                                   batch_size=self.config['batch_size'],
                                                                                   num_workers=0, ddp=self.ddp)
        else:
            self.train_loader = dataloader.generate_dataloader(mode='train',
                                                               batch_size=self.config['batch_size'], 
                                                               num_workers=0)
            
        self.eval_loader = dataloader.generate_dataloader(mode='eval', batch_size=64)
        
    def init_engine(self, Engine, **kwargs):
        optimizer = torch.optim.AdamW(self.model.parameters(), **self.config['Adam_params'])
        scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)

        self.es = EarlyStop(patience=self.config['patience'], mode='max')

        self.engine = Engine(train_loader=self.train_loader, eval_loader=self.eval_loader, optimizer=optimizer, 
                             scheduler=scheduler, model=self.model, device=self.device, device_rank=self.rank, ddp=self.ddp, **kwargs)
        
    def train(self):
        '''
        rewrite this method to train model
        '''
        pass


def train_model(Trainer, Engine, Collator=None,
                model=None, lmdb_path=None, tokenizer_path=None, task=None, data_dir=None,
                model_save_path=None, device='cpu', ddp=False, rank=-1, config=None, **kwargs):
    
    trainer = Trainer(
        model=model,
        model_save_path=model_save_path,
        device=device,
        ddp=ddp,
        rank=rank,
        config=config,
        lmdb_path=lmdb_path,
        tokenizer_path=tokenizer_path,
        task=task,
        data_dir=data_dir,
        **kwargs,
    )
    trainer.init_dataset(Collator=Collator)
    trainer.init_engine(Engine, **kwargs)
    trainer.train()