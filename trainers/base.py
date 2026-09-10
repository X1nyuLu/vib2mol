from tqdm import tqdm
import torch

from sklearn import metrics

from torch.utils.tensorboard import SummaryWriter
from timm.scheduler import create_scheduler_v2

from utils.base import AverageMeter, EarlyStop, BaseEngine
from utils.dataloader import lmdbDataset, Dataloader
from utils.collators import BaseCollator


class BaseTrainer:
    def __init__(self,
                 model, model_save_path=None, device='cpu', ddp=False, rank=-1, config=None,
                 lmdb_path=None, tokenizer_path=None, task=None, data_dir=None,
                 smiles_augment=False, spectra_augment=False, use_residue=False, **kwargs):
        
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
        self.smiles_augment = smiles_augment
        self.spectra_augment = spectra_augment
        self.use_residue = use_residue
        
    def init_dataset(self, Collator=None):
        if Collator is not None:
            self.collator = Collator
        if self.rank == 0:
            self.writer = SummaryWriter(self.model_save_path.replace('checkpoints', 'runs'))

        split_delimiter = '-' if '-' in self.task else '_'
        target_keys = self.task.split(split_delimiter)
        spectral_types = [item for item in target_keys if item in ('ms', 'nmr', 'ir', 'raman')]
        if len(spectral_types) == 0:
            spectral_types = ['spectra']
        dataloader = Dataloader(lmdb_path=self.lmdb_path, 
                                data_dir=self.data_dir, 
                                target_keys=target_keys, 
                                collate_fn=self.collator(spectral_types=spectral_types, tokenizer_path=self.tokenizer_path, use_residue=self.use_residue), 
                                device=self.device)
        
        dataloader.collate_fn.smiles_augment = self.smiles_augment
        dataloader.collate_fn.spectra_augment = self.spectra_augment
        global_batch_size = self.config['batch_size']
        num_workers = self.config.get('num_workers', 8)
        if self.ddp:
            import torch.distributed as dist
            world_size = dist.get_world_size()
            if global_batch_size < world_size:
                raise ValueError(
                    f'Global batch size ({global_batch_size}) must be at least '
                    f'the DDP world size ({world_size}).'
                )
            if global_batch_size % world_size:
                raise ValueError(
                    f'Global batch size ({global_batch_size}) must be divisible '
                    f'by the DDP world size ({world_size}).'
                )
            per_device_batch_size = global_batch_size // world_size
            if self.rank == 0:
                print(
                    f'DDP batch size: global={global_batch_size}, '
                    f'per-device={per_device_batch_size}, world-size={world_size}'
                )
            self.train_loader, self.train_sampler = dataloader.generate_dataloader(mode='train',
                                                                                   batch_size=per_device_batch_size,
                                                                                   num_workers=num_workers, ddp=self.ddp)
        else:
            self.train_loader = dataloader.generate_dataloader(mode='train',
                                                               batch_size=global_batch_size,
                                                               num_workers=num_workers)
        dataloader.collate_fn.smiles_augment = False
        dataloader.collate_fn.spectra_augment = False
        eval_batch_size = min(64, per_device_batch_size) if self.ddp else 64
        self.eval_loader = dataloader.generate_dataloader(
            mode='eval', batch_size=eval_batch_size, num_workers=num_workers,
            ddp=self.ddp,
        )
        
    def init_engine(self, Engine, **kwargs):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=float(self.config['lr']))
        scheduler, _ = create_scheduler_v2(optimizer, sched=self.config['lr_sched'], 
                                        num_epochs=self.config['epoch'], warmup_epochs=self.config['warmup_epochs'],
                                        min_lr=float(self.config['min_lr']), warmup_lr=float(self.config['warmup_lr']))

        self.es = EarlyStop(patience=self.config['patience'], mode='max')

        self.engine = Engine(train_loader=self.train_loader, eval_loader=self.eval_loader, optimizer=optimizer, 
                             scheduler=scheduler, model=self.model, device=self.device, device_rank=self.rank, ddp=self.ddp, **kwargs)

    def should_stop(self):
        """Broadcast rank-0 early-stopping decision to every DDP process."""
        if not self.ddp:
            return self.es.early_stop
        stop = torch.tensor(
            int(self.es.early_stop) if self.rank == 0 else 0,
            device=self.device,
        )
        torch.distributed.broadcast(stop, src=0)
        return bool(stop.item())
        
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
