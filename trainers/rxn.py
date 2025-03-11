import logging
from tqdm import tqdm

import torch
from torch.utils.tensorboard import SummaryWriter

from trainers import register_function
from trainers.base import BaseTrainer, train_model
from utils.base import BaseEngine, AverageMeter, compute_recall

from utils.collators import BaseCollator
from utils.dataloader import Dataloader


'''===================== RXN collator ================='''
class RXNCollator(BaseCollator):
    def __init__(self, tokenizer_path, spectral_types=None, smiles_types=None, mix_ratio=0.8):
        super().__init__(tokenizer_path)
        self.spectral_types = spectral_types
        self.smiles_types = smiles_types
        self.mix_ratio = mix_ratio
        
    def __call__(self, batch):
        batch_data = {}
        cache_data = {}
        for spectral_type in self.spectral_types:
            # spectral_types = ['reactant1_raman', 'reactant2_raman', 'product_raman']   
            spectra = self.process_spectra(batch, spectral_types=spectral_type)
            cache_data[spectral_type] = spectra
        
        smiles = [item['product_kekule_smiles'] for item in batch]
        smiles = self.tokenizer(smiles, padding=True, return_tensors='pt', truncation=True)
        batch_data['smiles'] = smiles
        
        batch_size = len(cache_data['reactant1_raman'])
        
        if self.mix_ratio != False: # generate three random numbers
            rxn_seed = int(torch.sum(cache_data['reactant1_raman']))
            rng = torch.Generator()
            rng.manual_seed(rxn_seed)
            random_weights = torch.rand(batch_size, generator=rng)
            random_weights = self.mix_ratio + (1 - self.mix_ratio) * random_weights
            batch_data['raman'] = (0.5 - 0.5 * random_weights).reshape(-1, 1, 1) * (cache_data['reactant1_raman'] + cache_data['reactant2_raman']) + random_weights.reshape(-1, 1, 1) * cache_data['product_raman']
        
        else:
            yields = torch.as_tensor([item['Yield'] for item in batch]).to(spectra.dtype)
            batch_data['raman'] = (0.5 - 0.5 * yields).reshape(-1, 1, 1) * (cache_data['reactant1_raman'] + cache_data['reactant2_raman']) + yields.reshape(-1, 1, 1) * cache_data['product_raman']
        return {'batch_size':batch_size, 'target':None, 'data': batch_data}


'''===================== RXN engine====================='''

class Engine(BaseEngine):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.phase = kwargs['phase']
        
    @torch.no_grad()
    def eval_epoch(self, epoch):

        eval_losses = AverageMeter()
        eval_losses_clip = AverageMeter()
        eval_losses_mlm = AverageMeter()
        eval_losses_lm = AverageMeter()
        eval_acc = AverageMeter()

        self.model.eval()
        
        all_smiles_embeddings = []
        all_spectra_embeddings = []

        bar = tqdm(self.eval_loader) if self.device_rank == 0 else self.eval_loader

        for batch in bar:
            data = batch['data']
            data = self._put_on_device(data)

            if self.phase == 1:
                output = self.model(data, return_proj_output=True)
                eval_losses_clip.update(output['clip_loss'].item(), batch['batch_size'])
                eval_losses.update(output['loss'].item(), batch['batch_size'])
                all_smiles_embeddings.append(output['molecular_proj_output'].detach().cpu())
                all_spectra_embeddings.append(output['spectral_proj_output'].detach().cpu())

                if self.device_rank == 0:
                    bar.set_description(
                        f'Epoch{epoch:4d}, valid loss:{eval_losses.avg:6f}')
                    
            elif self.phase == 2:
                output = self.model(data, return_proj_output=False)
                eval_losses_mlm.update(output['mlm_loss'].item(), batch['batch_size'])
                eval_losses_lm.update(output['lm_loss'].item(), batch['batch_size'])
                eval_losses.update(output['loss'].item(), batch['batch_size'])
                
                if epoch % 5 == 0:
                    if self.ddp:
                        output = self.model.module.infer_lm(data, max_len=data['smiles']['input_ids'].size(-1),
                                            return_metrics=True, target_ids=data['smiles']['input_ids'])
                    else:
                        output = self.model.infer_lm(data, max_len=data['smiles']['input_ids'].size(-1),
                                            return_metrics=True, target_ids=data['smiles']['input_ids'])
                    acc = output['metrics'].cpu().numpy()
                    eval_acc.update(acc, batch['batch_size'])
            
                if self.device_rank == 0:
                    bar.set_description(
                    f'Epoch{epoch:4d}, valid loss:{eval_losses.avg:6f}, valid acc:{eval_acc.avg:6f}')

        if self.phase == 1:
            all_smiles_embeddings = torch.cat(all_smiles_embeddings, dim=0)
            all_spectra_embeddings = torch.cat(all_spectra_embeddings, dim=0)
            simi_matrix = torch.mm(
                all_smiles_embeddings, all_spectra_embeddings.T)
            smiles_to_spectrum_recall = compute_recall(
                simi_matrix, k=1)
            spectrum_to_smiles_recall = compute_recall(
                simi_matrix.T, k=1)
            if self.device_rank == 0:    
                logging.info(
                    f'Epoch{epoch:4d}, eval loss:{eval_losses.avg:6f}, smiles_to_spectrum_recall:{smiles_to_spectrum_recall:6f}, spectrum_to_smiles_recall:{spectrum_to_smiles_recall:6f}')  
            return {'loss':eval_losses.avg, 'clip_loss':eval_losses_clip.avg, 'metrics':spectrum_to_smiles_recall}

        elif self.phase == 2:
            if self.device_rank == 0:    
                logging.info(
                    f'Epoch{epoch:4d}, valid loss:{eval_losses.avg:6f}, valid acc:{eval_acc.avg:6f}')
            return {'loss':eval_losses.avg, 'mlm_loss':eval_losses_mlm.avg, 'lm_loss':eval_losses_lm.avg, 'metrics':eval_acc.avg}


class Trainer(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mix_ratio = kwargs['mix_ratio']
        self.phase = kwargs['phase']
        
    def init_dataset(self, Collator=None):
        if Collator is not None:
            self.collator = Collator
        if self.rank == 0:
            self.writer = SummaryWriter(self.model_save_path.replace('checkpoints', 'runs'))

        split_delimiter = '-' if '-' in self.task else '_'
        target_keys = ['reactant1_kekule_smiles','reactant1_raman',
                   'reactant2_kekule_smiles','reactant2_raman',
                   'product_kekule_smiles','product_raman','Yield']
        spectral_types = ['reactant1_raman', 'reactant2_raman', 'product_raman']
        smiles_types = ['product_kekule_smiles']
    
        if len(spectral_types) == 0:
            spectral_types = ['spectra']
        dataloader = Dataloader(lmdb_path=self.lmdb_path, 
                                data_dir=self.data_dir, 
                                target_keys=target_keys, 
                                collate_fn=self.collator(spectral_types=spectral_types, tokenizer_path=self.tokenizer_path, mix_ratio=self.mix_ratio), 
                                device=self.device)
        
        if self.ddp:
            self.train_loader, self.train_sampler = dataloader.generate_dataloader(mode='train',
                                                                                   batch_size=self.config['batch_size'],
                                                                                   num_workers=12, ddp=ddp)
        else:
            self.train_loader = dataloader.generate_dataloader(mode='train',
                                                               batch_size=self.config['batch_size'], 
                                                               num_workers=12)
            
        self.eval_loader = dataloader.generate_dataloader(mode='eval', batch_size=64)
        
    
    def train(self):
        for epoch in range(self.config['epoch']):
            if self.ddp:
                self.train_sampler.set_epoch(epoch)
            train_loss = self.engine.train_epoch(epoch)
            eval_output = self.engine.eval_epoch(epoch)

            if self.engine.device_rank == 0:
                self.writer.add_scalar('train_loss', train_loss, epoch)
                self.writer.add_scalar('eval_loss', eval_output['loss'], epoch)  
                
                if self.phase == 1:
                    self.writer.add_scalar('eval_recall', eval_output['metrics'], epoch)
                    self.writer.add_scalar('eval_clip_loss', eval_output['clip_loss'], epoch)
                    save_path = f"{self.model_save_path}/epoch{epoch}_recall{eval_output['metrics']*100:.0f}.pth"
                
                elif self.phase == 2:
                    self.writer.add_scalar('eval_accuracy', eval_output['metrics'], epoch)            
                    self.writer.add_scalar('eval_lm_loss', eval_output['lm_loss'], epoch)
                    self.writer.add_scalar('eval_mlm_loss', eval_output['mlm_loss'], epoch)
                    save_path = f"{self.model_save_path}/epoch{epoch}_acc{eval_output['metrics']*100:.0f}.pth"
                          
                if 'metrics' in eval_output:
                    self.es(eval_output['metrics'], self. model,save_path)
                else:
                    assert False, 'No eval metrics'
            if self.es.early_stop:
                break
        print(self.es.val_score)
        torch.save(self.model.state_dict(), f'{self.model_save_path}/epoch{epoch}.pth')

        if self.rank == 0:
            self.writer.close()
            
        
@register_function('rxn')
def train_rxn_model(*args, **kwargs):
    return train_model(Trainer, Engine, RXNCollator, *args, **kwargs)