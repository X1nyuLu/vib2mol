import logging

from tqdm import tqdm
import torch
from trainers import register_function
from trainers.base import BaseTrainer, train_model
from utils.base import BaseEngine, AverageMeter, compute_recall


class Engine(BaseEngine):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @torch.no_grad()
    def eval_epoch(self, epoch):

        eval_losses = AverageMeter()
        eval_losses_clip = AverageMeter()
        eval_losses_mlm = AverageMeter()
        self.model.eval()
        
        all_smiles_embeddings = []
        all_spectra_embeddings = []

        bar = tqdm(self.eval_loader) if self.device_rank == 0 else self.eval_loader

        for batch in bar:
            data = batch['data']
            data = self._put_on_device(data)

            output = self.model(data, return_proj_output=True)
            eval_losses.update(output['loss'].item(), batch['batch_size'])
            eval_losses_clip.update(output['clip_loss'].item(), batch['batch_size'])
            eval_losses_mlm.update(output['mlm_loss'].item(), batch['batch_size'])

            all_smiles_embeddings.append(output['molecular_proj_output'].detach().cpu())
            all_spectra_embeddings.append(output['spectral_proj_output'].detach().cpu())

            if self.device_rank == 0:
                bar.set_description(
                    f'Epoch{epoch:4d}, valid loss:{eval_losses.avg:6f}')

        all_smiles_embeddings = torch.cat(all_smiles_embeddings, dim=0)
        all_spectra_embeddings = torch.cat(all_spectra_embeddings, dim=0)
        simi_matrix = torch.mm(all_smiles_embeddings, all_spectra_embeddings.T)
        smiles_to_spectrum_recall = compute_recall(simi_matrix, k=1)
        spectrum_to_smiles_recall = compute_recall(simi_matrix.T, k=1)
        if self.device_rank == 0:    
            logging.info(
                f'Epoch{epoch:4d}, eval loss:{eval_losses.avg:6f}, smiles_to_spectrum_recall:{smiles_to_spectrum_recall:6f}, spectrum_to_smiles_recall:{spectrum_to_smiles_recall:6f}')
                        
        return {'loss':eval_losses.avg, 'clip_loss':eval_losses_clip.avg, 'mlm_loss':eval_losses_mlm.avg, 'metrics':spectrum_to_smiles_recall}
    
    

class Trainer(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def train(self):
        for epoch in range(self.config['epoch']):
            if self.ddp:
                self.train_sampler.set_epoch(epoch)
            train_loss = self.engine.train_epoch(epoch)
            eval_output = self.engine.eval_epoch(epoch)

            if self.engine.device_rank == 0:
                self.writer.add_scalar('train_loss', train_loss, epoch)
                self.writer.add_scalar('eval_loss', eval_output['loss'], epoch)
                self.writer.add_scalar('eval_recall', eval_output['metrics'], epoch)
                self.writer.add_scalar('eval_clip_loss', eval_output['clip_loss'], epoch)
                self.writer.add_scalar('eval_mlm_loss', eval_output['mlm_loss'], epoch)
                if 'metrics' in eval_output:
                    self.es(eval_output['metrics'], self.model,
                    f"{self.model_save_path}/epoch{epoch}_recall{eval_output['metrics']*100:.0f}.pth")
                else:
                    assert False, 'No eval metrics'
            if self.es.early_stop:
                break
        print(self.es.val_score)
        torch.save(self.model.state_dict(), f'{self.model_save_path}/epoch{epoch}.pth')

        if self.rank == 0:
            self.writer.close()
            
        
@register_function('cl_mlm')
def train_cl_mlm_model(*args, **kwargs):
    return train_model(Trainer, Engine, *args, **kwargs)