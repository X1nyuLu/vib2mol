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
        eval_losses_cl = AverageMeter()
        eval_losses_match = AverageMeter()
        eval_acc = AverageMeter()

        self.model.eval()
        
        all_smiles_embeddings = []
        all_spectra_embeddings = []

        bar = tqdm(self.eval_loader) if self.device_rank == 0 else self.eval_loader

        for batch in bar:
            data = batch['data']
            data = self._put_on_device(data)

            output = self.model(data, return_proj_output=True)
            eval_losses.update(output['loss'].item(), batch['batch_size'])
            eval_losses_cl.update(output['cl_loss'].item(), batch['batch_size'])
            eval_losses_match.update(output['matching_loss'].item(), batch['batch_size'])

            all_smiles_embeddings.append(output['molecular_proj_output'].detach().cpu())
            all_spectra_embeddings.append(output['spectral_proj_output'].detach().cpu())

            matching_accuracy = output['matching_accuracy'].cpu().numpy()
            eval_acc.update(matching_accuracy, batch['batch_size'])
            
            if self.device_rank == 0:
                bar.set_description(
                    f'Epoch{epoch:4d}, valid loss:{eval_losses.avg:6f}, valid acc:{eval_acc.avg:6f}')
                
        all_smiles_embeddings = torch.cat(all_smiles_embeddings, dim=0)
        all_spectra_embeddings = torch.cat(all_spectra_embeddings, dim=0)
        simi_matrix = torch.mm(all_smiles_embeddings, all_spectra_embeddings.T)
        smiles_to_spectrum_recall = compute_recall(simi_matrix, k=1)
        spectrum_to_smiles_recall = compute_recall(simi_matrix.T, k=1)
        if self.device_rank == 0:    
            logging.info(
                f'Epoch{epoch:4d}, eval loss:{eval_losses.avg:6f}, smiles_to_spectrum_recall:{smiles_to_spectrum_recall:6f}, spectrum_to_smiles_recall:{spectrum_to_smiles_recall:6f}, valid acc:{eval_acc.avg:6f}')
    
        return {'loss':eval_losses.avg, 'cl_loss':eval_losses_cl.avg, 'matching_loss':eval_losses_match.avg, 'recall':spectrum_to_smiles_recall, 'acc':eval_acc.avg}
    
    

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
                self.writer.add_scalar('eval_cl_loss', eval_output['cl_loss'], epoch)
                self.writer.add_scalar('eval_matching_loss', eval_output['matching_loss'], epoch)
                self.writer.add_scalar('eval_recall', eval_output['recall'], epoch)
                self.writer.add_scalar('eval_match_accuracy', eval_output['acc'], epoch)

                if 'recall' in eval_output:
                    self.es(eval_output['recall'], self.model,
                    f"{self.model_save_path}/epoch{epoch}_recall{eval_output['recall']*100:.0f}.pth")
                else:
                    assert False, 'No eval metrics'
            if self.es.early_stop:
                break
        print(self.es.val_score)
        if self.rank == 0:
            torch.save(self.model.state_dict(), f'{self.model_save_path}/epoch{epoch}.pth')

        if self.rank == 0:
            self.writer.close()
            
        
@register_function('matching')
def train_matching_model(*args, **kwargs):
    return train_model(Trainer, Engine, *args, **kwargs)
