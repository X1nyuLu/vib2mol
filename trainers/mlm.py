import logging

from tqdm import tqdm
import torch

from sklearn import metrics
from trainers import register_function
from trainers.base import BaseTrainer, train_model
from utils.base import BaseEngine, AverageMeter, compute_recall


class Engine(BaseEngine):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @torch.no_grad()
    def eval_epoch(self, epoch):

        eval_losses = AverageMeter()
        eval_losses_mlm = AverageMeter()
        eval_acc = AverageMeter()
        self.model.eval()

        bar = tqdm(self.eval_loader) if self.device_rank == 0 else self.eval_loader

        for batch in bar:
            data = batch['data']
            data = self._put_on_device(data)

            output = self.model(data, return_proj_output=True)
            eval_losses.update(output['loss'].item(), batch['batch_size'])
            eval_losses_mlm.update(output['mlm_loss'].item(), batch['batch_size'])

            pred_tokens = torch.max(output['pred_logits'], dim=-1)[1]
            acc = metrics.accuracy_score(pred_tokens.cpu().numpy(), output['tgt_tokens'].cpu().numpy())
            eval_acc.update(acc, batch['batch_size'])
            if self.device_rank == 0:
                bar.set_description(
                    f'Epoch{epoch:4d}, valid loss:{eval_losses.avg:6f}, valid acc:{eval_acc.avg:6f}')

        if self.device_rank == 0:    
            logging.info(
                f'Epoch{epoch:4d}, eval loss:{eval_losses.avg:6f}')
                        
        return {'loss':eval_losses.avg, 'mlm_loss':eval_losses_mlm.avg, 'metrics':eval_acc.avg}


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
                self.writer.add_scalar('eval_accuracy', eval_output['metrics'], epoch)
                self.writer.add_scalar('eval_mlm_loss', eval_output['mlm_loss'], epoch)
                if 'metrics' in eval_output:
                    self.es(eval_output['metrics'], self.model,
                    f"{self.model_save_path}/epoch{epoch}_acc{eval_output['metrics']*100:.0f}.pth")
                else:
                    assert False, 'No eval metrics'
            if self.es.early_stop:
                break
        print(self.es.val_score)
        torch.save(self.model.state_dict(), f'{self.model_save_path}/epoch{epoch}.pth')

        if self.rank == 0:
            self.writer.close()
            
        
@register_function('mlm')
def train_mlm_model(*args, **kwargs):
    return train_model(Trainer, Engine, *args, **kwargs)