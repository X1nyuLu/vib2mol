#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File        :main.py
@Description :
@InitTime    :2024/07/29 19:48:45
@Author      :XinyuLu
@EMail       :xinyulu@stu.xmu.edu.cn

'''



import uuid
import re
import os
import time
import json
import torch
import logging
import argparse
import yaml

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from utils.base import seed_everything, load_state

import models
import trainers
import warnings


warnings.simplefilter("ignore", FutureWarning)
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def get_args_parser():
    parser = argparse.ArgumentParser('vib2mol', add_help=False)

    # basic params
    parser.add_argument('--model', default='vib2mol',
                        help="Choose network")
    parser.add_argument('--launch', default='matching',
                        help="Choose losses for training")
    parser.add_argument('--ds', default='mols',
                        help="Choose dataset")
    parser.add_argument('--task', default='raman-kekule_smiles',
                        help='Chose the task of this dataset')

    parser.add_argument('--train', '-train', action='store_true',
                        help="start train")
    parser.add_argument('--test', '-test', action='store_true',
                        help="start test")
    parser.add_argument('--debug', '-debug', action='store_true',
                        default=1,
                        help="start debug")
    
    parser.add_argument('--device', default='cpu',
                        help="Choose GPU device")
    parser.add_argument('--base_model_path', 
                        # default='',
                        help="Choose base model for fine-tune")
    parser.add_argument('--test_model_path',
                        help="Choose timestamp for test")
    parser.add_argument('--seed', default=624,
                        help="Random seed")
    parser.add_argument('-ddp', '--ddp', action='store_true',
                        default=False,
                        help="Use DistributedDataParallel")
    
    # params of strategy
    parser.add_argument('--batch_size',
                        help="batch size for training")
    parser.add_argument('--epoch',
                        help="epochs for training")
    parser.add_argument('--lr',
                        help="learning rate")
    parser.add_argument('--mask_prob',
                        default=0.45,
                        help="mask probability")
    parser.add_argument('--smiles_augment', action='store_true',
                        default=False,
                        help="augment smiles or not")
    parser.add_argument('--spectra_augment', action='store_true',
                        default=False,
                        help="augment spectra or not")                        
    parser.add_argument('--frozen_encoder', action='store_true',
                        default=False,
                        help="frozen encoders or not")
    parser.add_argument('--use_yield', action='store_true',
                        default=False,
                        help="introducing yield of product")
    parser.add_argument('--use_residue', action='store_true',
                        default=False,
                        help="introducing residue for tokenization")
    
    args = parser.parse_args()
    return args


def init_logs(local_rank):

    os.makedirs(f'logs/{args.ds}/{args.task}/{args.model}', exist_ok=True)

    if local_rank == 0:
        logging.basicConfig(
            filename=f'logs/{args.ds}/{args.task}/{args.model}/{ts}-{random_id}.log',
            format='%(levelname)s:%(message)s',
            level=logging.INFO)

        logging.info({k: v for k, v in args.__dict__.items() if v})
        print(f'logging save path: ./logs/{args.ds}/{args.task}/{args.model}/{ts}-{random_id}.log')

def init_device():
    if args.ddp:   # set up distributed device
        local_rank = int(os.environ["LOCAL_RANK"])
        ddp_device = torch.device("cuda", local_rank)
        return ddp_device
    else:
        return args.device
    
        
def init_model(local_rank):
    phase = 2  if 'spt' in args.launch else 1
    args.launch = 'rxn' if 'rxn' in args.launch else args.launch
    
    if args.train:
        if local_rank == 0:
            os.makedirs(f"checkpoints/{args.ds}/{args.task}/{args.model}/{ts}-{random_id}", exist_ok=True)

    with open('config.yaml', "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        
    defaults = config.pop('defaults')
    task_config = config[args.launch]
    params = defaults.copy()
    params.update(task_config)

    if args.batch_size:
        params['batch_size'] = int(args.batch_size)
    if args.epoch:
        params['epoch'] = int(args.epoch)
    if args.lr:
        params["lr"] = float(args.lr)
    
    if 'ir' in args.task and 'raman' in args.task:
        spectral_channel = 2
    else:
        spectral_channel = 1
    
    model = models.build_model(args.model, spectral_channel=spectral_channel, mask_prob=float(args.mask_prob), phase=phase)
        
    if 'cuda' in args.device and not args.ddp:
        model = model.to(device)
        
    base_model_path = args.base_model_path
    if base_model_path:
        ckpt = torch.load(base_model_path, map_location='cpu', weights_only=True)
        ckpt = {k.replace('module.', ''): v for k, v in ckpt.items()}
        model.load_state_dict(ckpt, strict=False)
    
    if phase == 2 and args.frozen_encoder:
        frozen_modules = [model.spectral_encoding, model.molecular_encoding, model.spectral_encoder, model.molecular_encoder]
        for module in frozen_modules:
            for name, param in module.named_parameters():
                if 'mask_token' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

    if args.ddp:   # set up distributed device
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(rank % torch.cuda.device_count())
        dist.init_process_group(backend="nccl")
        ddp_device = torch.device("cuda", local_rank)

        print(f"[init] == local rank: {local_rank}, global rank: {rank} ==")
        if torch.multiprocessing.get_start_method(allow_none=True) is None:
            torch.multiprocessing.set_start_method('spawn')
        model = model.to(ddp_device)
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    return model, params, phase

def catch_exception():
    import traceback
    import shutil

    traceback.print_exc()
    
    if os.path.exists(f'logs/{args.ds}/{args.task}/{args.model}/{ts}-{random_id}.log'):
        os.remove(f'logs/{args.ds}/{args.task}/{args.model}/{ts}-{random_id}.log') 
        print('unexpected log has been deleted')
    if os.path.exists(f'runs/{args.ds}/{args.task}/{args.model}/{ts}-{random_id}'):
        shutil.rmtree(f'runs/{args.ds}/{args.task}/{args.model}/{ts}-{random_id}')
        print('unexpected tensorboard record has been deleted')


if __name__ == "__main__":

    args = get_args_parser()
    device = init_device()
    local_rank = 0 if not args.ddp else int(os.environ["LOCAL_RANK"])

    seed_everything(int(args.seed))
    
    ts = time.strftime('%Y-%m-%d-%H-%M', time.localtime())
    random_id = uuid.uuid4().hex[:6]
    
    model_save_path = f"checkpoints/{args.ds}/{args.task}/{args.model}/{ts}-{random_id}"
    
    try:
        model, params, phase = init_model(local_rank)
        init_logs(local_rank)
        logging.info({k: v for k, v in params.items()})

        tokenizer_path = './models/MolTokenizer' if 'sequence' not in args.task else './models/MolTokenizer'
        if args.train or args.debug:
            trainers.launch_training(args.launch, model=model, lmdb_path=args.ds, task=args.task, 
                            tokenizer_path=tokenizer_path, data_dir='./datasets/vibench',
                            model_save_path=model_save_path, device=device, ddp=args.ddp, rank=local_rank, config=params,
                            phase=phase, smiles_augment=args.smiles_augment, spectra_augment=args.spectra_augment, 
                            use_yield=args.use_yield, use_residue=args.use_residue)
        
        elif args.test:
            raise 'use notebook for evaluation'

    except Exception as e:
        print(e)
        catch_exception()
