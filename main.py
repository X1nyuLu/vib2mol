#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File        :main.py
@Description :
@InitTime    :2024/07/29 19:48:45
@Author      :XinyuLu
@EMail       :xinyulu@stu.xmu.edu.cn

'''



import re
import os
import time
import json
import torch
import logging
import argparse
import config

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from utils.base import seed_everything, load_state

from models import build_model
from models import PretrainModel_CL, PretrainModel_MLM, PretrainModel_LM
from models import PretrainModel_CL_MLM, PretrainModel_CL_LM, PretrainModel_ALL, PretrainModel_Phase

from trainers import launch_training

os.environ["TOKENIZERS_PARALLELISM"] = "false"

'''
CL,         PretrainModel_CL,       vib2mol_cl
MLM,        PretrainModel_MLM,      vib2mol_mlm
LM,         PretrainModel_LM,       vib2mol_lm
CL_MLM,     PretrainModel_CL_MLM,   vib2mol_cl_mlm
CL_LM,      PretrainModel_CL_LM,    vib2mol_cl_lm
CL_MLM_LM   PretrainModel_ALL       vib2mol_all
SPT,        PretrainMdeol_Phase,    vib2mol_phase
'''

def get_args_parser():
    parser = argparse.ArgumentParser('vib2mol', add_help=False)

    # basic params
    parser.add_argument('--model', default='vib2mol_phase',
                        help="Choose network")
    parser.add_argument('--launch', default='cl',
                        help="Choose losses for training")
    parser.add_argument('--ds', default='nist_ir',
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
    parser.add_argument('--mix_ratio', # only for rxn model
                        default=False,
                        help="Minimum percentage of products")
    parser.add_argument('--phase', # only for rxn model
                        default=1,
                        help="select phase for training")
    args = parser.parse_args()
    return args


def init_logs():

    os.makedirs(f'logs/{args.ds}/{args.task}/{args.model}', exist_ok=True)

    if args.train or args.debug:
        mode = "train"
    elif args.test:
        mode = "test"

    os.makedirs(f'logs/{args.ds}/{args.task}/{args.model}', exist_ok=True)
    logging.basicConfig(
        filename=f'logs/{args.ds}/{args.task}/{args.model}/{ts}_{mode}.log',
        format='%(levelname)s:%(message)s',
        level=logging.INFO)

    logging.info({k: v for k, v in args.__dict__.items() if v})
    return mode


def init_device():
    if args.ddp:   # set up distributed device
        local_rank = int(os.environ["LOCAL_RANK"])
        ddp_device = torch.device("cuda", local_rank)
        return ddp_device
    else:
        return args.device
    
        
def init_model():
    if args.launch == 'spt':
        phase = 2  
    elif args.launch == 'rxn':
        phase = float(args.phase)
    else:
        phase = 1
    
    if args.train:
        os.makedirs(f"checkpoints/{args.ds}/{args.task}/{args.model}/{ts}", exist_ok=True)

    params = {'net': config.NET,
              'strategy': config.STRATEGY['train'] if args.train or args.debug else config.STRATEGY['tune']}

    if args.batch_size:
        params['strategy']['batch_size'] = int(args.batch_size)
    if args.epoch:
        params['strategy']['epoch'] = int(args.epoch)
    if args.lr:
        params['strategy']['Adam_params']["lr"] = float(args.lr)
    
    if 'ir' in args.task and 'raman' in args.task:
        spectral_channel = 2
    else:
        spectral_channel = 1
    
    model = build_model(args.model, spectral_channel=spectral_channel, mask_prob=float(args.mask_prob), phase=phase)
        
    if 'cuda' in args.device and not args.ddp:
        model = model.to(device)
        
    base_model_path = args.base_model_path
    if base_model_path:
        ckpt = torch.load(base_model_path, map_location='cpu', weights_only=True)
        ckpt = {k.replace('module.', ''): v for k, v in ckpt.items()}
        model.load_state_dict(ckpt, strict=False)
    
    if phase == 2:
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
    
    if os.path.exists(f'logs/{args.ds}/{args.task}/{args.model}/{ts}_{mode}.log'):
        os.remove(f'logs/{args.ds}/{args.task}/{args.model}/{ts}_{mode}.log') 
        print('unexpected log has been deleted')
    if os.path.exists(f'runs/{args.ds}/{args.task}/{args.model}/{ts}'):
        shutil.rmtree(f'runs/{args.ds}/{args.task}/{args.model}/{ts}')
        print('unexpected tensorboard record has been deleted')


if __name__ == "__main__":

    args = get_args_parser()
    device = init_device()
    local_rank = 0 if not args.ddp else int(os.environ["LOCAL_RANK"])

    seed_everything(int(args.seed))
    ts = time.strftime('%Y-%m-%d_%H:%M', time.localtime())
    model_save_path = f"checkpoints/{args.ds}/{args.task}/{args.model}/{ts}"

    mode = init_logs()
    
    try:
        model, params, phase = init_model()
        mix_ratio = eval(args.mix_ratio) if type(args.mix_ratio) == str else args.mix_ratio
        tokenizer_path = './models/MolTokenizer' if 'sequence' not in args.task else './models/PepTokenizer'
        if args.train or args.debug:
            launch_training(args.launch, model=model, lmdb_path=args.ds, task=args.task, 
                            tokenizer_path=tokenizer_path, data_dir='./datasets/vibench',
                            model_save_path=model_save_path, device=device, ddp=args.ddp, rank=local_rank, config=params['strategy'],
                            mix_ratio=mix_ratio, phase=phase)
        
        elif args.test:
            raise 'use notebook for evaluation'

    except Exception as e:
        print(e)
        catch_exception()
