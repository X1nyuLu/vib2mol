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
import tempfile
import yaml

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from utils.base import PHASE_ALIGN, PHASE_GENERATE, seed_everything, load_state

import models
import trainers
import warnings


warnings.simplefilter("ignore", FutureWarning)
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def get_args_parser():
    parser = argparse.ArgumentParser('vib2mol', add_help=True)

    # basic params
    parser.add_argument('--model', default='vib2mol',
                        help="Choose network")
    parser.add_argument('--launch', default='matching',
                        help="Choose losses for training")
    parser.add_argument('--ds', default='mols',
                        help="Choose dataset")
    parser.add_argument('--task', default='raman-kekule_smiles',
                        help='Chose the task of this dataset')

    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--train', '-train', action='store_true',
                            help="start training")
    mode_group.add_argument('--debug', '-debug', action='store_true',
                            help="run one debug epoch without persistent artifacts")
    mode_group.add_argument('--test', '-test', action='store_true',
                            help="start test")
    
    parser.add_argument('--device', default='cuda:0',
                        help="Choose GPU device")
    parser.add_argument('--base-model-path', '--base_model_path', dest='base_model_path',
                        # default='',
                        help="Choose base model for fine-tune")
    parser.add_argument('--test-model-path', '--test_model_path', dest='test_model_path',
                        help="Choose timestamp for test")
    parser.add_argument('--seed', default=624, type=int,
                        help="Random seed")
    parser.add_argument('-ddp', '--ddp', action='store_true',
                        default=False,
                        help="Use DistributedDataParallel")
    
    # params of strategy
    parser.add_argument('--batch-size', '--batch_size', dest='batch_size', type=int,
                        help="global batch size across all DDP processes")
    parser.add_argument('--num-workers', '--num_workers', dest='num_workers', type=int,
                        help="DataLoader workers per process")
    parser.add_argument('--epoch', type=int,
                        help="epochs for training")
    parser.add_argument('--lr', type=float,
                        help="learning rate")
    parser.add_argument('--mask-prob', '--mask_prob', dest='mask_prob',
                        default=0.45, type=float,
                        help="mask probability")
    parser.add_argument(
        '--gpu-align', '--gpu_align', dest='gpu_align', action='store_true',
        help='use cross-GPU negatives for contrastive learning',
    )
    parser.add_argument('--smiles-augment', '--smiles_augment', dest='smiles_augment', action='store_true',
                        default=False,
                        help="augment smiles or not")
    parser.add_argument('--spectra-augment', '--spectra_augment', dest='spectra_augment', action='store_true',
                        default=False,
                        help="augment spectra or not")                        
    parser.add_argument('--frozen-encoder', '--frozen_encoder', dest='frozen_encoder', action='store_true',
                        default=False,
                        help="frozen encoders or not")
    parser.add_argument('--use-yield', '--use_yield', dest='use_yield', action='store_true',
                        default=False,
                        help="introducing yield of product")
    parser.add_argument('--use-residue', '--use_residue', dest='use_residue', action='store_true',
                        default=False,
                        help="introducing residue for tokenization")
    
    args = parser.parse_args()
    return args


def init_logs(local_rank):
    if args.debug:
        return

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
    phase = PHASE_GENERATE if 'spt' in args.launch else PHASE_ALIGN
    args.launch = 'rxn' if 'rxn' in args.launch else args.launch
    
    if args.train and not args.debug:
        if local_rank == 0:
            os.makedirs(f"checkpoints/{args.ds}/{args.task}/{args.model}/{ts}-{random_id}", exist_ok=True)

    with open('config.yaml', "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        
    defaults = config.pop('defaults')
    task_config = config[args.launch]
    params = defaults.copy()
    params.update(task_config)

    if args.batch_size:
        params['batch_size'] = args.batch_size
    if args.num_workers is not None:
        params['num_workers'] = args.num_workers
    if args.epoch:
        params['epoch'] = args.epoch
    if args.lr:
        params["lr"] = args.lr
    if args.debug:
        params['epoch'] = 1
    
    if 'ir' in args.task and 'raman' in args.task:
        spectral_channel = 2
    else:
        spectral_channel = 1
    
    model = models.build_model(
        args.model, spectral_channel=spectral_channel,
        mask_prob=args.mask_prob, phase=phase, gpu_align=args.gpu_align,
    )
        
    if 'cuda' in args.device and not args.ddp:
        model = model.to(device)
        
    base_model_path = args.base_model_path
    if base_model_path:
        ckpt = torch.load(base_model_path, map_location='cpu', weights_only=True)
        ckpt = {k.replace('module.', ''): v for k, v in ckpt.items()}
        model.load_state_dict(ckpt, strict=True)
    
    if phase == PHASE_GENERATE and args.frozen_encoder:
        # frozen_modules = [model.spectral_encoding, model.molecular_encoding, model.spectral_encoder, model.molecular_encoder]
        frozen_modules = [model.molecular_encoding, model.formula_encoding, model.molecular_encoder, model.molecular_decoder, model.multimodal_encoder]
        for module in frozen_modules:
            for name, param in module.named_parameters():
                if 'mask_token' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

    if args.ddp:   # set up distributed device
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        ddp_device = torch.device("cuda", local_rank)

        print(f"[init] == local rank: {local_rank}, global rank: {rank} ==")
        if torch.multiprocessing.get_start_method(allow_none=True) is None:
            torch.multiprocessing.set_start_method('spawn')
        model = model.to(ddp_device)
        model = DDP(
            model, device_ids=[local_rank], output_device=local_rank,
            find_unused_parameters=True, broadcast_buffers=False,
        )

    return model, params, phase

if __name__ == "__main__":

    args = get_args_parser()
    if args.test:
        raise SystemExit('Use scripts/infer_retrieval.py or scripts/infer_generation.py for evaluation.')
    device = init_device()
    local_rank = 0 if not args.ddp else int(os.environ["LOCAL_RANK"])
    global_rank = 0 if not args.ddp else int(os.environ["RANK"])

    seed_everything(args.seed)
    
    ts = time.strftime('%Y-%m-%d-%H-%M', time.localtime())
    random_id = uuid.uuid4().hex[:6]
    
    debug_workspace = None
    if args.debug:
        debug_workspace = tempfile.TemporaryDirectory(prefix='vib2mol-debug-')
        model_save_path = debug_workspace.name
        print('Debug mode: running 1 epoch without persistent logs or checkpoints.')
    else:
        model_save_path = f"checkpoints/{args.ds}/{args.task}/{args.model}/{ts}-{random_id}"
    
    try:
        model, params, phase = init_model(global_rank)
        init_logs(global_rank)
        logging.info({k: v for k, v in params.items()})

        tokenizer_path = './models/MolTokenizer' if 'sequence' not in args.task else './models/MolTokenizer'
        if args.train or args.debug:
            trainers.launch_training(args.launch, model=model, lmdb_path=args.ds, task=args.task, 
                            tokenizer_path=tokenizer_path, data_dir='./datasets/vibench',
                            model_save_path=model_save_path, device=device, ddp=args.ddp, rank=global_rank, config=params,
                            phase=phase, smiles_augment=args.smiles_augment, spectra_augment=args.spectra_augment, 
                            use_yield=args.use_yield, use_residue=args.use_residue)
        
    finally:
        if debug_workspace is not None:
            debug_workspace.cleanup()
        if args.ddp and dist.is_initialized():
            dist.destroy_process_group()
