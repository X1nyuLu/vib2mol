import pickle
import sys
from pathlib import Path

# Support direct execution from the repository root as well as python -m.
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from tqdm import tqdm, trange
import argparse

import torch
import lmdb

from models import build_model
from utils.dataloader import Dataloader
from utils.collators import BaseCollator
from utils.base import seed_everything
from utils.lmdb_utils import iter_lmdb_records

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader

class BaseDataset(torch.utils.data.Dataset):
    """Custom Dataset for re-ranking."""
    def __init__(self, smiles_list: list):
        self.smiles_list = smiles_list

    def __len__(self) -> int:
        return len(self.smiles_list)

    def __getitem__(self, idx: int) -> dict:
        data = self.smiles_list[idx]
        output = {'smiles': data}
        return output


class BaseCollator:
    def __init__(self, tokenizer_path=None, **kwargs):

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path) if tokenizer_path is not None else None

    def process_smiles(self, batch, smiles_type='smiles'):

        smiles = [item[smiles_type] for item in batch]
        smiles = self.tokenizer(smiles, padding=True, return_tensors='pt', truncation=True)
        return smiles

    def __call__(self, batch):
        batch_data = {}
        batch_data['smiles'] = self.process_smiles(batch)
        batch_size = len(batch_data['smiles']['input_ids'])
        return {'batch_size':batch_size, 'target':None, 'data': batch_data}

def put_on_device(data, device):
    if isinstance(data, dict):
        return {k: put_on_device(v, device) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        return type(data)(put_on_device(item, device) for item in data)
    elif hasattr(data, 'to'):
        return data.to(device)
    else:
        return data

def calculate_similarity_matrix(embedding_query, embedding_key):
    embedding_query = torch.nn.functional.normalize(embedding_query, p=2, dim=1)
    embedding_key = torch.nn.functional.normalize(embedding_key, p=2, dim=1)
    similarity_matrix = torch.matmul(embedding_query, embedding_key.t())
    return similarity_matrix

def reject_sample(pred_smiles, target_formula):
    output_smiles = []

    for item in pred_smiles:
        try:
            pred_formula = rdMolDescriptors.CalcMolFormula(Chem.MolFromSmiles(item))
            if pred_formula == target_formula:
                output_smiles.append(item)
            else:
                output_smiles.append('*')
        except TypeError as e:
            continue
    return output_smiles


def get_args():
    parser = argparse.ArgumentParser('vib2mol-pubchem-retrieval')
    parser.add_argument('--num-samples', type=int,
                        help="number of PubChem candidates to add; default uses all")
    parser.add_argument('--model', default='vib2mol_mmm')
    parser.add_argument('--dataset-path', default='datasets/vibench/sdbs/sdbs_test.lmdb')
    parser.add_argument('--tokenizer-path', default='models/MolTokenizer')
    parser.add_argument('--test-model-path', required=True)
    parser.add_argument('--pubchem-embeds', default='datasets/pubchem/unique_pubchem.pt')
    parser.add_argument('--pubchem-dataframe', default='datasets/pubchem/unique_pubchem_df.pickle')
    parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--retrieval-topk', type=int, default=20)
    parser.add_argument('--rerank-topk', type=int, default=10)
    parser.add_argument('--seed', type=int, default=624,
                        help="number of pubchem for additional retrieval")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    seed_everything(args.seed)
    spectral_types = ['raman', 'ir']
    device = args.device

    db_path = args.dataset_path
    tokenizer_path = args.tokenizer_path
    ckpt_path = args.test_model_path


    model = build_model(args.model, spectral_channel=2).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    ckpt = {k.replace('module.', ''): v for k, v in ckpt.items()}
    model.load_state_dict(ckpt, strict=True)


    db = lmdb.open(db_path, subdir=False, lock=False, readonly=True, map_size=int(1e11))
    with db.begin() as txn:
        test_data = [pickle.loads(item[1]) for item in iter_lmdb_records(txn)]

    test_df = pd.DataFrame(test_data)
    kekule_smiles = test_df['kekule_smiles'].tolist()

    molecular_dataset = BaseDataset(kekule_smiles)
    molecular_dataloader = DataLoader(molecular_dataset, batch_size=args.batch_size, collate_fn=BaseCollator(tokenizer_path=tokenizer_path), shuffle=False)


    model.eval()
    all_smiles_embeddings = []

    bar = tqdm(molecular_dataloader, desc="generate molecular embeddings library...")
    with torch.no_grad():
        for batch in bar:
            data = batch['data']
            data = put_on_device(data, device)

            molecular_output = model.get_molecular_embeddings(data, use_cls_token=True)
            all_smiles_embeddings.append(molecular_output['proj_output'].detach().cpu())
        all_smiles_embeddings = torch.cat(all_smiles_embeddings, dim=0)


    # load pre-calculated pubchem embeds
    pubchem_embeds = torch.load(args.pubchem_embeds, map_location='cpu', weights_only=True)
    pubchem_df = pd.read_pickle(args.pubchem_dataframe)
    if args.num_samples is not None:
        pubchem_embeds = pubchem_embeds[:args.num_samples]
        pubchem_df = pubchem_df.iloc[:args.num_samples]


    # merge all molecular embeddings
    all_molecular_embeds = torch.cat([all_smiles_embeddings, pubchem_embeds], dim=0)
    all_kekule_smiles = kekule_smiles + pubchem_df.kekule_smiles.to_list()

    target_inchis = []
    for s in tqdm(test_df['kekule_smiles']):
        mol = Chem.MolFromSmiles(s)
        target_inchis.append(Chem.MolToInchi(mol) if mol else '*')

    model.eval()
    all_test_embeds = []
    batch_size_spectra = args.batch_size

    with torch.no_grad():
        for i in trange(0, len(test_df), batch_size_spectra):
            batch_df = test_df.iloc[i : i + batch_size_spectra]

            raman = torch.as_tensor(batch_df['raman'].tolist()).unsqueeze(1).to(device, dtype=torch.float32)
            ir = torch.as_tensor(batch_df['ir'].tolist()).unsqueeze(1).to(device, dtype=torch.float32)

            output = model.get_spectral_embeddings({'raman': raman, 'ir': ir}, raman_only=False, ir_only=False)
            all_test_embeds.append(output['proj_output'])

    all_test_embeds = torch.cat(all_test_embeds, dim=0) # [N_test, D]


    # calculating similarity
    sim_matrix = calculate_similarity_matrix(all_test_embeds.cpu(), all_molecular_embeds) # [N_test, N_db]

    # rerank
    top_1, top_5 = 0, 0
    collator = BaseCollator(tokenizer_path=tokenizer_path)

    pbar = tqdm(range(len(test_df)), desc="Evaluating")
    for i in pbar:
        retrieved_ids = torch.argsort(sim_matrix[i], descending=True)[:args.retrieval_topk]
        retrieved_smiles_raw = [all_kekule_smiles[idx] for idx in retrieved_ids.cpu().numpy()]

        # filtering by formula
        target_formula = test_df.iloc[i]['formula']
        filtered_smiles = reject_sample(retrieved_smiles_raw, target_formula)
        filtered_smiles = [s for s in filtered_smiles if s != "*"]

        if not filtered_smiles:
            continue

        matching_scores_list = []
        sampled_raman = torch.as_tensor(test_df.iloc[i]['raman']).reshape(1, 1, -1).to(device, dtype=torch.float32)
        sampled_ir = torch.as_tensor(test_df.iloc[i]['ir']).reshape(1, 1, -1).to(device, dtype=torch.float32)

        with torch.no_grad():
            for j in range(0, len(filtered_smiles), args.batch_size):
                batch_smiles = filtered_smiles[j : j + args.batch_size]
                data = collator([{'smiles': s} for s in batch_smiles])['data']
                data['raman'] = sampled_raman.repeat(len(batch_smiles), 1, 1)
                data['ir'] = sampled_ir.repeat(len(batch_smiles), 1, 1)
                data = put_on_device(data, device)

                scores = model.matching(data, raman_only=False, ir_only=False)
                matching_scores_list.append(torch.softmax(scores, dim=1)[:, 1])

        matching_scores_reshaped = torch.cat(matching_scores_list)
        rerank_ids = torch.argsort(matching_scores_reshaped, descending=True)

        # check validation
        top_k_indices = rerank_ids[:args.rerank_topk].cpu().numpy()
        target_inchi = target_inchis[i]

        found_at = -1
        for rank, idx in enumerate(top_k_indices):
            smi = filtered_smiles[idx]
            mol = Chem.MolFromSmiles(smi)
            if mol and Chem.MolToInchi(mol) == target_inchi:
                found_at = rank
                break

        if found_at == 0: top_1 += 1
        if 0 <= found_at < 5: top_5 += 1

        pbar.set_postfix({
            'Top-1': f"{top_1/(i+1):.4f}",
            'Top-5': f"{top_5/(i+1):.4f}",
        })

    total = len(test_df)
    print('\nPubChem Retrieval Evaluation Summary')
    print(f'Top-1: {top_1 / total:.5f}')
    print(f'Top-5: {top_5 / total:.5f}')
