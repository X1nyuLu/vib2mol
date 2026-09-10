import argparse
import pickle
import sys
from pathlib import Path

# Support direct execution from the repository root as well as python -m.
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import lmdb

from models import build_model
from utils.dataloader import Dataloader
from utils.collators import BaseCollator
from utils.base import seed_everything
from utils.lmdb_utils import iter_lmdb_records, read_lmdb_length

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

def get_args_parser() -> argparse.ArgumentParser:
    """
    Parses command-line arguments for the vib2mol inference script.

    Returns:
        argparse.ArgumentParser: Configured argument parser.
    """
    parser = argparse.ArgumentParser('vib2mol-inference', add_help=True)

    # Basic parameters
    parser.add_argument('--model', default='vib2mol',
                        help="Choose network architecture.")
    parser.add_argument('--ds', default='mols',
                        help="Choose dataset name (e.g., 'qm9s').")
    parser.add_argument('--spectral-types', '--spectral_types', dest='spectral_types', default='raman',
                        choices=('ir', 'raman', 'ir-raman', 'raman-ir'),
                        help="Hyphen-separated spectral types (e.g., 'raman', 'ir-raman').")
    parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu',
                        help="Choose GPU device (e.g., 'cuda:0', 'cpu').")
    parser.add_argument('--test-model-path', '--test_model_path', dest='test_model_path', type=str, required=True,
                        help="Path to the checkpoint for retrieval/inference.")
    parser.add_argument('--rerank', '-rerank', action='store_true',
                        help="Enable re-ranking of retrieval results.")
    parser.add_argument('--topk', type=int, default=None,
                        help="Number of top-k results to consider for re-ranking.")
    parser.add_argument('--batch-size', '--batch_size', dest='batch_size', type=int, default=128,
                        help="Inference batch size.")
    parser.add_argument('--use-formula', '--use_formula', dest='use_formula', action='store_true', default=False,
                        help="introduce formula or not.")
    parser.add_argument('--rank-model-path', '--rank_model_path', dest='rank_model_path', type=str, default=None,
                        help="Path to the checkpoint for the re-ranking model.")
    modality = parser.add_mutually_exclusive_group()
    modality.add_argument('--raman-only', '--raman_only', dest='raman_only', action='store_true')
    modality.add_argument('--ir-only', '--ir_only', dest='ir_only', action='store_true')
    args = parser.parse_args()
    return args


class TestDataset(torch.utils.data.Dataset):
    """Custom Dataset for re-ranking."""
    def __init__(self, dataframe: pd.DataFrame, spectral_types: list[str]):
        self.dataframe = dataframe
        self.spectral_types = spectral_types
        self.target_keys = spectral_types + ['kekule_smiles']

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, idx: int) -> dict:
        data = self.dataframe.iloc[idx]
        output = {}
        for k in self.target_keys:
            if k == 'kekule_smiles':
                output['smiles'] = str(data[k]) # Ensure smiles is string
            elif k in ('raman', 'ir'):
                # Ensure spectral data is a tensor
                output[k] = torch.as_tensor(data[k], dtype=torch.float32)
        return output

        
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

def compute_recall(topk_indices, num_queries, k_val):
    """
    Compute Recall@k for a given set of top-k indices.
    topk_indices: shape (num_queries, max_k)
    num_queries: total number of queries
    k_val: the K for which to compute recall (e.g., 1, 3, 5)
    """
    if k_val > topk_indices.shape[1]:
        print(f"Warning: k_val ({k_val}) is greater than the number of available ranked items ({topk_indices.shape[1]}). "
              "Adjusting k_val to available items.")
        k_val = topk_indices.shape[1]

    correct = 0
    for i in range(num_queries):
        # Check if the query's true index (i) is within the top k_val predictions for that query
        if i in topk_indices[i, :k_val]:
            correct += 1
    recall_at_k = correct / num_queries
    return recall_at_k


def evaluate_recall(topk_indices, num_queries, k_values=(1, 3, 5)):
    return {
        f'Recall@{k}': compute_recall(topk_indices, num_queries, k)
        for k in k_values if k <= topk_indices.shape[1]
    }


def print_metrics_table(stage_metrics):
    columns = ('Recall@1', 'Recall@3', 'Recall@5', 'Recall@10')
    rows = []
    for method, metrics in stage_metrics.items():
        row = {'Method': method}
        row.update({column: metrics.get(column, np.nan) for column in columns})
        rows.append(row)
    table = pd.DataFrame(rows, columns=('Method',) + columns)
    formatters = {
        column: (lambda value: '-' if pd.isna(value) else f'{value:.5f}')
        for column in columns
    }
    print('\nRetrieval Evaluation Summary')
    print(table.to_string(index=False, formatters=formatters, na_rep='-'))

def check_beam_mols_topk(pred_smiles_list: list[str], tgt_smiles: str, k: int) -> int:
    """
    Checks if the target molecule is present in the top-k predicted SMILES
    (based on InChIKey). Removes duplicates from predictions before checking.
    """
    pred_inchi_keys = set()
    for s in pred_smiles_list[:k]: 
        mol = Chem.MolFromSmiles(s)
        if mol is not None:
            try:
                pred_inchi_keys.add(Chem.MolToInchiKey(mol))
            except Exception:
                pass 

    tgt_mol = Chem.MolFromSmiles(tgt_smiles)
    if tgt_mol is None:
        return 0

    try:
        tgt_inchi_key = Chem.MolToInchiKey(tgt_mol)
        return 1 if tgt_inchi_key in pred_inchi_keys else 0
    except Exception:
        return 0

def reject_sample(pred_smiles, tgt_smiles):
    output_smiles = []
    target_formula = rdMolDescriptors.CalcMolFormula(Chem.MolFromSmiles(tgt_smiles))
    for item in pred_smiles:
        try:
            pred_formula = rdMolDescriptors.CalcMolFormula(Chem.MolFromSmiles(item))
            if pred_formula == target_formula:
                output_smiles.append(item)
            else:
                output_smiles.append('*')
        except (TypeError, ValueError):
            output_smiles.append('*')
    return output_smiles

def retrieval(model, ds='qm9s', spectral_types=None, device='cuda', batch_size=128, raman_only=False, ir_only=False, topk=5):
    spectral_types = spectral_types or ['raman']

    dataloader = Dataloader(lmdb_path = ds,
                            data_dir = 'datasets/vibench',
                            target_keys = spectral_types+['kekule_smiles'],
                            collate_fn = BaseCollator(spectral_types = spectral_types,
                                                    tokenizer_path='models/MolTokenizer'),
                            device=device)
    test_loader = dataloader.generate_dataloader(mode='test', batch_size=batch_size)

    model.eval()
    all_smiles_embeddings = []
    all_spectra_embeddings = []

    print(f"Performing retrieval on dataset: {ds} with spectral types: {spectral_types}")
    bar = tqdm(test_loader, desc="Retrieving embeddings")
    with torch.no_grad():
        for batch in bar:
            data = batch['data']
            data = put_on_device(data, device)

            spectra_output = model.get_spectral_embeddings(data, raman_only=raman_only, ir_only=ir_only)
            molecular_output = model.get_molecular_embeddings(data, use_cls_token=True)

            all_spectra_embeddings.append(spectra_output['proj_output'].detach().cpu())
            all_smiles_embeddings.append(molecular_output['proj_output'].detach().cpu())

        if not all_spectra_embeddings:
            raise ValueError('The test dataset is empty')
        all_spectra_embeddings = torch.cat(all_spectra_embeddings, dim=0)
        all_smiles_embeddings = torch.cat(all_smiles_embeddings, dim=0)

    similarity_matrix = calculate_similarity_matrix(all_spectra_embeddings, all_smiles_embeddings)


    num_queries = similarity_matrix.size(0)
    max_k_eval = min(max(5, topk), num_queries)
    _, initial_topk_indices = similarity_matrix.topk(max_k_eval, dim=1, largest=True, sorted=True)

    metrics = evaluate_recall(initial_topk_indices, num_queries)
    return similarity_matrix, initial_topk_indices, metrics


def rerank(model, similarity_matrix, topk=3, ds='qm9s', spectral_types=None, device='cuda', batch_size=128, use_formula=False, raman_only=False, ir_only=False):
    spectral_types = spectral_types or ['raman']
    if not 1 <= topk <= similarity_matrix.size(1):
        raise ValueError('topk must be between 1 and the candidate library size')

    # Get top-k indices from the initial similarity matrix
    _, idx = torch.topk(similarity_matrix, k=topk, dim=1)
    sample_idx = idx.flatten() 

    # Load test data from LMDB
    db_path = f'datasets/vibench/{ds}/{ds}_test.lmdb'
    db = lmdb.open(db_path, subdir=False, lock=False, readonly=True, map_size=int(1e11))
    with db.begin() as txn:
        test_data = [
            pickle.loads(item[1])
            for item in tqdm(
                iter_lmdb_records(txn), total=read_lmdb_length(txn),
                desc='Loading Test Records'
            )
        ]
    db.close()

    test_df = pd.DataFrame(test_data)
    if len(test_df) != similarity_matrix.size(0) or len(test_df) != similarity_matrix.size(1):
        raise ValueError('Similarity matrix and test dataset sizes do not match')
    test_df_repeated = test_df.loc[test_df.index.repeat(topk)].reset_index(drop=True)

    original_kekule_smiles = test_df['kekule_smiles'].tolist()

    if use_formula:
        retrieved_smiles = [[original_kekule_smiles[i.item()] for i in id] for id in idx]

        df_beam = pd.DataFrame({'tgt_smiles': original_kekule_smiles, 'pred_smiles_list': retrieved_smiles})
        selected_smiles_list = [
            reject_sample(pred_item, tgt_item)
            for pred_item, tgt_item in tqdm(
                zip(retrieved_smiles, original_kekule_smiles),
                total=len(original_kekule_smiles),
                desc='Formula Filtering',
            )
        ]
        df_beam['selected_smiles_list'] = selected_smiles_list

        formula_metrics = {}
        for k_val in [1, 3, 5, 10]:
            if k_val > topk:
                continue
            col_name = f'top_{k_val}'
            results = [
                check_beam_mols_topk(predictions, target, k=k_val)
                for predictions, target in tqdm(
                    zip(selected_smiles_list, original_kekule_smiles),
                    total=len(original_kekule_smiles),
                    desc=f'Evaluating Formula Recall@{k_val}',
                )
            ]
            df_beam[col_name] = results
            formula_metrics[f'Recall@{k_val}'] = float(np.mean(results))

        test_df_repeated['kekule_smiles'] = [subitem for item in selected_smiles_list for subitem in item]
        
    else:
        formula_metrics = None
        repeated_smiles = [original_kekule_smiles[idx_val.item()] for idx_val in sample_idx]
        test_df_repeated['kekule_smiles'] = repeated_smiles
            

    test_dataset = TestDataset(test_df_repeated, spectral_types=spectral_types)
    test_collator = BaseCollator(spectral_types=spectral_types, tokenizer_path='models/MolTokenizer')

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, collate_fn=test_collator, shuffle=False)
    test_bar = tqdm(test_loader, desc="Re-ranking Candidates")

    matching_scores_list = []
    model.eval()
    with torch.no_grad():
        for batch in test_bar:
            data = batch['data']
            data = put_on_device(data, device)
            tmp_scores = model.matching(data, raman_only=raman_only, ir_only=ir_only)
            matching_scores_list.append(tmp_scores.cpu()) # Move to CPU immediately

    matching_scores = torch.cat(matching_scores_list, dim=0)
    matching_scores = torch.softmax(matching_scores, dim=1) # Apply softmax to get probabilities
    # Take the probability of the positive class (assuming it's the second column)
    matching_scores_reshaped = matching_scores[:, 1].reshape(-1, topk)
    if use_formula:
        valid_candidates = torch.tensor([
            [smiles != '*' for smiles in row] for row in selected_smiles_list
        ], dtype=torch.bool)
        matching_scores_reshaped = matching_scores_reshaped.masked_fill(~valid_candidates, -torch.inf)

    num_queries = similarity_matrix.size(0) # Total number of original queries

    _, local_reranked_topk_indices = matching_scores_reshaped.topk(k=topk, dim=1, largest=True, sorted=True)
    reranked_topk_indices_global = sample_idx.reshape(num_queries, topk).gather(1, local_reranked_topk_indices)
    if use_formula:
        valid_ranked = valid_candidates.gather(1, local_reranked_topk_indices)
        reranked_topk_indices_global = reranked_topk_indices_global.masked_fill(~valid_ranked, -1)

    rerank_metrics = evaluate_recall(
        reranked_topk_indices_global, num_queries, k_values=(1, 3, 5, 10)
    )
    return reranked_topk_indices_global, rerank_metrics, formula_metrics


if __name__ == "__main__":

    seed_everything(624)
    args = get_args_parser()
    if args.batch_size < 1 or (args.topk is not None and args.topk < 1):
        raise SystemExit('--batch-size and --topk must be positive')
    if args.use_formula and not args.rerank:
        raise SystemExit('--use-formula requires --rerank in retrieval')

    spectral_types = args.spectral_types.split('-')
    spectral_channel = len(spectral_types)
    if (args.ir_only or args.raman_only) and args.model not in {'vib2mol_mmm', 'vib2mol_matching_shared_mask'}:
        raise SystemExit('--ir-only/--raman-only require --model vib2mol_mmm')
    if (args.ir_only or args.raman_only) and spectral_channel != 2:
        raise SystemExit('MMM channel masking requires --spectral-types ir-raman')
    if args.rerank and not args.rank_model_path:
        raise SystemExit('--rank-model-path is required with --rerank')
    model = build_model(args.model, spectral_channel=spectral_channel).to(args.device)
    ckpt = torch.load(args.test_model_path, map_location=args.device, weights_only=True)
    ckpt = {k.replace('module.', ''): v for k, v in ckpt.items()}
    model.load_state_dict(ckpt, strict=True)

    similarity_matrix, initial_topk_indices, initial_metrics = retrieval(
        model, ds=args.ds, spectral_types=spectral_types, device=args.device,
        batch_size=args.batch_size,
        topk=args.topk or 5,
        raman_only=args.raman_only, ir_only=args.ir_only,
    )
    stage_metrics = {'Initial Retrieval': initial_metrics}

    if args.rerank:
        # Check if args.topk is provided for reranking, otherwise use a default
        if args.topk is None:
            print("Warning: --topk not specified for re-ranking. Using default topk=3 for re-ranking.")
            rerank_topk = 3
        else:
            rerank_topk = args.topk

        # Ensure that the initial retrieval's topk is at least as large as the rerank_topk
        # If not, we might not have enough candidates for reranking
        if rerank_topk > initial_topk_indices.shape[1]:
            print(f"Error: --topk for re-ranking ({rerank_topk}) is larger than the initial retrieval's top-k candidates ({initial_topk_indices.shape[1]}). "
                  "Please ensure initial retrieval considers at least 'topk' candidates for re-ranking.")
            raise SystemExit(1)

        rank_model = build_model(args.model, spectral_channel=spectral_channel).to(args.device)
        ckpt = torch.load(args.rank_model_path, map_location=args.device, weights_only=True)
        ckpt = {k.replace('module.', ''): v for k, v in ckpt.items()}
        rank_model.load_state_dict(ckpt, strict=True)
        _, rerank_metrics, formula_metrics = rerank(
            rank_model, similarity_matrix, topk=rerank_topk, ds=args.ds,
            spectral_types=spectral_types, device=args.device,
            batch_size=args.batch_size,
            use_formula=args.use_formula, raman_only=args.raman_only,
            ir_only=args.ir_only,
        )
        if formula_metrics is not None:
            stage_metrics['Formula Filtering'] = formula_metrics
        stage_metrics['Re-ranking'] = rerank_metrics

    print_metrics_table(stage_metrics)
