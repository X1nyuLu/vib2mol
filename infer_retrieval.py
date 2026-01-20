import argparse
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import lmdb

from models import build_model
from utils.dataloader import Dataloader
from utils.collators import BaseCollator
from utils.base import seed_everything

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
    parser = argparse.ArgumentParser('vib2mol-inference', add_help=False)

    # Basic parameters
    parser.add_argument('--model', default='vib2mol',
                        help="Choose network architecture.")
    parser.add_argument('--ds', default='mols',
                        help="Choose dataset name (e.g., 'qm9s').")
    parser.add_argument('--spectral_types', default='raman',
                        help="Comma-separated spectral types (e.g., 'raman', 'ir-raman').")
    parser.add_argument('--device', default="cuda:0" if torch.cuda.is_available() else "cpu",
                        help="Choose GPU device (e.g., 'cuda:0', 'cpu').")
    parser.add_argument('--test_model_path', type=str,
                        help="Path to the checkpoint for retrieval/inference.")
    parser.add_argument('--rerank', '-rerank', action='store_true',
                        help="Enable re-ranking of retrieval results.")
    parser.add_argument('--topk', type=int, default=None,
                        help="Number of top-k results to consider for re-ranking.")
    parser.add_argument('--use_formula', '-use_formula', action='store_true', default=False,
                        help="introduce formula or not.")
    parser.add_argument('--rank_model_path', type=str, default=None,
                        help="Path to the checkpoint for the re-ranking model.")
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
        except TypeError as e:
            continue
    return output_smiles

def retrieval(model, ds='qm9s', spectral_types=['raman'], device='cuda'):

    dataloader = Dataloader(lmdb_path = ds,
                            data_dir = 'datasets/vibench',
                            target_keys = spectral_types+['kekule_smiles'],
                            collate_fn = BaseCollator(spectral_types = spectral_types,
                                                    tokenizer_path='models/MolTokenizer'),
                            device=device)
    test_loader = dataloader.generate_dataloader(mode='test', batch_size=128)

    model.eval()
    all_smiles_embeddings = []
    all_spectra_embeddings = []

    print(f"Performing retrieval on dataset: {ds} with spectral types: {spectral_types}")
    bar = tqdm(test_loader, desc="Retrieving embeddings")
    with torch.no_grad():
        for batch in bar:
            data = batch['data']
            data = put_on_device(data, device)

            spectra_output = model.get_spectral_embeddings(data)
            molecular_output = model.get_molecular_embeddings(data, use_cls_token=True)

            all_spectra_embeddings.append(spectra_output['proj_output'].detach().cpu())
            all_smiles_embeddings.append(molecular_output['proj_output'].detach().cpu())

        all_spectra_embeddings = torch.cat(all_spectra_embeddings, dim=0)
        all_smiles_embeddings = torch.cat(all_smiles_embeddings, dim=0)

    similarity_matrix = calculate_similarity_matrix(all_spectra_embeddings, all_smiles_embeddings)


    num_queries = similarity_matrix.size(0)
    max_k_eval = 5 # Default for initial retrieval display, or args.topk if it's larger
    if args.rerank and args.topk is not None:
        max_k_eval = max(max_k_eval, args.topk)
    _, initial_topk_indices = similarity_matrix.topk(max_k_eval, dim=1, largest=True, sorted=True)

    top1_recall_initial = compute_recall(initial_topk_indices, num_queries, k_val=1)
    top3_recall_initial = compute_recall(initial_topk_indices, num_queries, k_val=3)
    top5_recall_initial = compute_recall(initial_topk_indices, num_queries, k_val=5)

    print(f"Retrieval Recall@1: {top1_recall_initial:.4f}")
    print(f"Retrieval Recall@3: {top3_recall_initial:.4f}")
    print(f"Retrieval Recall@5: {top5_recall_initial:.4f}")

    return similarity_matrix, initial_topk_indices


def rerank(model, similarity_matrix, topk=3, ds='qm9s', spectral_types=['raman'], device='cuda', use_formula=False):

    print(f"Performing re-ranking with top-{topk}")

    # Get top-k indices from the initial similarity matrix
    _, idx = torch.topk(similarity_matrix, k=topk, dim=1)
    sample_idx = idx.flatten() 

    # Load test data from LMDB
    db_path = f'datasets/vibench/{ds}/{ds}_test.lmdb'
    db = lmdb.open(db_path, subdir=False, lock=False, readonly=True, map_size=int(1e11))
    with db.begin() as txn:
        test_data = [pickle.loads(item[1]) for item in txn.cursor()]
    db.close()

    test_df = pd.DataFrame(test_data)
    test_df_repeated = test_df.loc[test_df.index.repeat(topk)].reset_index(drop=True)

    original_kekule_smiles = test_df['kekule_smiles'].tolist()

    if use_formula:
        retrieved_smiles = [[original_kekule_smiles[i.item()] for i in id] for id in idx]

        df_beam = pd.DataFrame({'tgt_smiles': original_kekule_smiles, 'pred_smiles_list': retrieved_smiles})
        selected_smiles_list = [reject_sample(pred_item, tgt_item) for pred_item, tgt_item in zip(retrieved_smiles, original_kekule_smiles)]
        df_beam['selected_smiles_list'] = selected_smiles_list

        metrics = {}
        for k_val in [1, 3, 5, 10]:
            if k_val > topk:
                continue
            col_name = f'top_{k_val}'
            df_beam[col_name] = df_beam.apply(
                lambda row: check_beam_mols_topk(row['selected_smiles_list'], row['tgt_smiles'], k=k_val),
                axis=1,
            )
            metrics[col_name] = df_beam[col_name].mean()

        print("(After formula filtering) Top-K Recall:")
        for k, acc in metrics.items():
            print(f"{k.replace('_', '-')}:\t{acc:.5f}")

        test_df_repeated['kekule_smiles'] = [subitem for item in selected_smiles_list for subitem in item]
        
    else:
        repeated_smiles = [original_kekule_smiles[idx_val.item()] for idx_val in sample_idx]
        test_df_repeated['kekule_smiles'] = repeated_smiles
            

    test_dataset = TestDataset(test_df_repeated, spectral_types=spectral_types)
    test_collator = BaseCollator(spectral_types=spectral_types, tokenizer_path='models/MolTokenizer')

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=topk * 32, collate_fn=test_collator, shuffle=False)
    test_bar = tqdm(test_loader, desc="Calculating matching scores")

    matching_scores_list = []
    model.eval()
    with torch.no_grad():
        for batch in test_bar:
            data = batch['data']
            data = put_on_device(data, device)
            tmp_scores = model.matching(data)
            matching_scores_list.append(tmp_scores.cpu()) # Move to CPU immediately

    matching_scores = torch.cat(matching_scores_list, dim=0)
    matching_scores = torch.softmax(matching_scores, dim=1) # Apply softmax to get probabilities
    # Take the probability of the positive class (assuming it's the second column)
    matching_scores_reshaped = matching_scores[:, 1].reshape(-1, topk)

    num_queries = similarity_matrix.size(0) # Total number of original queries

    _, local_reranked_topk_indices = matching_scores_reshaped.topk(k=topk, dim=1, largest=True, sorted=True)
    reranked_topk_indices_global = sample_idx.reshape(num_queries, topk).gather(1, local_reranked_topk_indices)

    # Now calculate Recall@1, Recall@3, Recall@5 for the re-ranked results
    reranked_top1_recall = compute_recall(reranked_topk_indices_global, num_queries, k_val=1)
    reranked_top3_recall = compute_recall(reranked_topk_indices_global, num_queries, k_val=3)
    reranked_top5_recall = compute_recall(reranked_topk_indices_global, num_queries, k_val=5)

    print(f"Re-ranked Recall@1: {reranked_top1_recall:.4f}")
    print(f"Re-ranked Recall@3: {reranked_top3_recall:.4f}")
    print(f"Re-ranked Recall@5: {reranked_top5_recall:.4f}")

    return reranked_topk_indices_global


if __name__ == "__main__":

    seed_everything(624)
    args = get_args_parser()

    device = args.device
    use_formula = args.use_formula

    spectral_types = args.spectral_types.split('-')
    spectral_channel = len(spectral_types)
    model = build_model(args.model, spectral_channel=spectral_channel).to(device)
    ckpt = torch.load(args.test_model_path, map_location=device, weights_only=True)
    ckpt = {k.replace('module.', ''): v for k, v in ckpt.items()}
    model.load_state_dict(ckpt, strict=False)

    similarity_matrix, initial_topk_indices = retrieval(model, ds=args.ds, spectral_types=spectral_types, device=device)

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
            exit() # Or handle this error appropriately

        rank_model = build_model('vib2mol', spectral_channel=spectral_channel).to(device)
        ckpt = torch.load(args.rank_model_path, map_location=device, weights_only=True)
        ckpt = {k.replace('module.', ''): v for k, v in ckpt.items()}
        rank_model.load_state_dict(ckpt, strict=False)
        rerank(rank_model, similarity_matrix, topk=rerank_topk, ds=args.ds, spectral_types=spectral_types, device=device, use_formula=use_formula)