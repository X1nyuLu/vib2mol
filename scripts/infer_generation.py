import argparse
import pickle
import sys
from pathlib import Path

# Support direct execution from the repository root as well as python -m.
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import lmdb

import torch
from transformers import AutoTokenizer
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from tqdm import tqdm

from models import build_model
from utils.base import seed_everything
from utils.lmdb_utils import iter_lmdb_records

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

PROGRESS_LABEL_WIDTH = 24
PROGRESS_BAR_WIDTH = 140


def progress_bar(iterable, step, label, **kwargs):
    """Create consistently aligned progress bars for the three decoding stages."""
    return tqdm(
        iterable,
        desc=f'[Step {step}/3] {label:<{PROGRESS_LABEL_WIDTH}}',
        ncols=PROGRESS_BAR_WIDTH,
        dynamic_ncols=False,
        **kwargs,
    )

def get_args_parser() -> argparse.ArgumentParser:
    """
    Parses command-line arguments for the vib2mol inference script.
    """
    parser = argparse.ArgumentParser('vib2mol-inference', add_help=True)

    # Basic parameters
    parser.add_argument('--model', default='vib2mol',
                        help="Choose network architecture.")
    parser.add_argument('--ds', default='qm9s',
                        help="Choose dataset name (e.g., 'qm9s').")
    parser.add_argument('--spectral-types', '--spectral_types', dest='spectral_types', default='raman',
                        choices=('ir', 'raman', 'ir-raman', 'raman-ir'),
                        help="Hyphen-separated spectral types (e.g., 'raman' or 'ir-raman').")
    parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu',
                        help="Choose GPU device (e.g., 'cuda:0', 'cpu').")
    parser.add_argument('--test-model-path', '--test_model_path', dest='test_model_path', type=str, required=True,
                        help="Path to the checkpoint for molecular generation inference.")
    parser.add_argument('--beam-size', '--beam_size', dest='beam_size', type=int, default=10,
                        help="Beam size for molecular generation.")
    parser.add_argument('--batch-size', '--batch_size', dest='batch_size', type=int, default=128,
                        help="Inference batch size.")
    parser.add_argument('--use-formula', '--use_formula', dest='use_formula', action='store_true', default=False,
                        help="Condition generation on the molecular formula.")
    parser.add_argument('--rerank', '-rerank', action='store_true', 
                        help="Re-rank generated beam candidates with the matching model.")
    parser.add_argument('--topk', type=int, default=5, 
                        help="Number of top-k results to consider for re-ranking.")
    parser.add_argument('--rank-model-path', '--rank_model_path', dest='rank_model_path', type=str,
                        help="Path to the checkpoint for the re-ranking model.")
    modality = parser.add_mutually_exclusive_group()
    modality.add_argument('--raman-only', '--raman_only', dest='raman_only', action='store_true',
                          help="Evaluate a two-channel MMM model with Raman only.")
    modality.add_argument('--ir-only', '--ir_only', dest='ir_only', action='store_true',
                          help="Evaluate a two-channel MMM model with IR only.")
    args = parser.parse_args()
    return args


class TestDataset(torch.utils.data.Dataset):
    def __init__(self, raman: list, ir: list, formula: list, smiles: list):
        self.raman = raman
        self.ir = ir
        self.formula = formula
        self.smiles = smiles

    def __len__(self) -> int:
        return len(self.raman)

    def __getitem__(self, idx: int) -> tuple:
        return self.raman[idx], self.ir[idx], self.formula[idx], self.smiles[idx]


class TestCollator:
    def __init__(self, device: torch.device):
        self.formula_tokenizer = AutoTokenizer.from_pretrained('models/FormulaTokenizer')
        self.smiles_tokenizer = AutoTokenizer.from_pretrained('models/MolTokenizer')
        self.device = device

    def __call__(self, batch: list) -> dict:
        tmp_raman, tmp_ir, tmp_formula, tmp_smiles = zip(*batch)
        
        # Convert to tensor and move to device
        raman_tensor = torch.as_tensor(np.array(tmp_raman), dtype=torch.float32).unsqueeze(1).to(self.device)
        ir_tensor = torch.as_tensor(np.array(tmp_ir), dtype=torch.float32).unsqueeze(1).to(self.device)
        
        formula_encoded = self.formula_tokenizer(tmp_formula, padding=True, return_tensors='pt', truncation=True)
        formula_encoded = {k: v.to(self.device) for k, v in formula_encoded.items()}

        smiles_encoded = self.smiles_tokenizer(tmp_smiles, padding=True, return_tensors='pt', truncation=True)
        smiles_encoded = {k: v.to(self.device) for k, v in smiles_encoded.items()}
        
        return {'raman': raman_tensor, 'ir': ir_tensor, 'formula': formula_encoded, 'smiles': smiles_encoded}


def canonicalize_smiles(smiles: str) -> str:
    """
    Converts a SMILES string to its canonical kekule form.
    Returns '*' if the SMILES is invalid or cannot be canonicalized.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            return Chem.MolToSmiles(mol, isomericSmiles=False, kekuleSmiles=True, canonical=True)
        else:
            return '*'
    except Exception:
        return '*'


def check_mols_inchi(pred_smiles: str, tgt_smiles: str) -> int:
    """
    Checks if two SMILES strings represent the same molecule based on InChIKey.
    Returns 1 if they are identical, 0 otherwise. Handles invalid SMILES gracefully.
    """
    pred_mol = Chem.MolFromSmiles(pred_smiles)
    tgt_mol = Chem.MolFromSmiles(tgt_smiles)

    if pred_mol is None or tgt_mol is None:
        return 0 

    try:
        pred_inchi_key = Chem.MolToInchiKey(pred_mol)
        tgt_inchi_key = Chem.MolToInchiKey(tgt_mol)
        return 1 if pred_inchi_key == tgt_inchi_key else 0
    except Exception:
        return 0

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


def evaluate_topk(predictions, targets, topk_values, step, label):
    correct = {k: 0 for k in topk_values}
    pairs = zip(predictions, targets)
    for pred_list, target in progress_bar(
        pairs, step, label, total=len(targets)
    ):
        for k in topk_values:
            correct[k] += check_beam_mols_topk(pred_list, target, k=k)
    return {f'top-{k}': correct[k] / len(targets) for k in topk_values}

def reject_sample(pred_smiles, tgt_smiles):
    output_smiles = []
    target_formula = rdMolDescriptors.CalcMolFormula(Chem.MolFromSmiles(tgt_smiles))
    for item in pred_smiles:
        try:
            pred_formula = rdMolDescriptors.CalcMolFormula(Chem.MolFromSmiles(item))
            if pred_formula == target_formula:
                output_smiles.append(item)
                
        except TypeError as e:
            continue
    return output_smiles
    
def molecular_generation(
    model: torch.nn.Module,
    ds: str,
    spectral_types: list[str],
    device: torch.device,
    beam_size: int,
    batch_size: int,
    use_formula: bool,
    raman_only: bool = False,
    ir_only: bool = False,
):
    """
    Performs molecular generation from spectral data and evaluates the accuracy.

    Args:
        model: The trained molecular generation model.
        ds: Dataset name (e.g., 'qm9s').
        spectral_types: List of spectral types to use (e.g., ['raman'], ['ir', 'raman']).
        device: The PyTorch device to run inference on.
        beam_size: The beam size for beam search during generation.
        use_formula: Whether to use formula information.
    """
    print(f"Starting molecular generation for {ds} with spectral types: {', '.join(spectral_types)}")

    db_path = f'datasets/vibench/{ds}/{ds}_test.lmdb'
    db = lmdb.open(db_path, subdir=False, lock=False, readonly=True, map_size=int(1e11))
    with db.begin() as txn:
        test_data = [pickle.loads(item[1]) for item in iter_lmdb_records(txn)]
    db.close()

    test_df = pd.DataFrame(test_data)
    if test_df.empty:
        raise ValueError('The test dataset is empty')
    all_smiles = test_df['kekule_smiles'].to_list()
    max_len = max(len(s) for s in all_smiles) + 2

    if use_formula and 'formula' not in test_df.columns:
        raise ValueError(
            f"Dataset '{ds}' does not contain a formula field. "
            "Use a dataset containing molecular formulas or remove "
            "--use-formula."
        )
    formulas = (
        test_df['formula'].to_list()
        if 'formula' in test_df.columns
        else [''] * len(test_df)
    )
    test_dataset = TestDataset(
        test_df['raman'].to_list(),
        test_df['ir'].to_list(),
        formulas,
        all_smiles,
    )
    test_collator = TestCollator(device)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, collate_fn=test_collator)

    model.eval()
    
    # --- Greedy Decoding ---    
    all_pred_smiles_greedy = []
    with torch.no_grad():
        for data in progress_bar(test_loader, 1, 'Greedy Decoding'):
            input_data = data.copy()
            if 'raman' not in spectral_types:
                input_data.pop('raman')
            if 'ir' not in spectral_types:
                input_data.pop('ir')
            if not use_formula:
                input_data.pop('formula')

            generation_kwargs = {'max_len': max_len}
            if raman_only or ir_only:
                generation_kwargs.update(
                    raman_only=raman_only,
                    ir_only=ir_only,
                )
            pred_smiles_ids = model.infer_lm(
                input_data, **generation_kwargs
            )['pred_ids']
            
            tokenizer = test_collator.smiles_tokenizer 
            decoded_smiles = tokenizer.batch_decode(pred_smiles_ids, skip_special_tokens=False)
            cleaned_smiles = [item.split('</s>')[0].replace('<s>', '').strip() for item in decoded_smiles]
            all_pred_smiles_greedy.extend(cleaned_smiles)

    correct_list_greedy = []
    pairs = zip(all_pred_smiles_greedy, all_smiles)
    for pred, tgt in progress_bar(
        pairs, 1, 'Greedy Metrics', total=len(all_smiles)
    ):
        correct = check_mols_inchi(pred, tgt)
        correct_list_greedy.append(correct)
    stage_metrics = {
        'Greedy Decoding': {
            'top-1': sum(correct_list_greedy) / len(correct_list_greedy)
        }
    }

    # --- Beam Search Decoding ---
    all_pred_smiles_beam_list = [] 
    with torch.no_grad():
        for data in progress_bar(test_loader, 2, 'Beam Search'):
            input_data = data.copy() # Avoid modifying original data
            if 'raman' not in spectral_types:
                input_data.pop('raman')
            if 'ir' not in spectral_types:
                input_data.pop('ir')
            if not use_formula:
                input_data.pop('formula')

            generation_kwargs = {
                'max_len': max_len,
                'beam_size': beam_size,
                'temperature': 3.5,
            }
            if raman_only or ir_only:
                generation_kwargs.update(
                    raman_only=raman_only,
                    ir_only=ir_only,
                )
            pred_smiles_ids_list = model.beam_infer_lm(
                input_data, **generation_kwargs
            )['pred_ids']

            batch_pred_smiles_for_topk = []
            for query_beam_ids in pred_smiles_ids_list:
                decoded_smiles_for_query = tokenizer.batch_decode(query_beam_ids, skip_special_tokens=False)
                cleaned_smiles_for_query = [item.split('</s>')[0].replace('<s>', '').strip() for item in decoded_smiles_for_query]
                batch_pred_smiles_for_topk.append(cleaned_smiles_for_query)
            all_pred_smiles_beam_list.extend(batch_pred_smiles_for_topk)

    # Evaluate Top-K accuracy for beam search
    all_pred_smiles_beam_list = [list(dict.fromkeys(item)) for item in all_pred_smiles_beam_list]
    
    # Canonicalize all predicted smiles once
    canonicalized_beam_smiles_list = []
    for pred_list in progress_bar(
        all_pred_smiles_beam_list, 2, 'Canonicalization'
    ):
        canonicalized_beam_smiles_list.append(
            list(dict.fromkeys(canonicalize_smiles(s) for s in pred_list))
        )

    df_beam = pd.DataFrame({'tgt_smiles': all_smiles, 'pred_smiles_list': canonicalized_beam_smiles_list})

    topk_values = [k for k in (1, 3, 5, 10) if k <= beam_size]
    stage_metrics['Beam Search'] = evaluate_topk(
        canonicalized_beam_smiles_list, all_smiles, topk_values,
        step=2,
        label='Beam Metrics',
    )
    
    # --- Reject Sampling by Formula ---    
    if use_formula:
        selected_smiles_list = []
        pairs = zip(canonicalized_beam_smiles_list, all_smiles)
        for pred_item, tgt_item in progress_bar(
            pairs, 2, 'Formula Filtering', total=len(all_smiles)
        ):
            selected_smiles_list.append(reject_sample(pred_item, tgt_item))
        df_beam['selected_smiles_list'] = selected_smiles_list
        stage_metrics['Formula Reject Sampling'] = evaluate_topk(
            selected_smiles_list, all_smiles, topk_values,
            step=2,
            label='Filtered Metrics',
        )
        return selected_smiles_list, test_df, stage_metrics
    else:
        return canonicalized_beam_smiles_list, test_df, stage_metrics

def rerank(
    model: torch.nn.Module,
    all_pred_smiles: list[list[str]],
    test_df: pd.DataFrame,
    topk: int,
    spectral_types: list[str],
    device: torch.device,
    batch_size: int,
    use_formula: bool,
    raman_only: bool = False,
    ir_only: bool = False,
):
    """
    Reranks predicted SMILES based on a matching model and evaluates the accuracy.

    Args:
        model: The re-ranking model.
        all_pred_smiles: List of lists of predicted SMILES from molecular generation.
        test_df: DataFrame containing test data.
        topk: Number of top-k results to consider for re-ranking.
        spectral_types: List of spectral types to use.
        device: The PyTorch device to run inference on.
        use_formula: Whether to use formula information.
    """
    pred_smiles_unique_and_canonical = []
    for sublist in all_pred_smiles:
        # Get unique SMILES, then canonicalize and remove duplicates again (e.g., invalid canonicalizations)
        unique_canonical = list(dict.fromkeys(canonicalize_smiles(s) for s in sublist))
        pred_smiles_unique_and_canonical.append(unique_canonical)

    repeated_smiles_padded = []
    for smiles_list in pred_smiles_unique_and_canonical:
        if len(smiles_list) < topk:
            repeated_smiles_padded.append(smiles_list + ['*' for _ in range(topk - len(smiles_list))])
        else:
            repeated_smiles_padded.append(smiles_list[:topk])

    expanded_raman = []
    expanded_ir = []
    expanded_formula = []

    for i in range(len(test_df)):
        tmp_raman = test_df.iloc[i]['raman']
        tmp_ir = test_df.iloc[i]['ir']
        tmp_formula = test_df.iloc[i]['formula']

        expanded_raman.extend([tmp_raman] * topk)
        expanded_ir.extend([tmp_ir] * topk)
        expanded_formula.extend([tmp_formula] * topk)

    expanded_smiles = [smile for sublist in repeated_smiles_padded for smile in sublist]
            
    test_dataset = TestDataset(expanded_raman, expanded_ir, expanded_formula, expanded_smiles)
    test_collator = TestCollator(device) # Pass device to collator
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, collate_fn=test_collator, shuffle=False)
    test_bar = progress_bar(test_loader, 3, 'Re-ranking')

    matching_scores = []
    model.eval()
    with torch.no_grad():
        for data in test_bar:
            input_data = data.copy()
            if 'raman' not in spectral_types:
                input_data.pop('raman')
            if 'ir' not in spectral_types:
                input_data.pop('ir')
            if not use_formula:
                input_data.pop('formula')
            
            tmp_scores = model.matching(input_data, raman_only=raman_only, ir_only=ir_only)
            matching_scores.append(tmp_scores)
        
    matching_scores = torch.cat(matching_scores)
    matching_scores = torch.softmax(matching_scores, dim=1)
    matching_scores = matching_scores[:, 1].reshape(-1, topk)

    _, topk_indices = matching_scores.topk(k=topk, dim=1, largest=True, sorted=True)
    
    # Extract the re-ranked SMILES
    out_smiles = [[repeated_smiles_padded[i][idx] for idx in idxs] for i, idxs in enumerate(topk_indices)]
    df_rank = pd.DataFrame({'tgt_smiles': test_df['kekule_smiles'].to_list(), 'pred_smiles_list': out_smiles})

    topk_values = [k for k in (1, 3, 5, 10) if k <= topk]
    return evaluate_topk(
        out_smiles,
        test_df['kekule_smiles'].to_list(),
        topk_values,
        step=3,
        label='Re-ranking Metrics',
    )


def print_metrics_table(stage_metrics):
    columns = ('top-1', 'top-3', 'top-5', 'top-10')
    rows = []
    for method, metrics in stage_metrics.items():
        row = {'Method': method}
        row.update({column: metrics.get(column) for column in columns})
        rows.append(row)
    table = pd.DataFrame(rows, columns=('Method',) + columns)
    formatters = {
        column: (lambda value: '-' if pd.isna(value) else f'{value:.5f}')
        for column in columns
    }
    print('\nGeneration Evaluation Summary')
    print(table.to_string(index=False, formatters=formatters, na_rep='-'))


if __name__ == "__main__":
    seed_everything(624)
    args = get_args_parser()

    device = torch.device(args.device)
    spectral_types = args.spectral_types.split('-')
    spectral_channel = len(spectral_types)
    use_formula = args.use_formula
    if (args.ir_only or args.raman_only) and args.model not in {'vib2mol_mmm', 'vib2mol_matching_shared_mask'}:
        raise SystemExit('--ir-only/--raman-only require --model vib2mol_mmm')
    if (args.ir_only or args.raman_only) and spectral_channel != 2:
        raise SystemExit('MMM channel masking requires --spectral-types ir-raman')
    if min(args.batch_size, args.beam_size, args.topk) < 1:
        raise SystemExit('--batch-size, --beam-size and --topk must be positive')
    if args.rerank and args.topk > args.beam_size:
        raise SystemExit('--topk cannot exceed --beam-size')
    if args.rerank and not args.rank_model_path:
        raise SystemExit('--rank-model-path is required with --rerank')

    model = None # Initialize model to None
    try:
        model = build_model(args.model, spectral_channel=spectral_channel).to(device)
        print(f"Loading generation model from: {args.test_model_path}")
        ckpt = torch.load(args.test_model_path, map_location=device, weights_only=True)
        ckpt = {k.replace('module.', ''): v for k, v in ckpt.items()}
        model.load_state_dict(ckpt, strict=True)

    except Exception as e:
        print(f"Error loading generation model: {e}")
        exit(1)

    all_pred_smiles, test_df, stage_metrics = molecular_generation(
        model=model,
        ds=args.ds,
        spectral_types=spectral_types,
        device=device,
        beam_size=args.beam_size,
        batch_size=args.batch_size,
        use_formula=use_formula,
        raman_only=args.raman_only,
        ir_only=args.ir_only,
    )

    if args.rerank:
        rank_model = None # Initialize rank_model to None
        try:
            rank_model = build_model(args.model, spectral_channel=spectral_channel).to(device)
            print(f"Loading re-ranking model from: {args.rank_model_path}")
            ckpt = torch.load(args.rank_model_path, map_location=device, weights_only=True)
            ckpt = {k.replace('module.', ''): v for k, v in ckpt.items()}
            rank_model.load_state_dict(ckpt, strict=True)
        except Exception as e:
            print(f"Error loading re-ranking model: {e}")
            exit(1)
        
        if rank_model and args.topk is not None: # Ensure rank_model is loaded and topk is specified
            stage_metrics['Re-ranking'] = rerank(
                model=rank_model,
                all_pred_smiles=all_pred_smiles,
                test_df=test_df,
                topk=args.topk,
                spectral_types=spectral_types,
                device=device,
                batch_size=args.batch_size,
                use_formula=use_formula,
                raman_only=args.raman_only,
                ir_only=args.ir_only,
            )
        else:
            print("Re-ranking was requested but either the re-ranking model could not be loaded or --topk was not specified.")

    print_metrics_table(stage_metrics)
