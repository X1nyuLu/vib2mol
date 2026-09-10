import argparse
import csv
import pickle
import sys
from pathlib import Path

# Support direct execution from the repository root as well as python -m.
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import lmdb
import numpy as np
import pandas as pd
import torch
from rdkit import Chem, RDLogger
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from models import build_model
from utils.base import seed_everything
from utils.lmdb_utils import iter_lmdb_records, read_lmdb_length


RDLogger.DisableLog('rdApp.*')

PROGRESS_STAGE_WIDTH = 10
PROGRESS_LABEL_WIDTH = 28
PROGRESS_BAR_WIDTH = 140


def progress_bar(iterable, stage, label, **kwargs):
    """Create consistently aligned progress bars for MLM evaluation."""
    return tqdm(
        iterable,
        desc=(
            f'{f"[{stage}]":<{PROGRESS_STAGE_WIDTH}} '
            f'{label:<{PROGRESS_LABEL_WIDTH}}'
        ),
        ncols=PROGRESS_BAR_WIDTH,
        dynamic_ncols=False,
        **kwargs,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate spectrum-conditioned masked-SMILES reconstruction.'
    )
    parser.add_argument('--test-model-path', '--test_model_path', dest='test_model_path', required=True)
    parser.add_argument('--model', default='vib2mol')
    parser.add_argument('--ds', default='mols')
    parser.add_argument('--split', choices=('train', 'eval', 'test'), default='test')
    parser.add_argument('--data-dir', '--data_dir', dest='data_dir', default='datasets/vibench')
    parser.add_argument('--tokenizer-path', '--tokenizer_path', dest='tokenizer_path', default='models/MolTokenizer')
    parser.add_argument('--formula-tokenizer-path', '--formula_tokenizer_path', dest='formula_tokenizer_path', default='models/FormulaTokenizer')
    parser.add_argument('--spectral-types', '--spectral_types', dest='spectral_types', default='raman')
    parser.add_argument('--mode', choices=('token', 'branch', 'both'), default='both')
    parser.add_argument('--mask-prob', '--mask_prob', dest='mask_prob', type=float, default=0.75)
    parser.add_argument('--batch-size', '--batch_size', dest='batch_size', type=int, default=64)
    parser.add_argument('--max-length', '--max_length', dest='max_length', type=int, default=256)
    parser.add_argument('--num-workers', '--num_workers', dest='num_workers', type=int, default=0)
    parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=624)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--output-csv', '--output_csv', dest='output_csv', help='Optional branch-level prediction table.')
    parser.add_argument(
        '--deduplicate-branches', '--deduplicate_branches', dest='deduplicate_branches', action='store_true',
        help='Evaluate only the first occurrence of each branch text per SMILES.',
    )
    parser.add_argument(
        '--use-formula', '--use_formula', dest='use_formula', action='store_true',
        help='Condition MLM reconstruction on the molecular formula.',
    )
    return parser.parse_args()


def extract_branches(smiles):
    """Return all parenthesized spans, including nested spans."""
    branches = []
    stack = []
    for index, char in enumerate(smiles):
        if char == '(':
            stack.append(index)
        elif char == ')' and stack:
            start = stack.pop()
            branches.append((smiles[start + 1:index], start, index))
    branches.sort(key=lambda item: item[1])
    return branches


def load_records(args):
    path = Path(args.data_dir) / args.ds / f'{args.ds}_{args.split}.lmdb'
    if not path.is_file():
        raise FileNotFoundError(f'LMDB file not found: {path}')
    db = lmdb.open(str(path), subdir=False, lock=False, readonly=True, readahead=False)
    with db.begin() as txn:
        records = [
            pickle.loads(value)
            for _, value in progress_bar(
                iter_lmdb_records(txn), 'Setup', 'Loading LMDB Records',
                total=read_lmdb_length(txn),
            )
        ]
    db.close()
    if args.limit is not None:
        records = records[:args.limit]
    return records


class MLMData(Dataset):
    def __init__(self, records, spectral_types, use_formula=False):
        self.records = records
        self.spectral_types = spectral_types
        self.use_formula = use_formula

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        smiles = record.get('kekule_smiles', record.get('smiles'))
        if smiles is None:
            raise KeyError('Record has neither kekule_smiles nor smiles')
        item = {
            'smiles': smiles,
            **{name: record[name] for name in self.spectral_types},
        }
        if self.use_formula:
            item['formula'] = record['formula']
        return item


class RandomMaskCollator:
    def __init__(
        self, tokenizer, spectral_types, mask_prob, max_length,
        formula_tokenizer=None,
    ):
        self.tokenizer = tokenizer
        self.spectral_types = spectral_types
        self.mask_prob = mask_prob
        self.max_length = max_length
        self.formula_tokenizer = formula_tokenizer

    def add_formula(self, data, batch):
        if self.formula_tokenizer is not None:
            data['formula'] = self.formula_tokenizer(
                [item['formula'] for item in batch], padding=True,
                truncation=True, return_tensors='pt',
            )

    def __call__(self, batch):
        target = self.tokenizer(
            [item['smiles'] for item in batch], padding='max_length',
            max_length=self.max_length, truncation=True, return_tensors='pt',
            return_special_tokens_mask=True,
        )
        mask = torch.bernoulli(
            torch.full(target['input_ids'].shape, self.mask_prob)
        ).bool()
        mask[target['special_tokens_mask'].bool()] = False
        masked = target['input_ids'].clone()
        masked[mask] = self.tokenizer.mask_token_id
        data = {
            'smiles': {
                'input_ids': masked,
                'attention_mask': target['attention_mask'],
            }
        }
        data.update(stack_spectra(batch, self.spectral_types))
        self.add_formula(data, batch)
        return data, target['input_ids'], mask


class SpanMaskCollator(RandomMaskCollator):
    """Mask one contiguous molecular-token span at the requested ratio."""

    def __call__(self, batch):
        target = self.tokenizer(
            [item['smiles'] for item in batch], padding='max_length',
            max_length=self.max_length, truncation=True, return_tensors='pt',
            return_special_tokens_mask=True,
        )
        special_tokens = target['special_tokens_mask'].bool()
        mask = torch.zeros_like(target['input_ids'], dtype=torch.bool)
        for row in range(mask.size(0)):
            token_positions = (~special_tokens[row]).nonzero(
                as_tuple=False
            ).flatten()
            token_count = token_positions.numel()
            if token_count == 0 or self.mask_prob == 0:
                continue
            span_length = min(
                token_count,
                max(1, int(token_count * self.mask_prob + 0.5)),
            )
            max_start = token_count - span_length
            start = torch.randint(max_start + 1, (1,)).item()
            mask[row, token_positions[start:start + span_length]] = True

        masked = target['input_ids'].clone()
        masked[mask] = self.tokenizer.mask_token_id
        data = {
            'smiles': {
                'input_ids': masked,
                'attention_mask': target['attention_mask'],
            }
        }
        data.update(stack_spectra(batch, self.spectral_types))
        self.add_formula(data, batch)
        return data, target['input_ids'], mask


class BranchData(Dataset):
    def __init__(
        self, records, spectral_types, tokenizer, max_length,
        deduplicate_branches=False,
    ):
        self.items = []
        for record in progress_bar(
            records, 'Step 3/3', 'Creating Branch Masks'
        ):
            target = record.get('kekule_smiles', record.get('smiles'))
            encoded = tokenizer(
                target, padding='max_length', max_length=max_length,
                truncation=True, return_offsets_mapping=True,
                return_special_tokens_mask=True,
            )
            target_ids = torch.tensor(encoded['input_ids'], dtype=torch.long)
            attention_mask = torch.tensor(
                encoded['attention_mask'], dtype=torch.long
            )
            offsets = encoded['offset_mapping']
            special_tokens = encoded['special_tokens_mask']

            branches = extract_branches(target)
            if deduplicate_branches:
                seen = set()
                unique_branches = []
                for branch in branches:
                    if branch[0] not in seen:
                        seen.add(branch[0])
                        unique_branches.append(branch)
                branches = unique_branches

            # By default, every occurrence is evaluated independently, even
            # when two branches contain identical text.
            for _, start, end in branches:
                content_start, content_end = start + 1, end
                branch_mask = torch.tensor([
                    not is_special
                    and token_start >= content_start
                    and token_end <= content_end
                    and token_end > token_start
                    for (token_start, token_end), is_special
                    in zip(offsets, special_tokens)
                ], dtype=torch.bool)
                if not branch_mask.any():
                    continue

                # A truncated or partially aligned branch must not become a
                # misleading evaluation sample.
                selected_offsets = [
                    offset for offset, selected in zip(offsets, branch_mask)
                    if selected
                ]
                if (selected_offsets[0][0] != content_start
                        or selected_offsets[-1][1] != content_end):
                    continue

                masked_ids = target_ids.clone()
                masked_ids[branch_mask] = tokenizer.mask_token_id
                masked_display = (
                    target[:content_start] + '*' + target[content_end:]
                )
                self.items.append({
                    'record': record,
                    'target': target,
                    'masked': masked_display,
                    'input_ids': masked_ids,
                    'attention_mask': attention_mask,
                })
        self.spectral_types = spectral_types

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


class BranchCollator:
    def __init__(self, spectral_types, formula_tokenizer=None):
        self.spectral_types = spectral_types
        self.formula_tokenizer = formula_tokenizer

    def __call__(self, batch):
        data = {'smiles': {
            'input_ids': torch.stack([item['input_ids'] for item in batch]),
            'attention_mask': torch.stack([
                item['attention_mask'] for item in batch
            ]),
        }}
        spectral_batch = [
            {name: item['record'][name] for name in self.spectral_types}
            for item in batch
        ]
        data.update(stack_spectra(spectral_batch, self.spectral_types))
        if self.formula_tokenizer is not None:
            data['formula'] = self.formula_tokenizer(
                [item['record']['formula'] for item in batch], padding=True,
                truncation=True, return_tensors='pt',
            )
        return (
            data,
            [item['target'] for item in batch],
            [item['masked'] for item in batch],
        )


def stack_spectra(batch, spectral_types):
    return {
        name: torch.as_tensor(
            np.asarray([item[name] for item in batch]), dtype=torch.float32
        ).unsqueeze(1)
        for name in spectral_types
    }


def to_device(value, device):
    if isinstance(value, dict):
        return {key: to_device(item, device) for key, item in value.items()}
    return value.to(device) if hasattr(value, 'to') else value


def load_model(args, spectral_types, device):
    checkpoint = torch.load(args.test_model_path, map_location='cpu', weights_only=True)
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        checkpoint = checkpoint['model']
    checkpoint = {key.removeprefix('module.'): value for key, value in checkpoint.items()}

    embedding_key = 'molecular_encoding.molecular_embedding.weight'
    if embedding_key not in checkpoint:
        raise KeyError(f'Checkpoint does not contain {embedding_key}')
    num_embeddings = checkpoint[embedding_key].shape[0]
    model = build_model(
        args.model,
        spectral_channel=len(spectral_types),
        num_embeddings=num_embeddings,
        mask_prob=args.mask_prob,
    )
    model.load_state_dict(checkpoint, strict=True)
    return model.to(device).eval()


@torch.no_grad()
def evaluate_tokens(
    model, records, tokenizer, spectral_types, args, device,
    collator, stage, label, accuracy_name,
):
    loader = DataLoader(
        MLMData(records, spectral_types, use_formula=args.use_formula),
        batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers,
        collate_fn=collator,
    )
    correct = total = valid = samples = 0
    for data, targets, mask in progress_bar(
        loader, stage, label
    ):
        predictions = model.infer_mlm(to_device(data, device)).argmax(dim=-1).cpu()
        correct += (predictions[mask] == targets[mask]).sum().item()
        total += mask.sum().item()
        reconstructed = data['smiles']['input_ids'].clone()
        reconstructed[mask] = predictions[mask]
        decoded = tokenizer.batch_decode(
            reconstructed, skip_special_tokens=True
        )
        valid += sum(Chem.MolFromSmiles(smiles) is not None for smiles in decoded)
        samples += len(decoded)
    accuracy = correct / total if total else float('nan')
    return {
        'Samples': samples,
        'Masked Tokens': total,
        'Mask Prob.': args.mask_prob,
        accuracy_name: accuracy,
        'Valid SMILES Rate': valid / samples if samples else float('nan'),
    }


def same_molecule(prediction, target):
    pred_mol = Chem.MolFromSmiles(prediction)
    target_mol = Chem.MolFromSmiles(target)
    if pred_mol is None or target_mol is None:
        return False
    try:
        return Chem.MolToInchiKey(pred_mol) == Chem.MolToInchiKey(target_mol)
    except Exception:
        return False


@torch.no_grad()
def evaluate_branches(
    model, records, tokenizer, formula_tokenizer, spectral_types, args, device,
):
    dataset = BranchData(
        records, spectral_types, tokenizer, args.max_length,
        deduplicate_branches=args.deduplicate_branches,
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers,
        collate_fn=BranchCollator(spectral_types, formula_tokenizer),
    )
    rows = []
    valid = correct = 0
    for data, targets, masked_smiles in progress_bar(
        loader, 'Step 3/3', 'Branch Reconstruction'
    ):
        device_data = to_device(data, device)
        predictions = model.infer_mlm(device_data).argmax(dim=-1).cpu()
        output_ids = data['smiles']['input_ids'].clone()
        mask = output_ids == tokenizer.mask_token_id
        output_ids[mask] = predictions[mask]
        decoded = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        for target, masked, prediction in zip(targets, masked_smiles, decoded):
            is_valid = Chem.MolFromSmiles(prediction) is not None
            is_correct = same_molecule(prediction, target)
            valid += int(is_valid)
            correct += int(is_correct)
            rows.append({
                'target': target,
                'masked': masked,
                'prediction': prediction,
                'valid': int(is_valid),
                'correct': int(is_correct),
            })

    count = len(rows)
    output_path = None
    if args.output_csv:
        output_path = Path(args.output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open('w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=rows[0].keys() if rows else [
                'target', 'masked', 'prediction', 'valid', 'correct'
            ])
            writer.writeheader()
            writer.writerows(rows)
    return {
        'Samples': count,
        'Valid SMILES Rate': valid / count if count else float('nan'),
        'Molecular Accuracy': correct / count if count else float('nan'),
    }, output_path


def print_metrics_table(stage_metrics):
    columns = (
        'Samples', 'Masked Tokens', 'Mask Prob.', 'Token Accuracy',
        'Span Accuracy', 'Valid SMILES Rate', 'Molecular Accuracy',
    )
    rows = []
    for method, metrics in stage_metrics.items():
        row = {'Method': method}
        row.update({column: metrics.get(column, np.nan) for column in columns})
        rows.append(row)
    table = pd.DataFrame(rows, columns=('Method',) + columns)
    formatters = {
        'Samples': lambda value: '-' if pd.isna(value) else f'{int(value)}',
        'Masked Tokens': lambda value: (
            '-' if pd.isna(value) else f'{int(value)}'
        ),
        **{
            column: (lambda value: '-' if pd.isna(value) else f'{value:.6f}')
            for column in columns[2:]
        },
    }
    print('\nMLM Evaluation Summary')
    print(table.to_string(index=False, formatters=formatters, na_rep='-'))


def main():
    args = parse_args()
    if not 0 <= args.mask_prob <= 1:
        raise ValueError('--mask-prob must be between 0 and 1')
    spectral_types = [item for item in args.spectral_types.split('-') if item]
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is not available; use --device cpu')

    seed_everything(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    formula_tokenizer = None
    if args.use_formula:
        formula_tokenizer = AutoTokenizer.from_pretrained(
            args.formula_tokenizer_path
        )
    records = load_records(args)
    model = load_model(args, spectral_types, device)
    stage_metrics = {}
    output_path = None
    if args.mode in ('token', 'both'):
        stage_metrics['Random-mask MLM'] = evaluate_tokens(
            model, records, tokenizer, spectral_types, args, device,
            collator=RandomMaskCollator(
                tokenizer, spectral_types, args.mask_prob, args.max_length,
                formula_tokenizer=formula_tokenizer,
            ),
            stage='Step 1/3',
            label='Random-mask Reconstruction',
            accuracy_name='Token Accuracy',
        )
        stage_metrics['Contiguous-span MLM'] = evaluate_tokens(
            model, records, tokenizer, spectral_types, args, device,
            collator=SpanMaskCollator(
                tokenizer, spectral_types, args.mask_prob, args.max_length,
                formula_tokenizer=formula_tokenizer,
            ),
            stage='Step 2/3',
            label='Span Reconstruction',
            accuracy_name='Span Accuracy',
        )
    if args.mode in ('branch', 'both'):
        branch_metrics, output_path = evaluate_branches(
            model, records, tokenizer, formula_tokenizer, spectral_types,
            args, device,
        )
        branch_method = 'Branch Reconstruction'
        if args.deduplicate_branches:
            branch_method += ' (Deduplicated)'
        stage_metrics[branch_method] = branch_metrics

    if output_path is not None:
        print(f'predictions_csv: {output_path}')
    print_metrics_table(stage_metrics)


if __name__ == '__main__':
    main()
