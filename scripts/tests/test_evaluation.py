"""Regression tests using synthetic data, without checkpoints or a GPU."""

import contextlib
import io
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import lmdb
import torch

from scripts import infer_retrieval as retrieval
from scripts import run_inference as runner
from utils.lmdb_utils import iter_lmdb_records, read_lmdb_length


class EvaluationTests(unittest.TestCase):
    def test_numeric_lmdb_order_with_and_without_metadata(self):
        for metadata in (False, True):
            with self.subTest(metadata=metadata), tempfile.TemporaryDirectory() as directory:
                with lmdb.open(str(Path(directory) / 'test.lmdb'), subdir=False) as db:
                    with db.begin(write=True) as txn:
                        for index in range(12):
                            txn.put(str(index).encode(), pickle.dumps(index))
                        if metadata:
                            txn.put(b'length', pickle.dumps(12))
                    with db.begin() as txn:
                        self.assertEqual(read_lmdb_length(txn), 12)
                        self.assertEqual([pickle.loads(v) for _, v in iter_lmdb_records(txn)], list(range(12)))

    def test_missing_numeric_record_is_an_error(self):
        with tempfile.TemporaryDirectory() as directory:
            with lmdb.open(str(Path(directory) / 'test.lmdb'), subdir=False) as db:
                with db.begin(write=True) as txn:
                    txn.put(b'0', pickle.dumps({}))
                    txn.put(b'2', pickle.dumps({}))
                with db.begin() as txn, self.assertRaisesRegex(ValueError, 'Missing LMDB record'):
                    list(iter_lmdb_records(txn))

    def test_invalid_formula_candidate_keeps_its_slot(self):
        self.assertEqual(retrieval.reject_sample(['invalid', 'CC', 'C'], 'CC'), ['*', 'CC', '*'])

    def test_reranking_uses_the_same_sample_ids_as_embeddings(self):
        # Each record encodes its own index in the spectrum. The mock matching
        # model exposes any numeric-vs-lexicographic permutation immediately.
        seen = []

        class Collator:
            def __init__(self, **kwargs):
                pass

            def __call__(self, batch):
                return {'data': batch}

        class MatchingModel:
            def eval(self):
                return self

            def matching(self, batch, **kwargs):
                seen.extend((int(row['ir'][0]), row['smiles']) for row in batch)
                return torch.tensor([[0., 1.] for _ in batch])

        with tempfile.TemporaryDirectory() as directory:
            db_path = str(Path(directory) / 'test.lmdb')
            with lmdb.open(db_path, subdir=False) as db:
                with db.begin(write=True) as txn:
                    for index in range(12):
                        txn.put(str(index).encode(), pickle.dumps({'ir': [index], 'kekule_smiles': 'C' * (index + 1)}))
                    txn.put(b'length', pickle.dumps(12))
            real_open = lmdb.open
            with patch.object(retrieval.lmdb, 'open', side_effect=lambda *a, **kw: real_open(db_path, **kw)), \
                 patch.object(retrieval, 'BaseCollator', Collator), contextlib.redirect_stdout(io.StringIO()):
                indices, metrics, _ = retrieval.rerank(
                    MatchingModel(), torch.eye(12), topk=1,
                    spectral_types=['ir'], device='cpu', batch_size=4,
                )
            self.assertEqual(seen, [(index, 'C' * (index + 1)) for index in range(12)])
            self.assertEqual(metrics['Recall@1'], 1.)
            self.assertEqual(indices.flatten().tolist(), list(range(12)))

    def test_command_preserves_zero_and_omits_false_flags(self):
        command = runner.build_command('scripts/infer_mlm.py', {'--num-workers': 0, '--use-formula': False, '--limit': None})
        self.assertEqual(command[2:], ['--num-workers', '0'])

    def test_every_manifest_uses_a_final_model_and_existing_entrypoint(self):
        for path in runner.CONFIG_DIR.glob('*.yaml'):
            with self.subTest(suite=path.stem):
                config = runner.load_suite(path.stem)
                entrypoint = runner.REPO_ROOT / config['entrypoint']
                self.assertTrue(entrypoint.is_file())
                self.assertEqual(entrypoint.parent, runner.REPO_ROOT / 'scripts')
                for spec in config['experiments'].values():
                    options = runner.merge_options(config.get('defaults', {}), spec)
                    self.assertIn(options['--model'], ('vib2mol', 'vib2mol_mmm'))
                    self.assertIn('--test-model-path', options)


if __name__ == '__main__':
    unittest.main()
