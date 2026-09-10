"""CPU compatibility checks for the two maintained model architectures."""

import importlib
import unittest

import torch

import models
from utils.base import PHASE_ALIGN, PHASE_GENERATE, normalize_phase


def _small_model(module_name, **kwargs):
    module = importlib.import_module(module_name)
    return module.Vib2Mol(
        d_model=16,
        nhead=4,
        d_ff=32,
        encoder_nlayer=1,
        decoder_nlayer=1,
        multimodal_nlayer=1,
        **kwargs,
    )


class ModelCompatibilityTests(unittest.TestCase):
    def test_only_final_models_and_legacy_aliases_are_registered(self):
        self.assertEqual(set(models.model_registry), {
            'vib2mol', 'vib2mol_matching_shared',
            'vib2mol_mmm', 'vib2mol_matching_shared_mask',
        })

    def test_phase_aliases_are_backward_compatible(self):
        self.assertEqual(normalize_phase(1), PHASE_ALIGN)
        self.assertEqual(normalize_phase(2), PHASE_GENERATE)
        self.assertEqual(normalize_phase(PHASE_ALIGN), PHASE_ALIGN)
        self.assertEqual(normalize_phase(PHASE_GENERATE), PHASE_GENERATE)

    def test_spectral_channel_order_is_raman_then_ir(self):
        base = importlib.import_module('models.base').BaseModel()
        raman = torch.full((2, 1, 8), 1.0)
        ir = torch.full((2, 1, 8), 2.0)
        spectra = base.load_spectra({'ir': ir, 'raman': raman})
        torch.testing.assert_close(spectra[:, 0], raman[:, 0])
        torch.testing.assert_close(spectra[:, 1], ir[:, 0])

    def test_mmm_checkpoint_cannot_silently_load_into_standard_model(self):
        standard = _small_model('models.vib2mol')
        mmm = _small_model('models.vib2mol_mmm')
        self.assertNotIn('spectral_mask_token', standard.state_dict())
        self.assertIn('spectral_mask_token', mmm.state_dict())
        with self.assertRaises(RuntimeError):
            standard.load_state_dict(mmm.state_dict(), strict=True)

    def test_standard_model_inference_paths(self):
        # Tiny, randomly initialized network: verifies execution, not accuracy.
        for modalities in [('ir',), ('raman',), ('ir', 'raman')]:
            with self.subTest(modalities=modalities), torch.no_grad():
                model = _small_model('models.vib2mol', spectral_channel=len(modalities)).eval()
                inputs = {key: torch.randn(2, 1, 32) for key in modalities}
                inputs['smiles'] = {
                    'input_ids': torch.tensor([[0, 5, 2], [0, 6, 2]]),
                    'attention_mask': torch.ones(2, 3, dtype=torch.long),
                }
                self.assertEqual(model.matching(inputs).shape, (2, 2))
                spectral = model.get_spectral_embeddings(inputs)['proj_output']
                molecular = model.get_molecular_embeddings(inputs, use_cls_token=True)['proj_output']
                self.assertEqual(spectral.shape, molecular.shape)
                self.assertTrue(torch.isfinite(spectral).all())
                for with_formula in (False, True):
                    if with_formula:
                        inputs['formula'] = inputs['smiles']
                    greedy = model.infer_lm(inputs, max_len=4)['pred_ids']
                    beam = model.beam_infer_lm(inputs, max_len=4, beam_size=2)['pred_ids']
                    self.assertEqual(len(greedy), 2)
                    self.assertEqual(len(beam), 2)
                    self.assertTrue(torch.isfinite(model.infer_mlm(inputs)).all())


if __name__ == '__main__':
    unittest.main()
