# Vib2Mol: from vibrational spectra to molecular structures-a unified deep learning framework

<p align="center">
  <a href="https://doi.org/10.1039/D6SC04559F"><img src="https://img.shields.io/badge/Chemical%20Science-2026-blue?style=flat-square"></a>
  <a href="https://huggingface.co/xinyulu/vib2mol"><img src="https://img.shields.io/badge/🤗-Checkpoints-yellow?style=flat-square"></a>
  <a href="https://huggingface.co/datasets/xinyulu/vibench"><img src="https://img.shields.io/badge/🤗-ViBench-yellow?style=flat-square"></a>
  <a href="https://doi.org/10.6084/m9.figshare.28579832"><img src="https://img.shields.io/badge/Figshare-Data-blue?style=flat-square"></a>
</p>

## Overview

**Vib2Mol** is a unified framework for molecular retrieval and generation from IR/Raman spectra, optionally conditioned on prior knowledge such as molecular formulas.

<p align="center">
  <img src="./docs/results.svg" alt="Vib2Mol results" width="100%">
</p>


>We recently developed **Vib2Conf**, a framework for retrieving molecular conformations from vibrational spectra, published in [*Analytical Chemistry*](https://doi.org/10.1021/acs.analchem.6c02845).

## Quick start

### 1. Installation

```bash
python -m pip install -r requirements.txt

# Optional: run CPU regression tests
python -m unittest discover -s scripts/tests -v
```

The reference environment uses PyTorch 2.7.0. Full model evaluation is intended for NVIDIA GPUs; install the PyTorch build compatible with your CUDA environment.

### 2. Download datasets and checkpoints

**Dataset:** [ViBench](https://huggingface.co/datasets/xinyulu/vibench)
**Checkpoints:** [Vib2Mol](https://huggingface.co/xinyulu/vib2mol)

Using the Hugging Face CLI:

```bash
# Dataset
hf download xinyulu/vibench \
    --repo-type=dataset \
    --local-dir ./datasets/vibench

# Model checkpoints
hf download xinyulu/vib2mol \
    --local-dir ./checkpoints
```

The full repositories can require substantial disk space. You may instead download only the files required by the experiment manifests in [`scripts/configs/`](scripts/configs/).

For example, QM9S IR retrieval expects:

```text
datasets/vibench/qm9s/qm9s_test.lmdb
checkpoints/qm9s/ir-kekule_smiles/vib2mol/2025-07-08-02-55-db43f7/epoch999.pth
```

### 3. Run retrieval or generation

List available experiments:

```bash
python scripts/run_inference.py retrieval --list
python scripts/run_inference.py generation --list
```

Validate paths and inspect the resolved command without running inference:

```bash
python scripts/run_inference.py retrieval qm9s_ir --dry-run
```

Run QM9S IR retrieval:

```bash
CUDA_VISIBLE_DEVICES=0 \
python scripts/run_inference.py retrieval qm9s_ir \
    --device cuda:0 \
    --batch-size 32
```

Run QM9S IR generation:

```bash
CUDA_VISIBLE_DEVICES=0 \
python scripts/run_inference.py generation qm9s_ir \
    --device cuda:0 \
    --batch-size 32
```

Run formula-conditioned IR + Raman generation:

```bash
CUDA_VISIBLE_DEVICES=0 \
python scripts/run_inference.py generation qm9s_ir_raman_formula \
    --device cuda:0 \
    --batch-size 32
```

Reduce `--batch-size` if GPU memory is insufficient.

For custom checkpoints and advanced options:

```bash
python scripts/infer_retrieval.py --help
python scripts/infer_generation.py --help
```

The YAML manifests in [`scripts/configs/`](scripts/configs/) are the recommended interface for reproducing the provided experiments.

## Additional evaluation

### MMM

MMM uses checkpoints distinct from the standard Vib2Mol models.

```bash
python scripts/run_inference.py retrieval_mmm --list
python scripts/run_inference.py generation_mmm --list
```

The released MMM checkpoints have been evaluated on the full QM9S and MOLS test splits for supported single-/dual-spectrum retrieval and dual-spectrum, formula-conditioned generation settings, including re-ranking.


See [`scripts/README.md`](scripts/README.md) for masking, candidate-selection, and evaluation options.

## Training

Vib2Mol is trained in two stages.

### Stage 1: spectrum–molecule alignment and matching

```bash
python main.py \
    --train \
    --launch matching \
    --model vib2mol \
    --ds mols \
    --task ir-raman-kekule_smiles
```

### Stage 2: molecular generation

Replace the checkpoint path below with the corresponding Stage-1 weights:

```bash
torchrun --nproc_per_node=4 main.py \
    --train \
    --ddp \
    --launch spt \
    --model vib2mol \
    --ds mols \
    --task ir-raman-kekule_smiles-formula \
    --smiles-augment \
    --base-model-path path/to/stage1/epoch999.pth
```

The Stage-1 checkpoint must match the Stage-2 architecture and number of spectral channels. `config.yaml` contains the default training configuration, and `--batch-size` denotes the global batch size across DDP workers.

<details>
<summary><strong>Repository structure</strong></summary>

<br>

| Path                 | Description                                                                                            |
| -------------------- | ------------------------------------------------------------------------------------------------------ |
| `models/`            | Vib2Mol and MMM architectures, shared neural-network modules, and molecular/formula/peptide tokenizers |
| `trainers/`          | Training loops for alignment, matching, generation, and related workflows                              |
| `utils/`             | Dataset loading, LMDB handling, collation, checkpoint loading, seeds, and training utilities           |
| `scripts/`           | Inference and evaluation entry points, experiment runner, batch scripts, and MMM workflows             |
| `scripts/configs/`   | YAML experiment manifests defining datasets, checkpoints, model variants, and evaluation settings      |
| `scripts/tests/`     | CPU regression tests for model compatibility, manifests, evaluation logic, and CLI startup             |
| `notebooks/`         | Figure preparation and molecular dataset construction notebooks                                        |
| `docs/`              | README/project-page figures and related assets                                                         |
| `logs/`              | Original experiment and training logs                                                                  |
| `runs/`              | Original TensorBoard records and training summaries                                                    |
| `.github/workflows/` | GitHub Actions for automated CPU regression tests                                                      |

Repository-level files include:

```text
main.py
config.yaml
requirements.txt
README.md
LICENSE
```

</details>

<details>
<summary><strong>Research notebooks</strong></summary>

<br>

| Notebook                                                                             | Purpose                                                                                                        |
| ------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------- |
| [`figures.ipynb`](notebooks/figures.ipynb)                                           | Prepare paper and analysis figures from recorded metrics and supplementary inputs                              |
| [`generate_pah_dataset.ipynb`](notebooks/generate_pah_dataset.ipynb)                 | Enumerate substituted benzene, naphthalene, and anthracene structures and export optimized XYZ geometries      |
| [`generate_tripeptides_dataset.ipynb`](notebooks/generate_tripeptides_dataset.ipynb) | Construct modified peptide candidates, including phosphorylation/sulfation variants, and export XYZ geometries |

Run each notebook's setup cell first to resolve repository-relative paths.

Some notebooks require supplementary inputs that are not distributed with the repository. Their saved outputs should be treated as historical research records rather than freshly recomputed benchmark results.

</details>

<details>
<summary><strong>Reproducibility notes</strong></summary>

<br>

* Standard retrieval compares test spectra against molecules in the corresponding test split; it does not search PubChem by default.
* Dense retrieval similarity matrices require memory proportional to the square of the candidate-set size.
* PubChem evaluation requires a compatible candidate embedding index and dataframe.
* MMM checkpoints are separate from standard Vib2Mol checkpoints.
* CPU regression tests verify execution and compatibility, not reproduction of the reported paper metrics.
* Auxiliary MLM and PubChem suites have not undergone the same full-test GPU acceptance procedure as the main released evaluation workflows.
* LMDB records use Python pickle. Only load datasets obtained from trusted sources.
* Checkpoints are loaded with `weights_only=True` and strict parameter matching.
* Full retraining and DDP convergence were not revalidated as part of the repository release cleanup.

</details>

## Original experiment records

The `logs/` and `runs/` directories preserve the original experiment records used during development.

TensorBoard logs can be inspected with:

```bash
tensorboard --logdir ./runs
```

The original experiments used four NVIDIA A800 GPUs. VB-Mols pretraining required approximately 85 hours across the two training stages under the original setup.

Training logs and TensorBoard records are not directly interchangeable with inference metrics obtained after beam search or re-ranking.

## Citation

If you find Vib2Mol useful in your research, please cite:

```bibtex
@article{Lu2026Vib2Mol,
    author  = {Lu, Xinyu and Ma, Hao and Li, Hui and Li, Jia and Rong, Yi and Li, Yuqiang and Zhu, Tong and Liu, Guokun and Ren, Bin},
    title   = {Vib2Mol: from vibrational spectra to molecular structures—a unified deep learning framework},
    journal = {Chemical Science},
    year    = {2026},
    issn    = {2041-6520},
    doi     = {10.1039/D6SC04559F},
    url     = {https://doi.org/10.1039/D6SC04559F}
}
```

## Acknowledgements

This work was supported by the National Natural Science Foundation of China (Grant Nos. 22227802, 22021001, 22474117, and 22272139), the Fundamental Research Funds for the Central Universities (20720220009 and 20720250005), and Shanghai Innovation Institute.

## Contact

Questions, suggestions, and issues are welcome.

**Xinyu Lu**
Email: [xinyulu@stu.xmu.edu.cn](mailto:xinyulu@stu.xmu.edu.cn)
