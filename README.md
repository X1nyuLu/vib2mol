# Vib2Mol: from vibrational spectra to molecular structures—a versatile deep learning model
[![arXiv](https://img.shields.io/badge/arXiv-2503.07014-c72c2c.svg)](https://arxiv.org/abs/2503.07014)
[![](https://img.shields.io/badge/huggingface-vib2mol-dd9029)](https://huggingface.co/xinyulu/vib2mol)
[![](https://img.shields.io/badge/figshare-10.6084/m9.figshare.28579832-2243da)](https://doi.org/10.6084/m9.figshare.28579832)

## Abstract
There will be a paradigm shift in chemical and biological research, to be enabled by autonomous, closed-loop, real-time self-directed decision-making experimentation. Spectrum-to-structure correlation, which is to elucidate molecular structures with spectral information, is the core step in understanding the experimental results and to close the loop. However, current approaches usually divide the task into either database-dependent retrieval and database-independent generation and neglect the inherent complementarity between them. In this study, we proposed Vib2Mol, a general deep learning model designed to flexibly handle diverse spectrum-to-structure tasks according to the available prior knowledge by bridging the retrieval and generation. It achieves state-of-the-art performance, even for the most demanding Raman spectra, over previous models in predicting reaction products and sequencing peptides as well as analyzing experimental spectra and integrating multi-modal spectral data. Vib2Mol enables vibrational spectroscopy a real-time guide for autonomous scientific discovery workflows.


## Getting Started
```python
pip install -r requirements.txt

# start to train
python main.py -train --model vib2mol_cl --launch cl --ds mols --task raman-kekule_smiles  --epoch 1000

# start to train with ddp
torchrun --nproc_per_node=8 main.py -train --ddp --model vib2mol_cl  --launch cl --ds mols --task raman-kekule_smiles  --epoch 1000

# fine-tuning (take PAHs as an example)
python main.py -train --model vib2mol_cl --launch cl --ds pahs --task raman-kekule_smiles  --epoch 1000 --base_model_path 'path/to/your/checkpoint'
```

## Evaluation
all evaluation results and scripts are in the folder `evaluate`. You can check them for more details.

## Datasets and Checkpoints
We provide the datasets and checkpoints in the following links:
heckpoints
All checkpoints are available in the [Hugging Face](https://huggingface.co/xinyulu/vib2mol).  
All datasets are available in [figshare](https://doi.org/10.6084/m9.figshare.28579832). There are test sets only, and we will release the training and validation sets upon acceptance of the paper.


## Hardware
Eight NVIDIA GPUs were employed for experiments, while pretraining on VB-mols costs almost 35 hours (stage 1 for ~11.5 hours and stage 2 for ~23.5 hours).

## Citation
Cite our work as followes:
```
@article{lu2025vib2mol,
      title={Vib2Mol: from vibrational spectra to molecular structures-a versatile deep learning model}, 
      author={Xinyu Lu, Hao Ma, Hui Li, Jia Li, Tong Zhu, Guokun Liu, Bin Ren},
      year={2025},
      url={https://arxiv.org/abs/2503.07014}, 
}
```

## Acknowledgements
This work was supported by the National Natural Science Foundation (Grant No: 22227802, 22021001, 22474117, 22272139) of China and the Fundamental Research Funds for the Central Universities (20720220009) and Shanghai Innovation Institute.

## Contact
Welcome to contact us or raise issues if you have any questions. 
Email: xinyulu@stu.xmu.edu.cn