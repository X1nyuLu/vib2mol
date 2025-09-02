'''
collators for data loading
'''
import torch
from transformers import AutoTokenizer
from rdkit import Chem
import random
import numpy as np

def smiles_augment(smiles, augmentation_ratio=0.5):
    if random.random() < augmentation_ratio:
        mol = Chem.MolFromSmiles(smiles)
        augments = Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True, kekuleSmiles=True, doRandom=True)
        return augments
    else:
        return smiles

def spectra_augment(spec, theta=0.01, alpha=1, augmentation_ratio=0.5):
    """
    theta: threshold of points which will have noise
    alpha: extent of noise.
    """
    if random.random() < augmentation_ratio:
        random_values = np.zeros(spec.shape)
        indices =  spec > theta # Only add noise on peaks, in case it changed the meaning of the sepctrum
        random_values[indices] = (np.random.rand(sum(indices))*alpha - 1) # Uniformly sample random number between [-1, alpha-1)
        spec = spec * (1 + random_values)
    return spec

class BaseCollator:
    def __init__(self, tokenizer_path=None, spectral_types=None, use_residue=False, **kwargs):
        formula_tokenizer_path = tokenizer_path.replace('MolTokenizer', 'FormulaTokenizer')
        tokenizer_path = tokenizer_path.replace('MolTokenizer', 'PepTokenizer') if use_residue else tokenizer_path
        
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path) if tokenizer_path is not None else None
        self.formula_tokenizer = AutoTokenizer.from_pretrained(formula_tokenizer_path)
        
        self.spectral_types = spectral_types
        self.smiles_augment = False
        self.spectra_augment = False

    def process_smiles(self, batch, smiles_type='smiles'):
        if self.smiles_augment:
            smiles = [smiles_augment(item[smiles_type]) for item in batch]
        else:
            smiles = [item[smiles_type] for item in batch]
        smiles = self.tokenizer(smiles, padding=True, return_tensors='pt', truncation=True)
        return smiles
    
    def process_sequence(self, batch):
        sequence = [item['sequence'] for item in batch]
        sequence = self.tokenizer(sequence, padding=True, return_tensors='pt', truncation=True)
        return sequence
    
    def process_spectra(self, batch, spectral_types=None):
        if type(spectral_types) == str:
            if self.spectra_augment:
                spectra = torch.stack([torch.as_tensor(spectra_augment(item[spectral_types])) for item in batch]).unsqueeze(1).to(torch.float32)
            else:
                spectra = torch.stack([torch.as_tensor(item[spectral_types]) for item in batch]).unsqueeze(1).to(torch.float32)
        elif type(spectral_types) == list:
            spectra = {}
            for spectral_type in spectral_types:
                if self.spectra_augment:
                    spec = torch.stack([torch.as_tensor(spectra_augment(item[spectral_type])) for item in batch]).unsqueeze(1).to(torch.float32)
                else:  
                    spec = torch.stack([torch.as_tensor(item[spectral_type]) for item in batch]).unsqueeze(1).to(torch.float32)
                spectra[spectral_type] = spec
        return spectra
    
    def process_formula(self, batch):
        formula = [item['formula'] for item in batch]
        formula = self.formula_tokenizer(formula, padding=True, return_tensors='pt', truncation=True)
        return formula
    
    def __call__(self, batch):
        batch_data = {}
        batch_data.update(self.process_spectra(batch, spectral_types=self.spectral_types))
        
        if 'sequence' not in batch[0]:
            batch_data['smiles'] = self.process_smiles(batch)
        else:
            batch_data['smiles'] = self.process_sequence(batch)
            
        if 'formula' in batch[0]:
            batch_data['formula'] = self.process_formula(batch)
            
        batch_size = len(batch_data['smiles']['input_ids'])
        return {'batch_size':batch_size, 'target':None, 'data': batch_data}


class RXNCollator(BaseCollator):
    def __init__(self, tokenizer_path, spectral_types=None, smiles_types=None, mix_ratio=0.8, **kwargs):
        super().__init__(tokenizer_path)
        self.spectral_types = spectral_types
        self.smiles_types = smiles_types
        self.mix_ratio = mix_ratio
        
    def __call__(self, batch):
        batch_data = {}
        cache_data = {}
        cache_data.update(self.process_spectra(batch, spectral_type=self.spectral_types))
                
        smiles = []
        for smiles_type in self.smiles_types:
            # smiles_types = ['product_kekule_smiles']
            smiles += [item[smiles_type] for item in batch]
        smiles = self.tokenizer(smiles, padding=True, return_tensors='pt', truncation=True)
        batch_data['smiles'] = smiles
        
        batch_size = len(cache_data['reactant1_raman'])
        # generate three random numbers
        rxn_seed = int(torch.sum(cache_data['reactant1_raman']))
        rng = torch.Generator()
        rng.manual_seed(rxn_seed)
        random_weights = torch.rand(batch_size, generator=rng)
        random_weights = self.mix_ratio + (1 - self.mix_ratio) * random_weights
        
        batch_data['raman'] = (0.5 - 0.5 * random_weights).reshape(-1, 1, 1) * (cache_data['reactant1_raman'] + cache_data['reactant2_raman']) + random_weights.reshape(-1, 1, 1) * cache_data['product_raman']

        return {'batch_size':batch_size, 'target':None, 'data': batch_data}

