'''
collators for data loading
'''
import torch
from transformers import AutoTokenizer

class BaseCollator:
    def __init__(self, tokenizer_path=None, spectral_types=None, **kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path) if tokenizer_path is not None else None
        self.spectral_types = spectral_types
        
    def process_smiles(self, batch, smiles_type='smiles'):
        smiles = [item[smiles_type] for item in batch]
        smiles = self.tokenizer(smiles, padding=True, return_tensors='pt', truncation=True)
        return smiles
    
    def process_sequence(self, batch):
        sequence = [item['sequence'] for item in batch]
        sequence = self.tokenizer(sequence, padding=True, return_tensors='pt', truncation=True)
        return sequence
    
    def process_spectra(self, batch, spectral_types='spectra'):
        if type(spectral_types) == str:
            spectra = torch.stack([torch.as_tensor(item[spectral_types]) for item in batch]).unsqueeze(1).to(torch.float32)
        elif type(spectral_types) == list:
            spectra = {}
            for spectral_type in spectral_types:
                spec = torch.stack([torch.as_tensor(item[spectral_type]) for item in batch]).unsqueeze(1).to(torch.float32)
                spectra[spectral_type] = spec
        return spectra
    
    def __call__(self, batch):
        batch_data = {}
        batch_data.update(self.process_spectra(batch, spectral_types=self.spectral_types))
        if 'sequence' not in batch[0]:
            batch_data['smiles'] = self.process_smiles(batch)
        else:
            batch_data['smiles'] = self.process_sequence(batch)
        batch_size = len(batch_data['smiles']['input_ids'])
        return {'batch_size':batch_size, 'target':None, 'data': batch_data}


class RXNCollator(BaseCollator):
    def __init__(self, tokenizer_path, spectral_types=None, smiles_types=None, mix_ratio=0.8):
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


class PeptideCollator(BaseCollator):
    def __init__(self, tokenizer_path, spectral_types=None):
        super().__init__(tokenizer_path, spectral_types)        
    
    def __call__(self, batch):
        batch_data = {}
        batch_data.update(self.process_spectra(batch, spectral_type=self.spectral_types))
        
        if 'sequence' in batch[0]:
            sequence = self.process_sequence(batch)
            batch_data['smiles'] = sequence # replace smiles with sequence 
        else:
            smiles = self.process_smiles(batch)
            batch_data['smiles'] = smiles
            
        batch_size = len(batch_data['smiles']['input_ids'])
        return {'batch_size':batch_size, 'target':None, 'data': batch_data}


class RankCollator(BaseCollator):
    def __init__(self, tokenizer_path, spectral_types=None):
        super().__init__(tokenizer_path)
        self.spectral_types = spectral_types
    
    def process_smiles_list(self, batch):
        max_len = max([len(item['smiles']) for item in batch])
        out_smiles_list = []
        for item in batch:
            if len(item['smiles']) < max_len:
                tmp_list = item['smiles'] + [''] * (max_len - len(item['smiles']))
            else:
                tmp_list = item['smiles']
            out_smiles_list += tmp_list
        out_smiles_list = self.tokenizer(out_smiles_list, padding=True, return_tensors='pt', truncation=True)
        return out_smiles_list, max_len
    
    def process_sequence_list(self, batch):
        max_len = max([len(item['sequence']) for item in batch])
        out_sequence_list = []
        for item in batch:
            if len(item['sequence']) < max_len:
                tmp_list = item['sequence'] + [''] * (max_len - len(item['sequence']))
            else:
                tmp_list = item['sequence']
            out_sequence_list += tmp_list
        out_sequence_list = self.tokenizer(out_sequence_list, padding=True, return_tensors='pt', truncation=True)
        return out_sequence_list, max_len
    
    def __call__(self, batch):
        batch_data = {}
        for spectral_type in self.spectral_types:
            spectra = self.process_spectra(batch, spectral_type=spectral_type)
            batch_data[spectral_type] = spectra
        if 'smiles' in batch[0]:
            batch_data['smiles'], batch_data['max_smiles_len'] = self.process_smiles_list(batch)
        elif 'sequence' in batch[0]:
            batch_data['smiles'], batch_data['max_smiles_len'] = self.process_sequence_list(batch)
        batch_size = len(batch_data['smiles']['input_ids']) // batch_data['max_smiles_len']
        return {'batch_size':batch_size, 'target':None, 'data': batch_data}
