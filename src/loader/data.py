import os 
import torch 
from torch.utils.data import Dataset, DataLoader, IterableDataset
from typing import List
import numpy as np 


def _load_data(data_path):
    try:
        with open(data_path, "r") as f:
            data = f.read().splitlines()
    except:
        raise FileNotFoundError
    
    input_texts = [line.split(":")[0].strip() for line in data]
    target_texts = [line.split(":")[1].strip() for line in data]

    dataset = DictDataset(input_texts, target_texts)
    
    return dataset

def load_data(data_path, 
            encoding='prefix', 
            batch_sizes=[4, 100], 
            return_dataloader=True, 
            extensions=['train', 'test'], 
            do_shuffle=[True, False], 
            tokenizer=None,
            continuous_coefficient=True,
            continuous_exponent=False,
            support_learning=False,):
    
    
    ret = []
    for ext, batch_size, shuffle in zip(extensions, batch_sizes, do_shuffle):
        path = f"{data_path}.{ext}"
        print(f'loading ... {path}')
        if encoding: path = path + f'.{encoding}'
        dataset = _load_data(path)

        if return_dataloader: 
            data_collator = DataCollator(tokenizer, continuous_coefficient=continuous_coefficient, continuous_exponent=continuous_exponent, support_learning=support_learning)
            print(f'content of batch_size: {batch_size}', flush=True)
            dataset = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=os.cpu_count(), pin_memory=True, collate_fn=data_collator)

        ret.append(dataset)

    return ret[0] if len(ret) == 1 else ret

class DictDataset(Dataset):
    def __init__(self, input_texts, target_texts, tokenizer=None):
        self.tokenizer = tokenizer

        input_ = input_texts if tokenizer is None else tokenizer(input_texts, padding='longest', return_tensors='pt')
        target = target_texts if tokenizer is None else tokenizer(target_texts, padding='longest', return_tensors='pt')
        
        self.input = input_ if tokenizer is None else input_['input_ids']
        self.input_mask = None if tokenizer is None else input_['attention_mask'].bool()
        self.target = target if tokenizer is None else target['input_ids']
        self.target_mask = None if tokenizer is None else target['attention_mask'].bool()
        
    def __len__(self):
        return len(self.input)

    def __getitem__(self, idx):
        return {
            "input": self.input[idx],
            "target": self.target[idx],
            "input_mask": self.input_mask[idx] if self.tokenizer is not None else None,
            "target_mask": self.target_mask[idx] if self.tokenizer is not None else None,
        }

def str_to_float(s):
    try:
        return float(s)
    except:
        if '/' in s: 
            a, b = s.split('/')
            return float(a) / float(b)

        raise ValueError(f'invalid string: {s}')

class SimpleDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    @torch.no_grad()
    def __call__(self, batch):
        eos = self.tokenizer.eos_token
        input_texts = [item["input"] + f' {eos}' for item in batch]
        target_texts = [item["target"] + f' {eos}' for item in batch]
        
        input_encodings = self.tokenizer(input_texts, padding='longest', return_tensors='pt')
        target_encodings = self.tokenizer(target_texts, padding='longest', return_tensors='pt')

        return {
            'encoder_input': input_encodings['input_ids'] ,
            'decoder_input': target_encodings['input_ids'],
            'encoder_padding_mask': ~input_encodings['attention_mask'].bool(),  # NOTE: attantion mask given by tokenizer is 0/1 multiplicative mask (0 for no attention) but transformer use bool mask (True for no attention)
            'decoder_padding_mask': ~target_encodings['attention_mask'].bool(),
            'labels': target_encodings['input_ids'].contiguous(),
        }
    


def _preprocess_coefficients(input_text: str):
    
    tokens = input_text.split()
    c_labels = [str_to_float(t[1:]) if t[0] == 'C' else np.nan for t in tokens]
    
    for i, _ in enumerate(tokens):
        if c_labels[i] is not np.nan:
            tokens[i] = '[C]'

    text = ' '.join(tokens)
    
    return (text, c_labels)
    
def preprocess_coefficients(input_texts: List[str]):
    return [_preprocess_coefficients(it) for it in input_texts]


class HybridDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        input_texts = [item["input"] for item in batch]
        target_texts = [item["target"] for item in batch]

        eos = self.tokenizer.eos_token
        input_texts = [item["input"] + f' {eos}' for item in batch]
        target_texts = [item["target"] + f' {eos}' for item in batch]
        
        input_texts, input_coeff_labels = list(zip(*preprocess_coefficients(input_texts)))
        target_texts, target_coeff_labels = list(zip(*preprocess_coefficients(target_texts)))
        
        input_encodings     = self.tokenizer(input_texts, padding='longest', return_tensors='pt')
        target_encodings    = self.tokenizer(target_texts, padding='longest', return_tensors='pt')
        
        input_ids = input_encodings['input_ids'] 
        target_ids = target_encodings['input_ids']             
        
        length_in, length_tar = input_ids.shape[-1], target_ids.shape[-1]
        
        input_continuous_labels  = torch.tensor([t + [np.nan]*(length_in - len(t) ) for t in input_coeff_labels]).contiguous()
        target_continuous_labels = torch.tensor([t + [np.nan]*(length_tar - len(t)) for t in target_coeff_labels]).contiguous()
        # input_continuous_labels.append(input_coeff_labels.unsqueeze(-1))
        # target_continuous_labels.append(target_coeff_labels.unsqueeze(-1))
        
        return {
            'encoder_input': input_encodings['input_ids'] ,
            'decoder_input': target_encodings['input_ids'],
            'encoder_padding_mask': ~input_encodings['attention_mask'].bool(),  # NOTE: attantion mask given by tokenizer is 0/1 multiplicative mask (0 for no attention) but transformer use bool mask (True for no attention)
            'decoder_padding_mask': ~target_encodings['attention_mask'].bool(),
            'labels': target_encodings['input_ids'].contiguous(),
            'labels_for_regression': target_continuous_labels,
            'encoder_input_labels': input_continuous_labels,
            'decoder_input_labels': target_continuous_labels,
        }

        return {
            "input_ids"                 : input_ids,
            "attention_mask"            : attention_mask,
            "decoder_input_ids"         : target_ids[:, :-1].contiguous(),
            "decoder_attention_mask"    : target_attention_mask,
            "labels"                    : labels.contiguous(),
            "continuous_labels"         : continuous_labels,
            "input_continuous_labels"   : input_continuous_labels,
            "target_continuous_labels"  : target_continuous_labels,
        }
