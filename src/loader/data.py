import os 
import torch 
from torch.utils.data import Dataset, DataLoader, IterableDataset
from typing import List
import numpy as np 

def get_datacollator(model_name):
    if model_name == 'bart':
        return BartDataCollator
    elif model_name == 'bart+':
        return BartPlusDataCollator
    else:
        raise ValueError(f'invalid model name: {model_name}')

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
        input_texts = [item["input"] for item in batch]
        target_texts = [item["target"] for item in batch]
        
        input_encodings = self.tokenizer(input_texts, padding='longest', return_tensors='pt')
        target_encodings = self.tokenizer(target_texts, padding='longest', return_tensors='pt')

        return {
            'encoder_input': input_encodings['input_ids'] ,
            'decoder_input': target_encodings['input_ids'],
            'encoder_padding_mask': ~input_encodings['attention_mask'].bool(),  # NOTE: attantion mask given by tokenizer is 0/1 multiplicative mask (0 for no attention) but transformer use bool mask (True for no attention)
            'decoder_padding_mask': ~target_encodings['attention_mask'].bool(),
            'labels': target_encodings['input_ids'].contiguous(),
        }
    


# def _preprocess_coefficients(input_text: str):
    
#     tokens = input_text.split()
#     c_labels = [str_to_float(t[1:]) if t[0] == 'C' else np.nan for t in tokens]
    
#     for i, _ in enumerate(tokens):
#         if c_labels[i] is not np.nan:
#             tokens[i] = '[C]'

#     text = ' '.join(tokens)
    
#     return (text, c_labels)
    
# def preprocess_coefficients(input_texts: List[str]):
#     return [_preprocess_coefficients(it) for it in input_texts]


def _preprocess_coefficients(input_text: str, continuous_coefficient=False, continuous_exponent=False):
    
    tokens = input_text.split()
    c_labels, e_labels = None, None
    if continuous_coefficient:    
        c_labels = [str_to_float(t[1:]) if t[0] == 'C' else np.nan for t in tokens]
    if continuous_exponent: 
        e_labels = [str_to_float(t[1:]) if t[0] == 'E' else np.nan for t in tokens]
    
    for i, _ in enumerate(tokens):
        if continuous_coefficient and c_labels[i] is not np.nan:
            tokens[i] = '[C]'
        if continuous_exponent and e_labels[i] is not np.nan:
            tokens[i] = '[E]'

    text = ' '.join(tokens)
    
    return text, c_labels, e_labels
    
def preprocess_coefficients(input_texts: List[str], continuous_coefficient=False, continuous_exponent=False):
    return [_preprocess_coefficients(it, 
                                     continuous_coefficient=continuous_coefficient, 
                                     continuous_exponent=continuous_exponent) 
            for it in input_texts]
    
    

class HybridDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        input_texts = [item["input"] for item in batch]
        target_texts = [item["target"] for item in batch]

        eos = self.tokenizer.eos_token
        input_texts = [item["input"] for item in batch]
        target_texts = [item["target"] for item in batch]
        
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


class BartDataCollator:
    def __init__(self, tokenizer, **kwargs):
        self.tokenizer = tokenizer

    @torch.no_grad()
    def __call__(self, batch):
        input_texts = [item["input"] for item in batch]
        target_texts = [item["target"] for item in batch]
        
        input_encodings = self.tokenizer(input_texts, padding='longest', return_tensors='pt')
        target_encodings = self.tokenizer(target_texts, padding='longest', return_tensors='pt')

        return {
            'input_ids': input_encodings['input_ids'] ,
            'decoder_input_ids': target_encodings['input_ids'][:, :-1].contiguous(),
            'attention_mask': input_encodings['attention_mask'],  # NOTE: attantion mask given by tokenizer is 0/1 multiplicative mask (0 for no attention) but transformer use bool mask (True for no attention)
            'decoder_attention_mask': target_encodings['attention_mask'][:, :-1].contiguous(),
            'labels': target_encodings['input_ids'][:, 1:].contiguous(),
        }
        # return {
        #     "input_ids"             : input_encodings['input_ids'],
        #     "attention_mask"        : ~input_encodings['attention_mask'].bool(),  # NOTE: attantion mask given by tokenizer is 0/1 multiplicative mask (0 for no attention) but transformer use bool mask (True for no attention)
        #     "decoder_input_ids"     : target_encodings['input_ids'],
        #     "decoder_attention_mask": ~target_encodings['attention_mask'].bool(),
        #     "labels": input_encodings['input_ids']
        # }
    
    
class BartPlusDataCollator:
    def __init__(self, tokenizer, continuous_coefficient=False, continuous_exponent=False, support_learning=False):
        self.tokenizer = tokenizer
        self.continuous_coefficient = continuous_coefficient
        self.continuous_exponent    = continuous_exponent
        self.support_learning = support_learning

    def __call__(self, batch):
        input_texts = [item["input"] for item in batch]
        target_texts = [item["target"] for item in batch]

        # print(f'target text: {target_texts[0]}', flush=True)
        
        input_texts, input_coeff_labels, input_expo_labels = zip(*preprocess_coefficients(input_texts, 
                                                                                        continuous_coefficient=self.continuous_coefficient or self.support_learning, 
                                                                                        continuous_exponent=self.continuous_exponent)
                                                                                        )
        target_texts, target_coeff_labels, target_expo_labels = zip(*preprocess_coefficients(target_texts, 
                                                                                        continuous_coefficient=self.continuous_coefficient or self.support_learning, 
                                                                                        continuous_exponent=self.continuous_exponent)
                                                                                        )

        input_encodings     = self.tokenizer(input_texts, padding='longest', return_tensors='pt')
        target_encodings    = self.tokenizer(target_texts, padding='longest', return_tensors='pt')
        
        input_ids = input_encodings['input_ids'] 
        target_ids = target_encodings['input_ids']             
        
        attention_mask = input_encodings['attention_mask'].bool()  # NOTE: attantion mask given by tokenizer is 0/1 multiplicative mask (0 for no attention) but transformer use bool mask (True for no attention)
        target_attention_mask = target_encodings['attention_mask'].bool()[:, :-1].contiguous()

        length_in, length_tar = input_ids.shape[-1], target_ids.shape[-1]
        input_continuous_labels, target_continuous_labels = [], []
        
        if self.continuous_coefficient:
            input_coeff_labels  = torch.tensor([[np.nan] + t + [np.nan]*(length_in - len(t) - 1) for t in input_coeff_labels])
            target_coeff_labels = torch.tensor([[np.nan] + t + [np.nan]*(length_tar - len(t) - 1) for t in target_coeff_labels])
            input_continuous_labels.append(input_coeff_labels.unsqueeze(-1))
            target_continuous_labels.append(target_coeff_labels.unsqueeze(-1))

        if self.continuous_exponent:
            input_expo_labels  = torch.tensor([[np.nan] + t + [np.nan]*(length_in - len(t) - 1) for t in input_expo_labels])
            target_expo_labels = torch.tensor([[np.nan] + t + [np.nan]*(length_tar - len(t) - 1) for t in target_expo_labels])
            input_continuous_labels.append(input_expo_labels.unsqueeze(-1))
            target_continuous_labels.append(target_expo_labels.unsqueeze(-1))
        
        if len(input_continuous_labels) > 0:
            input_continuous_labels = torch.cat(input_continuous_labels, dim=-1)
            target_continuous_labels = torch.cat(target_continuous_labels, dim=-1)
        else:
            input_continuous_labels = torch.empty(*input_ids.shape, 0)
            target_continuous_labels = torch.empty(*target_ids.shape, 0)
        
        if self.support_learning: 
            input_continuous_labels     = torch.empty(*input_ids.shape, 0)
            target_continuous_labels    = torch.empty(*target_ids.shape, 0)
        
        labels_for_regression       = target_continuous_labels[:, 1:].contiguous() 
        target_continuous_labels    = target_continuous_labels[:, :-1].contiguous()

        return {
            "input_ids"                 : input_ids,
            "attention_mask"            : attention_mask,
            "decoder_input_ids"         : target_ids[:, :-1].contiguous(),
            "decoder_attention_mask"    : target_attention_mask,
            "labels"                    : target_ids[:, 1:].contiguous(),
            "labels_for_regression"     : labels_for_regression,
            "input_continuous_labels"   : input_continuous_labels,
            "target_continuous_labels"  : target_continuous_labels,
        }
