import torch 
from torch.utils.data import DataLoader
from transformers import PretrainedConfig
from loader.model import load_model
from loader.checkpoint import load_pretrained_bag
from loader.data import _load_data, get_datacollator
import numpy as np
from tqdm import tqdm
from misc.utils import to_cuda
from time import time 

def generation(model, model_name, batch, tokenizer, max_length=1024, quantize_fn=None):
    
    if model_name == 'bart':
        outputs = model.generate(batch['input_ids'], attention_mask=batch['attention_mask'],
                                 max_length=batch['input_ids'].shape[-1] + max_length, num_beams=1, do_sample=False)
        preds = tokenizer.batch_decode(outputs.long().cpu().numpy(), skip_special_tokens=True)
        
    if model_name == 'bart+':
        outputs = model.generate(batch['input_ids'].cuda(), 
                                 input_continuous_labels = batch['input_continuous_labels'].cuda(),
                                 attention_mask=batch['attention_mask'].cuda(),
                                 max_length=max_length,
                                 quantize_fn=quantize_fn,
                                 continuous_token_ids=[tokenizer.vocab['[C]']])
        
        generation_output = model.postprocess_prediction(outputs, tokenizer, skip_special_tokens=True)
        preds = generation_output['prediction_texts']
        
        # cpreds = generation_output['continuous_prediction']
        # tar = batch['labels_for_regression']
        # print(cpreds[0][cpreds[0].isfinite()], tar[0][tar[0].isfinite()])
    
    return preds

def support_eq(pred, label):

    if isinstance(pred, str):
        pred, label = pred.split(' '), label.split(' ')
    
    if len(pred) != len(label):
        return False 
    
    return np.all([p == l for p, l in zip(pred, label) if l != '[C]' and l[0] != 'C' ])

def coeffs_eq(pred, labels_for_regression, th=0, modulo=None):
    target = np.array(labels_for_regression[np.isfinite(labels_for_regression)])
    pred = np.array([float(p[1:]) for p in pred.split(' ') if p[0] == 'C'])
    
    if len(pred) != len(target): 
        return False, np.array([])
    
    if modulo is not None: 
        pred = np.array(pred.round() % modulo, dtype=int)
        
    delta = np.abs(target - pred) 
    
    return np.all(delta <= th), delta
    

def polynomial_eq(pred, label, label_for_regression=None, th=0, modulo=None):
    support_hit = support_eq(pred, label)
    
    if label_for_regression is not None:
        coeff_hit, _ = coeffs_eq(pred, label_for_regression, th=th, modulo=modulo)
        hit = support_hit and coeff_hit
    else:
        hit = pred == label
        
    return bool(hit), bool(support_hit)
    
def accuracy_score(preds, labels, labels_for_regression=None, th=0, modulo=None):
    
    if labels_for_regression is None:
        labels_for_regression = [None] * len(labels)
    
    hits = []
    support_hits = []
    for pred, label, label_rg in zip(preds, labels, labels_for_regression):
        hit, support_hit = polynomial_eq(pred, label, label_for_regression=label_rg, th=th, modulo=modulo)
        hits.append(hit)
        support_hits.append(support_hit)
    
    acc = np.array(hits, dtype=float).mean()
    support_acc = np.array(support_hits, dtype=float).mean()
    
    return {'acc': acc, 
            'support_acc': support_acc,
            'hits': hits, 
            'support_hits': support_hits}

@torch.no_grad()
def generation_accuracy(model, dataloader, batch_size=8, max_length=1024, tokenizer=None, th=0, disable_tqdm=False, modulo=None, model_name=None, quantize_fn=None):
    
    # load model    
    if isinstance(model, str):
        bag = load_pretrained_bag(model)
        model, tokenizer, model_name = bag['model'], bag['tokenizer'], bag['model_name']
        
    if isinstance(dataloader, str):
        assert(tokenizer is not None)
        dataset = _load_data(dataloader)
        dc = get_datacollator(model_name)(tokenizer, continuous_coefficient=True)
        dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=dc, shuffle=False)

    
    model.cuda().eval()
    hits = []
    support_hits = []
    runtimes = []
    dataloader = tqdm(dataloader, disable=disable_tqdm) if not disable_tqdm else dataloader
    for batch in dataloader:
        batch = to_cuda(batch)
        
        max_length = min(max_length, batch['labels'].shape[-1] + 5)
        
        start = time()
        preds = generation(model, model_name, batch, tokenizer, max_length=max_length, quantize_fn=quantize_fn)
        end = time()
        runtime = end - start
        
        targets = tokenizer.batch_decode(batch['labels'], skip_special_tokens=True)
        
        labels_for_regression = batch['labels_for_regression'].detach().cpu().numpy() if 'labels_for_regression' in batch else None
        results = accuracy_score(preds, 
                                 targets, 
                                 labels_for_regression=labels_for_regression, 
                                 th=th,
                                 modulo=modulo)
        
        hits.extend(results['hits'])
        support_hits.extend(results['support_hits'])
        runtimes.extend([runtime] * len(preds))
        
    acc = np.array(hits, dtype=float).mean()
    support_acc = np.array(support_hits, dtype=float).mean()
    
    results = {'acc': float(acc), 
               'support_acc': float(support_acc), 
               'hits': hits, 
               'support_hits': support_hits,
               'batch_runtimes': runtimes,
               'runtime_per_batch': float(np.mean(runtimes))}
    
    
    
    return results 

        