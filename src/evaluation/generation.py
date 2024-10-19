import torch 
from torch.utils.data import DataLoader
from transformers import PretrainedConfig
from loader.model import load_model
from loader.checkpoint import load_pretrained_bag
from loader.data import _load_data, SimpleDataCollator
import numpy as np
from tqdm import tqdm

@torch.no_grad()
def generation_accuracy(model, dataloader, batch_size=8, max_length=1024, tokenizer=None, disable_tqdm=False):

    # load data
    # trainset = _load_data(f'{data_path}/data_{data_name}.train.lex.infix')
    # testset = _load_data(f'{data_path}/data_{data_name}.test.lex.infix')
    if isinstance(dataloader, str):
        dataset = _load_data(dataloader)
        dc = SimpleDataCollator(tokenizer)
        dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=dc, shuffle=False)
    
    # load model    
    if isinstance(model, str):
        bag = load_pretrained_bag(model)
        config, model, tokenizer = bag['config'], bag['model'], bag['tokenizer']
    else:
        assert(tokenizer is not None)
    
    model.cuda().eval()
    hits = []
    dataloader = tqdm(dataloader, disable=disable_tqdm) if not disable_tqdm else dataloader
    for batch in dataloader:
        max_length = min(max_length, batch['labels'].shape[1] + 1)
        outputs = model.greedy_generate(batch['encoder_input'].cuda(), 
                                        encoder_attention_mask=None,
                                        encoder_padding_mask=batch['encoder_padding_mask'].cuda(),
                                        max_length=max_length)
        pred = tokenizer.batch_decode(outputs.cpu().numpy(), skip_special_tokens=True)
        target = tokenizer.batch_decode(batch['labels'], skip_special_tokens=True)
        
        hits += [p == t for p, t in zip(pred, target)]
        
    ret = {'acc': np.array(hits, dtype=float).mean(), 'hits': hits}
    
    return ret 

        