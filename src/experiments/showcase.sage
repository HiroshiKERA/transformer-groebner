
import os
import sys
import argparse
from time import time
from pprint import pprint

import numpy as np
import pickle
import yaml
import timeout
from tqdm import tqdm
from torch.utils.data import DataLoader

os.environ["TOKENIZERS_PARALLELISM"] = "false"

sys.path.append('src')
from loader.data import _load_data, get_datacollator
from loader.checkpoint import load_pretrained_bag
from evaluation.generation import generation
from misc.utils import to_cuda
load('src/dataset/symbolic_utils.sage')
load('src/experiments/paper_writing_utils.sage')

def get_raw_dataset(dataset, ring):
    raw_dataset = []
    for sample in dataset:
        Fstr, Gstr = sample['input'], sample['target']
        F, G = [], []
        
        for fstr in Fstr.split(' [SEP] '):
            f = sequence_to_poly(fstr, ring)
            F.append(f)
        
        for gstr in Gstr.split(' [SEP] '):
            g = sequence_to_poly(gstr, ring)
            G.append(g)
            
        raw_dataset.append((F, G))

    return raw_dataset
        
        


def get_parser():
    """
    Generate a parameters parser.
    """
    # parse parameters
    parser = argparse.ArgumentParser(description="Gröbner basis computation and timing")

    # path
    parser.add_argument("--data_path", type=str, help="Path to the input data")
    parser.add_argument("--model_path", type=str, default=None, help="Path to model")
    parser.add_argument("--generation_results", type=str, help="File of generation results")

    # setup
    parser.add_argument("--field", type=str, default="QQ", help="Field for polynomial ring (e.g., QQ, F7, F31)")
    parser.add_argument("--num_variables", type=int, help="Number of variables in the polynomial ring")
    parser.add_argument("--max_length", type=int, default=1024, help="Maximum length for generation")
    
    parser.add_argument("--num_samples_to_show", type=int, default=10, help="Number of samples to show")
    parser.add_argument('--print_style', type=str, default='sage', choices=['sage', 'latex'], help='Print style')

    return parser


def main():
    
    parser = get_parser()
    args = parser.parse_args()

    # get model
    bag = load_pretrained_bag(args.model_path)
    model, tokenizer, model_name = bag['model'], bag['tokenizer'], bag['model_name']
    assert(model_name == 'bart')  # this following script does not assume hybrid embedding
    
    # define ring 
    num_variables = args.num_variables
    field = get_field(args.field)
    ring = PolynomialRing(field, num_variables, 'x', order='lex')
    
    # get data
    dataset = _load_data(args.data_path)
    raw_dataset = get_raw_dataset(dataset, ring)  # sage format
    dc = get_datacollator(model_name)(tokenizer, continuous_coefficient=True)
    
    # load results of generation.py
    print(args.generation_results)
    with open(args.generation_results, 'r', encoding='utf-8') as f:
        generation_results = yaml.load(f, Loader=yaml.FullLoader)
        
    hits = generation_results['hits']
    
    success_samples = [(i, (F, G, None)) for i, (F, G) in enumerate(raw_dataset) if hits[i]]
    
    failed_samples = [(i, (F, G, None)) for i, (F, G) in enumerate(raw_dataset) if not hits[i]]
    failed_dataset = [sample for i, sample in enumerate(dataset) if not hits[i]]
    dataloader = DataLoader(failed_dataset, batch_size=args.num_samples_to_show, collate_fn=dc, shuffle=False)
    
    batch = next(iter(dataloader))
    batch = to_cuda(batch)
    preds = generation(model, model_name, batch, tokenizer, max_length=args.max_length)
    Gpreds = [[sequence_to_poly(g, ring) for g in pred.split(' [SEP] ')] for pred in preds]
    
    for i, Gp in enumerate(Gpreds):
        id, (F, G, _) = failed_samples[i]
        failed_samples[i] = (id, (F, G, Gp))
        
    print('-----------------------------------')
    print('# Success samples (F-G view)')
    print('-----------------------------------')
    print_cases(success_samples[:args.num_samples_to_show], cmp_style='FG', latex_form=args.print_style == 'latex')
    print('\n\n\n')
    
    print('-----------------------------------')
    print('# Failsed samples (F-G view)')
    print('-----------------------------------')
    print_cases(failed_samples[:args.num_samples_to_show], cmp_style='FG', latex_form=args.print_style == 'latex')
    print('\n\n\n')
    
    print('-----------------------------------')
    print('# Failsed samples (G-G\' view)')  # GT vs Transformer
    print('-----------------------------------')
    print_cases(failed_samples[:args.num_samples_to_show], cmp_style='GG', latex_form=args.print_style == 'latex')
    print()
        

if __name__ == '__main__':
    main()
