import os, yaml
from time import time 
from joblib import Parallel, delayed
import numpy as np 
import pickle
from pprint import pprint
import timeout

import sys 
sys.path.append('src')

preparser(False)

from loader.data import _load_data
from loader.data import BartDataCollator
from torch.utils.data import DataLoader
from loader.checkpoint import load_pretrained_bag

from loader.data import _load_data

from joblib import Parallel, delayed

# load('src_dev/data/gbdataset.sage')

load('src/dataset/symbolic_utils.sage')
import argparse

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
        
def _timing_algo(dataset, gb_algorithm, verbose=True, timeout_limit=5, n_jobs=1):
    print(f'Using timeout with limit {timeout_limit}', flush=True)
    
    @timeout.timeout(duration=timeout_limit)
    def compute_gb(F, algorithm=''):
        ts = time()
        G = ideal(F).groebner_basis(algorithm=algorithm)
        runtime = time() - ts 
        return G, runtime
    
    def compute_gb_timeout(F, algorithm):
        try:
            G, runtime = compute_gb(F, algorithm=algorithm)
            return G, runtime, True
        except:
            return None, timeout_limit, False
    
    results = []
    for i, (F, G) in enumerate(dataset):
        if ideal(F).dimension() != 0: print(F)
        
        Gpred, runtime, success = compute_gb_timeout(F, gb_algorithm)
        if success: 
            assert(ideal(Gpred) == ideal(G))
            assert(ideal(Gpred) == ideal(F))
        
        if verbose and (i+1) % 100 == 0: print(f'[{i+1:04d} / {len(dataset)}]', flush=True)
        
        results.append({'id': int(i), 'F': str(F), 'G': str(G), 'runtime': float(runtime), 'success': success})    
    
    return results
        
        
def _timing_transformer(dataloader, model, tokenizer, verbose=True):
    # batch size is supposed to be 1

    results = []
    
    for i, batch in enumerate(dataloader):
        max_length = min(max_length, batch['labels'].shape[1] + 1)
        
        start = time()
        outputs = model.greedy_generate(batch['encoder_input'].cuda(), 
                                        encoder_attention_mask=None,
                                        encoder_padding_mask=batch['encoder_padding_mask'].cuda(),
                                        max_length=max_length)
        end = time()
        
        pred = tokenizer.batch_decode(outputs.cpu().numpy(), skip_special_tokens=True)
        target = tokenizer.batch_decode(batch['labels'], skip_special_tokens=True)

        runtime = end - start
        success = pred[0] == target[0]
        
        results.append({'id': int(i), 
                        # 'F': str(F), 
                        # 'G': str(G), 
                        'runtime': runtime, 
                        'success': success})
    
    return results
        
def timing_experiment(dataset, dataloader, gb_algorithms, model, tokenizer, verbose=True, timeout_limit=5):

    all_results = {}
    for gb_algorithm in gb_algorithms:
        print(f'### {gb_algorithm} ############################')
        results = _timing_algo(dataset, gb_algorithm, verbose=verbose, timeout_limit=timeout_limit)
        all_results[gb_algorithm] = results
        print()
        
        if verbose: pprint(results)
    
    print(f'### "Transformer" ############################')
    results = _timing_transformer(dataloader, model, tokenizer, verbose=verbose)
    
    all_results['transformer'] = results
    
    return all_results

def summarize_all_results(all_results, verbose=True):
    
    summary = {}
    for method, results in all_results.items():
        
        runtimes = [r['runtime'] for r in results]
        success_rate = np.mean([r['success'] for r in results])
        
        summary[f'fwd_runtime_{method}'] = float(np.sum(runtimes))
        summary[f'fwd_runtime_{method}_mean'] = float(np.mean(runtimes))
        summary[f'fwd_runtime_{method}_std'] = float(np.std(runtimes))
        summary[f'fwd_runtime_{method}_success_rate'] = float(success_rate)
            
    if verbose:
        pprint(summary)
        print('', flush=True)
        
    return summary


def get_parser():
    """
    Generate a parameters parser.
    """
    # parse parameters
    parser = argparse.ArgumentParser(description="Gröbner basis computation and timing")

    parser.add_argument("--dryrun", action='store_true', default=False)

    # path
    parser.add_argument("--data_path", type=str, help="Path to the input data")
    parser.add_argument("--model_path", type=str, default="results/", help="Path to model")
    parser.add_argument("--save_path", type=str, default="results/timing", help="Path to save results")

    # setup
    parser.add_argument("--field", type=str, default="QQ", help="Field for polynomial ring (e.g., QQ, F7, F31)")
    parser.add_argument("--num_variables", type=int, help="Number of variables in the polynomial ring")
    parser.add_argument("--timeout", type=float, default=5, help="Timeout duration for GB computation")

    return parser


def main():
    gb_algorithms=['libsingular:std', 'libsingular:slimgb', 'libsingular:stdfglm']
    
    parser = get_parser()
    args = parser.parse_args()

    if args.dryrun: 
        print('Dry run')
        return 
    
    num_variables = args.num_variables
    field = get_field(args.field)
    ring = PolynomialRing(field, num_variables, 'x', order='lex')

    dataset = _load_data(args.data_path)
    raw_dataset = get_raw_dataset(dataset, ring)
    
    os.makedirs(args.save_path, exist_ok=True)

    # bag = load_pretrained_bag(args.model_path)
    # model, tokenizer = bag['model'], bag['tokenizer']    
    # dc = BartDataCollator(tokenizer)
    # dataloader = DataLoader(dataset, batch_size=1, collate_fn=dc, shuffle=False)
    
    # all_results = timing_experiment(raw_dataset, dataloader, gb_algorithms, model, tokenizer, verbose=True, timeout_limit=args.timeout)
    # summary = summarize_all_results(all_results, verbose=True)
    
    all_results = {}
    for gb_algorithm in gb_algorithms:
        print(f'### {gb_algorithm} ############################')
        results = _timing_algo(raw_dataset, gb_algorithm, verbose=True, timeout_limit=args.timeout)
        all_results[gb_algorithm] = results
        print('')
        
    summary = summarize_all_results(all_results, verbose=True)
    
    pprint(summary)
        
    import yaml
    save_path = os.path.join(args.save_path, f'timing_details.yaml')
    with open(save_path, 'w', encoding='utf-8') as f:
        yaml.dump(all_results, f, allow_unicode=True)
    
    save_path = os.path.join(args.save_path, f'timing.yaml')
    with open(save_path, 'w', encoding='utf-8') as f:
        yaml.dump(summary, f, allow_unicode=True)


if __name__ == '__main__':
    main()
    
    
    # # postfix = '_density=1'
    # # postfix = ''
    # # assert len(sys.argv) > 1
    # # postfix = sys.argv[-1]
    # # assert postfix in ('', '_density=1')
    
    # postfix = '_density=1'
    # save_dir = 'results/timing/'
    # if postfix: save_dir = os.path.join(save_dir, postfix[1:])
    
    # N = 1000
    
    # gb_algorithms = ['libsingular:std', 
    #                  'libsingular:slimgb', 
    #                  'libsingular:stdfglm', 
    #                 #  'singular:std', 'singular:slimgb', 'singular:stdfglm',  # slow
    #                 ]
    
    # os.makedirs(save_dir, exist_ok=True)
        
    
    # if sys.argv[-1] != 'dryrun': 
    #     for field in ['QQ', 'F7', 'F31']:
    #         config_paths = [f'config/gb_dataset_n={n}_field={field}{postfix}.yaml' for n in range(2, 6)]
    #         filename_postfix = f'_field={field}{postfix}_N={N}'
            
    #         pprint(config_paths)
            
    #         timing_experiment(N, config_paths, 
    #                         gb_algorithms=gb_algorithms, 
    #                         verbose=True,
    #                         save_dir=save_dir,
    #                         filename_postfix=filename_postfix)

            
    #         print('\n')
