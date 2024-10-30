
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

os.environ["TOKENIZERS_PARALLELISM"] = "false"

sys.path.append('src')
from loader.data import _load_data
from evaluation.generation import generation_accuracy
load('src/dataset/symbolic_utils.sage')


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
        
def _timing_algo(dataset, gb_algorithm, verbose=True, timeout_limit=5):
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
        except Exception as e:
            return None, timeout_limit, False
    
    results = []
    iterator = tqdm(enumerate(dataset), disable=not verbose) if verbose else enumerate(dataset)
    for i, (F, G) in iterator:
        Gpred, runtime, success = compute_gb_timeout(F, gb_algorithm)
        results.append({'id': int(i), 'runtime': float(runtime), 'success': success, 'F': str(F), 'G': str(G)})    
    
    return results
        
        
def _timing_transformer(model_path, data_path, max_length, modulo, verbose=True):

    _results = generation_accuracy(model_path, 
                                   data_path, 
                                   batch_size=int(1),  # batch size is fixed to 1 here
                                   max_length=max_length,
                                   modulo=modulo,
                                   disable_tqdm=not verbose)
        
    results = [{'id': int(i), 
                'runtime': _results['batch_runtimes'][i], 
                'success': _results['hits'][i]} for i in range(len(_results['hits']))]
    
    return results

def find_transformer_supremacy(all_results):
    
    print('### Listing samples where either of the algorithms timeouts and transformer succeeds', flush=True)
    print('--------------------------------------------------', flush=True)
    print('method                 runtime (sec)   success')

    gb_methods = [m for m in all_results if m != 'transformer']
    num_samples = len(all_results['transformer'])
    
    hard_sample_results = []
    for i in range(num_samples):
        all_algo_success = np.all([all_results[m][i]['success'] for m in gb_methods])
        transformer_sucess = all_results['transformer'][i]['success']
        
        if not all_algo_success and transformer_sucess:
            print('--------------------------------------------------', flush=True)
            id = all_results['transformer'][i]['id']
            print(f'"#sample id" {id}', flush=True)
            for method in all_results:
                runtime = all_results[method][i]['runtime']
                success = all_results[method][i]['success']
                print(f'{method:<20s}: {runtime:>13.4f}   [{success}]')
            
            hard_sample_results.append({ 
                'id': id,
                'F': all_results[gb_methods[0]][i]['F'],
                'G': all_results[gb_methods[0]][i]['G'],
                'runtime': {m: all_results[m][i]['runtime'] for m in all_results},
                'success': {m: all_results[m][i]['success'] for m in all_results}
            })

    print('--------------------------------------------------\n', flush=True)
    
    return hard_sample_results
            
    
def summarize_all_results(all_results, verbose=True):
    
    summary = {}
    for method, results in all_results.items():
        
        runtimes = [r['runtime'] for r in results]
        success_rate = np.mean([int(r['success']) for r in results])
        
        summary[f'{method}_runtime'] = float(np.sum(runtimes))
        summary[f'{method}_runtime_mean'] = float(np.mean(runtimes))
        summary[f'{method}_runtime_std'] = float(np.std(runtimes))
        summary[f'{method}_success_rate'] = float(success_rate)
            
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
    parser.add_argument("--model_path", type=str, default=None, help="Path to model")
    parser.add_argument("--save_path", type=str, default="results/timing", help="Path to save results")

    # setup
    parser.add_argument("--field", type=str, default="QQ", help="Field for polynomial ring (e.g., QQ, F7, F31)")
    parser.add_argument("--num_variables", type=int, help="Number of variables in the polynomial ring")
    parser.add_argument("--timeout", type=float, default=5, help="Timeout duration for GB computation")
    
    parser.add_argument("--max_length", type=int, default=1024, help="Maximum length for generation")

    return parser


def main():
    gb_algorithms=['libsingular:std', 'libsingular:slimgb', 'libsingular:stdfglm']
    
    parser = get_parser()
    args = parser.parse_args()

    num_variables = args.num_variables
    field = get_field(args.field)
    ring = PolynomialRing(field, num_variables, 'x', order='lex')

    dataset = _load_data(args.data_path)
    raw_dataset = get_raw_dataset(dataset, ring)
    
    os.makedirs(args.save_path, exist_ok=True)

    all_results = {}
    if args.model_path is not None:
        print(f'### "Transformer" ############################')
        modulo = int(args.field[2:]) if args.field.startswith('GF') else None
        results = _timing_transformer(args.model_path, args.data_path, args.max_length, modulo, verbose=True)
        all_results['transformer'] = results
        print()
    
    for gb_algorithm in gb_algorithms:
        print(f'### {gb_algorithm} ############################')
        results = _timing_algo(raw_dataset, gb_algorithm, verbose=True, timeout_limit=args.timeout)
        all_results[gb_algorithm] = results
        print()
        
    save_path = os.path.join(args.save_path, f'timing_details.yaml')
    with open(save_path, 'w', encoding='utf-8') as f:
        yaml.dump(all_results, f, allow_unicode=True)


    summary = summarize_all_results(all_results, verbose=True)
    
    print('--------------------------------------------------')
    print('method                 runtime (sec)  success rate')
    print('--------------------------------------------------')
    for method in all_results:
        print(f'{method:<20s}  {summary[f"{method}_runtime"]:>13.2f}  {summary[f"{method}_success_rate"]:>12.3f}')
    print('--------------------------------------------------', flush=True)
    
    save_path = os.path.join(args.save_path, f'timing.yaml')
    with open(save_path, 'w', encoding='utf-8') as f:
        yaml.dump(summary, f, allow_unicode=True)

    ## hard sample
    if args.model_path is not None:
        hard_sample_results = find_transformer_supremacy(all_results)
        
        save_path = os.path.join(args.save_path, f'transformer_supermacy.yaml')
        with open(save_path, 'w', encoding='utf-8') as f:
            yaml.dump(hard_sample_results, f, allow_unicode=True)

        timeout_limit = args.timeout * 20
        
        dataset = [raw_dataset[s['id']] for s in hard_sample_results]
        for gb_algorithm in gb_algorithms:
            print(f'### {gb_algorithm} ############################')
            results = _timing_algo(dataset, gb_algorithm, verbose=True, timeout_limit=timeout_limit)
            print()
            
        for i, ret in enumerate(results):
            for gb_algorithm in gb_algorithms:
                hard_sample_results[i]['runtime'][gb_algorithm] = ret['runtime']
        
        save_path = os.path.join(args.save_path, f'transformer_supermacy_timeout={timeout_limit}.yaml')
        with open(save_path, 'w', encoding='utf-8') as f:
            yaml.dump(hard_sample_results, f, allow_unicode=True)
            
    with open(os.path.join(args.save_path, "args.yaml"), "w") as f:
        yaml.dump(vars(args), f)

if __name__ == '__main__':
    main()
