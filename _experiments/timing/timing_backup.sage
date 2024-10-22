import sys, os
import time 
from joblib import Parallel, delayed
import numpy as np 
from pprint import pprint
import timeout

load('src/data/gbdataset.sage')
import argparse

timeout_duration = 10

def get_parser():
    """
    Generate a parameters parser.
    """
    # parse parameters
    parser = argparse.ArgumentParser(description="Language transfer")

    # path
    parser.add_argument("--data_path", type=str, default="data/")
    parser.add_argument("--load_path", type=str, default="results/")
    parser.add_argument("--save_path", type=str, default="results/timing")
    
    # setup
    parser.add_argument("--field", type=str, default="QQ")
    parser.add_argument("--timeout_duration", type=int, default=10)

    parser.add_argument("--dryrun", action='store_true', default=False)
    
    return parser



def load_raw_dataset(path, ring):
    with open(path, 'r') as f:
        data = f.read().splitlines()
    input_texts = [line.split(":")[0].strip() for line in data]
    target_texts = [line.split(":")[1].strip() for line in data]
    
    Fs, Gs = [], []
    if ring is not None:
        for text in input_texts:
            F = text.split(' [SEP] ')
            F = [ring(f) for f in F]
        for text in target_texts:
            G = text.split(' [SEP] ')
            G = [ring(g) for g in G]
        
        Fs.append(F)
        Gs.append(G)
        
    return (Fs, Gs)
        
def compute_gb(F, algorithm=''):
    ts = time()
    G = ideal(F).groebner_basis(algorithm=algorithm)
    runtime = time() - ts 
    return G, runtime

@timeout.timeout(duration=timeout_duration)
def _compute_gb_with_to(F, algorithm=''):
    ts = time()
    G = ideal(F).groebner_basis(algorithm=algorithm)
    runtime = time() - ts 
    return G, runtime

def compute_gb_with_to(F, algorithm=''):
    try:
        G, runtime = _compute_gb_with_to(F, algorithm=algorithm)
    except:
        # print(sys.exc_info())
        G, runtime = None, timeout_duration
        pass
    
    return G, runtime, (G is not None)

def timing_experiment(timing_n, config_paths, gb_algorithms=[''], verbose=True, save_dir=None, filename_postfix=''):
    
    timing_results = []
    dataset_profiles = []
    for config_path in config_paths:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        timing_result = {
            'config_file': os.path.basename(config_path),
            'num_samples': int(timing_n)
            }
        dataset_profile = {
            'config_file': os.path.basename(config_path),
            'num_samples': int(timing_n)
            }
        
        char = config['order']
        field = GF(char)
        ring = PolynomialRing(field, config['num_var'], 'x', order='lex')    

        dbuilder = GBDataset_Builder(ring, 
                                    max_size           = config['max_size'], 
                                    max_degree         = config['max_degree'], 
                                    max_num_terms      = config['max_num_terms'], 
                                    max_Gdegree        = config['max_Gdegree'], 
                                    max_num_Gterms     = config['max_num_Gterms'], 
                                    num_duplicants     = config['num_duplicants'], 
                                    density            = config['density'], 
                                    with_permutation   = config['with_permutation'])

        dbuilder.run(timing_n, run_check=False, timing=False)
        dataset, runtimes = dbuilder.dataset, dbuilder.runtime_stat
        stats_F, stats_G = dbuilder.stats
        stats_F_sum = dbuilder.summarize_stats(stats_F)
        stats_G_sum = dbuilder.summarize_stats(stats_G)
        dataset_profile['F'] = stats_F_sum
        dataset_profile['G'] = stats_G_sum
        dataset_profiles.append(dataset_profile)
        
        if verbose:
            print('-- Datset profile -----')
            pprint(dataset_profile)
        
        timing_result['bwd_runtime'] = float(runtimes['bwd_runtime'])
        timing_result['bwd_runtime_mean'] = float(runtimes['bwd_runtime'] / timing_n)
        timing_result['bwd_runtime_std'] = float(runtimes['bwd_runtime_std'])
        
        for gb_algorithm in gb_algorithms:
            if config['num_var'] <= 3:
                ret = Parallel(n_jobs=-1, backend="multiprocessing", verbose=verbose)(delayed(compute_gb)(F, algorithm=gb_algorithm) for F, _ in dataset)
                _, runtimes = zip(*ret)
                runtimes = list(runtimes)
                success_rate = 1.0
            else: 
                print(f'Using timeout with limit {timeout_duration} for n > {3}')
                # print(f'The runtime is estimated from the frist 10 % samples')
                runtimes = []
                successes = []
                for i, (F, _) in enumerate(dataset):
                    _, runtime, success = compute_gb_with_to(F, algorithm=gb_algorithm)
                    runtimes.append(runtime)
                    successes.append(success)
                    if (i+1) % 10 == 0: print(f'[{i+1:04d} / {len(dataset)}]', flush=True)
                success_rate = np.mean(successes)
                
            # ret = Parallel(n_jobs=-1, backend="multiprocessing", verbose=verbose)(delayed(dbuilder.check_gb)(G) for G in GBs)
            # assert np.all(list(ret))
            
            timing_result[f'fwd_runtime_{gb_algorithm}'] = float(np.sum(runtimes))
            timing_result[f'fwd_runtime_{gb_algorithm}_mean'] = float(np.mean(runtimes))
            timing_result[f'fwd_runtime_{gb_algorithm}_std'] = float(np.std(runtimes))
            timing_result[f'fwd_runtime_{gb_algorithm}_success_rate'] = float(success_rate)
        
        if verbose:
            pprint(timing_result)
            print('', flush=True)
        timing_results.append(timing_result)
        
    if verbose:
        for timing_result in timing_results:
            print(f'# {timing_result["config_file"]} ({timing_result["num_samples"]} samples)')
            print(f' backward generation | {timing_result["bwd_runtime"]:.3f} [sec]')
            
            for gb_algorithm in gb_algorithms:
                print(f'   foward generation | {timing_result[f"fwd_runtime_{gb_algorithm}"]:.3f} [sec] --- {gb_algorithm}')

            print('',flush=True)
    
    all_results = {}
    for timing_result in timing_results:
        config_file = timing_result['config_file']
        all_results[config_file] = timing_result

    all_profiles = {}
    for dataset_profile in dataset_profiles:
        config_file = dataset_profile['config_file']
        all_profiles[config_file] = dataset_profile

    if save_dir:
        with open(os.path.join(save_dir, f'timing_results{filename_postfix}.yaml'), 'w') as f:
            yaml.dump(all_results, f, default_flow_style=False)
        with open(os.path.join(save_dir, f'dataset_profiles{filename_postfix}.yaml'), 'w') as f:
            yaml.dump(all_profiles, f, default_flow_style=False)
            
    return all_results, all_profiles
        

def timing_experiment_num_var_on_time(max_num_var=5, num_samples=100, order=5, max_degree=3, max_num_terms=2, max_Gdegree=5, max_num_Gterms=None, max_size=5, verbose=False):
    field = GF(order)

    ## For timing
    stats = {}
    for num_var in range(1, max_num_var+1):
        ring = PolynomialRing(field, num_var, 'x', order='lex')
        dbuilder = GBDataset_Builder(ring, max_size=max_size, max_degree=max_degree, max_num_terms=max_num_terms, max_Gdegree=max_Gdegree, max_num_Gterms=max_num_Gterms)
        _, runtime_d = dbuilder.run(num_samples, run_check=False, timing=True)

        stats[num_var] = runtime_d
    
        if verbose:
            print(f'## n = {num_var}')
            fwd_runtime = runtime_d['fwd_runtime']
            bwd_runtime = runtime_d['bwd_runtime']
            f_runtime = runtime_d['runtime_for_Fs']
            sg_runtime = runtime_d['runtime_for_SGs']
            
            print(f' backward generation | {bwd_runtime:.3f} [sec] ({sg_runtime:.3f} + {f_runtime:.3f}) --- {bwd_runtime/num_samples:.3f} [sec/sample]')
            print(f'   foward generation | {fwd_runtime:.3f} [sec] --- {fwd_runtime/num_samples:.3f} [sec/sample]', flush=True)
        
    return stats 

def main(gb_algorithms=['libsingular:std', 
                        'libsingular:slimgb',
                        'libsingular:stdfglm']):
    
    parser = get_parser()
    args = parser.parse_args()

    if args.field in ('QQ'):
        field = QQ
    if args.field[0] == 'F' and args.field[1:].isdigit():
        field = GF(int(args.field[1:]))
        
    os.makedirs(args.save_path, exist_ok=True)

    if args.dryrun:
        print('dryrun done!', flush=True)
    else:
        config_paths = [f'config/gb_dataset_n={n}_field={field}{postfix}.yaml' for n in range(2, 6)]
        filename_postfix = f'_field={field}{postfix}_N={timing_n}'
        
        pprint(config_paths)
        
        timing_experiment(timing_n, 
                          config_paths, 
                          gb_algorithms=gb_algorithms, 
                          verbose=True,
                          save_path=args.save_path,
                          filename_postfix=filename_postfix)

            

if __name__ == '__main__':
    import os, pickle, yaml, sys 
    # postfix = '_density=1'
    # postfix = ''
    # assert len(sys.argv) > 1
    # postfix = sys.argv[-1]
    # assert postfix in ('', '_density=1')
    
    postfix = '_density=1'
    save_dir = 'results/timing/'
    if postfix: save_dir = os.path.join(save_dir, postfix[1:])
    
    timing_n = 1000
    
    gb_algorithms = ['libsingular:std', 
                     'libsingular:slimgb', 
                     'libsingular:stdfglm', 
                    #  'singular:std', 'singular:slimgb', 'singular:stdfglm',  # slow
                    ]
    
    os.makedirs(save_dir, exist_ok=True)
        
    
    if sys.argv[-1] != 'dryrun': 
        for field in ['QQ', 'F7', 'F31']:
            config_paths = [f'config/gb_dataset_n={n}_field={field}{postfix}.yaml' for n in range(2, 6)]
            filename_postfix = f'_field={field}{postfix}_N={timing_n}'
            
            pprint(config_paths)
            
            timing_experiment(timing_n, config_paths, 
                            gb_algorithms=gb_algorithms, 
                            verbose=True,
                            save_dir=save_dir,
                            filename_postfix=filename_postfix)

            
            print('\n')
