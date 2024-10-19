from collections import OrderedDict
from pprint import pprint 
import os, pickle, yaml, sys 

import argparse

def get_parser():
    """
    Generate a parameters parser.
    """
    # parse parameters
    parser = argparse.ArgumentParser(description="Language transfer")

    # path
    # parser.add_argument("--data_path", type=str, default="data/")
    parser.add_argument("--config_path", type=str, default="results/")
    parser.add_argument("--save_path", type=str, default="results/timing")
    
    # setup
    parser.add_argument("--field", type=str, default="QQ")
    parser.add_argument('--num_vars', required=True, nargs="*", type=float, help='a list of number of variables') 
    # parser.add_argument("--timeout_duration", type=int, default=10)
    parser.add_argument("--N", type=int, default=1000)

    parser.add_argument("--dryrun", action='store_true', default=False)
    
    return parser


def generate_dataset_profile_table(args):
    print('/////////////////////////')
    print(' Dataset Profile ')
    print('/////////////////////////')
    save_path = args.save_path 

    timing_n = 1000
    
    std_l = "$^{(\pm "
    std_r = ")}$"
    get_std_str = lambda n: std_l + f'{n:.2g}' + std_r
    get_mean_str = lambda n: f'{n:.2g}'
    get_rep = lambda m, std: get_mean_str(m) + get_std_str(std)

    metrics = ['size', 'max_degree', 'min_degree', 'total_num_monoms', 'is_GB']
    row_names_F = ['Size of $|F|$', 'Max degree in $F$', 'Min degree in $F$', '# of terms in $F$', '\gb ratio']
    row_names_G = ['Size of $|G|$', 'Max degree in $G$', 'Min degree in $G$', '# of terms in $G$', '\gb ratio']
    label_map = {"F": dict(zip(metrics, row_names_F)),
                 "G": dict(zip(metrics, row_names_G))}

    config_paths = [f'config/gb_dataset_n={n}_char={char}{postfix}.yaml' for n in range(2, 6)]
    config_names = [os.path.basename(cp) for cp in config_paths]
    pprint(config_paths)
    
    for target in ['F', 'G']:
        for metric in metrics: 
            filename_postfix = f'_char={char}{postfix}_N={timing_n}'
            with open(os.path.join(save_path, f'dataset_profiles{filename_postfix}.yaml'), 'r') as f:
                runtimes = yaml.safe_load(f)
            
            rows = []

            means = [runtimes[cp][target][metric + '_mean'] for cp in config_names]
            stds = [runtimes[cp][target][metric + '_std'] for cp in config_names]
            reps = [get_rep(mean, std) for mean, std in zip(means, stds)]
            row = ' & '.join(reps)
            row = label_map[target][metric] + ' & ' + row
            
            print(row)
            # rows.append(row)
            # rows.append('\hline')

        # for row in rows:
        #     print(row)

        print('')

def generate_runtime_table(postfix):
    print('/////////////////////////')
    print(' Runtime Summary ')
    print('/////////////////////////')
    
    save_path = 'results/timing/'
    timing_n = 1000
    
    get_mean_str = lambda n: f'{n:.3g}'
    get_rep = lambda m: get_mean_str(m)

    gb_algorithms = ['libsingular:std', 'libsingular:slimgb', 'libsingular:stdfglm']

    metrics_F = [f'fwd_runtime_{gb_alg}' for gb_alg in gb_algorithms]
    metrics_G = [f'bwd_runtime']
    row_names_F = [f'Foward ({gb_alg.split(":")[-1]})' for gb_alg in gb_algorithms]
    row_names_G = ['Backward (ours)']
    label_map = {"F": dict(zip(metrics_F, row_names_F)),
                 "G": dict(zip(metrics_G, row_names_G))
                 }

    for char in [7, 31]:
        config_paths = [f'config/gb_dataset_n={n}_char={char}{postfix}.yaml' for n in range(2, 6)]
        config_names = [os.path.basename(cp) for cp in config_paths]
        pprint(config_paths)
        for target, metrics in zip(["F", "G"], [metrics_F, metrics_G]):
            for metric in metrics: 
                filename_postfix = f'_char={char}{postfix}_N={timing_n}'
                with open(os.path.join(save_path, f'timing_results{filename_postfix}.yaml'), 'r') as f:
                    runtimes_stats = yaml.safe_load(f)
                
                rows = []
                runtimes = [runtimes_stats[cp][metric] for cp in config_names]
                reps = [get_rep(t) for t in runtimes]
                row = ' & '.join(reps)
                row = label_map[target][metric] + ' & ' + row
                rows.append(row)
                
                for row in rows:
                    print(row)
            
        print('')
        
        
def main():
    parser = get_parser()
    args = parser.parse_args()
        
    generate_dataset_profile_table(postfix)
    generate_runtime_table(postfix)
        
        
if __name__ == '__main__':
    # import sys 
    # assert len(sys.argv) > 1
    # postfix = sys.argv[-1]
    # assert postfix in ('', '_density=1')

    # postfix = '_density=1'
    postfix = ''
    generate_dataset_profile_table(postfix)
    generate_runtime_table(postfix)