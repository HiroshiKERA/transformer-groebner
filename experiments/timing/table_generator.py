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
    parser.add_argument('--profile_paths', required=False, nargs="*", type=str) 
    parser.add_argument('--runtime_paths', required=False, nargs="*", type=str) 
    # parser.add_argument("--timeout_duration", type=int, default=10)
    parser.add_argument("--N", type=int, default=1000)

    parser.add_argument("--dryrun", action='store_true', default=False)
    
    return parser


def generate_dataset_profile_table(args):
    print('/////////////////////////')
    print(' Dataset Profile ')
    print('/////////////////////////')
    
    std_l = "$^{(\pm "
    std_r = ")}$"
    get_std_str = lambda n: std_l + f'{n:.2g}' + std_r
    get_mean_str = lambda n: f'{n:.2g}'
    get_rep = lambda m, std: get_mean_str(m) + get_std_str(std)

    metrics = ['size', 'max_degree', 'min_degree', 'total_num_monoms', 'is_GB']
    row_names_F = ['Size of $F$', 'Max degree in $F$', 'Min degree in $F$', '# of terms in $F$', '\gb ratio']
    row_names_G = ['Size of $G$', 'Max degree in $G$', 'Min degree in $G$', '# of terms in $G$', '\gb ratio']
    label_map = {"F": dict(zip(metrics, row_names_F)),
                 "G": dict(zip(metrics, row_names_G))}

    profiles = []
    for path in args.profile_paths:
        with open(path, 'r') as f:
            profile = yaml.safe_load(f)
            profiles.append(profile)

    for target in ['F', 'G']:
        for metric in metrics: 
            
            means = [profile[target][metric + '_mean'] for profile in profiles]
            stds = [profile[target][metric + '_std'] for profile in profiles]
            reps = [get_rep(mean, std) for mean, std in zip(means, stds)]
            row = ' & '.join(reps)
            row = label_map[target][metric] + ' & ' + row
            
            print(row + ' \\\\')
            # rows.append(row)
            # rows.append('\hline')

        # for row in rows:
        #     print(row)

        print('')

def generate_runtime_table(args):
    print('/////////////////////////')
    print(' Runtime Summary ')
    print('/////////////////////////')
    
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

    runtime_stats = []
    for path in args.runtime_paths:
        with open(path, 'r') as f:
            runtime_stat = yaml.safe_load(f)
            runtime_stats.append(runtime_stat)

    for target, metrics in zip(["F", "G"], [metrics_F, metrics_G]):
        for metric in metrics:             
            rows = []
            runtimes = [runtime_stat[metric] for runtime_stat in runtime_stats]
            reps = [get_rep(t) for t in runtimes]
            row = ' & '.join(reps)
            row = label_map[target][metric] + ' & ' + row
            rows.append(row)
            
            for row in rows:
                print(row)
    print('')

def generate_success_rate_table(args):
    print('/////////////////////////')
    print(' Success Rate Summary ')
    print('/////////////////////////')
    
    get_mean_str = lambda n: f'{n:.3g}'
    get_rep = lambda m: get_mean_str(m)

    gb_algorithms = ['libsingular:std', 'libsingular:slimgb', 'libsingular:stdfglm']

    metrics_F = [f'fwd_runtime_{gb_alg}_success_rate' for gb_alg in gb_algorithms]
    metrics_G = []
    row_names_F = [f'Foward ({gb_alg.split(":")[-1]})' for gb_alg in gb_algorithms]
    row_names_G = ['Backward (ours)']
    label_map = {"F": dict(zip(metrics_F, row_names_F)),
                 "G": dict(zip(metrics_G, row_names_G))
                 }

    runtime_stats = []
    for path in args.runtime_paths:
        with open(path, 'r') as f:
            runtime_stat = yaml.safe_load(f)
            runtime_stats.append(runtime_stat)

    for target, metrics in zip(["F", "G"], [metrics_F, metrics_G]):
        for metric in metrics:             
            rows = []
            runtimes = [runtime_stat[metric] for runtime_stat in runtime_stats]
            reps = [get_rep(t) for t in runtimes]
            row = ' & '.join(reps)
            row = label_map[target][metric] + ' & ' + row
            rows.append(row)
            
            for row in rows:
                print(row)
                
def main():
    parser = get_parser()
    args = parser.parse_args()
    
    if args.profile_paths:
        generate_dataset_profile_table(args)
    if args.runtime_paths:
        generate_runtime_table(args)
        generate_success_rate_table(args)
        
        
if __name__ == '__main__':
    main()
    # import sys 
    # assert len(sys.argv) > 1
    # postfix = sys.argv[-1]
    # assert postfix in ('', '_density=1')

    # postfix = '_density=1'
    # postfix = ''
    # generate_dataset_profile_table(postfix)
    # generate_runtime_table(postfix)