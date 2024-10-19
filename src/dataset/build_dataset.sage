import os
import time
import argparse
from pprint import pprint

import numpy as np
import yaml
from joblib import Parallel, delayed

load('src/dataset/symbolic_utils.sage')
load('src/dataset/groebner.sage')

np.random.seed((os.getpid() * int(time())) % 123456789)

def summarize_stats(stats, metric=['mean', 'std', 'max', 'min', 'median']):
    summary = {}
    for k in stats[0]:
        if isinstance(stats[0][k], list): continue
        if 'mean' in metric:
            summary[f'{k}_mean'] = float(np.mean([item[k] for item in stats]))
        if 'median' in metric:    
            summary[f'{k}_median'] = float(np.median([item[k] for item in stats]))
        if 'std' in metric:    
            summary[f'{k}_std'] = float(np.std([item[k] for item in stats]))
        if 'max' in metric:    
            summary[f'{k}_max'] = float(np.max([item[k] for item in stats]))
        if 'min' in metric:    
            summary[f'{k}_min'] = float(np.min([item[k] for item in stats]))
    return summary

class Writer():
    def __init__(self, local_separator = ' [SEP] ', global_separator = ' : '):
        self.local_separator = local_separator
        self.global_separator = global_separator

    def _preprocess(self, F, G, encoding=None, ring=None, n_jobs=-1, split_rational=True):

        Fstr = [poly_to_sequence(f, split_rational=split_rational) for f in F]  # encoding not implemented
        Gstr = [poly_to_sequence(g, split_rational=split_rational) for g in G]  # encoding not implemented          
            
        Fstr = self.local_separator.join(Fstr)
        Gstr = self.local_separator.join(Gstr)
        s = Fstr + self.global_separator + Gstr 
        
        num_tokens_F = int(len(Fstr.split()))
        num_tokens_G = int(len(Gstr.split()))
        stats = {
            'num_tokens_F': num_tokens_F,
            'num_tokens_G': num_tokens_G,
            'num_tokens': num_tokens_F + num_tokens_G
        }
        
        return s, stats

    def write(self, filename, dataset, n_jobs=-1, encoding='raw', ring=None, split_rational=True):
        filename = f'{filename}.{encoding}'

        start = time()
        ret = Parallel(n_jobs=n_jobs, backend="multiprocessing", verbose=True)(delayed(self._preprocess)(F, G, encoding=encoding, ring=ring, split_rational=split_rational) for F, G in dataset)
        
        dataset_str, stats = zip(*ret)
        dataset_str = "\n".join(dataset_str)
        with open(filename, 'w') as f:
            f.write(dataset_str)
            # f.write("\n".join(dataset_str))
            # joblib.dump(dataset_str, f, compress=False)
            
        summary = summarize_stats(stats)
        with open(filename + '_token_stats.yaml', "w") as f:
            yaml.dump(summary, f)

            
    def save_stats(self, summaries, save_path='', postfix=''):
        if postfix: postfix = '_' + postfix
        stats_F_sum, stats_G_sum = summaries
        
        print('---- dataset statistics -----------')
        print('# F')
        pprint(stats_F_sum)
        print('# G')
        pprint(stats_G_sum)
        print('')
        
        if save_path:
            # with open(os.path.join(save_path, f"dataset_stats{postfix}.pickle"), "w") as f:
            #     yaml.dump(stats, f)

            with open(os.path.join(save_path, f"dataset_stats_F_summary{postfix}.yaml"), "w") as f:
                yaml.dump(stats_F_sum, f)
                
            with open(os.path.join(save_path, f"dataset_stats_G_summary{postfix}.yaml"), "w") as f:
                yaml.dump(stats_G_sum, f)


def get_parser():
    """
    Generate a parameters parser.
    """
    # parse parameters
    parser = argparse.ArgumentParser(description="Language transfer")

    # main parameters
    parser.add_argument("--data_path", type=str, default="./data/diff_dataset",
                        help="Experiment dump path")
    parser.add_argument("--data_encoding", type=str, default="infix")
    parser.add_argument("--save_path", type=str, default="./dumped",
                        help="Experiment dump path")
    parser.add_argument("--config_path", type=str, default="./config")
    # parser.add_argument("--task", type=str, default='shape', choices=['shape', 'cauchy'])
    parser.add_argument("--testset_only", action='store_true', default=False)
    parser.add_argument("--strictly_conditioned", action='store_true', default=False)

    return parser

def build_dataset(get_dataset_builder, config, save_dir, tag='test', encoding='infix', n_jobs=-1, timing=False, strictly_conditioned=True):
    
    field_name = config['field']
    if field_name == 'QQ':
        field = QQ
    elif field_name == 'RR':
        field = RR
    elif field_name == 'ZZ':
        field = ZZ
    elif field_name[:2] == 'GF':
        order = int(field_name[2:])
        field = GF(order)

    num_samples  = config[f'num_samples_{tag}']  
    num_variables = int(config['num_var'])
    ring = PolynomialRing(field, num_variables, 'x', order='lex')
    
    
    dbuilder = get_dataset_builder(ring, config)
    dbuilder.run(num_samples, n_jobs=n_jobs, timing=timing, strictly_conditioned=strictly_conditioned, degree_sampling=config['degree_sampling'], term_sampling=config['term_sampling'])
    
    dataset = dbuilder.dataset
    stats = (dbuilder.stats[0], dbuilder.stats[1])

    summaries = (summarize_stats(stats[0]), summarize_stats(stats[1]))

    writer = Writer()
    writer.write(os.path.join(save_dir, f'data_{field_name}_n={num_variables}.{tag}'), 
                dataset, encoding=f'lex.{encoding}', n_jobs=n_jobs, ring=ring, split_rational=True) 
    writer.save_stats(summaries, save_path=save_dir, postfix=f'data_{field_name}_n={num_variables}_{tag}')
    
    if field_name == 'QQ':
        writer.write(os.path.join(save_dir, f'data_{field_name}_n={num_variables}.{tag}'), 
                    dataset, encoding=f'lex.{encoding}+', n_jobs=n_jobs, ring=ring, split_rational=False) 
        writer.save_stats(summaries, save_path=save_dir, postfix=f'data_{field_name}_n={num_variables}_{tag}+')
        
        

def main():
    parser = get_parser()
    params = parser.parse_args()
    
    # gb_type = params.task
    save_dir = params.save_path

    print(f'## dataset geneartion with config {params.config_path} ##########')
    print(params)
    
    os.makedirs(save_dir, exist_ok=True)

    with open(params.config_path, 'r') as f:
        config = yaml.safe_load(f)

    print(config)
    
    get_dataset_builder = Dataset_Builder_Groebner

    encoding = params.data_encoding
    strictly_conditioned = params.strictly_conditioned
    
    print(f'-- Test set ({config["num_samples_test"]} samples)')        
    build_dataset(get_dataset_builder, config, save_dir, tag='test', encoding=encoding, n_jobs=-1, timing=False, strictly_conditioned=strictly_conditioned)

    if not params.testset_only:
        print(f'-- Train set ({config["num_samples_train"]} samples)')        
        build_dataset(get_dataset_builder, config, save_dir, tag='train', encoding=encoding, n_jobs=-1, timing=False, strictly_conditioned=strictly_conditioned)

    print('done!')

if __name__ == '__main__':
    main()

    