import os, yaml
from time import time 
from joblib import Parallel, delayed
import numpy as np 
import pickle
from pprint import pprint
import timeout

import sys 
sys.path.append('src')

from loader.data import _load_data
from loader.data import BartDataCollator
from torch.utils.data import DataLoader
from loader.checkpoint import load_pretrained_bag
from misc.utils import to_cuda

from loader.data import _load_data

from joblib import Parallel, delayed

import argparse

from loader.checkpoint import load_pretrained_bag
from loader.data import _load_data, get_datacollator
from evaluation.generation import generation_accuracy, generation, accuracy_score


def get_parser():
    """
    Generate a parameters parser.
    """
    # parse parameters
    parser = argparse.ArgumentParser(description="Gröbner basis computation")

    # path
    parser.add_argument("--data_path", type=str, help="Path to the input data")
    parser.add_argument("--model_path", type=str, default="results/", help="Path to model")
    parser.add_argument("--save_path", type=str, default="results/prediction", help="Path to save results")

    # generation setup 
    # parser.add_argument("--model_name", type=str, help="Model name")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for generation")
    parser.add_argument("--max_length", type=int, default=1024, help="Maximum length for generation")
    
    # evaluation setup
    parser.add_argument("--th", type=float, default=0.5, help="Threshold for accuracy evaluation")

    # setup
    parser.add_argument("--field", type=str, default="QQ", help="Field for polynomial ring (e.g., QQ, F7, F31)")
    
    # misc 
    parser.add_argument("--disable_tqdm", action='store_true', default=False, help="Use tqdm for progress bar")

    return parser


def main():
    
    parser = get_parser()
    args = parser.parse_args()

    os.makedirs(args.save_path, exist_ok=True)
    
    modulo = int(args.field[2:]) if args.field.startswith('GF') else None
    quantize_fn = (lambda x: x.round() % modulo) if args.field.startswith('GF') else None
    
    # try:
    results = generation_accuracy(args.model_path, 
                                    args.data_path, 
                                    batch_size=args.batch_size, 
                                    max_length=args.max_length,
                                    th=args.th, 
                                    modulo=modulo,
                                    disable_tqdm=args.disable_tqdm,
                                    quantize_fn=quantize_fn)

    print(f'Accuracy: {results["acc"]:.4f}, Support Accuracy: {results["support_acc"]:.4f}, Runtime: {results["runtime_per_batch"]:.4f} sec/batch')

    with open(os.path.join(args.save_path, 'generation_results.yaml'), 'w') as f:
        yaml.dump(results, f)

    # except Exception as e:
    #     print(f'Error: {e}')
    

    # from torch.utils.data import DataLoader
    # from misc.utils import to_cuda
    # import yaml, argparse
    # with open(os.path.join(args.save_path, 'params.yaml'), 'r') as f:
    #     params = yaml.safe_load(f)
    #     params = argparse.Namespace(**params)

    # bag = load_pretrained_bag(args.save_path, cuda=True)
    # model, tokenizer = bag['model'], bag['tokenizer']
    # model_name = params.model
    # print(f'Loaded model {model_name} from {args.save_path}')

    # data_encoding = 'infix+' if params.encoding_method == 'hybrid' and params.field == 'QQ' else 'infix'
    # testset     = _load_data(f'{params.data_path}.test.lex.{data_encoding}')
    # dc = get_datacollator(params.model)(tokenizer, continuous_coefficient=True)

    # testloader = DataLoader(testset, batch_size=1, collate_fn=dc, shuffle=False)
    
    # results = generation_accuracy(model_name, model, testloader, tokenizer=tokenizer, th=0, disable_tqdm=False)
    
    


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
