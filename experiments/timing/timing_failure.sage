from sage.all_cmdline import *   # import sage library

import pickle
from time import time 
import os, sys 
import argparse
import numpy as np
os.environ["TOKENIZERS_PARALLELISM"] = "true"
import timeout

sys.path.append('src_dev')

from loader.data import load_data
from loader.checkpoint import load_trained_bag

load('src/data/symbolic_utils.sage')
# load('src/loader/data.py')
# load('src/loader/checkpoint.py')

timeout_duration = 100

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
    parser.add_argument("--data_path", type=str, default="data/")
    parser.add_argument("--load_path", type=str, default="results/")
    parser.add_argument("--success_table_path", type=str, default="results/timing")

    # setup
    parser.add_argument("--num_variables", type=int, default=2)
    parser.add_argument("--field", type=str, default="QQ")
    parser.add_argument("--data_encoding", type=str, default="prefix")
    parser.add_argument("--term_order", type=str, default="lex")

    # parser.add_argument("--timeout_duration", type=int, default=10)

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
            Fs.append(F)
            
        for text in target_texts:
            G = text.split(' [SEP] ')
            G = [ring(g) for g in G]
            Gs.append(G)
        
    return (Fs, Gs)


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

def compute_gb(F, algorithm=''):
    ts = time()
    G = ideal(F).groebner_basis(algorithm=algorithm)
    runtime = time() - ts 
    return G, runtime


def main():
    parser = get_parser()
    args = parser.parse_args()

    if args.dryrun:
        print('dryrun done!', flush=True)
        return None

    field = args.field
    n = args.num_variables
    
    print(f'{field}, n={n}')

    data_path = args.data_path
    load_path = args.load_path
    success_table_path = args.success_table_path

    bag = load_trained_bag(load_path, from_checkpoint=False)
    model = bag['model'] 
    tokenizer = bag['tokenizer']
    
    if field == 'QQ': bfield = QQ
    if field == 'RR': bfield = RR
    if field[0] == 'F': bfield = GF(int(field[1:]))
    ring = PolynomialRing(bfield, n, 'x', order='lex')
    test_loader = load_data(data_path, encoding='prefix', batch_sizes=[int(1)], return_dataloader=False, extensions=['test.lex'], do_shuffle=[False], tokenizer=None)
    Fs, Gs = load_raw_dataset(data_path + '.test.lex.raw', ring)


    with open(success_table_path, 'rb') as f:
        sucecss_table = pickle.load(f)

    for algorithm in sucecss_table:
        failed_id = np.where(~sucecss_table[algorithm])[0]
        if len(failed_id) > 0:
            print(f'{algorithm:20}: {failed_id}')
        else:
            print(f'{algorithm:20}: -')

        failed_id = np.hstack([np.where(~sucecss_table[algorithm])[0] for algorithm in sucecss_table])
        failed_id = np.sort(list(set(failed_id)))

    
        # failed_id = np.where(~sucecss_table[algorithm])[0]
        # if len(failed_id) > 0:
        #     print(f'{algorithm:20}: {failed_id}')
        # else:
        #     print(f'{algorithm:20}: -')

    for fid in failed_id:
        ## test by Transformer
        x_text = test_loader[fid]['input']
        y_text = test_loader[fid]['target']
        x = tokenizer(x_text, return_tensors='pt')['input_ids'].cuda()
        y = tokenizer(y_text, return_tensors='pt')['input_ids'].cuda()
        start = time()
        output_ids = model.generate(x, max_length=y.shape[-1], num_beams=1, do_sample=False)
        end = time()

        y_pred = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        result = 'success' if y_text == y_pred else 'failure'
        method = 'Transformer'
        print(f'[{fid:03}] {method:20} -- {end-start} sec [{result}]')

            ## test by Math
        F = Fs[fid]
        for algorithm in sucecss_table:
            G, runtime, succees = compute_gb_with_to(F, algorithm=algorithm)
            # G, runtime = compute_gb(F, algorithm=algorithm)
            result = 'success' if succees else 'failure'
            print(f'[{fid:03}] {algorithm:20} -- {runtime} sec [{result}]')
            
    print('')


if __name__ == '__main__':
    main()
    