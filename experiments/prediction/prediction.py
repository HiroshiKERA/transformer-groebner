import os, sys 
from time import time
from tqdm import tqdm 

import torch 

sys.path.append('src_dev')
from loader.checkpoint import load_trained_bag
from evalution.evaluators import eval_prediction
from loader.data import load_data
import numpy as np 
import argparse

# os.environ['TOKENIZERS_PARALLELISM'] = 'false'

def get_parser():
    """
    Generate a parameters parser.
    """
    # parse parameters
    parser = argparse.ArgumentParser(description="Language transfer")

    # path
    parser.add_argument("--data_path", type=str, default="data/")
    parser.add_argument("--test_data_path", type=str, default=None)
    parser.add_argument("--load_path", type=str, default="results/")
    parser.add_argument("--save_path", type=str, default="results/")

    # setup
    parser.add_argument("--data_encoding", type=str, default="prefix")
    parser.add_argument("--term_order", type=str, default="lex")
    parser.add_argument("--field", type=str, default='QQ', help='QQ or FP with some integer P (e.g., F7).')

    # evaluation parameters
    parser.add_argument("--batch_size", type=int, default=8,)
    parser.add_argument("--continuous_threshold", type=float, default=0.1,)
    parser.add_argument("--support_only", action='store_true', default=False)


    return parser

@torch.no_grad()
def evaluate(data_path, load_dir, save_dir, term_order='lex', encoding='prefix', batch_size=8, use_tqdm=True, from_check_point=False, verbose=True, field=None, continuous_th=0.01, support_only=False):

    bag         = load_trained_bag(load_dir, from_checkpoint=from_check_point)
    model       = bag['model'] 
    tokenizer   = bag['tokenizer']
    params      = bag['params']

    has_continuous_tokens   = model.input_embedding.has_continuous_embedding
    continuous_vocab_size   = int(has_continuous_tokens) if not support_only else 0
    continuous_token_ids    = [tokenizer.vocab['[C]']][:continuous_vocab_size]

    quantize_fn = None
    if field is not None:
        if field[0] == 'F': quantize_fn = lambda x: torch.round(x).clamp(0, int(field[1:]))
        # if field == 'QQ': 
        #     import itertools as it
        #     nums = list(range(-5, 6))  # QQ num_bounds is 5  TO DO: make it a parameter
        #     qqs = [num/denom for num, denom in it.product(nums, nums) if denom != 0]
        #     qqs = torch.tensor(list(set(qqs))).cuda()
        #     _quantize_fn = lambda x: torch.argmin((x.unsqueeze(1) - qqs.unsqueeze(0)).abs(), dim=-1)
        #     quantize_fn = lambda x: qqs[_quantize_fn(x)]
    
    # testset = load_data(data_path, extensions=[f'test'], encoding=f'_{term_order}.prefix', do_shuffle=[False], return_dataloader=False)
    test_loader = load_data(data_path, 
                            batch_sizes             = [batch_size], 
                            encoding                = f'{term_order}.{encoding}', 
                            extensions              = ['test'], 
                            do_shuffle              = [False], 
                            return_dataloader       = True, 
                            continuous_coefficient  = has_continuous_tokens,
                            continuous_exponent     = False,
                            tokenizer               = tokenizer,
                            support_learning        = support_only,)
    
    write_mode_correct = 'w'
    write_mode_incorrect = 'w'
    filename_correct = os.path.join(save_dir, 'prediction_correct.txt')
    filename_incorrect = os.path.join(save_dir, 'prediction_incorrect.txt')
    
    iterator = tqdm(test_loader, desc='Evaluating', disable=not use_tqdm)
    hits, hits_supp, runtime = [], [], 0

    for batch_id, batch in enumerate(iterator):
        for k in batch: batch[k] = batch[k].cuda()

        y       = batch['decoder_input_ids']
        y_cont  = batch['target_continuous_labels']

        # compute GB by Transformer
        start = time()
        prediction = model.generate(input_ids               = batch['input_ids'], 
                                    input_continuous_labels = batch['input_continuous_labels'],
                                    attention_mask          = batch['attention_mask'], 
                                    continuous_token_ids    = continuous_token_ids,
                                    max_length              = y.shape[-1],
                                    quantize_fn             = quantize_fn,)
        continuous_prediction = prediction['continuous_prediction']
        z_texts = model.postprocess_prediction(prediction, tokenizer, skip_special_tokens=True)['prediction_texts']
        runtime += time() - start

        y_texts = model.decode_continuous_ids(y, y_cont, tokenizer, quantized=quantize_fn is not None, continuous_vocab_size=continuous_vocab_size, skip_special_tokens=True)

        # for support accuracy
        z_texts_supp = tokenizer.batch_decode(prediction['prediction'].long(), skip_special_tokens=True, num_beams=1, max_length=y.shape[-1])
        y_texts_supp = tokenizer.batch_decode(y, skip_special_tokens=True, num_beams=1, max_length=y.shape[-1])

        for i, (yt, zt) in enumerate(zip(y_texts, z_texts)):
            image_id = batch_id * batch_size + i
            hit      = yt == zt
            hit_supp = y_texts_supp[i] == z_texts_supp[i]

            if has_continuous_tokens and hit_supp:
                seq_len = continuous_prediction.shape[1]
                is_continuous = ~y_cont[i, :seq_len].isnan()
                # rloss = (y_cont[i, is_continuous] - continuous_prediction[i, is_continuous].view(-1)).abs().max().item()
                
                # rloss = (y_cont[i, :seq_len][is_continuous] - continuous_prediction[i, is_continuous].view(-1)).pow(2).mean().item()
                
                try:
                    rloss = (y_cont[i, :seq_len][is_continuous] - continuous_prediction[i, is_continuous].view(-1)).pow(2).mean().item()
                except:
                    print(y_cont[i, :seq_len].shape, continuous_prediction[i].shape, is_continuous.shape)

                
                # print()
                # print(y_cont[i, is_continous])
                # print(continuous_prediction[i, is_continous].view(-1))
                # print(f'relative loss = {rloss:.4f} (th = {continuous_th})')
                hit = rloss < continuous_th

            hits.append(hit)
            hits_supp.append(hit_supp)

            if hit:
                with open(filename_correct, write_mode_correct) as f: 
                    line = f'{image_id:04} >> {zt} \n'
                    f.writelines(line)
                    write_mode_correct = 'a'
            else:                
                with open(filename_incorrect, write_mode_incorrect) as f: 
                    line = f'{image_id:04} >> {zt} \n'
                    f.writelines(line)
                    write_mode_incorrect = 'a'

    acc         = sum(hits) / len(hits)
    acc_supp    = sum(hits_supp) / len(hits)

    if verbose:
        if has_continuous_tokens:
            print(f"full/supp accuracy = {acc:.3f}/{acc_supp:.3f} [processed {len(hits)} samples] in {runtime:.1f} sec")
        else:
            print(f"accuracy = {acc:.3f} [processed {len(hits)} samples] in {runtime:.1f} sec")

    return acc

def main():
    parser = get_parser()
    args = parser.parse_args()

    os.makedirs(args.save_path, exist_ok=True)

    print(f'Results will be saved to {os.path.basename(args.save_path)}.')

    evaluate(args.data_path, args.load_path, args.save_path, 
            term_order      = args.term_order,
            encoding        = args.data_encoding, 
            batch_size      = args.batch_size, 
            use_tqdm        = True, 
            verbose         = True, 
            from_check_point= False,
            field           = args.field,
            continuous_th   = args.continuous_threshold,
            support_only    = args.support_only)

if __name__ == '__main__':
    main()
