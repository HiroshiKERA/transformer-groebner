import argparse
import numpy as np
import yaml, os 
import torch 
from time import time 
import datetime
from zoneinfo import ZoneInfo

# from trainer import Trainer
from loader.data import _load_data, SimpleDataCollator, HybridDataCollator, BartDataCollator, BartPlusDataCollator
from loader.model import load_model 
from trainer.utils import compute_metrics, preprocess_logits_for_metrics, LimitStepsCallback
from misc.utils import count_cuda_devices
from evaluation.generation import generation_accuracy
# np.seterr(all='raise')

import warnings
warnings.filterwarnings("ignore", message="Was asked to gather along dimension 0")
warnings.filterwarnings("ignore", message="The PyTorch API of nested tensors is in prototype stage")


# import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def get_parser():
    """
    Generate a parameters parser.
    """
    # parse parameters
    parser = argparse.ArgumentParser(description="Language transfer")

    # main parameters
    parser.add_argument("--data_path",      type=str, default="./data/data_sum", help="Experiment dump path")
    parser.add_argument("--data_config_path",      type=str, default="./data/data_sum", help="Experiment dump path")
    parser.add_argument("--data_encoding",  type=str, default="infix")
    parser.add_argument("--save_path",      type=str, default="./dumped", help="Experiment dump path")
    parser.add_argument("--save_periodic",  type=int, default=0, help="Save the model periodically (0 to disable)")
    parser.add_argument("--exp_name",       type=str, default="debug", help="Experiment name")
    parser.add_argument("--exp_id",         type=str, default="", help="Experiment ID")
    parser.add_argument("--group",          type=str, default="", help="Experiment group")
    parser.add_argument("--task",           type=str, default="sum", help="Task name")    

    # float16 / AMP API
    parser.add_argument("--fp16",           type=bool, default=True, help="Run model with float16")
    parser.add_argument("--amp",            type=int, default=2, help="Use AMP wrapper for float16 / distributed / gradient accumulation. Level of optimization. -1 to disable.")

    # model parameters
    parser.add_argument("--model",              type=str, default='bart', help="Embedding layer size")
    parser.add_argument("--d_model",            type=int, default=512, help="Embedding layer size")
    parser.add_argument("--dim_feedforward",    type=int, default=2048, help="feedforward layer size")
    parser.add_argument("--num_encoder_layers", type=int, default=6, help="Number of Transformer layers in the encoder")
    parser.add_argument("--num_decoder_layers", type=int, default=6, help="Number of Transformer layers in the decoder")
    parser.add_argument("--nhead",              type=int, default=8, help="Number of Transformer heads")
    parser.add_argument("--dropout",            type=float, default=0.1, help="Dropout")
    parser.add_argument("--attention_dropout",  type=float, default=0, help="Dropout in the attention layer")

    parser.add_argument("--continuous_embedding_model", type=str, default='ffn')
    parser.add_argument("--encoding_method",    type=str, default='standard')
    # parser.add_argument("--token_register_size", type=int, default=2)
    # parser.add_argument("--positional_encoding", type=str, default='sinusoidal', choices=['sinusoidal', 'embedding'])
    
    # vocab and tokenizer parameters
    parser.add_argument("--num_variables", type=int, default=2)
    parser.add_argument("--field", type=str, default='QQ', help='QQ, RR, or GFP with some prime P (e.g., GF7).')
    parser.add_argument("--max_coefficient", type=int, default=1000, help='The maximum coefficients')
    parser.add_argument("--max_degree", type=int, default=10, help='The maximum degree')


    # training parameters
    parser.add_argument("--max_sequence_length", type=int, default=10000,
                        help="Maximum sequences length")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Number of sentences per batch")
    parser.add_argument("--test_batch_size", type=int, default=256,
                        help="Number of sentences per batch")
    parser.add_argument("--optimizer", type=str, default="adamw_apex_fused",
                        help="Optimizer (SGD / RMSprop / Adam, etc.)")
    parser.add_argument("--learning_rate", type=float, default=0.0001,
                        help="learning rate (default 0.0001)")
    parser.add_argument("--weight_decay", type=float, default=0.0,
                        help="weight decay (default 0)")
    parser.add_argument("--clip_grad_norm", type=float, default=5,
                        help="Clip gradients norm (0 to disable)")
    parser.add_argument("--epochs", type=int, default=5,
                        help="number of epochs")
    parser.add_argument("--max_steps_per_epoch", type=int, default=-1, help="Maximum epoch size")
    parser.add_argument("--warmup_ratio", type=float, default=0.0)
    parser.add_argument("--stopping_criterion", type=str, default="",
                        help="Stopping criterion, and number of non-increase before stopping the experiment")
    parser.add_argument("--validation_metrics", type=str, default="",
                        help="Validation metrics")
    parser.add_argument("--accumulate_gradients", type=int, default=1,
                        help="Accumulate model gradients over N iterations (N times larger batch sizes)")
    parser.add_argument("--num_workers", type=int, default=8,
                        help="Number of CPU workers for DataLoader")

    # environment parameters
    parser.add_argument("--resume_from_checkpoint", action='store_true', default=False)
    parser.add_argument("--wandb_id", type=str, default=None)
    parser.add_argument("--dryrun", action='store_true', default=False)
    
    ## Dev
    # parser.add_argument('--regression_weight', type=float, default=0.1)
    parser.add_argument('--regression_weights', nargs="*", type=float)
    # parser.add_argument("--continuous_coefficient", action='store_true', default=False)
    # parser.add_argument("--continuous_exponent", action='store_true', default=False)
    # parser.add_argument("--support_learning", action='store_true', default=False)
    

    return parser


def main():
    import wandb 
    from transformers import TrainingArguments
    from trainer.trainer import CustomTrainer as Trainer
    from trainer.trainer import CustomTrainingArguments as TrainingArguments

    parser = get_parser()
    params = parser.parse_args()            

    os.makedirs(params.save_path, exist_ok=True)

    ## Load data
    with open(params.data_config_path, 'r') as f:
        data_config = yaml.safe_load(f)

    if params.encoding_method == 'hybrid' and params.field == 'QQ':
        data_encoding = 'infix+'
    else:
        data_encoding = 'infix'

    trainset    = _load_data(f'{params.data_path}.train.lex.{data_encoding}')
    testset     = _load_data(f'{params.data_path}.test.lex.{data_encoding}')

    if params.dryrun:
        trainset.input = trainset.input[:1000]
        trainset.target = trainset.target[:1000]
        params.epochs = 8
        params.save_path = os.path.join(os.path.dirname(params.save_path), 'dryrun')
        params.exp_name = 'dryrun'

    ## Load model
    ### standard embedding
    # if 'standard' in params.encoding_method:
    from dataset.tokernizer import set_tokenizer, set_vocab
    
    use_continous_token = params.encoding_method == 'hybrid'
    vocab = set_vocab(params.num_variables, 
                        field=params.field, 
                        max_coeff=params.max_coefficient, 
                        max_degree=params.max_degree, 
                        continuous_coefficient=use_continous_token, 
                        continuous_exponent=False)
    tokenizer = set_tokenizer(vocab, max_seq_length=params.max_sequence_length)
    model = load_model(params.model, params, vocab=vocab, tokenizer=tokenizer)
    

    if params.model == 'bart':
        dc = BartDataCollator(tokenizer)
        label_names = ['labels']
        
    elif params.model == 'bart+':
        dc = BartPlusDataCollator(tokenizer, continuous_coefficient=use_continous_token)
        
        if use_continous_token:
            label_names = ['labels', 'labels_for_regression']
        else:
            label_names = ['labels']
    else:
        raise ValueError(f'unknown model: {params.model}')
    
    # if not use_continous_token:
    #     dc = SimpleDataCollator(tokenizer) 
    #     label_names = ['labels']
    # else:
    #     dc = HybridDataCollator(tokenizer)
    #     label_names = ['labels', 'labels_for_regression']

    tokenizer.save_pretrained(os.path.join(params.save_path, "tokenizer.json"))


    ## Save parameters 
    with open(os.path.join(params.save_path, "params.yaml"), "w") as f:
        yaml.dump(vars(params), f)

    now = datetime.datetime.now(ZoneInfo("Europe/Berlin"))
    datetime_str = now.strftime("%Y%m%d_%H%M%S")
    run_name = f'{params.exp_id}_{datetime_str}'

    trainer_config = TrainingArguments(
        bf16                        = True,
        dataloader_num_workers      = 4,
        dataloader_pin_memory       = True,
        disable_tqdm                = True,
        eval_steps                  = 1000,
        eval_strategy               = 'steps',
        label_names                 = label_names,
        logging_steps               = 50,
        # max_steps_per_epoch         = params.max_steps_per_epoch,
        num_train_epochs            = params.epochs,
        # optim                       = params.optimizer,
        output_dir                  = params.save_path,
        per_device_train_batch_size = params.batch_size // count_cuda_devices(),
        remove_unused_columns       = False,
        report_to                   = "wandb",
        run_name                    = run_name,
        # save_steps                  = 100,
        save_total_limit            = 1,
        # torch_compile               = True,
    )

    # limit_steps_callback = LimitStepsCallback(max_steps_per_epoch=params.max_steps_per_epoch)

    _compute_metrics = lambda x: compute_metrics(x, ignore_index=tokenizer.pad_token_id)
    trainer = Trainer(
        args                            = trainer_config,
        model                           = model,
        train_dataset                   = trainset,
        eval_dataset                    = testset,
        data_collator                   = dc,
        compute_metrics                 = _compute_metrics,
        preprocess_logits_for_metrics   = preprocess_logits_for_metrics,
        # callbacks                       = [limit_steps_callback]
        )

    ## Run training
    if params.resume_from_checkpoint is not None:
        wandb.init(project=params.exp_name, 
                name=run_name,
                group=params.group,
                config=trainer_config)
    else:
        if params.wandb_id is None: 
            raise ValueError('wandb_id is required when resuming from checkpoint')
        else:
            wandb.init(project=params.exp_name, resume='must', id=params.wandb_id)


    s = time()
    train_result = trainer.train(resume_from_checkpoint=params.resume_from_checkpoint)
    print(f'training time: [{time()-s:.1f} sec]')

    trainer.save_model()
    
    metrics = train_result.metrics
    dataset_metrics = trainer.evaluate(metric_key_prefix="test")
    metrics.update(dataset_metrics)

    testloader = trainer.get_eval_dataloader()

    modulo, th, quantize_fn = None, 0.0, None
    if params.encoding_method == 'hybrid':
        if params.field.startswith('GF'):
            modulo, th = int(params.field[2:]), 0.5
            quantize_fn = lambda x: x.round()
        else:
            th =0.05

    scores = generation_accuracy(model, testloader, 
                                 max_length=params.max_sequence_length, 
                                 tokenizer=tokenizer, 
                                 disable_tqdm=True,
                                 th=th,
                                 modulo=modulo,
                                 model_name=params.model,
                                 quantize_fn=quantize_fn)
    
    metrics.update({'test generation accuracy': scores['acc']})
    trainer.save_metrics("all", metrics)
    wandb.log(metrics)

    wandb.finish()

if __name__ == '__main__':
    main()    
