import torch
import yaml 
import os 
import argparse
import re 
from transformers import AutoModelForSeq2SeqLM, AutoConfig, BartConfig
from transformers import PretrainedConfig
from transformers import PreTrainedTokenizerFast
from .model import load_model
from safetensors import safe_open

# def load_args(save_dir):
#     config_file = os.path.join(save_dir, 'params.yaml')
#     with open(config_file, 'r') as f:
#         config = yaml.safe_load(f)
#     args = argparse.Namespace(**config)
#     return args 

# def load_config(save_dir):
#     config_file = os.path.join(save_dir, 'config.json')
#     with open(config_file, 'r') as f:
#         config = yaml.safe_load(f)
#     args = argparse.Namespace(**config)
#     return args 

def load_config(save_path):
    save_path            
    config_path = os.path.join(save_path, 'config.json')
    config = PretrainedConfig.from_json_file(config_path)

    config.regression_weight = 0.1
    return config

def load_pretrained_model(config, save_path, checkpoint_path=None, device_id=0):
    if checkpoint_path is None: 
        checkpoint_path = save_path
    
    if config.use_standard_embedding:
        tokenizer = load_tokenizer(checkpoint_path)
        model = load_model(config, vocab=tokenizer.vocab, tokenizer=tokenizer)
    else:
        ## TODO
        vocab_map = {'pad_token_id': 1,
                'bos_token_id': 2,
                'eos_token_id': 3,
                'sep_token_id': 4,
                'number_token_id': 0}

        model = load_model(config, vocab=vocab_map)
    model.cuda().eval()
    
    state_dict = {}
    with safe_open(f"{checkpoint_path}/model.safetensors", framework="pt", device=device_id) as f:
        for k in f.keys():
            state_dict[k] = f.get_tensor(k)
    model.load_state_dict(state_dict)
    model.eval().cuda()
    
    return model, tokenizer

def get_checkpoint_id(save_dir):
    cpt_file = [f for f in os.listdir(save_dir) if 'checkpoint' in f][0]
    cpid = int(re.search(r'checkpoint-(\d+)', cpt_file).group(1))
    return cpid 

def load_tokenizer(save_dir):
    tokenizer = PreTrainedTokenizerFast.from_pretrained(os.path.join(save_dir, f'tokenizer.json'))
    return tokenizer

# def load_pretrained_model(save_dir, tokenizer, model_name, from_checkpoint=False):
#     if from_checkpoint:
#         cpid = get_checkpoint_id(save_dir)
#         # config = AutoConfig.from_pretrained(os.path.join(save_dir, f'checkpoint-{cpid}/config.json'))
#         config = BartConfig.from_pretrained(os.path.join(save_dir, f'checkpoint-{cpid}/config.json'))
#         # model = AutoModelForSeq2SeqLM.from_pretrained(os.path.join(save_dir, f'checkpoint-{cpid}/pytorch_model.bin'), config=config)
#         # model = AutoModelForSeq2SeqLM.from_pretrained(os.path.join(save_dir, f'checkpoint-{cpid}/model.safetensors'), config=config, use_safetensors=True)
#         model = BartForPolynomialSystemGeneration.from_pretrained(os.path.join(save_dir, f'checkpoint-{cpid}/model.safetensors'), config=config, use_safetensors=True)
#         # model = load_model(config, tokenizer, model=model_name)
#         # model.from_pretrained(os.path.join(save_dir, f'checkpoint-{cpid}/model.safetensors'), config=config, use_safetensors=True)
#         # model = load_model(config, tokenizer, model=model_name)
#     else:
#         # config = AutoConfig.from_pretrained(os.path.join(save_dir, f'config.json'))
#         config = BartConfig.from_pretrained(os.path.join(save_dir, f'config.json'))
#         if not hasattr(config, 'continuous_embedding_model'):
#             setattr(config, 'continuous_embedding_model', 'ffn')

#         # print(config)
#         # exit()
#         # model = load_model(config, tokenizer, model=model_name)
#         model = BartForPolynomialSystemGeneration.from_pretrained(os.path.join(save_dir, f'model.safetensors'), config=config, use_safetensors=True)    
#         # model = AutoModelForSeq2SeqLM.from_pretrained(os.path.join(save_dir, f'pytorch_model.bin'), config=config)    
        
#     model.eval().cuda()
#     return model 

# def load_trained_bag(save_dir, from_checkpoint=False):
#     params = load_args(save_dir)
#     tokenizer = load_tokenizer(save_dir, from_checkpoint=from_checkpoint)
#     model = load_pretrained_model(save_dir, tokenizer, model_name=model_name, from_checkpoint=from_checkpoint)
#     model.to_bettertransformer()
#     bag = {'model': model, 'params': params, 'tokenizer': tokenizer}
    
#     return bag

def load_pretrained_bag(save_path, from_checkpoint=False):
    if from_checkpoint:
        cpid = get_checkpoint_id(save_path)
        ckpt_path = os.path.join(ckpt_path, f'checkpoint-{cpid}')
    else:
        ckpt_path = save_path

    config = load_config(ckpt_path)
    model, tokernizer = load_pretrained_model(config, save_path, checkpoint_path=ckpt_path)
    return {'model': model, 'config': config, 'tokenizer': tokernizer}
    
    