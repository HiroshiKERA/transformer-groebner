
def load_model(model_name, params, tokenizer=None, vocab=None, cuda=True):

    if model_name == 'base':
        from transformers import PretrainedConfig
        from loader.models.transformer_base import TransformerForPolynomials

        assert(vocab is not None)
        assert(tokenizer is not None)


        special_token_ids = dict(zip([k + '_id' for k in  tokenizer.special_tokens_map], 
                                    tokenizer.convert_tokens_to_ids(tokenizer.special_tokens_map.values()))
                                    )
        
        vocab_size = tokenizer.vocab_size

        config = PretrainedConfig.from_dict({
            'encoding_method'       : params.encoding_method,
            'd_model'               : params.d_model,
            'nhead'                 : params.nhead,
            'num_encoder_layers'    : params.num_encoder_layers,
            'num_decoder_layers'    : params.num_decoder_layers,
            'dim_feedforward'       : params.dim_feedforward,
            'dropout'               : params.dropout,
            'special_token_ids'     : special_token_ids,
            'vocab_size'            : vocab_size,
            'max_sequence_length'   : params.max_sequence_length,
            'positional_encoding'   : params.positional_encoding,
            'regression_weight'     : params.regression_weight,
        })

        model = TransformerForPolynomials(config)

    elif model_name == 'bart':
        from transformers import BartConfig, BartForConditionalGeneration

        config = BartConfig(
            encoder_layers=params.num_encoder_layers,
            encoder_attention_heads=params.nhead,
            decoder_layers=params.num_decoder_layers,
            decoder_attention_heads=params.nhead,
            vocab_size=len(tokenizer.vocab),
            d_model=params.d_model,
            encoder_ffn_dim=params.dim_feedforward,
            decoder_ffn_dim=params.dim_feedforward,
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            cls_token_id=tokenizer.cls_token_id,
            sep_token_id=tokenizer.sep_token_id,
            unk_token_id=tokenizer.unk_token_id,
            max_position_embeddings=params.max_sequence_length,
            decoder_start_token_id=tokenizer.bos_token_id,
        )
        
        model = BartForConditionalGeneration(config)
        
    elif model_name == 'bart+':
        from transformers import BartConfig 
        from .models.bart import BartForConditionalGenerationPlus
        
        assert(tokenizer is not None)

        # continuous_vocab_size = sum([params.continuous_coefficient, params.continuous_exponent])
        continuous_vocab_size = int(params.encoding_method == 'hybrid')
        
        config = BartConfig(
            encoder_layers=params.num_encoder_layers,
            encoder_attention_heads=params.nhead,
            decoder_layers=params.num_decoder_layers,
            decoder_attention_heads=params.nhead,
            vocab_size=len(tokenizer.vocab),
            d_model=params.d_model,
            encoder_ffn_dim=params.dim_feedforward,
            decoder_ffn_dim=params.dim_feedforward,
            regression_weights=params.regression_weights, 
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            cls_token_id=tokenizer.cls_token_id,
            sep_token_id=tokenizer.sep_token_id,
            unk_token_id=tokenizer.unk_token_id,
            max_position_embeddings=params.max_sequence_length,
            decoder_start_token_id=tokenizer.bos_token_id,
            continuous_vocab_size=continuous_vocab_size,
            continuous_embedding_model=params.continuous_embedding_model,
        )
        
        model = BartForConditionalGenerationPlus(config)
        
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
    if cuda: model.cuda()
    
    return model

