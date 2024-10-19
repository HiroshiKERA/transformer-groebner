from transformers import PretrainedConfig
from loader.models.transformer_base import TransformerForPolynomials

def load_model(params, vocab=None, tokenizer=None):

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

    model = TransformerForPolynomials(config).cuda()

    return model

