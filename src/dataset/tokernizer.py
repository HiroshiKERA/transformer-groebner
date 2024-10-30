from tokenizers import Tokenizer as TokenizerBase
from tokenizers import pre_tokenizers
from tokenizers.pre_tokenizers import CharDelimiterSplit
from transformers import PreTrainedTokenizerFast
from tokenizers.models import WordLevel
from tokenizers.processors import TemplateProcessing

SPECIAL_TOKENS = ['[PAD]', '<s>', '</s>', '[CLS]']  # tokens skipped when "skip_special_tokens=True" at model.generate
SPECIAL_TOKENS_PLUS = ['[PAD]', '<s>', '</s>', '[CLS]', '[SEP]', '[UNK]']
SPECIAL_TOKEN_MAP = dict(zip(['pad_token', 'bos_token', 'eos_token', 'cls_token'], SPECIAL_TOKENS))
# SPECIAL_TOKEN_PLUS_MAP = dict(zip(['pad_token', 'bos_token', 'eos_token', 'cls_token', 'sep_token', 'unk_token'], 
#                               SPECIAL_TOKENS_PLUS))

# vocab = set_vocab(params.num_variables, params.field_charasteristics, params.max_int, params.integer_encoding)
def set_vocab(num_vars, field='QQ', max_coeff=100, max_degree=10, continuous_coefficient=False, continuous_exponent=False):
    SYMBOLS = [f'x{i}' for i in range(num_vars)]
    OPS = ['+', '*', '^', '/']
    CONSTS = ['[C]']  
    ECONSTS = ['[E]']
    MISC = []
    
    if field in ('RR') and not continuous_coefficient: 
        raise ValueError('RR should use continuous_coefficient=True')
    
    if not continuous_coefficient:
        if field in ('QQ', 'ZZ'):
            CONSTS += [f'C{i}' for i in range(-max_coeff, max_coeff+1)]
        elif field[:2] == 'GF': 
            assert(field[2:].isdigit())
            p = int(field[2:])
            CONSTS += [f'C{i}' for i in range(p)]
        else:
            raise ValueError(f'unknown field: {field}')
    
    if not continuous_exponent:
        ECONSTS += [f'E{i}' for i in range(max_degree+1)]
    
    return SYMBOLS + CONSTS + ECONSTS + OPS + MISC + SPECIAL_TOKENS_PLUS

def set_tokenizer(vocab, max_seq_length=1024):
    if type(vocab) is list: 
        vocab = dict(zip(vocab, range(len(vocab))))
    # special_tokens = {'[PAD]':0, '<s>':1, '</s>':2, '[CLS]':3}
    tok = TokenizerBase(WordLevel(vocab))
    # tok.pre_tokenizer = pre_tokenizers.Sequence([Whitespace(), Digits(individual_digits=True)])
    tok.pre_tokenizer = pre_tokenizers.Sequence([CharDelimiterSplit(' ')])
    # tok.add_special_tokens(list(special_tokens.keys()))
    tok.add_special_tokens(SPECIAL_TOKENS)
    tok.enable_padding()
    tok.no_truncation()
    
    bos_token = SPECIAL_TOKEN_MAP['bos_token']
    eos_token = SPECIAL_TOKEN_MAP['eos_token']
    tok.post_processor = TemplateProcessing(
        single=f"{bos_token} $A {eos_token}",
        special_tokens=[(bos_token, tok.token_to_id(bos_token)), \
                        (eos_token, tok.token_to_id(eos_token))],
    )
        
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tok, 
                                        model_max_length=max_seq_length, 
                                        **SPECIAL_TOKEN_MAP)
    return tokenizer