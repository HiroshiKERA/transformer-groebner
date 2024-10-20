import torch
import torch.nn as nn
from transformers import PreTrainedModel

from loader.models.encoding import PositionalEncoding, PositionalEmbedding, HybridEmbedding


class TransformerForPolynomials(PreTrainedModel):
    # _tied_weights_keys = ['token_embedding', 'classifier.weight']
    
    def __init__(self, config):
        super().__init__(config)
        
        self.config = config
        self.d_model = config.d_model
        self.encoding_method = config.encoding_method
        
        self.special_token_ids = config.special_token_ids
        self.vocab_size = config.vocab_size
        self.max_sequence_length = config.max_sequence_length
        self.lambda_rg = config.regression_weight
        
        ## embedding layers
        if self.encoding_method == 'standard':
            self.token_embedding = nn.Embedding(self.vocab_size, self.d_model)
            self.embedding = nn.Identity()
        
        elif self.encoding_method == 'hybrid':
            self.token_embedding = HybridEmbedding(self.vocab_size, 
                                                    continuous_hidden_dim=self.d_model, 
                                                    embedding_dim=self.d_model, 
                                                    padding_idx=self.special_token_ids['pad_token_id'], 
                                                    continuous_embedding_model='ffn')
            self.embedding = nn.Identity()

        else:
            raise ValueError(f"Unknown encoding method: {self.encoding_method}")
        
        
        ## position encodings        
        if self.config.positional_encoding == 'sinusoidal':
            self.positional_encoding = PositionalEncoding(self.d_model, max_len=self.max_sequence_length)
        elif self.config.positional_encoding == 'embedding':
            self.positional_encoding = PositionalEmbedding(self.d_model, max_len=self.max_sequence_length)
        else:
            raise ValueError(f"Unknown positional encoding method: {self.config.positional_encoding}")
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, 
            nhead=config.nhead, 
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation=nn.GELU(),
            batch_first=True,
            norm_first=False
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=config.num_encoder_layers
        )
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.d_model, 
            nhead=config.nhead, 
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation=nn.GELU(),
            batch_first=True,
            norm_first=False
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer, 
            num_layers=config.num_decoder_layers
        )
        
        self.classifier = nn.Linear(self.d_model, self.vocab_size, bias=True)
        if self.encoding_method == 'hybrid':
            self.regression_head = nn.Linear(self.d_model, 1, bias=True)  
        
        
        self.loss_clf = nn.CrossEntropyLoss(ignore_index=self.special_token_ids['pad_token_id'])
        self.loss_rg = lambda x, y: nn.MSELoss(reduction='mean')(x.flatten(), y)
        
        self.post_init()

    def forward(
        self,
        encoder_input,
        decoder_input,
        encoder_input_labels=None,
        decoder_input_labels=None,
        labels=None,
        labels_for_regression=None,
        encoder_attention_mask=None,
        decoder_attention_mask=None,
        encoder_padding_mask=None,
        decoder_padding_mask=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # preembedded, perform_shift_right = False, True
        # if self.encoding_method == 'hybrid':
        #     encoder_input = self.token_embedding(encoder_input, labels_for_regression)
            
        #     # shifted_labels_for_regression = 
        #     print(labels_for_regression)
        #     decoder_input = self.token_embedding(self._shift_right(decoder_input), shifted_labels_for_regression)
        #     preembedded = True
        #     perform_shift_right = False
            
        encoder_output = self.encode(encoder_input, encoder_attention_mask, encoder_padding_mask, encoder_input_labels=encoder_input_labels)
        decoder_output = self.decode(decoder_input, encoder_output, decoder_attention_mask, encoder_padding_mask, decoder_padding_mask, encoder_attention_mask, decoder_input_labels=decoder_input_labels)

        logits = self.classifier(decoder_output)

        loss = None
        if labels is not None:
            loss_clf = self.loss_clf(logits.reshape(-1, self.vocab_size), labels.reshape(-1).long())
            
            loss_rg = torch.tensor(0.0).to(loss_clf.device)

            if labels_for_regression is not None:
                is_continuous_token = labels_for_regression.isfinite()
                logits_for_regression = self.regression_head(decoder_output)
                loss_rg = self.loss_rg(logits_for_regression[is_continuous_token], labels_for_regression[is_continuous_token])
            else:
                logits_for_regression = None
            
            loss = loss_clf + loss_rg * self.lambda_rg

        # if not return_dict:
        #     print('return_dict is False!!')
        #     output = (decoded_output,) + (encoder_output,)
        #     return ((loss,) + output) if loss is not None else output
        
        return {
            'loss': loss,
            'loss_clf': loss_clf,
            'loss_rg': loss_rg,
            'logits': logits,
            'logits_for_regression': logits_for_regression,
            'encoder_output': encoder_output,
        }

    def _shift_right(self, x, preembedded=False):
        
        shifted_input_embeds = torch.zeros_like(x)
        shifted_input_embeds[:, 0] = self.special_token_ids['bos_token_id']
        shifted_input_embeds[:, 1:] = x[:, :-1].clone()

        return shifted_input_embeds

    def generate_square_subsequent_mask(self, sz, dtype=float):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        if dtype == bool: mask = mask.isinf()
        return mask

    def encode(self, encoder_input, encoder_attention_mask=None, encoder_padding_mask=None, encoder_input_labels=None, preembedded=False):

        if not preembedded:
            if encoder_input_labels is not None:
                encoder_embedded = self.token_embedding(encoder_input.long(), encoder_input_labels)
            else:
                encoder_embedded = self.token_embedding(encoder_input.long())
        else:
            encoder_embedded = encoder_input
        
        pe = self.positional_encoding(encoder_embedded)
        encoder_embedded += pe
        
        if encoder_padding_mask is None:
            encoder_padding_mask = torch.zeros(encoder_input.shape[:2], dtype=torch.bool, device=encoder_input.device)
        
        encoder_output = self.encoder(
            encoder_embedded,
            src_key_padding_mask=encoder_padding_mask,
            mask=encoder_attention_mask
        )
        
        return encoder_output

    def decode(self, decoder_input, encoder_output, decoder_attention_mask=None, encoder_padding_mask=None, decoder_padding_mask=None, encoder_attention_mask=None, perform_shift_right=True, preembedded=False, decoder_input_labels=None):

        decoder_input_shifted = self._shift_right(decoder_input) if perform_shift_right else decoder_input

        if not preembedded:
            if decoder_input_labels is not None:
                shifted_decoder_input_labels = torch.cat([torch.full((decoder_input_labels.shape[0], 1), torch.inf, device=decoder_input_labels.device), decoder_input_labels[:, :-1]], dim=-1)
                decoder_embedded = self.token_embedding(decoder_input_shifted.long(), shifted_decoder_input_labels)
            else:
                decoder_embedded = self.token_embedding(decoder_input_shifted.long())
        else:
            decoder_embedded = decoder_input_shifted

        pe = self.positional_encoding(decoder_embedded)
        decoder_embedded += pe
        
        if decoder_padding_mask is None:
            decoder_padding_mask = torch.zeros(decoder_input.shape[:2], dtype=torch.bool, device=decoder_input.device)
        
        decoder_output = self.decoder(
            decoder_embedded,
            encoder_output,
            tgt_key_padding_mask=decoder_padding_mask,
            memory_key_padding_mask=encoder_padding_mask,
            tgt_mask=self.generate_square_subsequent_mask(decoder_input.size(1), dtype=bool).to(decoder_input.device),
            memory_mask=encoder_attention_mask,
            tgt_is_causal=True
        )
        
        return decoder_output

    @torch.no_grad()
    def greedy_generate(
        self,
        encoder_input,
        encoder_input_labels=None,
        max_length=100,
        encoder_attention_mask=None,
        encoder_padding_mask=None,
        continuous_token_ids = None,
        quantize_fn= lambda x: x,
    ):
        has_continuous_tokens = encoder_input_labels is not None
        
        batch_size = encoder_input.shape[0]
        device = encoder_input.device

        encoder_embedded = self.encode(encoder_input, encoder_attention_mask, encoder_padding_mask, encoder_input_labels=encoder_input_labels)
        decoder_input = torch.full((batch_size, max_length+1), self.special_token_ids['bos_token_id'], device=device) 
        if has_continuous_tokens:
            decoder_input_labels = torch.full((batch_size, max_length+1), torch.inf, device=device)

        eos = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for k in range(max_length):
            _decoder_input = decoder_input[:, :k+1][~eos]
            _decoder_input_labels = decoder_input_labels[~eos, :k+1] if has_continuous_tokens else None
            
            _encoder_embedded = encoder_embedded[~eos]
            _encoder_padding_mask = encoder_padding_mask[~eos] if encoder_padding_mask is not None else None
            _encoder_attention_mask = encoder_attention_mask[~eos] if encoder_attention_mask is not None else None
            
            decoder_output = self.decode(_decoder_input, _encoder_embedded, 
                                         encoder_padding_mask=_encoder_padding_mask, 
                                         encoder_attention_mask=_encoder_attention_mask,
                                         perform_shift_right=False,
                                         decoder_input_labels=_decoder_input_labels)
            
            
            logits = self.classifier(decoder_output[:, -1:, :])
            next_input = logits.argmax(dim=-1)

            if has_continuous_tokens:
                is_continuous_next_token = torch.isin(next_input, continuous_token_ids)
                logits_for_regression = self.regression_head(decoder_output[:, -1:, :])
                next_input_labels = quantize_fn(logits_for_regression).squeeze(-1)
                next_input_labels[~is_continuous_next_token] = torch.inf

            decoder_input[~eos, k+1:k+2] = next_input
            if has_continuous_tokens:
                decoder_input_labels[~eos, k+1:k+2] = next_input_labels

            eos[~eos] |= (next_input == self.special_token_ids['eos_token_id']).flatten()
            if eos.all(): 
                decoder_input = decoder_input[:, :k+2]
                decoder_input_labels = decoder_input_labels[:, :k+2] if has_continuous_tokens else None
                break
        
        return (decoder_input, decoder_input_labels) if has_continuous_tokens else decoder_input