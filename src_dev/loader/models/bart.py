import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union, List, Callable

import torch
import torch.nn as nn
from torch import Tensor, LongTensor, FloatTensor
from torch.nn import CrossEntropyLoss, MSELoss

from transformers import (
    BartPreTrainedModel,
    BartModel,
    BartConfig,
    BartForConditionalGeneration,
    GenerationMixin
)
from transformers.models.bart.modeling_bart import BartEncoder, BartDecoder
from transformers.modeling_outputs import ModelOutput
from transformers.modeling_utils import logging

from loader.models.encoding import PositionalEncoding, HybridEmbedding
from loader.models.io import Seq2SeqLMOutputPlus

class CustomBartModel(BartModel):
    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight"]

    def __init__(self, config: BartConfig):
        super().__init__(config)
        self.encoder = BartEncoder(config)  # we will feed input embeds instead of ids.
        self.decoder = BartDecoder(config)
        # Initialize weights and apply final processing
        self.post_init()


class BartForPolynomialSystemGeneration(BartPreTrainedModel, GenerationMixin):
    base_model_prefix = "model"
    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight"] #, "lm_head.weight"]
    _keys_to_ignore_on_load_missing = ["final_logits_bias"]

    def __init__(self, config: BartConfig):
        super().__init__(config)

        self.model = CustomBartModel(config)
        self.input_embedding = HybridEmbedding(discrete_vocab_size      = config.vocab_size,
                                               continuous_hidden_dim    = config.d_model, 
                                               embedding_dim            = config.d_model,
                                               padding_idx              = config.pad_token_id,
                                               continuous_vocab_size    = config.continuous_vocab_size,
                                               continuous_embedding_model = config.continuous_embedding_model)

        virtual_vocab_size = self.input_embedding.num_embeddings
        self.register_buffer("final_logits_bias", torch.zeros((1, virtual_vocab_size)))
        self.lm_head = nn.Linear(config.d_model, self.input_embedding.num_embeddings, bias=False)

        self.special_token_ids = {'pad_token_id': config.pad_token_id,
                                  'bos_token_id': config.bos_token_id,
                                  'eos_token_id': config.eos_token_id,
                                  'cls_token_id': config.cls_token_id,
                                  'sep_token_id': config.sep_token_id,
                                  'unk_token_id': config.unk_token_id}

        self.post_init()

    def forward(
        self,
        input_ids               : torch.LongTensor = None,
        attention_mask          : Optional[torch.Tensor] = None,
        decoder_input_ids       : Optional[torch.LongTensor] = None,
        decoder_attention_mask  : Optional[torch.LongTensor] = None,
        head_mask               : Optional[torch.Tensor] = None,
        decoder_head_mask       : Optional[torch.Tensor] = None,
        cross_attn_head_mask    : Optional[torch.Tensor] = None,
        encoder_outputs         : Optional[List[torch.FloatTensor]] = None,
        past_key_values         : Optional[List[torch.FloatTensor]] = None,
        inputs_embeds           : Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds   : Optional[torch.FloatTensor] = None,
        labels                  : Optional[torch.LongTensor] = None,
        input_continuous_labels : Optional[torch.FloatTensor] = None,
        target_continuous_labels: Optional[torch.FloatTensor] = None,
        labels_for_regression       : Optional[torch.FloatTensor] = None,
        use_cache               : Optional[bool] = None,
        output_attentions       : Optional[bool] = None,
        output_hidden_states    : Optional[bool] = None,
        return_dict             : Optional[bool] = None,
    ) -> Union[Tuple, Seq2SeqLMOutputPlus]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        inputs_embeds           = self.input_embedding(input_ids, input_continuous_labels) 
        decoder_inputs_embeds   = self.input_embedding(decoder_input_ids, target_continuous_labels) 


        outputs = self.model(
            None, # input_ids,
            attention_mask          = attention_mask,
            decoder_attention_mask  = decoder_attention_mask,
            head_mask               = head_mask,
            decoder_head_mask       = decoder_head_mask,
            cross_attn_head_mask    = cross_attn_head_mask,
            inputs_embeds           = inputs_embeds,
            decoder_inputs_embeds   = decoder_inputs_embeds,
        )

        lm_logits = self.lm_head(outputs[0])

        cv_size = self.input_embedding.continuous_vocab_size
        d_logits = lm_logits[:, :, :lm_logits.shape[-1]-cv_size]
        if cv_size > 0:
            c_logits = lm_logits[:, :, -cv_size:]

        masked_lm_loss = None
        if labels is not None:
            labels = labels.to(d_logits.device)
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            masked_lm_loss = loss_fct(d_logits.view(-1, self.config.vocab_size), labels.view(-1))

        regression_losses = None
        if labels_for_regression is not None and cv_size > 0:
            labels_for_regression = labels_for_regression.to(lm_logits.device)
            is_continuous = labels_for_regression.isfinite()
            
            loss_fct = nn.MSELoss()
            regression_losses = []
            for i in range(cv_size):
                is_continuous_i = is_continuous[:, :, i]
                c_logits_i = c_logits[:, :, i]
                labels_for_regression_i = labels_for_regression[:, :, i]
                mse_loss = loss_fct(c_logits_i[is_continuous_i], labels_for_regression_i[is_continuous_i])

                regression_losses.append(mse_loss)

            weights = self.config.regression_weights
            if len(weights) == 1: weights *= len(regression_losses)

        loss = masked_lm_loss
        loss_bag = [masked_lm_loss]

        if regression_losses is not None:
            loss_bag.extend(regression_losses)
            loss += sum([w * rl for w, rl in zip(weights, regression_losses)])

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((loss,) + output) if any(loss) else output

        return Seq2SeqLMOutputPlus(
            loss                        = loss, # masked_lm_loss,
            logits                      = d_logits,
            logits_for_regression       = c_logits if cv_size > 0 else torch.tensor([]),
            loss_bag                    = loss_bag,
            past_key_values             = outputs.past_key_values,
            decoder_hidden_states       = outputs.decoder_hidden_states,
            decoder_attentions          = outputs.decoder_attentions,
            cross_attentions            = outputs.cross_attentions,
            encoder_last_hidden_state   = outputs.encoder_last_hidden_state,
            encoder_hidden_states       = outputs.encoder_hidden_states,
            encoder_attentions          = outputs.encoder_attentions,
        )

    def get_encoder(self):
        return self.model.get_encoder()

    def get_decoder(self):
        return self.model.get_decoder()

    def resize_token_embeddings(self, new_num_tokens: int, pad_to_multiple_of: Optional[int] = None) -> nn.Embedding:
        new_embeddings = super().resize_token_embeddings(new_num_tokens, pad_to_multiple_of)
        self._resize_final_logits_bias(new_embeddings.weight.shape[0])
        return new_embeddings

    def _resize_final_logits_bias(self, new_num_tokens: int) -> None:
        old_num_tokens = self.final_logits_bias.shape[-1]
        if new_num_tokens <= old_num_tokens:
            new_bias = self.final_logits_bias[:, :new_num_tokens]
        else:
            extra_bias = torch.zeros((1, new_num_tokens - old_num_tokens), device=self.final_logits_bias.device)
            new_bias = torch.cat([self.final_logits_bias, extra_bias], dim=1)
        self.register_buffer("final_logits_bias", new_bias)

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            # cached cross_attention states don't have to be reordered -> they are always the same
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past[:2])
                + layer_past[2:],
            )
        return reordered_past

    @torch.no_grad()
    def generate(
        self,
        input_ids: Tensor,
        input_continuous_labels: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        max_length: Optional[int] = None,
        continuous_token_ids: Optional[List] = [],
        quantize_fn: Optional[Callable] = None,
    ) -> dict:
        
        has_continuous_tokens = self.input_embedding.has_continuous_embedding
        continuous_vocab_size = self.input_embedding.continuous_vocab_size
        vocab_size            = self.input_embedding.num_embeddings
        device                = self.device
        eos_token_id          = self.config.eos_token_id

        if len(continuous_token_ids) < continuous_vocab_size:
            raise ValueError(f'continuous_token_ids should be provided for all continuous tokens. {len(continuous_token_ids)} < {continuous_vocab_size}')

        input_ids = input_ids.to(device)
        input_continuous_labels = input_continuous_labels.to(device)

        if max_length == None: 
            max_length = self.config.max_position_embeddings
                           
        inputs_embeds = self.input_embedding(input_ids, input_continuous_labels)
        
        # decoderへの入力を作成
        dec_input = torch.full((input_ids.shape[0], max_length+1), self.config.pad_token_id, dtype=torch.float32).to(device) # fill with padding
        target_continuous_labels = torch.full((*dec_input.shape, continuous_vocab_size), torch.nan, dtype=torch.float32).to(device)

        dec_input[:, 0] = self.config.bos_token_id  # 先頭を開始タグにする
    
        # マスクを作成
        decoder_attention_mask = torch.full_like(dec_input, False, dtype=torch.bool).to(device)
        decoder_attention_mask[:, 0] = True

        finished = dec_input[:, 0] == eos_token_id
        finished_len = max_length - 1
        for i in range(max_length):
            decoder_inputs_embeds = self.input_embedding(dec_input[~finished], target_continuous_labels[~finished])
            outputs = self.model(
                None,
                attention_mask          = attention_mask[~finished],
                decoder_attention_mask  = decoder_attention_mask[~finished],
                inputs_embeds           = inputs_embeds[~finished],
                decoder_inputs_embeds   = decoder_inputs_embeds,)

            logits = self.lm_head(outputs[0])[:, i, :]  # not i+1

            if has_continuous_tokens:
                d_logits = logits[:, :-continuous_vocab_size]
                c_logits = logits[:, -continuous_vocab_size:]
            else:
                d_logits = logits

            preds = d_logits.argmax(dim=-1) 
            
            dec_input[~finished, i+1] = preds.to(d_logits.dtype) 
            decoder_attention_mask[:, i+1] = True

            for k in range(continuous_vocab_size):
                is_continuous_k = (preds == continuous_token_ids[k])
                if quantize_fn is not None: 
                    c_logits[is_continuous_k, k] = quantize_fn(c_logits[is_continuous_k, k])

                tmp = target_continuous_labels[~finished]
                tmp[is_continuous_k, i+1, k] = c_logits[is_continuous_k, k]
                target_continuous_labels[~finished] = tmp

            finished[~finished] = preds == eos_token_id
            
            if finished.all():
                finished_len = i
                break

        # if not finished.all():
        #     raise ValueError(f'max_length={max_length} is too short to finish the generation. {finished_len} < {max_length}')

        prediction = dec_input[:, :finished_len+1]
        continuous_prediction = target_continuous_labels[:, :finished_len+1]
        return {'prediction'            : prediction, 
                'continuous_prediction' : continuous_prediction,
                'continuous_vocab_size' : continuous_vocab_size,
                'continuous_token_ids'  : continuous_token_ids,
                'quantized'             : quantize_fn is not None}

    # def prepare_inputs_for_generation(
    #     self,
    #     decoder_input_ids,
    #     past_key_values         = None,
    #     attention_mask          = None,
    #     decoder_attention_mask  = None,
    #     head_mask               = None,
    #     decoder_head_mask       = None,
    #     cross_attn_head_mask    = None,
    #     use_cache               = None,
    #     encoder_outputs         = None,
    #     **kwargs,
    # ):
    #     # cut decoder_input_ids if past_key_values is used

    #     if past_key_values is not None:
    #         past_length = past_key_values[0][0].shape[2]

    #         # Some generation methods already pass only the last input ID
    #         if decoder_input_ids.shape[1] > past_length:
    #             remove_prefix_length = past_length
    #         else:
    #             # Default to old behavior: keep only final ID
    #             remove_prefix_length = decoder_input_ids.shape[1] - 1

    #         decoder_input_ids = decoder_input_ids[:, remove_prefix_length:]
        
    #     return {
    #         "input_ids": None,  # encoder_outputs is defined. input_ids not needed
    #         "encoder_outputs": encoder_outputs,
    #         "past_key_values": past_key_values,
    #         "decoder_input_ids": decoder_input_ids,
    #         "attention_mask": attention_mask,
    #         "decoder_attention_mask": decoder_attention_mask,
    #         "head_mask": head_mask,
    #         "decoder_head_mask": decoder_head_mask,
    #         "cross_attn_head_mask": cross_attn_head_mask,
    #         "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
    #     }


    @torch.no_grad()
    def postprocess_prediction(self, generation_output: dict, tokenizer, skip_special_tokens: bool = False):

        pred_texts = self.decode_continuous_ids(generation_output['prediction'], 
                                                generation_output['continuous_prediction'], 
                                                tokenizer, 
                                                quantized               = generation_output['quantized'], 
                                                continuous_vocab_size   = generation_output['continuous_vocab_size'], 
                                                continuous_tokens       = ['[C]', '[E]'], 
                                                skip_special_tokens     = skip_special_tokens)

        generation_output['prediction_texts'] = pred_texts
        return generation_output

    
    @staticmethod
    @torch.no_grad()
    def decode_continuous_ids(ids: LongTensor, continuous_ids: FloatTensor, tokenizer, quantized=False, continuous_vocab_size=0, continuous_tokens = ['[C]', '[E]'], skip_special_tokens=True):
        
        # set up special tokens
        special_tokens = [t for t in tokenizer.all_special_tokens if t != tokenizer.sep_token]
        end_tokens = [tokenizer.eos_token_id, tokenizer.pad_token_id]
        continuous_tokens = continuous_tokens[:continuous_vocab_size]

        if continuous_vocab_size == 0:
            # standard decoding
            texts = tokenizer.batch_decode(ids.long(), skip_special_tokens=skip_special_tokens)
            return texts

        # decode without skips
        texts = tokenizer.batch_decode(ids.long(), skip_special_tokens=False)
        
        for k, text in enumerate(texts):
            tokens = []
            for i, token in enumerate(text.split()):
                if skip_special_tokens and token in end_tokens: break 
                if skip_special_tokens and token in special_tokens: continue

                if token in continuous_tokens:
                    for j, cont_token in enumerate(continuous_tokens):
                        assert (cont_token[0] == '[' and cont_token[-1] == ']')
                        cont_value = continuous_ids[k, i, j]
                        if quantized: cont_value = int(cont_value)
                        if token == cont_token: tokens.append(f'{cont_token[1:-1]}{cont_value}')
                else:
                    tokens.append(token)

            texts[k] = ' '.join(tokens)

        return texts

    
def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    '''
    Shift input ids one token to the right.
    '''
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    if pad_token_id is None:
        raise ValueError("self.model.config.pad_token_id has to be defined.")
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids