from torch import nn
from transformers import AutoModel, AutoTokenizer, AutoConfig, BertModel
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions 
import json
from typing import List, Dict, Optional, Union, Tuple, Any
import os
from kobert_tokenizer import KoBERTTokenizer
import torch
#from tokenization_ranked import FullTokenizer as KBertRankedTokenizer



#model = SentenceTransformer(args.model_name_or_path) #model_name_or_path = bert_base

class Transformer(nn.Module):
    """Huggingface AutoModel to generate token embeddings.
    Loads the correct class, e.g. BERT / RoBERTa etc.

    :param model_name_or_path: Huggingface models name (https://huggingface.co/models)
    :param max_seq_length: Truncate any inputs longer than max_seq_length
    :param model_args: Arguments (key, value pairs) passed to the Huggingface Transformers model
    :param cache_dir: Cache dir for Huggingface Transformers to store/load models
    :param tokenizer_args: Arguments (key, value pairs) passed to the Huggingface Tokenizer model
    :param do_lower_case: Lowercase the input
    """
    def __init__(self, model_name_or_path: str, max_seq_length: int = 128,
                 model_args: Dict = {}, cache_dir: Optional[str] = None,
                 tokenizer_args: Dict = {}, do_lower_case: Optional[bool] = None,
                 attention_probs_dropout_prob: Optional[float] = None, hidden_dropout_prob: Optional[float] = None):
        super(Transformer, self).__init__()
        self.config_keys = ['max_seq_length']
        self.max_seq_length = max_seq_length

        if do_lower_case is not None:
            tokenizer_args['do_lower_case'] = do_lower_case

        config = AutoConfig.from_pretrained(model_name_or_path, **model_args, cache_dir=cache_dir)
        #config.vocab_size= 8002
        #dropout option
        if attention_probs_dropout_prob is not None:
            config.attention_probs_dropout_prob = attention_probs_dropout_prob
        if hidden_dropout_prob is not None:
            config.hidden_dropout_prob = hidden_dropout_prob
        self.auto_model = ConSERTModel.from_pretrained(model_name_or_path, config=config, cache_dir=cache_dir)
        if  "kobert" in model_name_or_path: #kobert
            self.tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1', sp_model_kwargs={'nbest_size': -1, 'alpha': 0.6, 'enable_sampling': True})
        else: # this could be krBERT or klue-bert
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, cache_dir=cache_dir, **tokenizer_args)
        


    def forward(self, features):
        """Returns token_embeddings, cls_token"""
        if "feature_cache" in self.__dict__:
            input_ids = features["input_ids"]
            attention_mask = features["attention_mask"]
            for input_id, mask in zip(input_ids, attention_mask):
                self.feature_cache.append({
                    "input_id": input_id.tolist(),
                    "attention_mask": mask.tolist()
                })
        
        output_states = self.auto_model(**features)
        # output_states = (last_layer, last_pooled_output, all_layers)
        output_tokens = output_states[0]
        #[batch, seq_len, dim]


        cls_tokens = output_tokens[:, 0, :]  # CLS token is first token
        # cls_tokens = [batch, dim]
        features.update({'token_embeddings': output_tokens, 'cls_token_embeddings': cls_tokens, 'attention_mask': features['attention_mask']})

        if self.auto_model.config.output_hidden_states:
            all_layer_idx = 2
            if len(output_states) < 3: # Some models only output last_hidden_states and all_hidden_states
                all_layer_idx = 1

            hidden_states = output_states[all_layer_idx]
            features.update({'all_layer_embeddings': hidden_states})

        return features

    def get_word_embedding_dimension(self) -> int:
        return self.auto_model.config.hidden_size

    # def tokenize(self, texts: Union[List[str], List[Dict], List[Tuple[str, str]]]):
    #         """
    #         Tokenizes a text and maps tokens to token-ids
    #         """
    #         output = {}
    #         if isinstance(texts[0], str):
    #             to_tokenize = [texts]
    #         elif isinstance(texts[0], dict):
    #             to_tokenize = []
    #             output['text_keys'] = []
    #             for lookup in texts:
    #                 text_key, text = next(iter(lookup.items()))
    #                 to_tokenize.append(text)
    #                 output['text_keys'].append(text_key)
    #             to_tokenize = [to_tokenize]
    #         else:
    #             batch1, batch2 = [], []
    #             for text_tuple in texts:
    #                 batch1.append(text_tuple[0])
    #                 batch2.append(text_tuple[1])
    #             to_tokenize = [batch1, batch2]

    #         #strip
    #         to_tokenize = [[str(s).strip() for s in col] for col in to_tokenize]

    #         #Lowercase
    #         if self.do_lower_case:
    #             to_tokenize = [[s.lower() for s in col] for col in to_tokenize]

    #         output.update(self.tokenizer(*to_tokenize, padding=True, truncation='longest_first', return_tensors="pt", max_length=self.max_seq_length))
    #         return output

    def tokenize(self, text: Union[str, List[str]]) -> List[int]:
        """
        Tokenizes a text and maps tokens to token-ids
        """
        if isinstance(text, str):
            return self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text))
        else:
            return [self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(t)) for t in text]

    def get_sentence_features(self, tokens: Union[List[int], List[List[int]]], pad_seq_length: int):
        """
        Convert tokenized sentence in its embedding ids, segment ids and mask

        :param tokens:
            a tokenized sentence
        :param pad_seq_length:
            the maximal length of the sequence. Cannot be greater than self.sentence_transformer_config.max_seq_length
        :return: embedding ids, segment ids and mask for the sentence
        """
        pad_seq_length = min(pad_seq_length, self.max_seq_length, self.auto_model.config.max_position_embeddings-3) + 3 #Add space for special tokens

        if len(tokens) == 0 or isinstance(tokens[0], int):
            return self.tokenizer.prepare_for_model(tokens, max_length=pad_seq_length, padding='max_length', return_tensors='pt', truncation=True, prepend_batch_axis=True)
        else:
            return self.tokenizer.prepare_for_model(tokens[0], tokens[1], max_length=pad_seq_length, padding='max_length', return_tensors='pt', truncation='longest_first', prepend_batch_axis=True)

    def get_config_dict(self):
        return {key: self.__dict__[key] for key in self.config_keys}

    def save(self, output_path: str):
        self.auto_model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)

        with open(os.path.join(output_path, 'sentence_bert_config.json'), 'w') as fOut:
            json.dump(self.get_config_dict(), fOut, indent=2)

    @staticmethod
    def load(input_path: str):
        #Old classes used other config names than 'sentence_bert_config.json'
        for config_name in ['sentence_bert_config.json', 'sentence_roberta_config.json', 'sentence_distilbert_config.json', 'sentence_camembert_config.json', 'sentence_albert_config.json', 'sentence_xlm-roberta_config.json', 'sentence_xlnet_config.json']:
            sbert_config_path = os.path.join(input_path, config_name)
            if os.path.exists(sbert_config_path):
                break

        with open(sbert_config_path) as fIn:
            config = json.load(fIn)
        return Transformer(model_name_or_path=input_path, **config)



class ConSERTModel(BertModel):
    """

    The model can behave as an encoder (with only self-attention) as well
    as a decoder, in which case a layer of cross-attention is added between
    the self-attention layers, following the architecture described in `Attention is all you need
    <https://arxiv.org/abs/1706.03762>`__ by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones,
    Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

    To behave as an decoder the model needs to be initialized with the
    :obj:`is_decoder` argument of the configuration set to :obj:`True`.
    To be used in a Seq2Seq model, the model needs to initialized with both :obj:`is_decoder`
    argument and :obj:`add_cross_attention` set to :obj:`True`; an
    :obj:`encoder_hidden_states` is then expected as an input to the forward pass.
    """


    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        encoder_hidden_states  (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
            if the model is configured as a decoder.
        encoder_attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask
            is used in the cross-attention if the model is configured as a decoder.
            Mask values selected in ``[0, 1]``:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        # Custom added, for data augmentation
        position_ids = self._replace_position_ids(input_ids, position_ids, attention_mask)  # replace the position ids, since data augmentation includes "shuffle"
        # ----- Custom added END ------------
        
        embedding_output = self.embeddings(
            input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
        )
        
        # Custom added, for data augmentation
        self._most_recent_embedding_output = embedding_output  # every time call forward, record the embedding output here
        embedding_output = self._replace_embedding_output(embedding_output, attention_mask)  # replace the embedding output, using different data augmentation strategies
        # ----- Custom added END ------------

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0] # (hidden_states, all_hidden_states, all_attentions) == (last layer, all_layers, none)
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]
            # ([batch, seq_len, dim], [batch, dim])  +  ()
            # encoder_outputs[1:] = all_layers_outputs,  not none when output_hidden_states is true

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

    # custom added functions for data augmentation
    def set_flag(self, key: str, value: Any):
        assert f"flag__{key}" not in self.__dict__
        self.__dict__[f"flag__{key}"] = value
    
    def unset_flag(self, key: str):
        assert f"flag__{key}" in self.__dict__
        del self.__dict__[f"flag__{key}"]
    
    def exists_flag(self, key: str):
        return f"flag__{key}" in self.__dict__
    
    def get_flag(self, key: str):
        assert f"flag__{key}" in self.__dict__
        return self.__dict__[f"flag__{key}"]
    
    def get_most_recent_embedding_output(self):
        return self._most_recent_embedding_output
    
    def _replace_embedding_output(self, embedding_output, attention_mask):
        bsz, seq_len, emb_size = embedding_output.shape
        if self.exists_flag("data_aug_cutoff"):
            direction = self.get_flag("data_aug_cutoff.direction")
            assert direction in ("row", "column", "random")  # "row" for the token level, "column" for the feature level, and "random" means randomly pick elements in embedding matrix and not restricted in row or column
            rate = self.get_flag("data_aug_cutoff.rate")
            assert isinstance(rate, float) and 0.0 < rate < 1.0
            self.unset_flag("data_aug_cutoff")
            self.unset_flag("data_aug_cutoff.direction")
            self.unset_flag("data_aug_cutoff.rate")
            embedding_after_cutoff = self._cutoff_embeddings(embedding_output, attention_mask, direction, rate)
            return embedding_after_cutoff
        elif self.exists_flag("data_aug_shuffle_embeddings"):
            self.unset_flag("data_aug_shuffle_embeddings")
            shuffled_embeddings = []
            for bsz_id in range(bsz):
                sample_embedding = embedding_output[bsz_id]
                sample_mask = attention_mask[bsz_id]
                num_tokens = sample_mask.sum().int().item()
                indexes = list(range(num_tokens))
                import random
                random.shuffle(indexes)
                rest_indexes = list(range(num_tokens, seq_len))
                total_indexes = indexes + rest_indexes
                shuffled_embeddings.append(torch.index_select(sample_embedding, 0, torch.tensor(total_indexes).to(device=embedding_output.device)).unsqueeze(0))
            return torch.cat(shuffled_embeddings, 0)
        else:
            return embedding_output
        
    def _replace_position_ids(self, input_ids, position_ids, attention_mask):
        bsz, seq_len = input_ids.shape
        if self.exists_flag("data_aug_shuffle"):
            self.unset_flag("data_aug_shuffle")
            
            if position_ids is None:
                position_ids = torch.arange(512).expand((bsz, -1))[:, :seq_len].to(device=input_ids.device)
            
            # shuffle position_ids
            shuffled_pid = []
            for bsz_id in range(bsz):
                sample_pid = position_ids[bsz_id]
                sample_mask = attention_mask[bsz_id]
                num_tokens = sample_mask.sum().int().item()
                indexes = list(range(num_tokens))
                import random
                random.shuffle(indexes)
                rest_indexes = list(range(num_tokens, seq_len))
                total_indexes = indexes + rest_indexes
                shuffled_pid.append(torch.index_select(sample_pid, 0, torch.tensor(total_indexes).to(device=input_ids.device)).unsqueeze(0))
            return torch.cat(shuffled_pid, 0)
        else:
            return position_ids


    ########## ADDED ###########
    def _cutoff_embeddings(self, embedding_output, attention_mask, direction, rate):
        bsz, seq_len, emb_size = embedding_output.shape
        cutoff_embeddings = []
        for bsz_id in range(bsz):
            sample_embedding = embedding_output[bsz_id]
            sample_mask = attention_mask[bsz_id]
            if direction == "row":
                num_dimensions = sample_mask.sum().int().item()  # number of tokens
                dim_index = 0
            elif direction == "column":
                num_dimensions = emb_size  # number of features
                dim_index = 1
            elif direction == "random":
                num_dimensions = sample_mask.sum().int().item() * emb_size
                dim_index = 0
            num_cutoff_indexes = int(num_dimensions * rate)
            indexes = list(range(num_dimensions))
            import random
            random.shuffle(indexes)
            cutoff_indexes = indexes[:num_cutoff_indexes]
            if direction == "random":
                sample_embedding = sample_embedding.reshape(-1)
            cutoff_embedding = torch.index_fill(sample_embedding, dim_index, torch.tensor(cutoff_indexes, dtype=torch.long).to(device=embedding_output.device), 0.0)
            if direction == "random":
                cutoff_embedding = cutoff_embedding.reshape(seq_len, emb_size)
            cutoff_embeddings.append(cutoff_embedding.unsqueeze(0))
            # cutoff_embedding = [seq_len, dim]
        cutoff_embeddings = torch.cat(cutoff_embeddings, 0)
        assert cutoff_embeddings.shape == embedding_output.shape, (cutoff_embeddings.shape, embedding_output.shape)
        return cutoff_embeddings
        ########## ADDED ###########
