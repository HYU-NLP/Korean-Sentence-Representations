import torch
from torch import Tensor
from torch import nn
from typing import Union, Tuple, List, Iterable, Dict, Optional
import os
import json
from transformers import AutoModel, AutoTokenizer, AutoConfig   #where these models from?>

class Pooling(nn.Module):
    """Performs pooling (max or mean) on the token embeddings.

    Using pooling, it generates from a variable sized sentence a fixed sized sentence embedding. This layer also allows to use the CLS token if it is returned by the underlying word embedding model.
    You can concatenate multiple poolings together.

    :param word_embedding_dimension: Dimensions for the word embeddings
    :param pooling_mode_cls_token: Use the first token (CLS token) as text representations
    :param pooling_mode_max_tokens: Use max in each dimension over all tokens.
    :param pooling_mode_mean_tokens: Perform mean-pooling
    :param pooling_mode_mean_sqrt_len_tokens: Perform mean-pooling, but devide by sqrt(input_length).
    """
    def __init__(self,
                 word_embedding_dimension: int,
                 pooling_mode_cls_token: bool = False,
                 pooling_mode_max_tokens: bool = False,
                 pooling_mode_mean_tokens: bool = True,
                 #Consert uses mean_tokens
                 pooling_mode_mean_sqrt_len_tokens: bool = False,
                 pooling_mode_mean_last_2_tokens: bool = False,
                 pooling_mode_mean_first_last_tokens: bool = False,  # same as bert-flow, see https://github.com/bohanli/BERT-flow/issues/11
                 pooling_mode_pad_max_tokens: bool = False,
                 pooling_mode_pad_mean_tokens: bool = False,
                 ):
        super(Pooling, self).__init__()

        self.config_keys = ['word_embedding_dimension',  'pooling_mode_cls_token', 'pooling_mode_mean_tokens', 'pooling_mode_max_tokens', 'pooling_mode_mean_sqrt_len_tokens', 'pooling_mode_mean_last_2_tokens']

        self.word_embedding_dimension = word_embedding_dimension
        self.pooling_mode_cls_token = pooling_mode_cls_token
        self.pooling_mode_mean_tokens = pooling_mode_mean_tokens
        self.pooling_mode_max_tokens = pooling_mode_max_tokens
        self.pooling_mode_mean_sqrt_len_tokens = pooling_mode_mean_sqrt_len_tokens
        self.pooling_mode_mean_last_2_tokens = pooling_mode_mean_last_2_tokens
        self.pooling_mode_mean_first_last_tokens = pooling_mode_mean_first_last_tokens
        self.pooling_mode_pad_max_tokens = pooling_mode_pad_max_tokens
        self.pooling_mode_pad_mean_tokens = pooling_mode_pad_mean_tokens

        pooling_mode_multiplier = sum([pooling_mode_cls_token, pooling_mode_max_tokens, pooling_mode_mean_tokens, pooling_mode_mean_sqrt_len_tokens])
        self.pooling_output_dimension = (pooling_mode_multiplier * word_embedding_dimension)

    def forward(self, features: Dict[str, Tensor]):
        token_embeddings = features['token_embeddings']
        cls_token = features['cls_token_embeddings']
        attention_mask = features['attention_mask']
#         assert features["all_layer_embeddings"][-1].sum() == features["token_embeddings"].sum()

        ## Pooling strategy
        output_vectors = []
        if self.pooling_mode_cls_token:
            output_vectors.append(cls_token)
        if self.pooling_mode_max_tokens:
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            token_embeddings[input_mask_expanded == 0] = -1e9  # Set padding tokens to large negative value
            max_over_time = torch.max(token_embeddings, 1)[0]
            output_vectors.append(max_over_time)
        if self.pooling_mode_mean_tokens or self.pooling_mode_mean_sqrt_len_tokens:
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)

            #If tokens are weighted (by WordWeights layer), feature 'token_weights_sum' will be present
            if 'token_weights_sum' in features:
                sum_mask = features['token_weights_sum'].unsqueeze(-1).expand(sum_embeddings.size())
            else:
                sum_mask = input_mask_expanded.sum(1)

            sum_mask = torch.clamp(sum_mask, min=1e-9)

            if self.pooling_mode_mean_tokens:
                output_vectors.append(sum_embeddings / sum_mask)
            if self.pooling_mode_mean_sqrt_len_tokens:
                output_vectors.append(sum_embeddings / torch.sqrt(sum_mask))
        
        if self.pooling_mode_mean_last_2_tokens and "all_layer_embeddings" in features: # avg of last 2 layers
            if "token_checker" in self.__dict__:
                token_ids = features['input_ids']
                new_mask = []
                for sample_token_ids in token_ids:
                    sample_mask = []
                    for token_id in sample_token_ids:
                        if self.token_checker(token_id.item()):
                            sample_mask.append(1)
                        else:
                            sample_mask.append(0)
                    new_mask.append(sample_mask)
                attention_mask = torch.tensor(new_mask).to(device=attention_mask.device, dtype=attention_mask.dtype)
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_mask = input_mask_expanded.sum(1)
            sum_mask = torch.clamp(sum_mask, min=1e-9)
            
            token_embeddings_last1 = features["all_layer_embeddings"][-1]
            sum_embeddings_last1 = torch.sum(token_embeddings_last1 * input_mask_expanded, 1)
            sum_embeddings_last1 = sum_embeddings_last1 / sum_mask
            
            token_embeddings_last2 = features["all_layer_embeddings"][-2]
            sum_embeddings_last2 = torch.sum(token_embeddings_last2 * input_mask_expanded, 1)
            sum_embeddings_last2 = sum_embeddings_last2 / sum_mask
            
            output_vectors.append((sum_embeddings_last1+sum_embeddings_last2) / 2)
            
        if self.pooling_mode_mean_first_last_tokens and "all_layer_embeddings" in features: # avg of the first and the last layers
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_mask = input_mask_expanded.sum(1)
            sum_mask = torch.clamp(sum_mask, min=1e-9)
            
            token_embeddings_first = features["all_layer_embeddings"][-0]
            sum_embeddings_first = torch.sum(token_embeddings_first * input_mask_expanded, 1)
            sum_embeddings_first = sum_embeddings_first / sum_mask
            
            token_embeddings_last = features["all_layer_embeddings"][-1]
            sum_embeddings_last = torch.sum(token_embeddings_last * input_mask_expanded, 1)
            sum_embeddings_last = sum_embeddings_last / sum_mask
            
            output_vectors.append((sum_embeddings_first + sum_embeddings_last) / 2)
            
        max_output, _ = torch.max(token_embeddings[:, 1:, :], dim=1)
        if self.pooling_mode_pad_max_tokens:
            output_vectors.append(max_output)
        features.update({'pad_max_tokens': max_output})
            
        mean_ouput = torch.mean(token_embeddings[:, 1:, :], 1)
        if self.pooling_mode_pad_mean_tokens:
            output_vectors.append(mean_output)
        features.update({'pad_mean_tokens': mean_ouput})

        output_vector = torch.cat(output_vectors, 1)
        features.update({'sentence_embedding': output_vector})
        return features

    def get_sentence_embedding_dimension(self):
        return self.pooling_output_dimension

    def get_config_dict(self):
        return {key: self.__dict__[key] for key in self.config_keys}

    def save(self, output_path):
        with open(os.path.join(output_path, 'config.json'), 'w') as fOut:
            json.dump(self.get_config_dict(), fOut, indent=2)

    @staticmethod
    def load(input_path):
        with open(os.path.join(input_path, 'config.json')) as fIn:
            config = json.load(fIn)

        return Pooling(**config)


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
        if attention_probs_dropout_prob is not None:
            config.attention_probs_dropout_prob = attention_probs_dropout_prob
        if hidden_dropout_prob is not None:
            config.hidden_dropout_prob = hidden_dropout_prob
        self.auto_model = AutoModel.from_pretrained(model_name_or_path, config=config, cache_dir=cache_dir)
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
        output_tokens = output_states[0]
        #[96, 47, 768]

        #output_states[1].shape = [96,768]

        cls_tokens = output_tokens[:, 0, :]  # CLS token is first token
        features.update({'token_embeddings': output_tokens, 'cls_token_embeddings': cls_tokens, 'attention_mask': features['attention_mask']})

        if self.auto_model.config.output_hidden_states:
            all_layer_idx = 2
            if len(output_states) < 3: #Some models only output last_hidden_states and all_hidden_states
                all_layer_idx = 1

            hidden_states = output_states[all_layer_idx]
            features.update({'all_layer_embeddings': hidden_states})

        return features

    def get_word_embedding_dimension(self) -> int:
        return self.auto_model.config.hidden_size

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







