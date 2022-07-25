from torch import nn
from transformers import AutoModel, AutoTokenizer, AutoConfig
import json
from typing import List, Dict, Optional, Union, Tuple
import os

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
        #[batch, seq_len, dim]

        # output_states = ([batch, seq_len, dim], [batch, dim])

        cls_tokens = output_tokens[:, 0, :]  # CLS token is first token
        # cls_tokens = [batch, dim]
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






