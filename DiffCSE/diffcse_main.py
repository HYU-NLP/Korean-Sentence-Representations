import logging
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Optional
import torch
from typing import Optional, Union, List, Dict, Tuple
from transformers.tokenization_utils_base import BatchEncoding, PaddingStrategy, PreTrainedTokenizerBase

import datasets
import numpy as np
from datasets import load_dataset

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
    BertForPreTraining
)
# from transformers.trainer_utils import get_last_checkpoint
from diffcse.diffcse_model import BertForDiffCSE
from diffcse.trainers import DiffCSETrainer

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    # Huggingface's original arguments
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization."
            "Don't set if you want to train a model from scratch."
        },
    )
    generator_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights of the generator model."
        },
    )

    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
   
    temp: float = field(
        default=0.05,
        metadata={
            "help": "Temperature for softmax."
        }
    )
    
    batchnorm: bool = field(
        default=True,
        metadata={
            "help": "Whether to use two-layer mlp for the pooler."
        }
    )
    lambda_weight: float = field(
        default=0.005,
        metadata={
            "help": "Weight for lambda."
        }
    )
    mlp_only_train: bool = field(
        default=True,
        metadata={
            "help": "Use MLP only during training"   # inference 때는 mlp 거치지않고 바로 pooling된 cls를 sentence embedding을 쓰나보네
        }
    )
    
    masking_ratio: float = field(
        default=0.15, 
        metadata={"help": "Ratio of tokens to mask for MLM"}
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    max_seq_length: int = field(
        default=32, # paper baseline
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to pad all samples to `max_seq_length`. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch."
            )
        },
    )

    train_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation data."}
    )

    def __post_init__(self):
        if self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, a json or a txt file."
                
    # test_file: Optional[str] = field(default=None, metadata={"help": "A csv or a json file containing the test data."}) senteval이라 필요없을듯




def main():
    ## HfparseArguments로 parsing하기
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")
    

    # Set seed before initializing model.
    set_seed(training_args.seed)

    
    
    ## config 정의
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
    )
    ## tokenizer 정의
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        use_fast=model_args.use_fast_tokenizer,
    )
    ## model 정의
    model = BertForDiffCSE.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        model_args=model_args
    )
    ############## Need to Check again #################
    # pretrained_model = BertForPreTraining.from_pretrained(model_args.model_name_or_path) # 왜 정의한거지?
    # # model.lm_head.load_state_dict(pretrained_model.cls.predictions.state_dict())
    # model.electra_head = torch.nn.Linear(768, 2)  # electra를 굳이 밖에서 정의?
    # model.discriminator.load_state_dict(pretrained_model.bert.state_dict(), strict=False) # discriminator에 pretrained_model 상태 복붙
    # # 아래에 electra 관련해서 넘겨줘야되나? 이게 가장큰 궁금증
    ############## Need to Check again #################

    model.resize_token_embeddings(len(tokenizer))  # len(tokenizer) = vocab_size = 30522

    # dataset 및 datacollator정의
    data_files = {}
    if data_args.train_file is not None:
        data_files["train"] = data_args.train_file
    extension = data_args.train_file.split(".")[-1] # extension = 'txt'
    if extension == "txt":
        extension = "text"
    if extension == "csv":
        datasets = load_dataset(extension, data_files=data_files, cache_dir="./data/", delimiter="\t" if "tsv" in data_args.train_file else ",")
    else:
        datasets = load_dataset(extension, data_files=data_files, cache_dir="./data/")

    # Prepare features
    column_names = datasets["train"].column_names
    # Unsupervised datasets
    sent0_cname = column_names[0] # 'text'
    sent1_cname = column_names[0] # 'text'
    sent2_cname = None

    def prepare_features(examples):
        # padding = longest (default)
        #   If no sentence in the batch exceed the max length, then use
        #   the max sentence length in the batch, otherwise use the 
        #   max sentence length in the argument and truncate those that
        #   exceed the max length.
        # padding = max_length (when pad_to_max_length, for pressure test)
        #   All sentences are padded/truncated to data_args.max_seq_length.
        total = len(examples[sent0_cname]) # 왜 2개씩 들어오지?  total = 2

        
        sentences = examples[sent0_cname] + examples[sent1_cname]
        # sentences = [s1, s2, s1, s2] 합치기


        sent_features = tokenizer(
            sentences,
            max_length=data_args.max_seq_length,
            truncation=True,
            padding="max_length" if data_args.pad_to_max_length else False, # padding = False
        )
        #sent_features = Dict{input_ids, token_type_ids, attention_mask}    'input_ids' = [[s1],[s2],[s1],[s2]]
        features = {}
        if sent2_cname is not None: # sent2_cname = None
            for key in sent_features:
                features[key] = [[sent_features[key][i], sent_features[key][i+total], sent_features[key][i+total*2]] for i in range(total)]
        else:
            for key in sent_features: # key = 'input_ids', 'token_type_ids', 'attention_mask'
                features[key] = [[sent_features[key][i], sent_features[key][i+total]] for i in range(total)]
                # [[s1],[s2],[s1],[s2]] => [ [[s1],[s1]], [[s2],[s2]] ]         
        return features    # feautures['input_ids] = [[[101, 26866, 1999],[101, 26866, 1999]], [[101, 2148, 2660], [101, 2148, 2660]]]


    # Data collator
    @dataclass
    class OurDataCollatorWithPadding:

        tokenizer: PreTrainedTokenizerBase
        padding: Union[bool, str, PaddingStrategy] = True
        max_length: Optional[int] = None
        pad_to_multiple_of: Optional[int] = None
        mlm: bool = True
        mlm_probability: float = 0.15

        def __call__(self, features: List[Dict[str, Union[List[int], List[List[int]], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
            special_keys = ['input_ids', 'attention_mask', 'token_type_ids', 'mlm_input_ids', 'mlm_labels']
            bs = len(features)
            if bs > 0:
                num_sent = len(features[0]['input_ids'])
            else:
                return
            flat_features = []   # features[] = [[s1], [s1]]
            for feature in features:
                for i in range(num_sent):
                    flat_features.append({k: feature[k][i] if k in special_keys else feature[k] for k in feature})

            batch = self.tokenizer.pad(
                flat_features,
                padding=self.padding,
                max_length=self.max_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors="pt",
            )
            batch["mlm_input_ids"], batch["mlm_labels"] = self.mask_tokens(batch["input_ids"])

            batch = {k: batch[k].view(bs, num_sent, -1) if k in special_keys else batch[k].view(bs, num_sent, -1)[:, 0] for k in batch}

            if "label" in batch:
                batch["labels"] = batch["label"]
                del batch["label"]
            if "label_ids" in batch:
                batch["labels"] = batch["label_ids"]
                del batch["label_ids"]

            return batch
        
        def mask_tokens(
            self, inputs: torch.Tensor, special_tokens_mask: Optional[torch.Tensor] = None
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
            """
            inputs = inputs.clone() # 이건 뭐지? 새로운 주소값을 부여?? 인자로 들어온 inputs의 영향을 주지않기위해서???;;
            labels = inputs.clone() 
            # labels = [bsz * 2, seq_len]
            # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
            probability_matrix = torch.full(labels.shape, model_args.masking_ratio) # probaility_matrix = [bsz * 2, seq_len]   every entry is 0.3
            if special_tokens_mask is None:
                special_tokens_mask = [
                    self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
                ]
                special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
            else:
                special_tokens_mask = special_tokens_mask.bool()

            probability_matrix.masked_fill_(special_tokens_mask, value=0.0) # special token은 cls또는 padding 된 부분
            masked_indices = torch.bernoulli(probability_matrix).bool() # 값에따라 확률적으로 0, 1 을 리턴 
            labels[~masked_indices] = -100  # We only compute loss on masked tokens

            # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
            indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices  # indices_replaced = [bsz *2 , seq_len]
            inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)  # inputs[indices_replaced]= [636 = 53 *12..?]

            # 10% of the time, we replace masked input tokens with random word
            indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced # 바꾸지 않은 부분에서 절반
            random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
            inputs[indices_random] = random_words[indices_random]

            # The rest of the time (10% of the time) we keep the masked input tokens unchanged
            return inputs, labels

    data_collator = OurDataCollatorWithPadding(tokenizer)

    train_dataset = datasets["train"].map( # 왜 두개씩 뱉을까,,, it still remains mistery
        prepare_features,
        batched=True,
        remove_columns=column_names, # column_names = 'text'
    )

    # ########### 임시 dataset ###########
    # train_dataset = train_dataset.train_test_split(test_size=0.997)
    # train_dataset = train_dataset['train']
    # ########### 임시 dataset ###########
    
    ## Trainer 정의 
    trainer = DiffCSETrainer( 
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    trainer.model_args = model_args
    
    ## do_train
    trainer.train()
    ## do_eval
    trainer.evaluate(while_training = False)
    


if __name__ == "__main__":
    main()