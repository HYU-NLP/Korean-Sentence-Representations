import logging
import sys
from dataclasses import dataclass, field
from typing import Union, Optional, List, Dict

import torch
import transformers
from datasets import load_dataset
from transformers import (
    HfArgumentParser,
    TrainingArguments,
    set_seed,
    BertTokenizer, BertConfig, PreTrainedTokenizerBase
)

from simcse.models import BertForCL, RobertaForCL, POOLER_TYPE_CLS, POOLER_TYPE_ALL
from simcse.trainers import CLTrainer

logger = logging.getLogger(__name__)

MODE_UNSUP = 'unsup'
MODE_SUP_HARD_NEG = 'sup'
MODE_ALL = [MODE_UNSUP, MODE_SUP_HARD_NEG]


class StreamToLogger(object):
    """
    Fake file-like stream object that redirects writes to a logger instance.
    """

    def __init__(self, logger, log_level=logging.INFO):
        self.logger = logger
        self.log_level = log_level
        self.line_buf = ''

    def write(self, buf):
        temp_line_buf = self.line_buf + buf
        self.line_buf = ''
        for line in temp_line_buf.splitlines(True):
            # From the io.TextIOWrapper docs:
            #   On output, if newline is None, any '\n' characters written
            #   are translated to the system default line separator.
            # By default sys.stdout.write() expects '\n' newlines and then
            # translates them so this is still cross platform.

            if line[-1] == '\n':
                self.logger.log(self.log_level, line.rstrip())
            else:
                self.line_buf += line

    def flush(self):
        if self.line_buf != '':
            self.logger.log(self.log_level, self.line_buf.rstrip())
        self.line_buf = ''


def logger_init():
    logging.basicConfig(
        level=logging.INFO
    )

    sys.stdout = StreamToLogger(logging.getLogger('stdout'), logging.INFO)
    # sys.stderr = StreamToLogger(logging.getLogger('stderr'), logging.ERROR)  # Don't redirect stderr because of tqdm


def log_args(used_args, unused_args):
    if used_args:
        logger.info('[List of used arguments]')
        used_args_sorted_key = sorted(used_args.__dict__)
        for key in used_args_sorted_key:
            logger.info(f'{key}: {used_args.__dict__[key]}')

    if unused_args:
        logger.info(f'[List of unused arguments]: {unused_args}')


def main():
    # Parser --
    logger_init()
    parser = HfArgumentParser(TrainingArguments)
    training_args, unused_args = parser.parse_args_into_dataclasses(return_remaining_strings=True)
    log_args(training_args, unused_args)

    # Seed --
    set_seed(training_args.seed)

    # Prepare tokenizer, dataset (+ dataloader), model, loss function, optimizer, etc --
    tokenizer = BertTokenizer.from_pretrained(training_args.model_name_or_path)

    if training_args.task_mode == MODE_UNSUP:
        datasets = load_dataset('text', data_files={'train': training_args.train_file})
        column_names = datasets['train'].column_names

        def preprocess_function(examples):
            column_name = column_names[0]  # The only column name in unsup dataset

            total = len(examples[column_name])  # Total len
            copied = examples[column_name] + examples[column_name]  # Repeat itself

            tokenized = tokenizer(copied, truncation=True, max_length=training_args.max_seq_length)

            result = {}
            for key in tokenized:
                result[key] = [[tokenized[key][i], tokenized[key][i + total]] for i in range(total)]

            return result

        train_dataset = datasets['train'].map(
            preprocess_function,
            batched=True,
            remove_columns=column_names,
        )

    elif training_args.task_mode == MODE_SUP_HARD_NEG:
        datasets = load_dataset('csv', data_files={'train': training_args.train_file}, delimiter=',')
        column_names = datasets['train'].column_names

        def preprocess_function(examples):
            total = len(examples[column_names[0]])  # Total len
            copied = examples[column_names[0]] + examples[column_names[1]] + examples[column_names[2]]

            tokenized = tokenizer(copied, truncation=True, max_length=training_args.max_seq_length)

            result = {}
            for key in tokenized:
                result[key] = [
                    [tokenized[key][i], tokenized[key][i + total], tokenized[key][i + total * 2]] for i in range(total)
                ]

            return result

        train_dataset = datasets['train'].map(
            preprocess_function,
            batched=True,
            remove_columns=column_names,
        )

    else:
        raise NotImplemented

    config = BertConfig.from_pretrained(training_args.model_name_or_path)

    if 'roberta' in training_args.model_name_or_path:
        # Work around when "loading best model" on transformers package 4.2.1 version
        RobertaForCL.temperature = training_args.temperature
        RobertaForCL.hard_negative_weight = training_args.hard_negative_weight
        RobertaForCL.pooler_type = training_args.pooler_type
        RobertaForCL.mlp_only_train = training_args.mlp_only_train

        model = RobertaForCL.from_pretrained(training_args.model_name_or_path, config=config)

    elif 'bert' in training_args.model_name_or_path:
        # Work around when "loading best model" on transformers package 4.2.1 version
        BertForCL.temperature = training_args.temperature
        BertForCL.hard_negative_weight = training_args.hard_negative_weight
        BertForCL.pooler_type = training_args.pooler_type
        BertForCL.mlp_only_train = training_args.mlp_only_train

        model = BertForCL.from_pretrained(training_args.model_name_or_path, config=config)
    else:
        raise NotImplementedError

    # Custom Data collator, because of data repeating in preprocess_function
    @dataclass
    class DataCollatorWithPadding:
        tokenizer: PreTrainedTokenizerBase
        padding = True
        max_length: Optional[int] = None
        pad_to_multiple_of: Optional[int] = None

        def __call__(
                self, features: List[Dict[str, Union[List[int], List[List[int]], torch.Tensor]]]
        ) -> Dict[str, torch.Tensor]:

            special_keys = ['input_ids', 'attention_mask', 'token_type_ids']

            bs = len(features)
            if bs == 0:
                raise ValueError('Dataset is empty')

            num_sent = len(features[0]['input_ids'])

            # flat
            flat_features = []
            for feature in features:
                for i in range(num_sent):
                    flat_features.append({k: feature[k][i] if k in special_keys else feature[k] for k in feature})

            batch = self.tokenizer.pad(
                flat_features,
                padding=self.padding,
                max_length=self.max_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors='pt',
            )

            # un-flat
            batch = {
                k: batch[k].view(bs, num_sent, -1) if k in special_keys else batch[k].view(bs, num_sent, -1)[:, 0]
                for k in batch
            }

            if 'label' in batch:
                batch['labels'] = batch['label']
                del batch['label']
            if 'label_ids' in batch:
                batch['labels'] = batch['label_ids']
                del batch['label_ids']

            return batch

    data_collator = DataCollatorWithPadding(tokenizer)

    trainer = CLTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
    )

    trainer.train()
    trainer.evaluate(while_training=False)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    STRATEGY = 'steps'
    STRATEGY_STEPS = 250

    task_mode: str = field(default=MODE_UNSUP)

    # Trainer Arguments --
    output_dir: str = field(default='./output_dir')
    overwrite_output_dir: bool = field(default=True)

    evaluation_strategy: str = field(default=STRATEGY)
    eval_steps: int = field(default=STRATEGY_STEPS)
    save_strategy: str = field(default=STRATEGY)
    save_steps: int = field(default=STRATEGY_STEPS)
    save_total_limit: int = field(default=2)
    logging_strategy: str = field(default=STRATEGY)
    logging_steps: int = field(default=STRATEGY_STEPS)
    load_best_model_at_end: bool = field(default=True)
    metric_for_best_model: str = field(default='stsb_spearman')
    report_to: str = field(default='tensorboard')

    num_train_epochs: int = field(default=1)
    per_device_train_batch_size: int = field(default=64)
    per_device_eval_batch_size: int = field(default=64)

    learning_rate: float = field(default=1e-5)

    # Non-Trainer Arguments --
    model_name_or_path: str = field(default='bert-base-uncased')
    max_seq_length: int = field(default=32)

    train_file: str = field(default='')  # Depend on simcse_mode
    pooler_type: str = field(default='')  # Depend on simcse_mode
    mlp_only_train: Optional[bool] = field(default=None)  # Depend on simcse_mode

    temperature: float = field(default=0.05)
    hard_negative_weight: float = field(default=0)

    def __post_init__(self):
        super().__post_init__()

        if self.task_mode not in MODE_ALL:
            raise ValueError(f'{self.task_mode} is not a valid simcse mode. Valid modes are {MODE_ALL}.')

        if self.task_mode == MODE_UNSUP:
            self.train_file = './data/wiki1m_for_simcse.txt'
            self.pooler_type = POOLER_TYPE_CLS
            self.mlp_only_train = True

        elif self.task_mode == MODE_SUP_HARD_NEG:
            self.train_file = './data/nli_for_simcse.csv'
            self.pooler_type = POOLER_TYPE_CLS
            self.mlp_only_train = False

        if self.pooler_type not in POOLER_TYPE_ALL:
            raise ValueError(f'{self.pooler_type} is not a valid pooler type. Valid types are {POOLER_TYPE_ALL}.')


if __name__ == '__main__':
    main()
