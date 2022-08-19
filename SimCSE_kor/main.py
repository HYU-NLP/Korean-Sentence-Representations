import csv
import logging
import sys
from dataclasses import dataclass, field
from typing import Union, Optional, List, Dict

import torch
import transformers
from datasets import load_dataset
from tqdm.contrib.logging import logging_redirect_tqdm
from transformers import (
    HfArgumentParser,
    TrainingArguments,
    set_seed,
    BertTokenizer, BertConfig, PreTrainedTokenizerBase
)

from simcse.models import BertForCL, RobertaForCL, POOLER_TYPE_CLS, POOLER_TYPE_ALL
from simcse.trainers import CLTrainer

logging.basicConfig(
    format='%(asctime)s:%(levelname)s:%(name)s:%(message)s',
    level=logging.INFO,
)

logger = logging.getLogger(__name__)

MODE_UNSUP = 'unsup'
MODE_SUP_HARD_NEG = 'sup'
MODE_ALL = [MODE_UNSUP, MODE_SUP_HARD_NEG]


def log_args(used_args, unused_args):
    if used_args:
        logger.info('[List of used arguments]')
        used_args_sorted_key = sorted(used_args.__dict__)
        for key in used_args_sorted_key:
            logger.info(f'{key}: {used_args.__dict__[key]}')

    if unused_args:
        logger.info(f'[List of unused arguments]: {unused_args}')


def main(default_params):
    # Parser --
    parser = HfArgumentParser(TrainingArguments)
    training_args, unused_args = parser.parse_args_into_dataclasses(default_params, return_remaining_strings=True)
    log_args(training_args, unused_args)

    # Seed --
    set_seed(training_args.seed)

    # Prepare tokenizer, dataset (+ dataloader), model, loss function, optimizer, etc --
    tokenizer = BertTokenizer.from_pretrained(training_args.model_name_or_path)

    if training_args.simcse_mode == MODE_SUP_HARD_NEG:
        train_dataset = load_dataset(
            'csv',
            data_files={'train': training_args.train_file},
            sep='\t',
            quoting=csv.QUOTE_NONE,
        )

        column_names = train_dataset['train'].column_names

        def preprocess_train_function(examples):
            total = len(examples[column_names[0]])

            # These kinds of data exist in snli_1.0_train.ko.tsv, which interpreted as None by load_dataset, which needs to be removed.
            # Examples)
            #  - 설명할 그림을 볼 수 없습니다. N/A neutral
            #  - 설명할 그림을 볼 수 없습니다. N/A entailment
            #  - 설명할 그림을 볼 수 없습니다. N/A contradiction
            abnormal_examples_index = []
            for i in range(total):
                if (
                        examples[column_names[0]][i] == None
                        or examples[column_names[1]][i] == None
                        or examples[column_names[2]][i] == None
                ):
                    logger.info(
                        f'Example that will be removed: {examples[column_names[0]][i]} {examples[column_names[1]][i]} {examples[column_names[2]][i]}'
                    )

                    abnormal_examples_index.append(i)

            # Remove from the end to avoid index shift.
            for i in reversed(abnormal_examples_index):
                del examples[column_names[0]][i]
                del examples[column_names[1]][i]
                del examples[column_names[2]][i]
                total -= 1

            copied = examples[column_names[0]] + examples[column_names[1]] + examples[column_names[2]]

            tokenized = tokenizer(copied, truncation=True, max_length=training_args.max_seq_length)

            result = {}
            for key in tokenized:
                result[key] = [
                    [tokenized[key][i], tokenized[key][i + total], tokenized[key][i + total * 2]] for i in range(total)
                ]

            return result

        with logging_redirect_tqdm():
            train_dataset = train_dataset['train'].map(
                preprocess_train_function,
                batched=True,
                remove_columns=column_names,
                num_proc=training_args.preprocessing_num_workers,
            )

        eval_dataset = load_dataset(
            'csv',
            data_files={'valid': training_args.eval_file},
            sep='\t',
            quoting=csv.QUOTE_NONE,
        )

        eval_dataset = eval_dataset['valid']

        test_dataset = load_dataset(
            'csv',
            data_files={'test': training_args.test_file},
            sep='\t',
            quoting=csv.QUOTE_NONE,
        )

        test_dataset = test_dataset['test']

    else:
        raise NotImplementedError

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

    # [Note]
    # Validation is hard coded via overriding, uses eval_file of our TrainingArguments.
    # See CLTrainer.
    trainer = CLTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset
    )

    logger.info("***** Running Evaluation before training *****")
    logger.info(trainer.evaluate())

    trainer.train()
    trainer.save_model()
    trainer.save_state()

    logger.info("***** Running Evaluate via testset via best model *****")
    logger.info(trainer.evaluate(eval_dataset=test_dataset))


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    model_name_or_path: str = field(default='bert-base-multilingual-uncased')
    max_seq_length: int = field(default=32)

    preprocessing_num_workers: int = field(default=8)

    temperature: float = field(default=0.05)
    hard_negative_weight: float = field(default=0)

    simcse_mode: str = field(default=MODE_SUP_HARD_NEG)
    train_file: str = field(default='./data/KorNLI/snli_1.0_train.ko.tsv')
    eval_file: str = field(default='./data/KorSTS/sts-dev.tsv')
    test_file: str = field(default='./data/KorSTS/sts-test.tsv')
    pooler_type: str = field(default=POOLER_TYPE_CLS)  # Depend on simcse_mode
    mlp_only_train: bool = field(default=False)  # Depend on simcse_mode

    def __post_init__(self):
        super().__post_init__()

        if self.pooler_type not in POOLER_TYPE_ALL:
            raise ValueError(f'{self.pooler_type} is not a valid pooler type. Valid types are {POOLER_TYPE_ALL}.')

        if self.simcse_mode not in MODE_ALL:
            raise ValueError(f'{self.simcse_mode} is not a valid simcse mode. Valid modes are {MODE_ALL}.')

        # Comment rules if you want to do differently from paper
        if self.simcse_mode == MODE_UNSUP:
            raise ValueError('Unsupervised mode is not implemented yet.')

        elif self.simcse_mode == MODE_SUP_HARD_NEG:
            if self.pooler_type != POOLER_TYPE_CLS:
                raise ValueError('pooler_type must be POOLER_TYPE_CLS when simcse_mode is MODE_SUP')

            if self.train_file in 'snli_1.0_train' and self.eval_file in 'sts-dev':
                raise ValueError(
                    f'{self.train_file} and {self.eval_file} are not valid for simcse_mode {self.simcse_mode}'
                )


if __name__ == '__main__':
    # Default params for TrainingArguments, can still be overridden by command-line
    fake_argv = [
        '--output_dir', './output_dir',
        '--overwrite_output_dir', 'True',

        '--evaluation_strategy', 'steps',
        '--eval_steps', '250',
        '--save_strategy', 'steps',
        '--save_steps', '250',
        '--logging_strategy', 'steps',
        '--logging_steps', '250',
        '--load_best_model_at_end', 'True',
        '--report_to', 'tensorboard',

        '--num_train_epochs', '1',
        '--per_device_train_batch_size', '64',
        '--per_device_eval_batch_size', '64',

        '--learning_rate', '1e-5',

        '--metric_for_best_model', 'kor_stsb_spearman',  # See CLTrainer
    ]

    fake_argv.extend(sys.argv[1:])

    main(fake_argv)
