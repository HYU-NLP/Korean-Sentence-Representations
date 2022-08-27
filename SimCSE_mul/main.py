import copy
import csv
import logging
import random
from dataclasses import dataclass, field
from typing import Union, Optional, List, Dict

import torch
import transformers
from datasets import load_dataset
from kobert_tokenizer import KoBERTTokenizer
from transformers import (
    HfArgumentParser,
    TrainingArguments,
    set_seed,
    BertTokenizer, BertConfig, PreTrainedTokenizerBase, BertModel
)

from simcse_mul.models import BertForCL, RobertaForCL, POOLER_TYPE_CLS, POOLER_TYPE_ALL
from simcse_mul.trainers import CLTrainer

logger = logging.getLogger(__name__)


def log_init():
    logging.basicConfig(
        format='%(asctime)s:%(levelname)s:%(name)s:%(message)s',
        level=logging.INFO,
    )


def log_args(used_args, unused_args):
    if used_args:
        logger.info('[List of used arguments]')
        used_args_sorted_key = sorted(used_args.__dict__)
        for key in used_args_sorted_key:
            logger.info(f'{key}: {used_args.__dict__[key]}')

    if unused_args:
        logger.info(f'[List of unused arguments]: {unused_args}')


def main():
    # Logger & Parser --
    log_init()
    parser = HfArgumentParser(TrainingArguments)
    training_args, unused_args = parser.parse_args_into_dataclasses(return_remaining_strings=True)
    log_args(training_args, unused_args)

    # Seed --
    set_seed(training_args.seed)

    # Prepare tokenizer, dataset (+ dataloader), model, loss function, optimizer, etc --
    if training_args.is_mbert_or_eng_base():
        tokenizer = BertTokenizer.from_pretrained(training_args.model_name_or_path)
    elif training_args.is_kobert_base():
        tokenizer = KoBERTTokenizer.from_pretrained(training_args.model_name_or_path)
    else:
        raise ValueError

    config = BertConfig.from_pretrained(training_args.model_name_or_path)

    train_dataset = None

    eval_dataset = load_dataset(
        'csv',
        data_files={'valid': training_args.eval_file},
        sep='\t',
        quoting=csv.QUOTE_NONE,
        split='valid',
    )

    test_dataset = load_dataset(
        'csv',
        data_files={'test': training_args.test_file},
        sep='\t',
        quoting=csv.QUOTE_NONE,
        split='test',
    )

    if training_args.is_mode_no_train():
        model = BertModel.from_pretrained(training_args.model_name_or_path)

    else:
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

        if (
                False
                or training_args.task_mode == TrainingArguments.MODE_KOR_MBERT_SUP_HARD_NEG
                or training_args.task_mode == TrainingArguments.MODE_KOR_MBERT_SUP_HARD_NEG_SAMPLE
        ):
            train_dataset = load_dataset(
                'csv',
                data_files={'train': training_args.train_file},
                sep='\t',
                quoting=csv.QUOTE_NONE,
                split='train',
            )

            column_names = train_dataset.column_names

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
                        [tokenized[key][i], tokenized[key][i + total], tokenized[key][i + total * 2]] for i in
                        range(total)
                    ]

                return result

            train_dataset = train_dataset.map(
                preprocess_train_function,
                batched=True,
                remove_columns=column_names,
                load_from_cache_file=False,  # FIXME improve this
            )

        elif (
                False
                or training_args.task_mode == TrainingArguments.MODE_ENG_BERT_UNSUP
                or training_args.task_mode == TrainingArguments.MODE_ENG_BERT_UNSUP_RAN
                or training_args.task_mode == TrainingArguments.MODE_KOR_MBERT_UNSUP
                or training_args.task_mode == TrainingArguments.MODE_KOR_MBERT_UNSUP_SAMPLE
                or training_args.task_mode == TrainingArguments.MODE_KOR_KOBERT_UNSUP
                or training_args.task_mode == TrainingArguments.MODE_KOR_KOBERT_UNSUP_RAN
                or training_args.task_mode == TrainingArguments.MODE_KOR_KOBERT_UNSUP_SAMPLE
                or training_args.task_mode == TrainingArguments.MODE_KOR_KOBERT_UNSUP_RAN_SAMPLE
        ):
            train_dataset = load_dataset(
                'text',
                data_files={'train': training_args.train_file},
                split='train',
            )

            column_names = train_dataset.column_names

            def preprocess_function(examples):
                column_name = column_names[0]  # The only column name in unsup dataset

                total = len(examples[column_name])  # Total len

                if (
                        False
                        or training_args.task_mode == TrainingArguments.MODE_ENG_BERT_UNSUP_RAN
                        or training_args.task_mode == TrainingArguments.MODE_KOR_KOBERT_UNSUP_RAN
                        or training_args.task_mode == TrainingArguments.MODE_KOR_KOBERT_UNSUP_RAN_SAMPLE
                ):
                    copied_examples = copy.deepcopy(examples[column_name])
                    permuted_examples = []
                    for example in copied_examples:
                        t = example.split()
                        random.shuffle(t)
                        permuted_examples.append(' '.join(t))

                    copied = examples[column_name] + permuted_examples

                else:
                    copied = examples[column_name] + examples[column_name]  # Repeat itself

                tokenized = tokenizer(copied, truncation=True, max_length=training_args.max_seq_length)

                result = {}
                for key in tokenized:
                    result[key] = [[tokenized[key][i], tokenized[key][i + total]] for i in range(total)]

                return result

            train_dataset = train_dataset.map(
                preprocess_function,
                batched=True,
                remove_columns=column_names,
                load_from_cache_file=False,  # FIXME improve this
            )

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

    # [Note] eval_dataset (validation and test dataset) is pre-processed on-the-fly in evaluate function in CLTrainer
    trainer = CLTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset
    )

    logger.info("***** Running Evaluate w/ valiation-set before training *****")
    logger.info(trainer.evaluate())

    if not training_args.is_mode_no_train():
        trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.save_model()
        trainer.save_state()

    logger.info("***** Running Evaluate w/ testset via best model *****")
    logger.info(trainer.evaluate(eval_dataset=test_dataset))


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    """
    Default arguments are assumed you are running Supervised SimCSE with korNLI with m-bert.
    """

    MODE_ENG_BERT = 'en_bert'
    MODE_ENG_BERT_UNSUP = MODE_ENG_BERT + '_unsup'
    MODE_ENG_BERT_UNSUP_RAN = MODE_ENG_BERT + '_unsup_ran'

    MODE_KOR_MBERT = 'mbert'
    MODE_KOR_MBERT_UNSUP = MODE_KOR_MBERT + '_unsup'
    MODE_KOR_MBERT_UNSUP_SAMPLE = MODE_KOR_MBERT + '_unsup_sample'
    MODE_KOR_MBERT_SUP_HARD_NEG = MODE_KOR_MBERT + '_sup',
    MODE_KOR_MBERT_SUP_HARD_NEG_SAMPLE = MODE_KOR_MBERT + '_sup_sample',

    MODE_KOR_KOBERT = 'kobert'
    MODE_KOR_KOBERT_UNSUP = MODE_KOR_KOBERT + '_unsup'
    MODE_KOR_KOBERT_UNSUP_RAN = MODE_KOR_KOBERT + '_unsup_ran'
    MODE_KOR_KOBERT_UNSUP_SAMPLE = MODE_KOR_KOBERT + '_unsup_sample'
    MODE_KOR_KOBERT_UNSUP_RAN_SAMPLE = MODE_KOR_KOBERT + '_unsup_sample_ran'

    MODE_ALL = [
        MODE_ENG_BERT_UNSUP,
        MODE_ENG_BERT_UNSUP_RAN,

        MODE_KOR_MBERT,
        MODE_KOR_MBERT_UNSUP,
        MODE_KOR_MBERT_UNSUP_SAMPLE,
        MODE_KOR_MBERT_SUP_HARD_NEG,
        MODE_KOR_MBERT_SUP_HARD_NEG_SAMPLE,

        MODE_KOR_KOBERT,
        MODE_KOR_KOBERT_UNSUP,
        MODE_KOR_KOBERT_UNSUP_SAMPLE,
        MODE_KOR_KOBERT_UNSUP_RAN,
        MODE_KOR_KOBERT_UNSUP_RAN_SAMPLE,
    ]

    STRATEGY = 'steps'
    STRATEGY_STEPS = 125

    task_mode: str = field(default=MODE_ENG_BERT_UNSUP)

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
    metric_for_best_model: str = field(default='stsb_spearman')  # See CLTrainer
    report_to: str = field(default='tensorboard')

    num_train_epochs: int = field(default=1)
    per_device_train_batch_size: int = field(default=64)
    per_device_eval_batch_size: int = field(default=64)

    learning_rate: float = field(default=1e-5)

    # Non-Trainer Arguments --
    model_name_or_path: str = field(default='')  # Depends on task_mode
    max_seq_length: int = field(default=32)

    temperature: float = field(default=0.05)
    hard_negative_weight: float = field(default=0)

    train_file: str = field(default='')  # Depends on task_mode
    eval_file: str = field(default='')  # Depends on task_mode
    test_file: str = field(default='')  # Depends on task_mode
    pooler_type: str = field(default=None)  # Depends on task_mode
    mlp_only_train: Optional[bool] = field(default=None)  # Depends on task_mode

    def is_mode_no_train(self):
        return self.task_mode == self.MODE_KOR_MBERT or self.task_mode == self.MODE_KOR_KOBERT

    def is_mbert_or_eng_base(self):
        return self.MODE_KOR_MBERT in self.task_mode or self.MODE_ENG_BERT in self.task_mode

    def is_kobert_base(self):
        return self.MODE_KOR_KOBERT in self.task_mode

    def __post_init__(self):
        super().__post_init__()

        # Set values by task_mode --

        if self.task_mode not in TrainingArguments.MODE_ALL:
            raise ValueError(
                f'{self.task_mode} is not a valid training_mode. Valid modes are {self.MODE_ALL}.'
            )

        elif (
                False
                or self.task_mode == TrainingArguments.MODE_ENG_BERT_UNSUP
                or self.task_mode == TrainingArguments.MODE_ENG_BERT_UNSUP_RAN
        ):
            self.model_name_or_path = 'bert-base-uncased'
            self.pooler_type = POOLER_TYPE_CLS

            self.eval_file = './data/eng/sts-dev.csv'
            self.test_file = './data/eng/sts-test.csv'

            self.train_file = './data/eng/wiki1m_for_simcse.txt'
            self.mlp_only_train = True

        elif self.task_mode == TrainingArguments.MODE_KOR_MBERT:
            self.model_name_or_path = 'bert-base-multilingual-uncased'
            self.pooler_type = POOLER_TYPE_CLS

            self.eval_file = './data/kor/KorSTS/sts-dev.tsv'
            self.test_file = './data/kor/KorSTS/sts-test.tsv'

        elif (
                False
                or self.task_mode == TrainingArguments.MODE_KOR_MBERT_SUP_HARD_NEG
                or self.task_mode == TrainingArguments.MODE_KOR_MBERT_SUP_HARD_NEG_SAMPLE
        ):
            self.model_name_or_path = 'bert-base-multilingual-uncased'
            self.pooler_type = POOLER_TYPE_CLS

            self.eval_file = './data/kor/KorSTS/sts-dev.tsv'
            self.test_file = './data/kor/KorSTS/sts-test.tsv'

            self.train_file = (
                './data/kor/KorNLI/snli_1.0_train.ko.tsv' if self.task_mode == TrainingArguments.MODE_KOR_MBERT_SUP_HARD_NEG
                else './data/kor/KorNLI/snli_1.0_train_sample.ko.tsv'
            )
            self.mlp_only_train = False

        elif (
                False
                or self.task_mode == TrainingArguments.MODE_KOR_MBERT_UNSUP
                or self.task_mode == TrainingArguments.MODE_KOR_MBERT_UNSUP_SAMPLE
        ):
            self.model_name_or_path = 'bert-base-multilingual-uncased'
            self.pooler_type = POOLER_TYPE_CLS

            self.eval_file = './data/kor/KorSTS/sts-dev.tsv'
            self.test_file = './data/kor/KorSTS/sts-test.tsv'

            self.train_file = (
                './data/kor/korean_news_data.txt' if self.task_mode == TrainingArguments.MODE_KOR_MBERT_UNSUP
                else './data/kor/korean_news_data.sample.txt'
            )
            self.mlp_only_train = True

        elif self.task_mode == TrainingArguments.MODE_KOR_KOBERT:
            self.model_name_or_path = 'skt/kobert-base-v1'
            self.pooler_type = POOLER_TYPE_CLS

            self.eval_file = './data/kor/KorSTS/sts-dev.tsv'
            self.test_file = './data/kor/KorSTS/sts-test.tsv'

        elif (
                False
                or self.task_mode == TrainingArguments.MODE_KOR_KOBERT_UNSUP
                or self.task_mode == TrainingArguments.MODE_KOR_KOBERT_UNSUP_SAMPLE
                or self.task_mode == TrainingArguments.MODE_KOR_KOBERT_UNSUP_RAN
                or self.task_mode == TrainingArguments.MODE_KOR_KOBERT_UNSUP_RAN_SAMPLE
        ):
            self.model_name_or_path = 'skt/kobert-base-v1'
            self.pooler_type = POOLER_TYPE_CLS

            self.eval_file = './data/kor/KorSTS/sts-dev.tsv'
            self.test_file = './data/kor/KorSTS/sts-test.tsv'

            self.mlp_only_train = True
            self.train_file = (
                './data/kor/korean_news_data.txt' if (self.task_mode == TrainingArguments.MODE_KOR_KOBERT_UNSUP
                                                      or self.task_mode == TrainingArguments.MODE_KOR_KOBERT_UNSUP_RAN)
                else './data/kor/korean_news_data.sample.txt'
            )

        # Check essential values --

        if self.pooler_type not in POOLER_TYPE_ALL:
            raise ValueError(f'{self.pooler_type} is not a valid pooler type. Valid types are {POOLER_TYPE_ALL}.')


if __name__ == '__main__':
    main()
