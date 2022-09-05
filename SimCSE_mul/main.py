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
from konlpy.tag import Okt
from transformers import (
    HfArgumentParser,
    TrainingArguments,
    set_seed,
    BertTokenizer, BertConfig, PreTrainedTokenizerBase, BertModel, EarlyStoppingCallback
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
    if training_args.is_kobert_base():
        tokenizer = KoBERTTokenizer.from_pretrained(training_args.model_name_or_path)
    else:
        tokenizer = BertTokenizer.from_pretrained(training_args.model_name_or_path)

    train_dataset = None

    if training_args.is_mode_eval_klue():
        def format_label(batch):
            return {'score': batch['labels']['label']}

        valid_dataset = load_dataset('klue', 'sts', split='train[90%:]').map(format_label)
        test_dataset = load_dataset('klue', 'sts', split='validation').map(format_label)

    else:
        valid_dataset = load_dataset(
            'csv',
            data_files={'valid': training_args.valid_file},
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
        config = BertConfig.from_pretrained(training_args.model_name_or_path)
        if training_args.is_mode_no_dropout():
            config.hidden_dropout_prob = 0
            config.classifier_dropout = 0
            config.attention_probs_dropout_prob = 0

        if 'roberta' in training_args.model_name_or_path:
            model = RobertaForCL.from_pretrained(
                training_args.model_name_or_path,
                config=config,
                args=training_args,
            )

        else:
            model = BertForCL.from_pretrained(
                training_args.model_name_or_path,
                config=config,
                args=training_args,
            )

        if training_args.is_mode_sup():
            train_dataset = load_dataset(
                'csv',
                data_files={'train': training_args.train_file},
                sep='\t',
                quoting=csv.QUOTE_NONE,
                split='train',
            )

            if training_args.shuffle_dataset != -1:
                train_dataset = train_dataset.shuffle(seed=training_args.shuffle_dataset)

            column_names = train_dataset.column_names

            def preprocess_function(examples):
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
                preprocess_function,
                batched=True,
                remove_columns=column_names,
            )

        elif training_args.is_mode_unsup():
            train_dataset = load_dataset(
                'text',
                data_files={'train': training_args.train_file},
                split='train',
            )

            if training_args.shuffle_dataset != -1:
                train_dataset = train_dataset.shuffle(seed=training_args.shuffle_dataset)

            column_names = train_dataset.column_names

            if training_args.is_mode_sov_alg1():
                okt = Okt()
                okt_josa = 'Josa'

            def preprocess_function(examples):
                column_name = column_names[0]  # The only column name in unsup dataset
                total = len(examples[column_name])  # Total len

                if training_args.is_mode_sov_ran():
                    copied_examples = copy.deepcopy(examples[column_name])
                    permuted_examples = []
                    for example in copied_examples:
                        t = example.split()
                        random.shuffle(t)
                        permuted_examples.append(' '.join(t))

                    copied = examples[column_name] + permuted_examples

                elif training_args.is_mode_sov_alg1():
                    # FIXME implement here
                    copied_examples = copy.deepcopy(examples[column_name])
                    permuted_examples = []
                    for example in copied_examples:
                        example_pos = okt.pos(example)

                        noun_josa = []
                        noun_verb = []
                        for pos in example_pos:
                            word, tag = pos
                            if tag == okt_josa:
                                pass

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
                load_from_cache_file=False,
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

            if training_args.is_mode_full_ran():
                # Because there is no padding, attention_mask is all 1
                # So only tokenized['input_ids'] shuffle is enough
                for feature in features:
                    random.shuffle(feature['input_ids'][0])

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
        eval_dataset=valid_dataset,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
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

    MODE_TRAIN_SUP = 'train-sup'
    MODE_TRAIN_UNSUP = 'train-unsup'
    MODE_TRAIN_NO = 'train-no'

    MODE_BERT = 'bert'
    MODE_KRBERT = 'krbert'
    MODE_KOBERT = 'kobert'
    MODE_KLUEBERT = 'kluebert'

    MODE_EVAL_KAKAO = 'kakao'
    MODE_EVAL_KLUE = 'klue'

    # Will not permute if no below option exist
    MODE_PERMUTE_FULL_RAN = 'permute-full-ran'
    MODE_PERMUTE_SOV_RAN = 'permute-sov-ran'
    MODE_PERMUTE_SOV_ALG1 = 'permute-sov-alg1'

    # Will do dropout if no below option exist
    MODE_DROPOUT_NO = 'dropout-no'

    STRATEGY = 'steps'
    STRATEGY_STEPS = 125

    task_mode: List[str] = field(default=None)  # Must put

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

    learning_rate: float = field(default=0)  # Must put

    # Non-Trainer Arguments --
    model_name_or_path: str = field(default='')  # Depends on task_mode
    max_seq_length: int = field(default=32)

    temperature: float = field(default=0.05)
    hard_negative_weight: float = field(default=0)

    train_file: str = field(default='')  # Must put
    valid_file: str = field(default='')  # Depends on task_mode
    test_file: str = field(default='')  # Depends on task_mode
    pooler_type: str = field(default=None)  # Depends on task_mode
    mlp_only_train: Optional[bool] = field(default=None)  # Depends on task_mode

    shuffle_dataset: int = field(default=-1)

    def is_mode_sup(self):
        return TrainingArguments.MODE_TRAIN_SUP in self.task_mode

    def is_mode_unsup(self):
        return TrainingArguments.MODE_TRAIN_UNSUP in self.task_mode

    def is_mode_full_ran(self):
        return TrainingArguments.MODE_PERMUTE_FULL_RAN in self.task_mode

    def is_mode_sov_ran(self):
        return TrainingArguments.MODE_PERMUTE_SOV_RAN in self.task_mode

    def is_mode_sov_alg1(self):
        return TrainingArguments.MODE_PERMUTE_SOV_ALG1 in self.task_mode

    def is_mode_no_train(self):
        return TrainingArguments.MODE_TRAIN_NO in self.task_mode

    def is_kobert_base(self):
        return TrainingArguments.MODE_KOBERT in self.task_mode

    def is_mode_no_dropout(self):
        return TrainingArguments.MODE_DROPOUT_NO in self.task_mode

    def is_mode_eval_klue(self):
        return TrainingArguments.MODE_EVAL_KLUE in self.task_mode

    def __post_init__(self):
        super().__post_init__()

        # Set values by task_mode --

        if TrainingArguments.MODE_TRAIN_UNSUP in self.task_mode:
            self.pooler_type = POOLER_TYPE_CLS
            self.mlp_only_train = True
        elif TrainingArguments.MODE_TRAIN_SUP in self.task_mode:
            self.pooler_type = POOLER_TYPE_CLS
            self.mlp_only_train = False
        elif TrainingArguments.MODE_TRAIN_NO in self.task_mode:
            pass
        else:
            raise ValueError

        if TrainingArguments.MODE_BERT in self.task_mode:
            self.model_name_or_path = 'bert-base-uncased'
        elif TrainingArguments.MODE_KOBERT in self.task_mode:
            self.model_name_or_path = 'skt/kobert-base-v1'
        elif TrainingArguments.MODE_KRBERT in self.task_mode:
            self.model_name_or_path = 'snunlp/KR-BERT-char16424'
        elif TrainingArguments.MODE_KLUEBERT in self.task_mode:
            self.model_name_or_path = 'klue/bert-base'
        else:
            raise ValueError

        if TrainingArguments.MODE_EVAL_KAKAO in self.task_mode:
            # Only when it is not MODE_BERT
            if TrainingArguments.MODE_BERT in self.task_mode:
                raise ValueError

            self.valid_file = './data/kor/KorSTS/sts-dev.tsv'
            self.test_file = './data/kor/KorSTS/sts-test.tsv'
        elif TrainingArguments.MODE_EVAL_KLUE in self.task_mode:
            # Only when it is not MODE_BERT
            if TrainingArguments.MODE_BERT in self.task_mode:
                raise ValueError

            # Eval, Test files will be loaded by huggingface
            self.valid_file = '-'
            self.test_file = '-'

        else:
            # Only when it is MODE_BERT
            if TrainingArguments.MODE_BERT not in self.task_mode:
                raise ValueError

            self.valid_file = './data/eng/sts-dev.csv'
            self.test_file = './data/eng/sts-test.csv'

        # Check essential values --

        if (
                self.pooler_type not in POOLER_TYPE_ALL
                or self.max_seq_length <= -1
                or self.learning_rate == 0
                or not self.train_file
                or not self.valid_file
                or not self.test_file
        ):
            raise ValueError


if __name__ == '__main__':
    main()
