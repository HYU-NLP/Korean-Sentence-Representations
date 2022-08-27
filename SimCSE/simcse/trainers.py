import math
import os
import sys
import time
from typing import List, Optional

import torch
import torch.distributed as dist
from packaging import version
from prettytable import PrettyTable
from torch import nn
from torch.utils.data import DataLoader, Dataset, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from tqdm.auto import tqdm
from transformers import Trainer, trainer, is_apex_available
from transformers.debug_utils import DebugOption, DebugUnderflowOverflow
from transformers.deepspeed import deepspeed_init
from transformers.integrations import (
    hp_params,
)
from transformers.trainer_callback import (
    TrainerState,
)
from transformers.trainer_pt_utils import (
    IterableDatasetShard,
)
from transformers.trainer_utils import (
    HPSearchBackend,
    ShardedDDPOption,
    TrainOutput,
    has_length,
    speed_metrics,
)
from transformers.utils import (
    is_sagemaker_mp_enabled,
    is_torch_tpu_available,
    logging,
)

if is_torch_tpu_available(check_device=False):
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    import torch_xla.distributed.parallel_loader as pl

if is_apex_available():
    from apex import amp

if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp

PATH_TO_SENTEVAL = './SentEval'  # Set path to SentEval
PATH_TO_DATA = './SentEval/data'  # Set path to SentEval data

sys.path.insert(0, PATH_TO_SENTEVAL)  # To Import SentEval
import senteval

logger = logging.get_logger(__name__)


def print_table(task_names, scores):
    tb = PrettyTable()
    tb.field_names = task_names
    tb.add_row(scores)
    print(tb)


def print_results(results):
    task_names = []
    scores = []
    for task in ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark', 'SICKRelatedness']:
        task_names.append(task)
        if task in results:
            if task in ['STS12', 'STS13', 'STS14', 'STS15', 'STS16']:
                scores.append("%.2f" % (results[task]['all']['spearman']['all'] * 100))
            else:
                scores.append("%.2f" % (results[task]['test']['spearman'].correlation * 100))
        else:
            scores.append("0.00")

    task_names.append("Avg.")
    scores.append("%.2f" % (sum([float(score) for score in scores]) / len(scores)))
    print_table(task_names, scores)
    task_names = []
    scores = []
    for task in ['MR', 'CR', 'SUBJ', 'MPQA', 'SST2', 'TREC', 'MRPC']:
        task_names.append(task)
        if task in results:
            scores.append("%.2f" % (results[task]['acc']))
        else:
            scores.append("0.00")
    task_names.append("Avg.")
    scores.append("%.2f" % (sum([float(score) for score in scores]) / len(scores)))
    print_table(task_names, scores)


class CLTrainer(Trainer):

    def evaluate(
            self,
            eval_dataset: Optional[Dataset] = None,
            ignore_keys: Optional[List[str]] = None,
            metric_key_prefix: str = "eval",
            while_training: bool = True,
    ):

        # SentEval prepare and batcher
        def prepare(params, samples):
            return

        def batcher(params, batch):
            sentences = [' '.join(s) for s in batch]
            batch = self.data_collator.tokenizer.batch_encode_plus(
                sentences,
                return_tensors='pt',
                padding=True,
            )

            for k in batch:
                batch[k] = batch[k].to(self.args.device)

            with torch.no_grad():
                outputs = self.model(**batch, output_hidden_states=True, return_dict=True, sent_emb=True)
                pooler_output = outputs.pooler_output

            return pooler_output.cpu()

        if not while_training:
            params = {
                'task_path': PATH_TO_DATA,
                'usepytorch': True,
                'kfold': 10,
                'classifier': {'nhid': 0, 'optim': 'adam', 'batch_size': 64, 'tenacity': 5, 'epoch_size': 4}
            }
        else:
            # Set params for SentEval (fastmode)
            params = {
                'task_path': PATH_TO_DATA,
                'usepytorch': True,
                'kfold': 5,
                'classifier': {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128, 'tenacity': 3, 'epoch_size': 2}
            }

        se = senteval.engine.SE(params, batcher, prepare)
        if not while_training:
            tasks = [
                'STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark', 'SICKRelatedness',
                'MR', 'CR', 'SUBJ', 'MPQA', 'SST2', 'TREC', 'MRPC'
            ]
        else:
            tasks = ['STSBenchmark', 'SICKRelatedness']

        self.model.eval()
        results = se.eval(tasks)

        if not while_training:
            print_results(results)
            return
        else:
            stsb_spearman = results['STSBenchmark']['dev']['spearman'][0]
            sickr_spearman = results['SICKRelatedness']['dev']['spearman'][0]

            metrics = {
                "eval_stsb_spearman": stsb_spearman,
                "eval_sickr_spearman": sickr_spearman,
                "eval_avg_sts": (stsb_spearman + sickr_spearman) / 2
            }

            self.log(metrics)
            return metrics
