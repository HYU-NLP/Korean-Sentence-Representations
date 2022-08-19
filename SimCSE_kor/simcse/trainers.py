import csv
import logging
from typing import List, Optional, Dict, Union, Callable, Tuple

import numpy as np
import torch
from datasets import load_dataset
from scipy.stats import spearmanr, pearsonr
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import Trainer, PreTrainedModel, TrainingArguments, DataCollator, PreTrainedTokenizerBase, \
    EvalPrediction, TrainerCallback

logger = logging.getLogger(__name__)


def cosine(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))


def similarity(s1, s2):
    return np.nan_to_num(cosine(np.nan_to_num(s1), np.nan_to_num(s2)))


class CLTrainer(Trainer):

    def evaluate(
            self,
            eval_dataset: Optional[Dataset] = None,
            ignore_keys: Optional[List[str]] = None,
            metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:

        if eval_dataset is None:
            eval_dataset = self.eval_dataset

        eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=self.args.eval_batch_size
        )

        # Model & Tokenizer --
        model = self.model
        tokenizer = self.data_collator.tokenizer

        def batcher(batch):

            batch = tokenizer(
                batch,
                return_tensors='pt',
                padding=True,
            )

            for k in batch:
                batch[k] = batch[k].to(self.args.device)

            with torch.no_grad():
                outputs = model(**batch, output_hidden_states=True, return_dict=True, sent_emb=True)
                pooler_output = outputs.pooler_output

            return pooler_output.cpu()

        # Evaluate --
        predict = []
        labels = []

        model.eval()
        for step, batch in enumerate(eval_dataloader):
            gs_scores = batch['score']
            enc1 = batcher(batch['sentence1'])
            enc2 = batcher(batch['sentence2'])

            for i in range(enc2.shape[0]):
                sys_score = similarity(enc1[i], enc2[i])
                predict.append(sys_score)
                labels.append(gs_scores[i])

        metrics = {
            'eval_kor_stsb_spearman': spearmanr(predict, labels)[0],
            'eval_kor_stsb_pearson': pearsonr(predict, labels)[0],
        }

        self.log(metrics)
        return metrics
