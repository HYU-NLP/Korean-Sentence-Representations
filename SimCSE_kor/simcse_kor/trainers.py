import logging
from typing import List, Optional, Dict

import numpy as np
import torch
from scipy.stats import spearmanr, pearsonr
from simcse_kor.models import Pooler
from torch.utils.data import Dataset, DataLoader
from transformers import Trainer

logger = logging.getLogger(__name__)


def cosine(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))


def similarity(s1, s2):
    return np.nan_to_num(cosine(np.nan_to_num(s1), np.nan_to_num(s2)))


# bsz : batch size (number of positive pairs)
# d   : latent dim
# x   : Tensor, shape=[bsz, d]
#       latents for one side of positive pairs
# y   : Tensor, shape=[bsz, d]
#       latents for the other side of positive pairs

def align_loss(x, y, alpha=2):
    return (x - y).norm(p=2, dim=1).pow(alpha).mean()


# x   : Tensor, shape=[bsz, d]
#       latents for one side of positive pairs
def uniform_loss(x, t=2):
    return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()


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
                if self.args.is_mode_mbert():
                    outputs = model(**batch, output_hidden_states=True, return_dict=True)

                    pooler = Pooler(self.args.pooler_type)
                    pooler_output = pooler(batch['attention_mask'], outputs)
                else:
                    outputs = model(**batch, output_hidden_states=True, return_dict=True, sent_emb=True)
                    pooler_output = outputs.pooler_output

            return pooler_output.cpu()

        # Evaluate --
        predict = []
        labels = []

        all_loss_align = []
        all_loss_uniform = []

        model.eval()
        for step, batch in enumerate(eval_dataloader):
            gs_scores = batch['score']
            enc1 = batcher(batch['sentence1'])
            enc2 = batcher(batch['sentence2'])

            # Calculate align & uniform --
            pos_indices = [i for i in range(len(gs_scores)) if gs_scores[i] >= 4.0]

            enc1_pos = enc1[pos_indices]
            enc2_pos = enc2[pos_indices]

            norm_enc1_pos = torch.nn.functional.normalize(enc1_pos, p=2, dim=1)
            norm_enc2_pos = torch.nn.functional.normalize(enc2_pos, p=2, dim=1)

            loss_align = align_loss(norm_enc1_pos, norm_enc2_pos)
            loss_uniform = uniform_loss(torch.cat((norm_enc1_pos, norm_enc2_pos), dim=0))

            all_loss_align.append(loss_align)
            all_loss_uniform.append(loss_uniform)

            batch_predict = []
            batch_labels = []

            # Calculate similarity --
            for i in range(enc2.shape[0]):
                sys_score = similarity(enc1[i], enc2[i])

                # batch_predict.append(sys_score)  # FIXME remove
                # batch_labels.append(gs_scores[i])  # FIXME remove

                predict.append(sys_score)
                labels.append(gs_scores[i])

            # spearman_for_batch = spearmanr(batch_predict, batch_labels)[0]  # FIXME remove
            # print(f'spearman_for_batch - {spearman_for_batch}')  # FIXME remove

        metrics = {
            'eval_kor_stsb_spearman': spearmanr(predict, labels)[0],
            'eval_kor_stsb_pearson': pearsonr(predict, labels)[0],
            'eval_align_loss': np.mean(all_loss_align, dtype=np.float64),
            'eval_uniform_loss': np.mean(all_loss_uniform, dtype=np.float64),
        }

        self.log(metrics)
        return metrics
