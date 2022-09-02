import pstats
from sklearn.feature_selection import SelectFdr
import torch
from torch import divide, nn, Tensor
from typing import Union, Tuple, List, Iterable, Dict, Set
from ..SentenceTransformer import SentenceTransformer
import logging
import copy

# for masking
LARGE_NUM = 1e9

class MyLoss(nn.Module):
    def __init__(self,
                args,
                 model: SentenceTransformer,
                 sentence_embedding_dimension: int,
                 num_labels: int = 1,
                 data_aug_strategy1: str =None,
                 data_aug_strategy2: str =None,
                 contrastive_loss_rate: float = 1.0,  # alpha in the paper in joint                
                 temperature: float = 1.0,                              
                ):
        super(MyLoss, self).__init__()
        self.model = model
        self.num_labels = num_labels
        self.args= args
        self.data_aug_strategy1 = data_aug_strategy1
        self.data_aug_strategy2 = data_aug_strategy2
        self.no_pair = args.no_pair
        self.contrastive_loss_rate = contrastive_loss_rate
        self.temperature = temperature
        
        # Wf +b
        self.classifier = nn.Linear(3* sentence_embedding_dimension, num_labels)
 
    def _reps_to_output(self, rep_a: torch.Tensor, rep_b: torch.Tensor):
        # this function works as f = Concat(r1, r1 |r1-r2|)
        sub = torch.abs(rep_a - rep_b)
        features = torch.cat([rep_a, rep_b, sub], dim=1)
        output = self.classifier(features)
        return output
    
    def _contrastive_loss_forward(self,
                                  hidden1: torch.Tensor,
                                  hidden2: torch.Tensor,
                                  temperature: float = 1.0):
        #this is from SimCLR paper 
        batch_size, hidden_dim = hidden1.shape
        # hidden1 = [batch size, dim]
        
        # norm
        hidden1 = torch.nn.functional.normalize(hidden1, p=2, dim=-1)
        hidden2 = torch.nn.functional.normalize(hidden2, p=2, dim=-1)

        hidden1_large = hidden1
        hidden2_large = hidden2
        #both are [batch size, dim]  

        labels = torch.arange(0, batch_size).to(device=hidden1.device)
        #labels = [0,1,2,3,4, ... , batch size]
        masks = torch.nn.functional.one_hot(torch.arange(0, batch_size), num_classes=batch_size).to(device=hidden1.device, dtype=torch.float)
        # [batch size, batch size] diagonal matrix

        logits_aa = torch.matmul(hidden1, hidden1_large.transpose(0, 1)) / temperature  # shape (bsz, bsz) # numerator
        logits_aa = logits_aa - masks * LARGE_NUM
        logits_bb = torch.matmul(hidden2, hidden2_large.transpose(0, 1)) / temperature  # shape (bsz, bsz)
        logits_bb = logits_bb - masks * LARGE_NUM
        logits_ab = torch.matmul(hidden1, hidden2_large.transpose(0, 1)) / temperature  # shape (bsz, bsz)
        logits_ba = torch.matmul(hidden2, hidden1_large.transpose(0, 1)) / temperature  # shape (bsz, bsz)

        loss_a = torch.nn.functional.cross_entropy(torch.cat([logits_ab, logits_aa], dim=1), labels)  
        loss_b = torch.nn.functional.cross_entropy(torch.cat([logits_ba, logits_bb], dim=1), labels)  
        loss = loss_a + loss_b
        # still confusing whether or not to divide 2
        # loss /= 2
        return loss

    def _recover_to_origin_keys(self, sentence_feature: Dict[str, Tensor], ori_keys: Set[str]):
        return {k: v for k, v in sentence_feature.items() if k in ori_keys}
    

    # #this could be the solution
    def _data_aug(self, sentence_feature, name, ori_keys):
        assert name in ("none", "shuffle", "token_cutoff", "feature_cutoff", "dropout")
        sentence_feature = self._recover_to_origin_keys(sentence_feature, ori_keys)
        cutoff_rate = self.args.cutoff_rate
        if name == "none":
            pass  # do nothing
        elif name == "shuffle":
            self.model[0].auto_model.set_flag("data_aug_shuffle", True)
        elif name == "token_cutoff":
            self.model[0].auto_model.set_flag("data_aug_cutoff", True)
            self.model[0].auto_model.set_flag("data_aug_cutoff.direction", "row")
            self.model[0].auto_model.set_flag("data_aug_cutoff.rate", cutoff_rate)
        elif name == "feature_cutoff":
            self.model[0].auto_model.set_flag("data_aug_cutoff", True)
            self.model[0].auto_model.set_flag("data_aug_cutoff.direction", "column")
            self.model[0].auto_model.set_flag("data_aug_cutoff.rate", cutoff_rate)
        elif name == "dropout":
            self.model[0].auto_model.set_flag("data_aug_cutoff", True)
            self.model[0].auto_model.set_flag("data_aug_cutoff.direction", "random")
            self.model[0].auto_model.set_flag("data_aug_cutoff.rate", cutoff_rate)
        rep = self.model(sentence_feature)["sentence_embedding"]
        return rep
    
    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        if self.args.train_way == "sup":
            reps = [self.model(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features]
            rep_a, rep_b = reps

            output = self._reps_to_output(rep_a, rep_b)
            
            loss_fct = nn.CrossEntropyLoss()

            if labels is not None:
                loss = loss_fct(output, labels.view(-1))
                return loss
            else:
                return reps, output
        else:  
            # making crossentropy input
            if self.args.train_way == "joint":
                if not self.no_pair:
                    sentence_feature_a, sentence_feature_b = sentence_features
                else:
                    sentence_feature_a = sentence_features[0] 
                    # sentenc_feature_b = None

                ori_feature_keys = set(sentence_feature_a.keys())  # record the keys since the features will be update
                rep_a = self._data_aug(sentence_feature_a, "none", ori_feature_keys)
                # ['input_ids', 'token_type_ids', 'attention_mask', 'token_embeddings', 'cls_token_embeddings', 'pad_max_tokens', 'pad_mean_tokens', 'sentence_embedding']
                sentence_feature_a = {k: v for k, v in sentence_feature_a.items() if k in ori_feature_keys}

                if not self.no_pair:
                    rep_b = self._data_aug(sentence_feature_b, "none", ori_feature_keys)
                    sentence_feature_b = {k: v for k, v in sentence_feature_b.items() if k in ori_feature_keys}
                else:
                    rep_b = None

            # for sup-unsup, joint-unsup, unsup
            if not self.no_pair:                        
                sentence_feature_a, sentence_feature_b = sentence_features
            else:
                sentence_feature_a = sentence_features[0] # sentenc_feature_b = None

            ori_feature_keys = set(sentence_feature_a.keys())  # record the keys since the features will be updated

            rep_a_view1 = self._data_aug(sentence_feature_a, self.data_aug_strategy1, ori_feature_keys)
            if not self.no_pair:
                rep_b_view1 = self._data_aug(sentence_feature_b, self.data_aug_strategy1, ori_feature_keys)
            else:
                rep_b_view1 = None

            rep_a_view2 = self._data_aug(sentence_feature_a, self.data_aug_strategy2, ori_feature_keys)
            if not self.no_pair:
                rep_b_view2 = self._data_aug(sentence_feature_b, self.data_aug_strategy2, ori_feature_keys)
            else:
                rep_b_view2 = None


            # loss calculation
            final_loss = 0
            
            if self.args.train_way in ["joint", "sup"]:
                match_output_n_n = self._reps_to_output(rep_a, rep_b)
                # match_outpput_n_n = [batch size, 3]    3 means num of classes (entail, contra, neutral)
                loss_fct = nn.CrossEntropyLoss()
                loss_n_n = loss_fct(match_output_n_n, labels.view(-1))
                final_loss += loss_n_n
                
            if self.args.train_way in ["unsup", "joint", "sup-unsup", "joint-unsup"]:
                contrastive_loss_a = self._contrastive_loss_forward(rep_a_view1 , rep_a_view2, temperature=self.temperature) 
                if not self.no_pair: #recheck
                    contrastive_loss_b = self._contrastive_loss_forward(rep_b_view1, rep_b_view2, temperature=self.temperature)
                else:
                    contrastive_loss_b = torch.tensor(0.0)
                contrastive_loss = contrastive_loss_a + contrastive_loss_b
                final_loss += self.contrastive_loss_rate * contrastive_loss
            return final_loss