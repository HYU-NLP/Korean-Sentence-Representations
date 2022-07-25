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

class myLoss(nn.Module):
    def __init__(self,
                args,
                 model: SentenceTransformer,
                 sentence_embedding_dimension: int,
                 num_labels: int,
                 data_aug_strategy1: str =None,
                 data_aug_strategy2: str =None,
                 contrastive_loss_rate: float = 1.0,  # alpha in the paper in joint                
                 temperature: float = 1.0,                              
                ):
        super(myLoss, self).__init__()
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
        # hidden1 = [96, 768], which means [batch size, dim]
        
        # why normalize?
        hidden1 = torch.nn.functional.normalize(hidden1, p=2, dim=-1)
        hidden2 = torch.nn.functional.normalize(hidden2, p=2, dim=-1)

        hidden1_large = hidden1
        hidden2_large = hidden2
        #both are [batch size, dim]  

        labels = torch.arange(0, batch_size).to(device=hidden1.device)
        #labels = [0,1,2,3,4, ... , 94,95]
        masks = torch.nn.functional.one_hot(torch.arange(0, batch_size), num_classes=batch_size).to(device=hidden1.device, dtype=torch.float)
        # [batch size, batch size] diagonal matrix

        logits_aa = torch.matmul(hidden1, hidden1_large.transpose(0, 1)) / temperature  # shape (bsz, bsz) # numerator
        logits_aa = logits_aa - masks * LARGE_NUM
        logits_bb = torch.matmul(hidden2, hidden2_large.transpose(0, 1)) / temperature  # shape (bsz, bsz)
        logits_bb = logits_bb - masks * LARGE_NUM
        logits_ab = torch.matmul(hidden1, hidden2_large.transpose(0, 1)) / temperature  # shape (bsz, bsz)
        logits_ba = torch.matmul(hidden2, hidden1_large.transpose(0, 1)) / temperature  # shape (bsz, bsz)

        loss_a = torch.nn.functional.cross_entropy(torch.cat([logits_ab, logits_aa], dim=1), labels)  #??? labels랑 비교?? 이게정답? 
        loss_b = torch.nn.functional.cross_entropy(torch.cat([logits_ba, logits_bb], dim=1), labels)  
        loss = loss_a + loss_b
        # still confusing whether or not to divide 2
        # loss /= 2
        return loss

    def _recover_to_origin_keys(self, sentence_feature: Dict[str, Tensor], ori_keys: Set[str]):
        return {k: v for k, v in sentence_feature.items() if k in ori_keys}
    

    # #this could be the solution
    # def _data_aug(self, sentence_feature, name, ori_keys, cutoff_rate):
    #     assert name in ("none", "shuffle", "token_cutoff", "feature_cutoff", "dropout")
    #     sentence_feature = self._recover_to_origin_keys(sentence_feature, ori_keys)
    #     if name == "none":
    #         pass  # do nothing
    #     elif name == "shuffle":
    #         self.model[0].auto_model.set_flag("data_aug_shuffle", True)
    #     elif name == "token_cutoff":
    #         self.model[0].auto_model.set_flag("data_aug_cutoff", True)
    #         self.model[0].auto_model.set_flag("data_aug_cutoff.direction", "row")
    #         self.model[0].auto_model.set_flag("data_aug_cutoff.rate", cutoff_rate)
    #     elif name == "feature_cutoff":
    #         self.model[0].auto_model.set_flag("data_aug_cutoff", True)
    #         self.model[0].auto_model.set_flag("data_aug_cutoff.direction", "column")
    #         self.model[0].auto_model.set_flag("data_aug_cutoff.rate", cutoff_rate)
    #     elif name == "dropout":
    #         self.model[0].auto_model.set_flag("data_aug_cutoff", True)
    #         self.model[0].auto_model.set_flag("data_aug_cutoff.direction", "random")
    #         self.model[0].auto_model.set_flag("data_aug_cutoff.rate", cutoff_rate)
    #     rep = self.model(sentence_feature)["sentence_embedding"]
    #     return rep
    
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
            ############## modified ##############
            # making crossentropy input
            # deepcopy,, 왜왜왜ㅙㅇ 아래도 sentence_feature_a 2번씩 쓰는데 이런문제가 왜 안생기지?;
            # 이러한 문제를 방지하기위해서 key reset을 하는거 같으넫,, 아닌가, 차피 forwarding이 된 상태니까
            if self.args.train_way == "joint":
                aux_sentence_features = copy.deepcopy(sentence_features)
                #this causes CUDA OOM. should come up with other solutions
                if not self.no_pair:
                    sentence_feature_a, sentence_feature_b = aux_sentence_features
                else:
                    sentence_feature_a = aux_sentence_features[0] 
                    # sentenc_feature_b = None

                ori_feature_keys = set(sentence_feature_a.keys())  # record the keys since the features will be updated
                rep_a= self.model(sentence_feature_a)['sentence_embedding'] # since self.data_aug_strategy1 == None
                # ['input_ids', 'token_type_ids', 'attention_mask', 'token_embeddings', 'cls_token_embeddings', 'pad_max_tokens', 'pad_mean_tokens', 'sentence_embedding']
                sentence_feature_a = {k: v for k, v in sentence_feature_a.items() if k in ori_feature_keys}

                if not self.no_pair:
                    rep_b= self.model(sentence_feature_b)['sentence_embedding']
                    sentence_feature_b = {k: v for k, v in sentence_feature_b.items() if k in ori_feature_keys}
                else:
                    rep_b = None

            # data augmentation generation
            #_data_aug 함수로 통합시키면 더 깔끔하지 않을까
            if self.data_aug_strategy1 == "None":
                if self.data_aug_strategy2 == "None": # (none, none)
                    pass

                elif self.data_aug_strategy2 == "shuffle": # (none, shuffle)
                    if not self.no_pair:
                        sentence_feature_a, sentence_feature_b = sentence_features
                    else:
                        sentence_feature_a = sentence_features[0] 
                        # sentenc_feature_b = None

                    ori_feature_keys = set(sentence_feature_a.keys())  # record the keys since the features will be updated
                    rep_a_view1 = self.model(sentence_feature_a)['sentence_embedding'] # since self.data_aug_strategy1 == None
                    sentence_feature_a = {k: v for k, v in sentence_feature_a.items() if k in ori_feature_keys}
                    if not self.no_pair:
                        rep_b_view1 = self.model(sentence_feature_b)['sentence_embedding']
                        sentence_feature_b = {k: v for k, v in sentence_feature_b.items() if k in ori_feature_keys}
                    else:
                        rep_b_view1 = None

                    self.model[0].auto_model.set_flag(f"data_aug_{self.data_aug_strategy2}", True)
                    rep_a_view2 = self.model(sentence_feature_a)['sentence_embedding']
                    if not self.no_pair:
                        self.model[0].auto_model.set_flag(f"data_aug_{self.data_aug_strategy2}", True)
                        rep_b_view2 = self.model(sentence_feature_b)['sentence_embedding']
                    else:
                        rep_b_view2 = None
                   
                elif self.data_aug_strategy2 == "token_cutoff": # (none, token_cutoff)
                    pass
                elif self.data_aug_strategy2 == "feature_cutoff": # (none, feature_cutoff)
                    pass
                
            elif self.data_aug_strategy1 == "shuffle": 

                if self.data_aug_strategy2 == "shuffle": # (shuffle, shuffle)
                    pass
                elif self.data_aug_strategy2 == "token_cutoff": # (shuffle, token_cutoff)
                    pass
                elif self.data_aug_strategy2 == "feature_cutoff": # (shuffle, feature_cutoff)
                    if not self.no_pair:                        sentence_feature_a, sentence_feature_b = sentence_features
                    else:
                        sentence_feature_a = sentence_features[0] # sentenc_feature_b = None

                    ori_feature_keys = set(sentence_feature_a.keys())  # record the keys since the features will be updated

                    self.model[0].auto_model.set_flag("data_aug_shuffle", True)
                    rep_a_view1 = self.model(sentence_feature_a)['sentence_embedding']
                    sentence_feature_a = {k: v for k, v in sentence_feature_a.items() if k in ori_feature_keys}
                    if not self.no_pair:
                        self.model[0].auto_model.set_flag("data_aug_shuffle", True)
                        rep_b_view1 = self.model(sentence_feature_b)['sentence_embedding']
                        sentence_feature_b = {k: v for k, v in sentence_feature_b.items() if k in ori_feature_keys}
                    else:
                        rep_b_view1 = None
                    #flag에 feature_cutoff는 없고 cutoff만 있음
                    self.model[0].auto_model.set_flag("data_aug_cutoff", True)
                    self.model[0].auto_model.set_flag("data_aug_cutoff.direction", "column")
                    self.model[0].auto_model.set_flag("data_aug_cutoff.rate", self.args.cutoff_rate)
                    rep_a_view2 = self.model(sentence_feature_a)['sentence_embedding']
                    if not self.no_pair:
                        self.model[0].auto_model.set_flag(f"data_aug_cutoff", True)
                        self.model[0].auto_model.set_flag("data_aug_cutoff.direction", "column")
                        self.model[0].auto_model.set_flag("data_aug_cutoff.rate", self.args.cutoff_rate)
                        rep_b_view2 = self.model(sentence_feature_b)['sentence_embedding']
                    else:
                        rep_b_view2 = None
                
            elif self.data_aug_strategy1 == "token_cutoff":
                if self.data_aug_strategy2 == "token_cutoff": # (token_cutoff, token_cutoff)
                    pass
                elif self.data_aug_strategy2 == "feature_cutoff": # (token_cutoff, feature_cutoff)
                    pass
              
            elif self.data_aug_strategy1 == "feature_cutoff":
                if self.data_aug_strategy2 == "feature_cutoff": # (feature_cutoff, feature_cutoff)
                    pass
            ############## modified ##############    

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