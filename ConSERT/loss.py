import pstats
from sklearn.feature_selection import SelectFdr
import torch
from torch import divide, nn, Tensor
from typing import Union, Tuple, List, Iterable, Dict, Set
from ..SentenceTransformer import SentenceTransformer
import logging

# for masking
LARGE_NUM = 1e9


# for warm-up scheduling
def scheduler0(cur_step, global_step):
    return 1.0, 1.0
def scheduler1(cur_step, global_step):
    """global_step=9814"""
    if cur_step < 7950:
        return 1.0, 1.0
    else:
        return 0.0, 1.0
def scheduler2(cur_step, global_step):
    """global_step=9814"""
    if cur_step < 7950:
        return 1.0, 1.0
    else:
        return 0.01, 1.0
def scheduler3(cur_step, global_step):
    """global_step=9814"""
    if cur_step < 7900:
        return 1.0, 1.0
    else:
        return 0.0, 1.0
def scheduler4(cur_step, global_step):
    """global_step=9814"""
    if cur_step < 7900:
        return 1.0, 1.0
    else:
        return 0.01, 1.0
def scheduler5(cur_step, global_step):
    """global_step=9814"""
    if cur_step < 8814:
        return 1.0, 1.0
    else:
        return 0.0, 0.1
def scheduler6(cur_step, global_step):
    """global_step=9814"""
    if cur_step < 8814:
        return 1.0, 1.0
    else:
        return 0.0, 0.03
def scheduler7(cur_step, global_step):
    """global_step=9814"""
    if cur_step < 8814:
        return 1.0, 1.0
    else:
        return 0.1, 0.1
def scheduler8(cur_step, global_step):
    """global_step=9814"""
    if cur_step < 8814:
        return 1.0, 1.0
    else:
        return 0.1, 0.03
def scheduler9(cur_step, global_step):
    level = cur_step // 1000
    rate = pow(0.5, level)
    return rate, 1.0
def scheduler10(cur_step, global_step):
    level = cur_step // 1000
    rate = pow(0.3, level)
    return rate, 1.0
def scheduler11(cur_step, global_step):
    level = cur_step // 1000
    rate1 = pow(0.5, level)
    rate2 = pow(0.7, level)
    return rate1, rate2
def scheduler12(cur_step, global_step):
    level = cur_step // 3000
    rate = pow(0.464, level)
    return rate, 1.0
def scheduler13(cur_step, global_step):
    level = cur_step // 3000
    rate = pow(0.215, level)
    return rate, 1.0
def scheduler14(cur_step, global_step):
    level = cur_step // 3000
    rate = pow(0.1, level)
    return rate, 1.0
def scheduler15(cur_step, global_step):
    level = cur_step // 4000
    rate = pow(0.316, level)
    return rate, 1.0
def scheduler16(cur_step, global_step):
    level = cur_step // 4000
    rate = pow(0.1, level)
    return rate, 1.0
def scheduler17(cur_step, global_step):
    level = cur_step // 4000
    rate = pow(0.032, level)
    return rate, 1.0
def scheduler18(cur_step, global_step):
    if cur_step < int(global_step * 0.8):
        return 1.0, 1.0
    else:
        return 0.0, 1.0
    

LOSS_RATE_SCHEDULERS = [
    scheduler0,
    scheduler1,
    scheduler2,
    scheduler3,
    scheduler4,
    scheduler5,
    scheduler6,
    scheduler7,
    scheduler8,
    scheduler9,
    scheduler10,
    scheduler11,
    scheduler12,
    scheduler13,
    scheduler14,
    scheduler15,
    scheduler16,
    scheduler17,
    scheduler18
]


def distance_to_center_mse_loss(x: torch.Tensor):
    """x: shape (batch_size, hidden_dim)"""
    bsz, hidden = x.shape
    center = torch.mean(x, dim=0)
    to_center_dist = torch.norm(x - center, p=2, dim=-1)
    return to_center_dist.pow(2).mean()
    
class myLoss(nn.Module):
    """
    :param model: SentenceTransformer model
    :param sentence_embedding_dimension: Dimension of your sentence embeddings
    :param num_labels: Number of different labels
    :param concatenation_sent_rep: Concatenate vectors u,v for the softmax classifier?
    :param concatenation_sent_difference: Add abs(u-v) for the softmax classifier?
    :param concatenation_sent_multiplication: Add u*v for the softmax classifier?

    train_loss = losses.AdvCLSoftmaxLoss(model=model, sentence_embedding_dimension=model.get_sentence_embedding_dimension(), 
                num_labels=len(label2int), concatenation_sent_max_square=args.concatenation_sent_max_square, use_contrastive_loss=args.add_cl, 
                contrastive_loss_type=args.cl_type, contrastive_loss_rate=args.cl_rate, temperature=args.temperature, 
                contrastive_loss_stop_grad=args.contrastive_loss_stop_grad, mapping_to_small_space=args.mapping_to_small_space, 
                add_contrastive_predictor=args.add_contrastive_predictor, projection_hidden_dim=args.projection_hidden_dim, 
                projection_use_batch_norm=args.projection_use_batch_norm, add_projection=args.add_projection, 
                projection_norm_type=args.projection_norm_type, contrastive_loss_only=args.cl_loss_only, 
                data_augmentation_strategy=args.data_augmentation_strategy, cutoff_direction=args.cutoff_direction, 
                cutoff_rate=args.cutoff_rate, no_pair=args.no_pair, regularization_term_rate=args.regularization_term_rate, 
                loss_rate_scheduler=args.loss_rate_scheduler, data_augmentation_strategy_final_1=args.da_final_1, data_augmentation_strategy_final_2=args.da_final_2, 
                cutoff_rate_final_1=args.cutoff_rate_final_1, cutoff_rate_final_2=args.cutoff_rate_final_2)
    """
    def __init__(self,
                args,
                 model: SentenceTransformer,
                 sentence_embedding_dimension: int,
                 num_labels: int,
                 
                 loss_rate_scheduler: int = 0,                          # 用来控制对比损失和主任务损失相对大小
                 
                 data_aug_strategy1: str =None,
                 data_aug_strategy2: str =None,

                 use_contrastive_loss: bool = False,                    # 是否加对比损失
                 contrastive_loss_only: bool = False,                   # 只使用对比损失进行（无监督）训练
                 no_pair: bool = False,                                 # 不使用配对的语料，避免先验信息
                 contrastive_loss_rate: float = 1.0,                    # 对比损失的系数
                 regularization_term_rate: float = 0.0,                 # 正则化项（同一个batch内分布的方差）所占的比率大小
                 do_hidden_normalization: bool = True,                  # 进行对比损失之前，是否对句子表示做正则化
                 temperature: float = 1.0,                              # 对比损失中的温度系数，仅对于交叉熵损失有效
                ):
        super(myLoss, self).__init__()
        self.model = model
        self.num_labels = num_labels
        self.args= args
        self.loss_rate_scheduler = loss_rate_scheduler
        self.data_aug_strategy1 = data_aug_strategy1
        self.data_aug_strategy2 = data_aug_strategy2

        self.use_contrastive_loss = use_contrastive_loss
        self.contrastive_loss_only = contrastive_loss_only
        self.no_pair = no_pair
        if no_pair:
            assert use_contrastive_loss and contrastive_loss_only
        
        self.contrastive_loss_rate = contrastive_loss_rate
        self.regularization_term_rate = regularization_term_rate
        self.do_hidden_normalization = do_hidden_normalization
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
                                  hidden_norm: bool = True,
                                  temperature: float = 1.0):
        #this is from SimCLR paper 
        batch_size, hidden_dim = hidden1.shape
        # hidden1 = [96, 768], which means [batch size, dim]
        
        if hidden_norm:
            hidden1 = torch.nn.functional.normalize(hidden1, p=2, dim=-1)
            hidden2 = torch.nn.functional.normalize(hidden2, p=2, dim=-1)

        hidden1_large = hidden1
        hidden2_large = hidden2
        #both are [96,768]

        labels = torch.arange(0, batch_size).to(device=hidden1.device)
        #labels = [0,1,2,3,4, ... , 94,95]
        masks = torch.nn.functional.one_hot(torch.arange(0, batch_size), num_classes=batch_size).to(device=hidden1.device, dtype=torch.float)
        #[96, 96] , 대각행렬 (대각선만 1)

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
    
    def _data_aug(self, sentence_feature, name, ori_keys, cutoff_rate):
        assert name in ("none", "shuffle", "token_cutoff", "feature_cutoff", "dropout")
        sentence_feature = self._recover_to_origin_keys(sentence_feature, ori_keys)
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
        return rep # sentence representation이구나
    
    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        
        if not self.use_contrastive_loss:  # for sup-unsup, need to train sup first
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
            total_step, cur_step = self.model.num_steps_total, self.model.global_step
            adv_rate, cl_rate = LOSS_RATE_SCHEDULERS[self.loss_rate_scheduler](cur_step, total_step)
            
            # data augmentation generation
            ############## modified ##############
            #아래 전체가 data aug를 위한부분인데, crossentropy쪽 rep 생성은 다른쪽에서 하는게 좀더 깔끔?
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
                    if not self.no_pair:
                        sentence_feature_a, sentence_feature_b = sentence_features
                    else:
                        sentence_feature_a = sentence_features[0] # sentenc_feature_b = None

                    ori_feature_keys = set(sentence_feature_a.keys())  # record the keys since the features will be updated

                    self.model[0].auto_model.set_flag(f"data_aug_{self.data_aug_strategy1}", True)
                    rep_a_view1 = self.model(sentence_feature_a)['sentence_embedding']
                    sentence_feature_a = {k: v for k, v in sentence_feature_a.items() if k in ori_feature_keys}
                    if not self.no_pair:
                        self.model[0].auto_model.set_flag(f"data_aug_{self.data_aug_strategy1}", True)
                        rep_b_view1 = self.model(sentence_feature_b)['sentence_embedding']
                        sentence_feature_b = {k: v for k, v in sentence_feature_b.items() if k in ori_feature_keys}
                    else:
                        rep_b_view1 = None
                    #flag에 feature cutoff는 없고 cutoff만 있음
                    self.model[0].auto_model.set_flag("data_aug_cutoff", True)
                    self.model[0].auto_model.set_flag("data_aug_cutoff.direction", "column")
                    self.model[0].auto_model.set_flag("data_aug_cutoff.rate", self.args.cutoff_rate)
                    rep_a_view2 = self.model(sentence_feature_a)['sentence_embedding']
                    if not self.no_pair:
                        self.model[0].auto_model.set_flag(f"data_aug_{self.data_aug_strategy2}", True)
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



            # if self.data_augmentation_strategy_final_1 is None: # final 과 그냥 strategy차이는?
            #     if self.use_contrastive_loss and self.data_augmentation_strategy == "shuffle": 
            #         if not self.no_pair:
            #             sentence_feature_a, sentence_feature_b = sentence_features
            #         else:
            #             sentence_feature_a = sentence_features[0]

            #         ori_feature_keys = set(sentence_feature_a.keys())  # record the keys since the features will be updated

            #         rep_a = self.model(sentence_feature_a)['sentence_embedding']
            #         sentence_feature_a = {k: v for k, v in sentence_feature_a.items() if k in ori_feature_keys}
            #         if not self.no_pair:
            #             rep_b = self.model(sentence_feature_b)['sentence_embedding']
            #             sentence_feature_b = {k: v for k, v in sentence_feature_b.items() if k in ori_feature_keys}
            #         else:
            #             rep_b = None

            #         self.model[0].auto_model.set_flag(f"data_aug_{self.data_augmentation_strategy}", True)
            #         rep_a_shuffle = self.model(sentence_feature_a)['sentence_embedding']
            #         if not self.no_pair:
            #             self.model[0].auto_model.set_flag(f"data_aug_{self.data_augmentation_strategy}", True)
            #             rep_b_shuffle = self.model(sentence_feature_b)['sentence_embedding']
            #         else:
            #             rep_b_shuffle = None

            #     elif self.use_contrastive_loss and self.data_augmentation_strategy == "cutoff": 
            #         if not self.no_pair:
            #             sentence_feature_a, sentence_feature_b = sentence_features
            #         else:
            #             sentence_feature_a = sentence_features[0]

            #         ori_feature_keys = set(sentence_feature_a.keys())  # record the keys since the features will be updated

            #         rep_a = self.model(sentence_feature_a)['sentence_embedding'] 
            #         #transformer and pooling   transformer forward에서 modeling.bert호출 + 이곳에서 data augmentation 진행 
            #         sentence_feature_a = {k: v for k, v in sentence_feature_a.items() if k in ori_feature_keys}
            #         if not self.no_pair:
            #             rep_b = self.model(sentence_feature_b)['sentence_embedding']
            #             sentence_feature_b = {k: v for k, v in sentence_feature_b.items() if k in ori_feature_keys}
            #         else:
            #             rep_b = None
                    
            #         #옵션을 키고 이제 다시 임베딩 통과,,,
            #         self.model[0].auto_model.set_flag("data_aug_cutoff", True)
            #         self.model[0].auto_model.set_flag("data_aug_cutoff.direction", self.cutoff_direction)
            #         self.model[0].auto_model.set_flag("data_aug_cutoff.rate", self.cutoff_rate)
            #         rep_a_cutoff = self.model(sentence_feature_a)['sentence_embedding']
            #         if not self.no_pair:
            #             self.model[0].auto_model.set_flag("data_aug_cutoff", True)
            #             self.model[0].auto_model.set_flag("data_aug_cutoff.direction", self.cutoff_direction)
            #             self.model[0].auto_model.set_flag("data_aug_cutoff.rate", self.cutoff_rate)
            #             rep_b_cutoff = self.model(sentence_feature_b)['sentence_embedding']
            #         else:
            #             rep_b_cutoff = None
            #     else: 
            #         reps = [self.model(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features]
            #         if not self.no_pair:
            #             rep_a, rep_b = reps
            #         else:
            #             rep_a, rep_b = reps[0], None
            # else:
            #     if not self.no_pair:
            #         sentence_feature_a, sentence_feature_b = sentence_features
            #     else:
            #         sentence_feature_a = sentence_features[0] #차피 0밖에없음
            #     #cutoff는 어디짜르는거지?
            #     ori_feature_keys = set(sentence_feature_a.keys())  # record the keys since the features will be updated
            #     rep_a_view1 = self._data_aug(sentence_feature_a, self.data_augmentation_strategy_final_1, ori_feature_keys, self.cutoff_rate_final_1)
            #     rep_a_view2 = self._data_aug(sentence_feature_a, self.data_augmentation_strategy_final_2, ori_feature_keys, self.cutoff_rate_final_2)
            #     if not self.no_pair:
            #         rep_b_view1 = self._data_aug(sentence_feature_b, self.data_augmentation_strategy_final_1, ori_feature_keys, self.cutoff_rate_final_1)
            #         rep_b_view2 = self._data_aug(sentence_feature_b, self.data_augmentation_strategy_final_2, ori_feature_keys, self.cutoff_rate_final_2)
            #     else:
            #         rep_b_view1 = None
            #         rep_b_view2 = None

            # loss calculation
            final_loss = 0
            
            if self.args.train_way == "joint" or self.args.train_way == "sup":
                match_output_n_n = self._reps_to_output(rep_a, rep_b)
                # match_outpput_n_n = [batch size, 3]    3 means num of classes (entail, contra, neutral)
                loss_fct = nn.CrossEntropyLoss()
                loss_n_n = loss_fct(match_output_n_n, labels.view(-1))
                final_loss += loss_n_n * adv_rate
                
            if self.args.train_way == "unsup" or self.args.train_way == "joint":
                contrastive_loss_a = self._contrastive_loss_forward(rep_a_view1 , rep_a_view2, hidden_norm=self.do_hidden_normalization, temperature=self.temperature) 
                if not self.no_pair: #recheck
                    contrastive_loss_b = self._contrastive_loss_forward(rep_b_view1, rep_b_view2, hidden_norm=self.do_hidden_normalization, temperature=self.temperature)
                else:
                    contrastive_loss_b = torch.tensor(0.0)
                contrastive_loss = contrastive_loss_a + contrastive_loss_b
                final_loss += self.contrastive_loss_rate * contrastive_loss * cl_rate
                if self.regularization_term_rate > 1e-10:
                    regularization_term = distance_to_center_mse_loss(rep_a_view1)  # note: only applied for rep_a_view1
                    final_loss += self.regularization_term_rate * regularization_term
            
            return final_loss