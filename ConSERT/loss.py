import torch
from torch import nn, Tensor
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
    
class AdvCLSoftmaxLoss(nn.Module):
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
                 model: SentenceTransformer,
                 sentence_embedding_dimension: int,
                 num_labels: int,
                 concatenation_sent_rep: bool = True,
                 concatenation_sent_difference: bool = True,
                 concatenation_sent_multiplication: bool = False,
                 concatenation_sent_max_square: bool = False,           # 拼接两个句子表示的max-square（如寐建议的一个trick）
                 normal_loss_stop_grad: bool = False,                   # 对于传统损失（句子对分类）是否加stop-grad
                 
                 use_adversarial_training: bool = False,                # 是否加对抗损失
                 adversarial_loss_rate: float = 1.0,                    # 对抗损失的系数
                 do_noise_normalization: bool = True,                   # 是否将对抗扰动（噪声）正则化
                 noise_norm: float = 0.01,                              # 对抗扰动的大小
                 normal_normal_weight: float = 0.25,                    # normal to normal句子对分类损失的系数
                 normal_adv_weight: float = 0.25,                       # normal to adv句子对分类损失的系数
                 adv_normal_weight: float = 0.25,                       # adv to normal句子对分类损失的系数
                 adv_adv_weight: float = 0.25,                          # adv to adv句子对分类损失的系数
                 adv_loss_stop_grad: bool = False,                      # 对于对抗损失（一系列的句子对分类）是否加stop-grad
                 loss_rate_scheduler: int = 0,                          # 用来控制对比损失和主任务损失相对大小
                 
                 use_contrastive_loss: bool = False,                    # 是否加对比损失
                 data_augmentation_strategy: str= None,               
                 cutoff_direction: str = None,                          # 如果使用cutoff作为数据增强方法，该参数表示cutoff是对行进行还是对列进行
                 cutoff_rate: float = None,                             # 如果使用cutoff作为数据增强方法，该参数表示cutoff的比率（0到1之间，类似dropout）
                 data_augmentation_strategy_final_1: str = None,        # 最终的五种数据增强方法（none、shuffle、token-cutoff、feature-cutoff、dropout），用于生成第一个view
                 data_augmentation_strategy_final_2: str = None,        # 最终的五种数据增强方法（none、shuffle、token-cutoff、feature-cutoff、dropout），用于生成第二个view
                 cutoff_rate_final_1: float = None,                           # 与第一个view对应的cutoff/dropout的rate
                 cutoff_rate_final_2: float = None,                           # 与第二个view对应的cutoff/dropout的rate
                 contrastive_loss_only: bool = False,                   # 只使用对比损失进行（无监督）训练
                 no_pair: bool = False,                                 # 不使用配对的语料，避免先验信息
                 contrastive_loss_type: str = "nt_xent",                # 加对比损失的形式（“nt_xent” or “cosine”）
                 contrastive_loss_rate: float = 1.0,                    # 对比损失的系数
                 regularization_term_rate: float = 0.0,                 # 正则化项（同一个batch内分布的方差）所占的比率大小
                 do_hidden_normalization: bool = True,                  # 进行对比损失之前，是否对句子表示做正则化
                 temperature: float = 1.0,                              # 对比损失中的温度系数，仅对于交叉熵损失有效
                ):
        super(AdvCLSoftmaxLoss, self).__init__()
        self.model = model
        self.num_labels = num_labels
        self.concatenation_sent_rep = concatenation_sent_rep
        self.concatenation_sent_difference = concatenation_sent_difference
        self.concatenation_sent_multiplication = concatenation_sent_multiplication
        self.concatenation_sent_max_square = concatenation_sent_max_square
        self.normal_loss_stop_grad = normal_loss_stop_grad
        
        self.use_adversarial_training = use_adversarial_training
        self.adversarial_loss_rate = adversarial_loss_rate
        self.do_noise_normalization = do_noise_normalization
        self.noise_norm = noise_norm
        self.normal_normal_weight = normal_normal_weight
        self.normal_adv_weight = normal_adv_weight
        self.adv_normal_weight = adv_normal_weight
        self.adv_adv_weight = adv_adv_weight
        self.adv_loss_stop_grad = adv_loss_stop_grad
        self.loss_rate_scheduler = loss_rate_scheduler
        
        self.use_contrastive_loss = use_contrastive_loss
        assert data_augmentation_strategy in ("none", "adv", "meanmax", "shuffle", "cutoff", "shuffle-cutoff", "shuffle+cutoff", "shuffle_embeddings")
        if data_augmentation_strategy in ("cutoff", "shuffle-cutoff", "shuffle+cutoff"):
            assert cutoff_direction is not None and cutoff_direction in ("row", "column", "random")
            assert cutoff_rate is not None and 0.0 < cutoff_rate < 1.0
            self.cutoff_direction = cutoff_direction
            self.cutoff_rate = cutoff_rate
        self.data_augmentation_strategy = data_augmentation_strategy
        self.data_augmentation_strategy_final_1 = data_augmentation_strategy_final_1
        self.data_augmentation_strategy_final_2 = data_augmentation_strategy_final_2
        self.cutoff_rate_final_1 = cutoff_rate_final_1
        self.cutoff_rate_final_2 = cutoff_rate_final_2
        self.contrastive_loss_only = contrastive_loss_only
        self.no_pair = no_pair
        if no_pair:
            assert use_contrastive_loss and contrastive_loss_only
        assert contrastive_loss_type in ("nt_xent", "cosine") #nt_xent 그냥 논문에있던 식비스무리한거
        self.contrastive_loss_type = contrastive_loss_type
        self.contrastive_loss_rate = contrastive_loss_rate
        self.regularization_term_rate = regularization_term_rate
        self.do_hidden_normalization = do_hidden_normalization
        self.temperature = temperature
        

        num_vectors_concatenated = 0 #크로스엔트로피,,!!! (r1, r2, |r1-r2|)
        if concatenation_sent_rep:
            num_vectors_concatenated += 2
        if concatenation_sent_difference:
            num_vectors_concatenated += 1
        if concatenation_sent_multiplication:
            num_vectors_concatenated += 1
        if concatenation_sent_max_square:
            num_vectors_concatenated += 1
        logging.info("Softmax loss: #Vectors concatenated: {}".format(num_vectors_concatenated))
        #크로스엔트로피 로스용
        self.classifier = nn.Linear(num_vectors_concatenated * sentence_embedding_dimension, num_labels)
 
    def _reps_to_output(self, rep_a: torch.Tensor, rep_b: torch.Tensor):
        vectors_concat = []
        if self.concatenation_sent_rep:
            vectors_concat.append(rep_a)
            vectors_concat.append(rep_b)

        if self.concatenation_sent_difference:
            vectors_concat.append(torch.abs(rep_a - rep_b))

        if self.concatenation_sent_multiplication:
            vectors_concat.append(rep_a * rep_b)
        
        if self.concatenation_sent_max_square:
            vectors_concat.append(torch.max(rep_a, rep_b).pow(2))

        features = torch.cat(vectors_concat, 1)

        output = self.classifier(features)
        return output
    
    def _contrastive_loss_forward(self,
                                  hidden1: torch.Tensor,
                                  hidden2: torch.Tensor,
                                  hidden_norm: bool = True,
                                  temperature: float = 1.0):
        #this is from SimCLR paper 
        """
        hidden1/hidden2: (bsz, dim)
        """
        batch_size, hidden_dim = hidden1.shape
        # hidden1 = [96, 768]
        
        if self.add_projection:
            hidden1 = self.projection_head(hidden1)
            hidden2 = self.projection_head(hidden2)
        if self.projection_mode in ("both", "normal"):
            hidden1 = self.projection(hidden1)
        if self.projection_mode in ("both", "adv"):
            hidden2 = self.projection(hidden2)
        
        if self.contrastive_loss_type == "cosine":
            hidden1 = torch.nn.functional.normalize(hidden1, p=2, dim=-1)
            hidden2 = torch.nn.functional.normalize(hidden2, p=2, dim=-1)
            
            scores = torch.einsum("bd,bd->b", hidden1, hidden2)
            neg_cosine_loss = -1.0 * scores.mean()
            return neg_cosine_loss
            
        elif self.contrastive_loss_type == "nt_xent":
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
        return rep #sentence representation이구나
    
    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        
        if not self.training:  # when is it false?
            reps = [self.model(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features]
            rep_a, rep_b = reps
            
            output = self._reps_to_output(rep_a, rep_b)
            
            loss_fct = nn.CrossEntropyLoss()

            if labels is not None:
                loss = loss_fct(output, labels.view(-1))
                return loss
            else:
                return reps, output
        elif not self.use_adversarial_training and not self.use_contrastive_loss:  # 仅使用传统的监督训练方法（baseline设定下）
            reps = [self.model(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features]
            rep_a, rep_b = reps
            if self.normal_loss_stop_grad:
                rep_b = rep_b.detach()
            
            output = self._reps_to_output(rep_a, rep_b)
            
            loss_fct = nn.CrossEntropyLoss()

            if labels is not None:
                loss = loss_fct(output, labels.view(-1))
                return loss
            else:
                return reps, output
        else:  # 使用对抗训练或对比损失训练
            total_step, cur_step = self.model.num_steps_total, self.model.global_step
            adv_rate, cl_rate = LOSS_RATE_SCHEDULERS[self.loss_rate_scheduler](cur_step, total_step)
            
            # data augmentation generation
            
            if self.data_augmentation_strategy_final_1 is None: #final 과 그냥 strategy차이는?
                if self.use_adversarial_training or (self.use_contrastive_loss and self.data_augmentation_strategy == "adv"):  # 若需要用到对抗训练，或对比学习需要生产对抗样本做数据增强，就生成对抗样本
                    # 1. normal forward
                    sentence_feature_a, sentence_feature_b = sentence_features

                    ori_feature_keys = set(sentence_feature_a.keys())  # record the keys since the features will be updated

                    rep_a = self.model(sentence_feature_a)['sentence_embedding']
                    embedding_output_a = self.model[0].auto_model.get_most_recent_embedding_output()
                    rep_b = self.model(sentence_feature_b)['sentence_embedding']
                    embedding_output_b = self.model[0].auto_model.get_most_recent_embedding_output()

                    sentence_feature_a = {k: v for k, v in sentence_feature_a.items() if k in ori_feature_keys}
                    sentence_feature_b = {k: v for k, v in sentence_feature_b.items() if k in ori_feature_keys}

                    output = self._reps_to_output(rep_a, rep_b)

                    loss_fct = nn.CrossEntropyLoss()

                    normal_loss = loss_fct(output, labels.view(-1))

                    # 2. adversarial backward
                    embedding_output_a.retain_grad()
                    embedding_output_b.retain_grad()
                    normal_loss.backward(retain_graph=True)
                    unnormalized_noise_a = embedding_output_a.grad.detach_()
                    unnormalized_noise_b = embedding_output_b.grad.detach_()
                    for p in self.parameters():
                        if p.grad is not None:
                            p.grad.detach_()
                            p.grad.zero_()  # clear the gradient on parameters

                    if self.do_noise_normalization:  # do normalization
                        norm_a = unnormalized_noise_a.norm(p=2, dim=-1)
                        normalized_noise_a = unnormalized_noise_a / (norm_a.unsqueeze(dim=-1) + 1e-10)  # add 1e-10 to avoid NaN
                        norm_b = unnormalized_noise_b.norm(p=2, dim=-1)
                        normalized_noise_b = unnormalized_noise_b / (norm_b.unsqueeze(dim=-1) + 1e-10)  # add 1e-10 to avoid NaN
                    else:  # no normalization
                        normalized_noise_a = unnormalized_noise_a
                        normalized_noise_b = unnormalized_noise_b

                    noise_a = self.noise_norm * normalized_noise_a
                    noise_b = self.noise_norm * normalized_noise_b

                    # 3. adversarial forward
                    noise_embedding_a = embedding_output_a + noise_a
                    noise_embedding_b = embedding_output_b + noise_b

                    self.model[0].auto_model.set_flag("data_aug_adv", True)
                    self.model[0].auto_model.set_flag("noise_embedding", noise_embedding_a)
                    adv_rep_a = self.model(sentence_feature_a)['sentence_embedding']
                    self.model[0].auto_model.set_flag("data_aug_adv", True)
                    self.model[0].auto_model.set_flag("noise_embedding", noise_embedding_b)
                    adv_rep_b = self.model(sentence_feature_b)['sentence_embedding']

                elif self.use_contrastive_loss and self.data_augmentation_strategy == "meanmax":  # 使用mean-max pooling的对比
                    rep_dicts = [self.model(sentence_feature) for sentence_feature in sentence_features]
                    reps_mean = [rep_dict['pad_mean_tokens'] for rep_dict in rep_dicts]
                    if not self.no_pair:
                        rep_a_mean, rep_b_mean = reps_mean
                    else:
                        rep_a_mean, rep_b_mean = reps_mean[0], None
                    reps_max = [rep_dict['pad_max_tokens'] for rep_dict in rep_dicts]
                    if not self.no_pair:
                        rep_a_max, rep_b_max = reps_max
                    else:
                        rep_a_max, rep_b_max = reps_max[0], None

                elif self.use_contrastive_loss and self.data_augmentation_strategy in ("shuffle", "shuffle_embeddings"):  # 随机打乱词序
                    if not self.no_pair:
                        sentence_feature_a, sentence_feature_b = sentence_features
                    else:
                        sentence_feature_a = sentence_features[0]

                    ori_feature_keys = set(sentence_feature_a.keys())  # record the keys since the features will be updated

                    rep_a = self.model(sentence_feature_a)['sentence_embedding']
                    sentence_feature_a = {k: v for k, v in sentence_feature_a.items() if k in ori_feature_keys}
                    if not self.no_pair:
                        rep_b = self.model(sentence_feature_b)['sentence_embedding']
                        sentence_feature_b = {k: v for k, v in sentence_feature_b.items() if k in ori_feature_keys}
                    else:
                        rep_b = None

                    self.model[0].auto_model.set_flag(f"data_aug_{self.data_augmentation_strategy}", True)
                    rep_a_shuffle = self.model(sentence_feature_a)['sentence_embedding']
                    if not self.no_pair:
                        self.model[0].auto_model.set_flag(f"data_aug_{self.data_augmentation_strategy}", True)
                        rep_b_shuffle = self.model(sentence_feature_b)['sentence_embedding']
                    else:
                        rep_b_shuffle = None

                elif self.use_contrastive_loss and self.data_augmentation_strategy == "cutoff":  # cutoff数据增强策略
                    if not self.no_pair:
                        sentence_feature_a, sentence_feature_b = sentence_features
                    else:
                        sentence_feature_a = sentence_features[0]

                    ori_feature_keys = set(sentence_feature_a.keys())  # record the keys since the features will be updated

                    rep_a = self.model(sentence_feature_a)['sentence_embedding'] 
                    #transformer and pooling   transformer forward에서 modeling.bert호출 + 이곳에서 data augmentation 진행 
                    sentence_feature_a = {k: v for k, v in sentence_feature_a.items() if k in ori_feature_keys}
                    if not self.no_pair:
                        rep_b = self.model(sentence_feature_b)['sentence_embedding']
                        sentence_feature_b = {k: v for k, v in sentence_feature_b.items() if k in ori_feature_keys}
                    else:
                        rep_b = None
                    
                    #옵션을 키고 이제 다시 임베딩 통과,,,
                    self.model[0].auto_model.set_flag("data_aug_cutoff", True)
                    self.model[0].auto_model.set_flag("data_aug_cutoff.direction", self.cutoff_direction)
                    self.model[0].auto_model.set_flag("data_aug_cutoff.rate", self.cutoff_rate)
                    rep_a_cutoff = self.model(sentence_feature_a)['sentence_embedding']
                    if not self.no_pair:
                        self.model[0].auto_model.set_flag("data_aug_cutoff", True)
                        self.model[0].auto_model.set_flag("data_aug_cutoff.direction", self.cutoff_direction)
                        self.model[0].auto_model.set_flag("data_aug_cutoff.rate", self.cutoff_rate)
                        rep_b_cutoff = self.model(sentence_feature_b)['sentence_embedding']
                    else:
                        rep_b_cutoff = None

                elif self.use_contrastive_loss and self.data_augmentation_strategy == "shuffle-cutoff":  # 分别用shuffle和cutoff来生成两个view
                    if not self.no_pair:
                        sentence_feature_a, sentence_feature_b = sentence_features
                    else:
                        sentence_feature_a = sentence_features[0]

                    ori_feature_keys = set(sentence_feature_a.keys())  # record the keys since the features will be updated

                    # shuffle strategy
                    self.model[0].auto_model.set_flag("data_aug_shuffle", True)
                    rep_a_shuffle = self.model(sentence_feature_a)['sentence_embedding']
                    sentence_feature_a = {k: v for k, v in sentence_feature_a.items() if k in ori_feature_keys}
                    if not self.no_pair:
                        self.model[0].auto_model.set_flag("data_aug_shuffle", True)
                        rep_b_shuffle = self.model(sentence_feature_b)['sentence_embedding']
                        sentence_feature_b = {k: v for k, v in sentence_feature_b.items() if k in ori_feature_keys}
                    else:
                        rep_b_shuffle = None

                    # cutoff strategy
                    self.model[0].auto_model.set_flag("data_aug_cutoff", True)
                    self.model[0].auto_model.set_flag("data_aug_cutoff.direction", self.cutoff_direction)
                    self.model[0].auto_model.set_flag("data_aug_cutoff.rate", self.cutoff_rate)
                    rep_a_cutoff = self.model(sentence_feature_a)['sentence_embedding']
                    if not self.no_pair:
                        self.model[0].auto_model.set_flag("data_aug_cutoff", True)
                        self.model[0].auto_model.set_flag("data_aug_cutoff.direction", self.cutoff_direction)
                        self.model[0].auto_model.set_flag("data_aug_cutoff.rate", self.cutoff_rate)
                        rep_b_cutoff = self.model(sentence_feature_b)['sentence_embedding']
                    else:
                        rep_b_cutoff = None
                    # for supervised loss
                    rep_a = rep_a_cutoff
                    rep_b = rep_b_cutoff

                elif self.use_contrastive_loss and self.data_augmentation_strategy == "shuffle+cutoff":  # 用shuffle和cutoff的组合作为一个view
                    if not self.no_pair:
                        sentence_feature_a, sentence_feature_b = sentence_features
                    else:
                        sentence_feature_a = sentence_features[0]

                    ori_feature_keys = set(sentence_feature_a.keys())  # record the keys since the features will be updated

                    rep_a = self.model(sentence_feature_a)['sentence_embedding']
                    sentence_feature_a = {k: v for k, v in sentence_feature_a.items() if k in ori_feature_keys}
                    if not self.no_pair:
                        rep_b = self.model(sentence_feature_b)['sentence_embedding']
                        sentence_feature_b = {k: v for k, v in sentence_feature_b.items() if k in ori_feature_keys}
                    else:
                        rep_b = None

                    self.model[0].auto_model.set_flag("data_aug_shuffle", True)
                    self.model[0].auto_model.set_flag("data_aug_cutoff", True)
                    self.model[0].auto_model.set_flag("data_aug_cutoff.direction", self.cutoff_direction)
                    self.model[0].auto_model.set_flag("data_aug_cutoff.rate", self.cutoff_rate)
                    rep_a_shuffle_cutoff = self.model(sentence_feature_a)['sentence_embedding']
                    if not self.no_pair:
                        self.model[0].auto_model.set_flag("data_aug_shuffle", True)
                        self.model[0].auto_model.set_flag("data_aug_cutoff", True)
                        self.model[0].auto_model.set_flag("data_aug_cutoff.direction", self.cutoff_direction)
                        self.model[0].auto_model.set_flag("data_aug_cutoff.rate", self.cutoff_rate)
                        rep_b_shuffle_cutoff = self.model(sentence_feature_b)['sentence_embedding']
                    else:
                        rep_b_shuffle_cutoff = None
                        
                else:  # 最原始的版本，只需获取rep_a和rep_b即可  # TODO: 在这里添加更多的数据增强策略
                    reps = [self.model(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features]
                    if not self.no_pair:
                        rep_a, rep_b = reps
                    else:
                        rep_a, rep_b = reps[0], None
            else:
                if not self.no_pair:
                    sentence_feature_a, sentence_feature_b = sentence_features
                else:
                    sentence_feature_a = sentence_features[0] #차피 0밖에없음
                #cutoff는 어디짜르는거지?
                ori_feature_keys = set(sentence_feature_a.keys())  # record the keys since the features will be updated
                rep_a_view1 = self._data_aug(sentence_feature_a, self.data_augmentation_strategy_final_1, ori_feature_keys, self.cutoff_rate_final_1)
                rep_a_view2 = self._data_aug(sentence_feature_a, self.data_augmentation_strategy_final_2, ori_feature_keys, self.cutoff_rate_final_2)
                if not self.no_pair:
                    rep_b_view1 = self._data_aug(sentence_feature_b, self.data_augmentation_strategy_final_1, ori_feature_keys, self.cutoff_rate_final_1)
                    rep_b_view2 = self._data_aug(sentence_feature_b, self.data_augmentation_strategy_final_2, ori_feature_keys, self.cutoff_rate_final_2)
                else:
                    rep_b_view1 = None
                    rep_b_view2 = None

            # loss calculation
            final_loss = 0
            
            if self.use_adversarial_training:
                if self.adv_loss_stop_grad:
                    rep_b = rep_b.detach()
                    adv_rep_b = adv_rep_b.detach()
                match_output_n_n = self._reps_to_output(rep_a, rep_b)
                match_output_n_a = self._reps_to_output(rep_a, adv_rep_b)
                match_output_a_n = self._reps_to_output(adv_rep_a, rep_b)
                match_output_a_a = self._reps_to_output(adv_rep_a, adv_rep_b)

                loss_n_n = loss_fct(match_output_n_n, labels.view(-1))
                loss_n_a = loss_fct(match_output_n_a, labels.view(-1))
                loss_a_n = loss_fct(match_output_a_n, labels.view(-1))
                loss_a_a = loss_fct(match_output_a_a, labels.view(-1))

                adv_training_loss = self.normal_normal_weight * loss_n_n + self.normal_adv_weight * loss_n_a + \
                                    self.adv_normal_weight * loss_a_n + self.adv_adv_weight * loss_a_a
                final_loss += self.adversarial_loss_rate * adv_training_loss * adv_rate
                self.model.tensorboard_writer.add_scalar(f"train_adv_loss", self.adversarial_loss_rate * adv_rate * adv_training_loss.item(), global_step=self.model.global_step)
            elif not self.contrastive_loss_only:
                match_output_n_n = self._reps_to_output(rep_a, rep_b)
                # match_outpput_n_n = [96, 3]    3 means num of classes (entail, contra, neutral)
                loss_fct = nn.CrossEntropyLoss()
                loss_n_n = loss_fct(match_output_n_n, labels.view(-1))
                final_loss += loss_n_n * adv_rate
                self.model.tensorboard_writer.add_scalar(f"train_normal_loss", loss_n_n.item() * adv_rate, global_step=self.model.global_step)
                
            if self.use_contrastive_loss:
                if self.data_augmentation_strategy_final_1 is None:
                    if self.data_augmentation_strategy == "adv":
                        rep_a_view1, rep_b_view1 = rep_a, rep_b
                        rep_a_view2, rep_b_view2 = adv_rep_a, adv_rep_b
                    elif self.data_augmentation_strategy == "none":
                        rep_a_view1, rep_b_view1 = rep_a, rep_b
                        rep_a_view2, rep_b_view2 = rep_a, rep_b
                    elif self.data_augmentation_strategy == "meanmax":
                        rep_a_view1, rep_b_view1 = rep_a_mean, rep_b_mean
                        rep_a_view2, rep_b_view2 = rep_a_max, rep_b_max
                    elif self.data_augmentation_strategy in ("shuffle", "shuffle_embeddings"):
                        rep_a_view1, rep_b_view1 = rep_a, rep_b
                        rep_a_view2, rep_b_view2 = rep_a_shuffle, rep_b_shuffle
                    elif self.data_augmentation_strategy == "cutoff":
                        rep_a_view1, rep_b_view1 = rep_a, rep_b
                        rep_a_view2, rep_b_view2 = rep_a_cutoff, rep_b_cutoff
                    elif self.data_augmentation_strategy == "shuffle-cutoff":
                        rep_a_view1, rep_b_view1 = rep_a_shuffle, rep_b_shuffle
                        rep_a_view2, rep_b_view2 = rep_a_cutoff, rep_b_cutoff
                    elif self.data_augmentation_strategy == "shuffle+cutoff":
                        rep_a_view1, rep_b_view1 = rep_a, rep_b
                        rep_a_view2, rep_b_view2 = rep_a_shuffle_cutoff, rep_b_shuffle_cutoff
                    else:
                        raise ValueError("Invalid data augmentation strategy")
                contrastive_loss_a = self._contrastive_loss_forward(rep_a_view1, rep_a_view2, hidden_norm=self.do_hidden_normalization, temperature=self.temperature)
                #rep_a_view1 : cutoff,  rep_a_view2 : shuffle
                self.model.tensorboard_writer.add_scalar(f"train_contrastive_loss_a", contrastive_loss_a.item(), global_step=self.model.global_step)
                if not self.no_pair: #recheck
                    contrastive_loss_b = self._contrastive_loss_forward(rep_b_view1, rep_b_view2, hidden_norm=self.do_hidden_normalization, temperature=self.temperature)
                else:
                    contrastive_loss_b = torch.tensor(0.0)
                self.model.tensorboard_writer.add_scalar(f"train_contrastive_loss_b", contrastive_loss_b.item(), global_step=self.model.global_step)
                contrastive_loss = contrastive_loss_a + contrastive_loss_b
                final_loss += self.contrastive_loss_rate * contrastive_loss * cl_rate
                self.model.tensorboard_writer.add_scalar(f"train_contrastive_loss_total", self.contrastive_loss_rate * cl_rate * contrastive_loss.item(), global_step=self.model.global_step)
                if self.regularization_term_rate > 1e-10:
                    regularization_term = distance_to_center_mse_loss(rep_a_view1)  # note: only applied for rep_a_view1
                    final_loss += self.regularization_term_rate * regularization_term
                    self.model.tensorboard_writer.add_scalar(f"contrastive_loss_regularization_term", self.regularization_term_rate * regularization_term.item(), global_step=self.model.global_step)
            
            return final_loss