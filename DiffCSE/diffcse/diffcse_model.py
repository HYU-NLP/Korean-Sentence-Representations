import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

import transformers
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel
from transformers.activations import gelu
from transformers.file_utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from transformers.modeling_outputs import SequenceClassifierOutput, BaseModelOutputWithPoolingAndCrossAttentions
from transformers import AutoModel
class ProjectionMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        in_dim = config.hidden_size
        hidden_dim = config.hidden_size * 2
        out_dim = config.hidden_size
        affine=False
        list_layers = [nn.Linear(in_dim, hidden_dim, bias=False),
                       nn.BatchNorm1d(hidden_dim),
                       nn.ReLU(inplace=True)]
        list_layers += [nn.Linear(hidden_dim, out_dim, bias=False),
                        nn.BatchNorm1d(out_dim, affine=affine)]
        self.net = nn.Sequential(*list_layers)

    def forward(self, x):
        return self.net(x) # x = [bsz, num_sent, dim]

class Similarity(nn.Module):
    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        sim = self.cos(x, y)
        return sim / self.temp


class Pooler(nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, attention_mask, outputs):
        last_hidden = outputs.last_hidden_state
        pooler_output = outputs.pooler_output
        hidden_states = outputs.hidden_states

        return last_hidden[:, 0]

def cl_forward(cls,
    encoder,
    input_ids=None,
    attention_mask=None,
    token_type_ids=None,
    position_ids=None,
    head_mask=None,
    inputs_embeds=None,
    labels=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
    mlm_input_ids=None,
    mlm_labels=None,
    cls_token=101,
):

    return_dict = return_dict if return_dict is not None else cls.config.use_return_dict
    batch_size = input_ids.size(0)

    # Number of sentences in one instance
    # 2: pair instance; 3: pair instance with a hard negative
    num_sent = input_ids.size(1)

    mlm_outputs = None
    # Flatten input for encoding
    input_ids = input_ids.view((-1, input_ids.size(-1))) # (bs * num_sent, len)
    attention_mask = attention_mask.view((-1, attention_mask.size(-1))) # (bs * num_sent len)
    if token_type_ids is not None:
        token_type_ids = token_type_ids.view((-1, token_type_ids.size(-1))) # (bs * num_sent, len)

    # Get raw embeddings
    outputs = encoder(
        input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states=False,
        return_dict=True,
    ) # outputs = (bsz * num_sent, seq_len, dim)
    # Pooling
    pooler_output = cls.pooler(attention_mask, outputs) # (bsz * 2 , dim)
    pooler_output = pooler_output.view((batch_size, num_sent, pooler_output.size(-1))) # (bsz, 2, dim)
    ## add an extra MLP layer
    pooler_output = pooler_output.view((batch_size*num_sent, pooler_output.size(-1))) # (bsz, num_sent, dim)
    pooler_output = cls.mlp(pooler_output) # (bs, num_sent, hidden)
    pooler_output = pooler_output.view((batch_size, num_sent, pooler_output.size(-1))) # (bsz, num_sent, dim)


    ## Produce MLM augmentations and perform conditional ELECTRA using the discriminator
    if mlm_input_ids is not None:
        mlm_input_ids = mlm_input_ids.view((-1, mlm_input_ids.size(-1)))
        with torch.no_grad():
            g_pred = cls.generator(mlm_input_ids, attention_mask)[0].argmax(-1)
        g_pred[:, 0] = cls_token
        replaced = (g_pred != input_ids) * attention_mask
        e_inputs = g_pred * attention_mask  # (bsz * 2, seq_len)
        mlm_outputs = cls.discriminator(
            e_inputs,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=False, # because we use [CLS]
            return_dict=True,
            cls_input=pooler_output.view((-1, pooler_output.size(-1))),  # (bsz * 2, dim)
        )

    # Separate representation
    z1, z2 = pooler_output[:,0], pooler_output[:,1]

    cos_sim = cls.sim(z1.unsqueeze(1), z2.unsqueeze(0))
    
    labels = torch.arange(cos_sim.size(0)).long().to(cls.device)
    loss_fct = nn.CrossEntropyLoss()

    loss = loss_fct(cos_sim, labels)

    # Calculate loss for conditional ELECTRA
    if mlm_outputs is not None and mlm_labels is not None:
        # mlm_labels = mlm_labels.view(-1, mlm_labels.size(-1))
        e_labels = replaced.view(-1, replaced.size(-1))
        # prediction_scores = cls.lm_head(mlm_outputs.last_hidden_state)
        prediction_scores = cls.electra_head(mlm_outputs.last_hidden_state)
        # masked_lm_loss = loss_fct(prediction_scores.view(-1, cls.config.vocab_size), mlm_labels.view(-1))
        masked_lm_loss = loss_fct(prediction_scores.view(-1, 2), e_labels.view(-1))
        loss = loss + cls.model_args.lambda_weight * masked_lm_loss

    if not return_dict:
        output = (cos_sim,) + outputs[2:]
        return ((loss,) + output) if loss is not None else output
    return SequenceClassifierOutput(
        loss=loss,
        logits=cos_sim,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )


def sentemb_forward(
    cls,
    encoder,
    input_ids=None,
    attention_mask=None,
    token_type_ids=None,
    position_ids=None,
    head_mask=None,
    inputs_embeds=None,
    labels=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
):

    return_dict = return_dict if return_dict is not None else cls.config.use_return_dict

    outputs = encoder(
        input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states= False,
        return_dict=True,
    )

    pooler_output = cls.pooler(attention_mask, outputs)
    if not cls.model_args.mlp_only_train: # cls.pooler_type == "cls" and 
        pooler_output = cls.mlp(pooler_output)

    if not return_dict:
        return (outputs[0], pooler_output) + outputs[2:]

    return BaseModelOutputWithPoolingAndCrossAttentions(
        pooler_output=pooler_output,
        last_hidden_state=outputs.last_hidden_state,
        hidden_states=outputs.hidden_states,
    )


class BertForDiffCSE(BertPreTrainedModel) :
    def __init__(self, config, *model_args, **model_kargs):
        super().__init__(config)

        self.model_args = model_kargs["model_args"]
        self.bert = BertModel(config, add_pooling_layer = False)

        self.pooler = Pooler()
        self.mlp = ProjectionMLP(config)
        self.sim = Similarity(temp=self.model_args.temp)
        self.init_weights() # new version 에서는 post_init

        ##### modified #####
        self.discriminator = AutoModel.from_pretrained(self.model_args.model_name_or_path, config=config, add_pooling_layer = False)
        self.electra_head = torch.nn.Linear(768, 2)
        ##### modified #####
        self.generator = transformers.DistilBertForMaskedLM.from_pretrained('distilbert-base-uncased') if self.model_args.generator_name is None else transformers.AutoModelForMaskedLM.from_pretrained(self.model_args.generator_name)
        
    def forward(self, input_ids=None, 
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        sent_emb=False,
        mlm_input_ids=None,
        mlm_labels=None,
    ):
        if sent_emb:
            return sentemb_forward(self, self.bert,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        else:
            return cl_forward(self, self.bert,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                mlm_input_ids=mlm_input_ids,
                mlm_labels=mlm_labels,
                cls_token=101,
            )