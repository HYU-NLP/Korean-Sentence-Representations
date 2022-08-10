from cmath import cos
import os
import argparse
import os
from datetime import datetime

import pandas as pd
import torch
from torch import cosine_similarity, nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import set_seed, BertTokenizer, BertModel, BertConfig

import numpy as np
from sklearn.metrics import accuracy_score, mean_squared_error
from scipy.stats import pearsonr, spearmanr
import copy
from datasets import load_dataset


class BertForSupervisedSimCse(nn.Module):
    def __init__(self, model_name, temperature, num_labels): # num_labels
        super(BertForSupervisedSimCse, self).__init__()


        self.config = BertConfig.from_pretrained(model_name)  # uses default dropout rate = 0.1
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.hidden_size = BertConfig.from_pretrained(model_name).hidden_size
        
        self.cosine_similarity = nn.CosineSimilarity(dim=-1)
        self.temperature = temperature
            
        self.linear = nn.Linear(in_features=self.hidden_size, out_features=num_labels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask, token_type_ids, **kwargs):
        batch_size = input_ids.shape[0]

        # flat, (batch_size * 3, seq_len)
        input_ids = input_ids.view((-1, input_ids.shape[-1]))
        attention_mask = attention_mask.view((-1, input_ids.shape[-1]))
        token_type_ids = token_type_ids.view((-1, input_ids.shape[-1]))

        # encode, (batch_size * 3, hidden_size)
        _, pooler_out = self.bert(input_ids, attention_mask, token_type_ids, return_dict=False) # [24, 3, 768]

        # revert flat, (batch_size, 3, hidden_size)
        pooler_out = pooler_out.view((batch_size, 3, pooler_out.shape[-1])) #[72, 768]

        # cos sim
        z1, z2, z3 = pooler_out[:, 0], pooler_out[:, 1], pooler_out[:, 2]
        z1_z2_cos = self.cosine_similarity(z1.unsqueeze(1), z2.unsqueeze(0)) / self.temperature
        z1_z3_cos = self.cosine_similarity(z1.unsqueeze(1), z3.unsqueeze(0)) / self.temperature
        cos_sim = torch.cat([z1_z2_cos, z1_z3_cos], dim=1)
        return cos_sim
    
    def encode(self, batch, **kwargs):
        input_ids_1, attention_mask_1, token_type_ids_1 = batch['input_ids_1'], batch['attention_mask_1'], batch['token_type_ids_1']
        input_ids_2, attention_mask_2, token_type_ids_2 = batch['input_ids_2'], batch['attention_mask_2'], batch['token_type_ids_2']
        _, pooler_out_1 = self.bert(input_ids_1, attention_mask_1, token_type_ids_1, return_dict=False)
        _, pooler_out_2 = self.bert(input_ids_2, attention_mask_2, token_type_ids_2, return_dict=False)

        return pooler_out_1.squeeze(), pooler_out_2.squeeze()



class SupervisedSimCseDataset(Dataset):
    def __init__(self, mode, data_frame, tokenizer, max_length, column_names=None):
        if column_names is None:
            if mode == "train" :
                column_names = ['sent0', 'sent1', 'hard_neg']
            elif mode == "val" :
                column_names = ['sentence1', 'sentence2', 'label']
                
        self.len = len(data_frame)
        self.data_frame = data_frame
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.column_names = column_names
        self.mode = mode

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        self.total = {}
        if self.mode == "nli" :
            premise = self.tokenizer(self.data_frame[self.column_names[0]][idx], padding='max_length', max_length=self.max_length, truncation=True, return_tensors="pt")
            hypothesis = self.tokenizer(self.data_frame[self.column_names[1]][idx], padding='max_length', max_length=self.max_length, truncation=True, return_tensors="pt")
            hard_neg = self.tokenizer(self.data_frame[self.column_names[2]][idx], padding='max_length', max_length=self.max_length, truncation=True, return_tensors="pt")

            return {
                'input_ids': torch.stack((premise['input_ids'].squeeze(), hypothesis['input_ids'].squeeze(), hard_neg['input_ids'].squeeze())),
                'attention_mask': torch.stack((premise['attention_mask'].squeeze(), hypothesis['attention_mask'].squeeze(), hard_neg['attention_mask'].squeeze())),
                'token_type_ids': torch.stack((premise['token_type_ids'].squeeze(), hypothesis['token_type_ids'].squeeze(), hard_neg['token_type_ids'].squeeze())),
            }
        elif self.mode == "sts-b" :
            self.total = {}
            self.sentence1_tokens = self.tokenizer(self.data_frame['sentence1'][idx], truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt")
            self.sentence2_tokens = self.tokenizer(self.data_frame['sentence2'][idx], truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt")
            # sentence1
            self.total['input_ids_1'] = self.sentence1_tokens['input_ids'].squeeze()
            self.total['token_type_ids_1'] = self.sentence1_tokens['token_type_ids'].squeeze()
            self.total['attention_mask_1'] = self.sentence1_tokens['attention_mask'].squeeze()
            # sentence2
            self.total['input_ids_2'] = self.sentence2_tokens['input_ids'].squeeze()
            self.total['token_type_ids_2'] = self.sentence2_tokens['token_type_ids'].squeeze()
            self.total['attention_mask_2'] = self.sentence2_tokens['attention_mask'].squeeze()
            self.total['labels'] = torch.Tensor(self.data_frame['label'])[idx]
            return self.total


def save_model_config(path, model_name, model_state_dict, model_config_dict):
    dirname = os.path.dirname(path)
    os.makedirs(dirname, exist_ok=True)
    torch.save({
        'model_name': model_name,
        'model_state_dict': model_state_dict,
        'model_config_dict': model_config_dict
    }, path)
    
def model_save_fn(args, pretrained_model):
    if pretrained_model != None : 
        save_model_config(f'checkpoint/{args.model_state_name}', args.model_name, pretrained_model.bert.state_dict(), pretrained_model.bert.config.to_dict())

def evaluate_model (device, dataloader, model, fn_loss, fn_score, temperature):
    model.to(device)
    fn_loss.to(device)

    eval_loss = 0
    eval_pred = []
    eval_label = []
    
    model.eval()
    with torch.no_grad():
        for val_batch in dataloader : # sts-b
            val_batch = {k: v.to(device) for k, v in val_batch.items()}
            # labels = batch['labels'].to(device)
            emb1, emb2 = model.encode(val_batch) # 24,1
            cos_sim = cosine_similarity(emb1, emb2) / temperature
            
            loss = fn_loss (cos_sim.unsqueeze(0), val_batch['labels'].unsqueeze(0)) #batch['labels].size() = torch.size([24]) #.float().unsqueeze(0)
            
            eval_loss += loss.item()
            eval_pred.extend(cos_sim.clone().cpu().tolist())
            eval_label.extend(val_batch['labels'].clone().cpu().tolist())
        
    eval_score = fn_score(eval_pred, eval_label) 
    return eval_score, eval_loss

def pretrain_model(epochs, device, train_dataloader, validation_dataloader, model, fn_loss, optimizer, fn_score, model_save_fn, args):
    model.to(device)
    fn_loss.to(device)

    best_val_loss = 0
    best_val_score = 0
    best_model = None
    
    for epoch in range(1, epochs + 1):
        model.train()

        train_loss = 0

        for i, batch in enumerate(tqdm(train_dataloader)):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels = torch.arange(batch['input_ids'].shape[0]).to(device) 

            optimizer.zero_grad()
            predict = model(input_ids, attention_mask, token_type_ids)
            loss = fn_loss(predict, labels) #labels.to(torch.longtensor()) -> label long으로 하기 
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            # eval
            if (i + (epoch - 1) * len(train_dataloader)) % 250 == 0:
                val_score, val_loss = evaluate_model(device, validation_dataloader, model, fn_loss, fn_score, args.temperature)
                if val_score > best_val_score :
                    best_model = copy.deepcopy(model)
                    best_val_score = val_score
                    
                print(f"\n{epoch}th epoch Validation loss / cur_val_score / best_val_score : {val_loss} / {val_score} / {best_val_score}")
    
    model_save_fn(args, best_model) 
    
    return best_model


def main():
    # Parser --
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='bert-base-uncased', type=str)  # simcse used bert-base-uncased and roberta-base-cased
    parser.add_argument('--batch_size', default=24, type=int)  # simcse used batch size {64, 128, 256, 512}
    parser.add_argument('--seq_max_length', default=128, type=int)
    parser.add_argument('--epochs', default=3, type=int)
    parser.add_argument('--lr', default=1e-5, type=float)  # simcse used 1e-5, 3e-5, 5e-5
    parser.add_argument('--gpu', default=1, type=int)
    parser.add_argument('--seed', default=4885, type=int)
    parser.add_argument('--task', default='sup_simcse', type=str)
    parser.add_argument('--model_state_name', default='sup_simcse_bert.pt', type=str)  # for downstream tasks, if 'model_state_name' exist, 'model_name' will be ignored
    parser.add_argument('--pretrain_dataset', default='nli_for_simcse.csv', type=str)  # for pretrained task
    parser.add_argument('--temperature', default=0.05, type=float)
    # parser.add_argument('--add_mlp_layer', default=True, type=bool)

    args = parser.parse_known_args()[0]
    setattr(args, 'device', f'cuda:{args.gpu}' if torch.cuda.is_available() and args.gpu >= 0 else 'cpu')
    setattr(args, 'time', datetime.now().strftime('%Y%m%d-%H:%M:%S'))

    print('[List of arguments]')
    for a in args.__dict__:
        print(f'{a}: {args.__dict__[a]}')

    # Device --
    device = args.device

    # Hyper parameter --
    set_seed(args.seed)
    model_name = args.model_name
    batch_size = args.batch_size
    seq_max_length = args.seq_max_length
    epochs = args.epochs
    learning_rate = args.lr
    task = args.task

    if task == "sup_simcse":
        # Prepare tokenizer, dataset (+ dataloader), model, loss function, optimizer, etc --
        temperature = args.temperature
        pretrain_dataset = args.pretrain_dataset
        model_state_name = args.model_state_name
        
        df_train = pd.read_csv(f'/home/skchajie/Proj-Sentence-Representation-main/Supervised-SimCSE/dataset/{pretrain_dataset}', sep=',')
        df_val = load_dataset('glue', 'stsb', split="validation")
        
        tokenizer = BertTokenizer.from_pretrained(model_name)
        
        train_dataset = SupervisedSimCseDataset("nli", df_train, tokenizer, seq_max_length)
        validation_dataset = SupervisedSimCseDataset("sts-b", df_val, tokenizer, seq_max_length)
    
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
        validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size)
        
        model = BertForSupervisedSimCse(model_name, temperature, 1)
        optimizer = AdamW(model.parameters(), lr=learning_rate)  # AdamW is used in SimCSE original project
        
        fn_loss = nn.CrossEntropyLoss()

        def fn_score(output, label):
            score = spearmanr(output, label)[0]
            return score
  
        best_model = pretrain_model(epochs, device, train_dataloader, validation_dataloader, model, fn_loss, optimizer, fn_score, model_save_fn, args)

        #evaluate sent eval transfer task (best model)
        #evaludate sent eval sts task (best model)
        
    else:
        ValueError(f"Unknown task: {task}")


if __name__ == "__main__":
    main()
