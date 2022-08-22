import argparse
import copy
from datetime import datetime
from scipy import stats
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
# import pandas as pd
import os

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from transformers import set_seed
from transformers import BertModel, BertConfig, BertTokenizer
from custom_optim import EarlyStopping, LinearLR

class BertForUnsupervisedSimCSE(nn.Module):
    def __init__(self, bert_model_name):
        super(BertForUnsupervisedSimCSE, self).__init__()

        self.hidden_size = BertConfig.from_pretrained(bert_model_name).hidden_size
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.linear = nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, batch):
        if len(batch) == 3 : # train batch data encoding
            input_ids, attention_mask, token_type_ids = batch['input_ids'], batch['attention_mask'], batch['token_type_ids']
            _, pooler_out = self.bert(input_ids, attention_mask, token_type_ids, return_dict=False)
            # linear_out = self.linear(pooler_out)
            # activation_out = self.activation(linear_out)
            return pooler_out.squeeze()
            # return activation_out.squeeze()
        else : # validation train batch data encoding, same as len(batch) == 7
            input_ids_1, attention_mask_1, token_type_ids_1 = batch['input_ids_1'], batch['attention_mask_1'], batch['token_type_ids_1']
            input_ids_2, attention_mask_2, token_type_ids_2 = batch['input_ids_2'], batch['attention_mask_2'], batch['token_type_ids_2']
            _, pooler_out_1 = self.bert(input_ids_1, attention_mask_1, token_type_ids_1, return_dict=False)
            _, pooler_out_2 = self.bert(input_ids_2, attention_mask_2, token_type_ids_2, return_dict=False)
            # linear_out_1 = self.linear(pooler_out_1)
            # linear_out_2 = self.linear(pooler_out_2)
            # activation_out_1 = self.activation(linear_out_1)
            # activation_out_2 = self.activation(linear_out_2)
            return pooler_out_1.squeeze(), pooler_out_2.squeeze()
            # return activation_out_1.squeeze(), activation_out_2.squeeze()

class wikiDataset(Dataset):
    def __init__(self, example_text, args, tokenizer):
        self.args = args
        if example_text == True: # 4개 말고 50 ~ 100개 정도로 늘여서 실험
            data = [
            "chocolates are my favourite items.",
            "The fish dreamed of escaping the fishbowl and into the toilet where he saw his friend go.",
            "The person box was packed with jelly many dozens of months later.",
            "I love chocolates."
                ]
        else :
            # dataset_df = pd.read_csv("Proj-Sentence-Representation/Unsupervised_SimCSE/wiki1m_for_simcse.txt", names=["text"], on_bad_lines='skip')
            # dataset_df.dropna(inplace=True)
            # data = list(dataset_df['train']["text"].values)
            
            dataset_df = load_dataset('text', data_files='Proj-Sentence-Representation/Unsupervised_SimCSE/wiki1m_for_simcse.txt')
            data = dataset_df['train']['text']
            for idx in range(len(data)):
                if data[idx] is None:
                    data[idx] = " "
        self.data = data
        self.len = len(data)
        self.tokenizer = tokenizer
        
    def __len__(self): 
        return self.len
    
    def __getitem__(self,idx) : 

        self.tokens = self.tokenizer(self.data[idx], truncation=True, padding="max_length", max_length=self.args.seq_max_length, return_tensors="pt")
        self.tokens['input_ids'] = self.tokens['input_ids'].squeeze()
        self.tokens['token_type_ids'] = self.tokens['token_type_ids'].squeeze()
        self.tokens['attention_mask'] = self.tokens['attention_mask'].squeeze()
        return self.tokens
    
class STSBenchmark(Dataset):
    def __init__(self, args, tokenizer):
        self.args = args
        data = load_dataset('glue', 'stsb', split="validation")
        self.data = data
        self.len = len(data)
        self.tokenizer = tokenizer
        
    def __len__(self): 
        return self.len
    
    # tokenize 2 sentences
    def __getitem__(self,idx) : 
        self.total_tokens = {}
        self.sentence1_tokens = self.tokenizer(self.data['sentence1'][idx], truncation=True, padding="max_length", max_length=self.args.seq_max_length, return_tensors="pt")
        self.sentence2_tokens = self.tokenizer(self.data['sentence2'][idx], truncation=True, padding="max_length", max_length=self.args.seq_max_length, return_tensors="pt")
        # sentence1
        self.total_tokens['input_ids_1'] = self.sentence1_tokens['input_ids'].squeeze()
        self.total_tokens['token_type_ids_1'] = self.sentence1_tokens['token_type_ids'].squeeze()
        self.total_tokens['attention_mask_1'] = self.sentence1_tokens['attention_mask'].squeeze()
        # sentence2
        self.total_tokens['input_ids_2'] = self.sentence2_tokens['input_ids'].squeeze()
        self.total_tokens['token_type_ids_2'] = self.sentence2_tokens['token_type_ids'].squeeze()
        self.total_tokens['attention_mask_2'] = self.sentence2_tokens['attention_mask'].squeeze()
        self.total_tokens['labels'] = torch.Tensor(self.data['label'])[idx]
        return self.total_tokens
        
def train_setting(args, tokenizer):
    batch_size = args.batch_size
    example_text = args.example_text
    set_seed(args.seed)

    train_dataset = wikiDataset(example_text, args, tokenizer)
    validation_dataset = STSBenchmark(args, tokenizer)
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size)

    return train_dataloader, validation_dataloader

def get_score(output, label):
    score = stats.spearmanr(output, label)[0]
    return score

def cosine_similarity(embeddings1, embeddings2, temperature=0.05): # sentence embedding size : [batch_size, 768]
    # unit_embed1 = embeddings1 / torch.norm(embeddings1)
    # unit_embed2 = embeddings2 / torch.norm(embeddings2)
    # unit_embed1 = embeddings1.clone().detach()
    # unit_embed2 = embeddings2.clone().detach()
    # for i in range(embeddings1.size()[0]):
    #     unit_embed1[i] = embeddings1[i] / torch.norm(embeddings1[i])
    #     unit_embed2[i] = embeddings2[i] / torch.norm(embeddings2[i])
    # similarity = torch.matmul(unit_embed1, torch.transpose(unit_embed2, 0, 1)) / temperature
    
    # qmin's solution; entire progress of normalization is same as mine
    unit_embed1 = torch.nn.functional.normalize(embeddings1, dim=-1)
    unit_embed2 = torch.nn.functional.normalize(embeddings2, dim=-1)
    similarity = torch.matmul(unit_embed1, torch.transpose(unit_embed2, 0, 1)) / temperature
    return similarity

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
        save_model_config(f'Proj-Sentence-Representation/Unsupervised_SimCSE/checkpoint/{args.model_state_name}', args.model_name, pretrained_model.bert.state_dict(), pretrained_model.bert.config.to_dict())
    
def unsupervised_train(args, train_dataloader, validation_dataloader, model, loss_fn):
    writer = SummaryWriter()
    device = args.device
    learning_rate = args.lr
    epochs = args.epochs
    temperature = args.temperature
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    scheduler = LinearLR(optimizer, start_factor=learning_rate, total_iters=4)
    
    model.to(device)
    loss_fn.to(device)
    best_val_score = 0
    best_model = None
    big_step = 0
    
    print("\n----------<\tUnsupervised SimCSE training start\t>----------")
    for t in range(epochs):
        print(f"Epoch {t+1} :")
        train_early_stop = EarlyStopping(patience=10)
        valid_early_stop = EarlyStopping(patience=10)
        model.train()
        for step, batch in enumerate(tqdm(train_dataloader)):
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            output1 = model(batch)
            output2 = model(batch)

            cos_sim = cosine_similarity(output1, output2, temperature)
            if cos_sim.dim() == 0 : 
                labels = torch.tensor(0)
            else :  
                labels = torch.arange(cos_sim.size(0))
            
            train_loss = loss_fn(cos_sim, labels.to(device))
            train_loss.backward()
            optimizer.step()
            scheduler.step()
            
            if (step + 1) % 250 == 0 or step == len(train_dataloader) - 1:
                # big_step += 1
                # print(f"Diagonal : {torch.diagonal(cos_sim)*temperature}")
                print(f'\n[Iteration {step + 1}] train loss: ({train_loss:.4})')
                
                model.eval()
                with torch.no_grad():
                    val_pred = []
                    val_label = []
                    val_score = 0

                    for _, val_batch in enumerate(tqdm(validation_dataloader)):
                        val_batch = {k: v.to(device) for k, v in val_batch.items()}
                        val_output1, val_output2 = model(val_batch)
                        # train에선 temperature를 넣지만 여기선 그러면 안됨
                        # train에서 embedding 간 거리 조절을 위해 넣은거지 validation에서는 그럴 필요가 없음
                        val_cos_sim = cosine_similarity(val_output1, val_output2, temperature = 1)
                        
                        if val_cos_sim.dim() == 0 : 
                            val_cos_sim = val_cos_sim.unsqueeze(dim=0)
                        
                        # val_loss = loss_fn(val_cos_sim, val_labels.to(device))
                        # val_loss로 똑같이 할 수 없으니까 spearman correlation으로 metric을 정한 것
                        # 따라서 val_loss의 가치가 없음
                        # val_loss += loss.item()
                        # val_pred.extend(val_cos_sim.clone().cpu().tolist())
                        val_pred.extend(torch.diagonal(val_cos_sim).clone().cpu().tolist())
                        val_label.extend(val_batch['labels'].clone().cpu().tolist())

                    val_score = get_score(np.array(val_pred), np.array(val_label))
                    if best_val_score < val_score:
                        best_val_score = val_score
                        best_model = copy.deepcopy(model)
                        
                    # write train loss on Tensorbeard
                    writer.add_scalar("loss/train step", train_loss, step)
                    writer.add_scalar("spearman correlation", val_score, step)
                    
                print(f"\n\t cur_val_score / best_val_score : {val_score} / {best_val_score}")
                
                # early stopping
                train_early_stop.step(train_loss.item())
                valid_early_stop.step(val_score.item())
                if train_early_stop.is_stop() or valid_early_stop.is_stop() : break
    return best_model

def main():
    parser = argparse.ArgumentParser() # 원래 setting을 변형해서라도 train loss가 유의미하게 떨어지는지 확인
    # 원 논문에선 저게 best라니 똑같이 시도해보고 만약 아니면 우리가 이 parameter를 직접 조정해서 성능 향상시켜야지
    parser.add_argument('--model_name', default='bert-base-uncased', type=str)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--seq_max_length', default=32, type=int)
    parser.add_argument('--epochs', default=1, type=int)
    parser.add_argument('--lr', default=3e-5, type=float)
    parser.add_argument('--gpu', default=1, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--model_state_name', default='unsupervised_simcse_bert_base.pt', type=str)
    parser.add_argument('--temperature', default=0.05, type=float)
    parser.add_argument('--example_text', default=False, type=str)
    parser.add_argument('--time', default=datetime.now().strftime('%Y%m%d-%H:%M:%S'), type=str)

    args = parser.parse_args()
    setattr(args, 'device', f'cuda:{args.gpu}' if torch.cuda.is_available() and args.gpu >= 0 else 'cpu')

    print('[List of arguments]')
    for a in args.__dict__:
        print(f'{a}: {args.__dict__[a]}')

    model_name = args.model_name
    
    tokenizer = BertTokenizer.from_pretrained(model_name)
    bert_model = BertForUnsupervisedSimCSE(model_name)
    loss_fn = nn.CrossEntropyLoss()
    
    train_dataloader, validation_dataloader = train_setting(args, tokenizer)        
    best_model = unsupervised_train(args, train_dataloader, validation_dataloader, bert_model, loss_fn)
    model_save_fn(args, best_model)

if __name__ == '__main__':
    main()