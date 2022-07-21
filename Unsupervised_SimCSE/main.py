import argparse
import copy
from datetime import datetime
from scipy import stats
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader

from transformers import set_seed
from transformers import BertModel, BertConfig, BertTokenizer


class BertForUnsupervisedSimCSE(nn.Module):
    def __init__(self, bert_model_name, num_labels):
        super(BertForUnsupervisedSimCSE, self).__init__()

        self.hidden_size = BertConfig.from_pretrained(bert_model_name).hidden_size
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.linear = nn.Linear(in_features=self.hidden_size, out_features=num_labels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, batch):
        input_ids, attention_mask, token_type_ids = batch['input_ids'], batch['attention_mask'], batch['token_type_ids']
        _, pooler_out = self.bert(input_ids, attention_mask, token_type_ids, return_dict=False)
        linear_out = self.linear(pooler_out)
        sigmoid_out = self.sigmoid(linear_out)
        return sigmoid_out.squeeze()

class wikiDataset(Dataset):
    def __init__(self, example_text, args, tokenizer):
        self.args = args
        if example_text == True:
            data = [
            "chocolates are my favourite items.",
            "The fish dreamed of escaping the fishbowl and into the toilet where he saw his friend go.",
            "The person box was packed with jelly many dozens of months later.",
            "I love chocolates"
                ]
        else :
            dataset_df = pd.read_csv("Proj-Sentence-Representation/Unsupervised_SimCSE/wiki1m_for_simcse.txt", names=["text"], on_bad_lines='skip')
            dataset_df.dropna(inplace=True)
            data = list(dataset_df["text"].values)
        self.data = data
        self.len = len(data)
        self.tokenizer = tokenizer
        
    def __len__(self): 
        return self.len
    
    # 여기서 텐서를 출력해야지
    def __getitem__(self,idx) : 
        self.tokens = self.tokenizer(self.data[idx], truncation=True, padding="max_length", max_length=self.args.seq_max_length, return_tensors="pt")
        self.tokens['input_ids'] = self.tokens['input_ids'].squeeze()
        self.tokens['token_type_ids'] = self.tokens['token_type_ids'].squeeze()
        self.tokens['attention_mask'] = self.tokens['attention_mask'].squeeze()
        return self.tokens
    
def train_setting(args, model, loss_fn, tokenizer):
    seq_max_length = args.seq_max_length
    batch_size = args.batch_size
    example_text = args.example_text
    set_seed(args.seed)

    train_dataset = wikiDataset(example_text, args, tokenizer)
    validation_dataset = load_dataset('glue', 'stsb', split="validation")

    def encode_input(examples):
        encoded = tokenizer(examples['sentence1'], examples['sentence2'], max_length=seq_max_length, truncation=True, padding='max_length')
        encoded['input_ids'] = list(map(float, encoded['input_ids']))
        return encoded

    def format_output(example):
        return {'labels': example['label']}
 
    validation_dataset = validation_dataset.map(encode_input).map(format_output)
    validation_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'token_type_ids', 'labels'])
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size)

    unsupervised_train(args, train_dataloader, validation_dataloader, model, loss_fn)

def get_score(output, label):
    score = stats.pearsonr(output, label)[0]
    return score

def cosine_similarity(embeddings1, embeddings2, temperature=0.05):
    unit_embed1 = embeddings1 / torch.norm(embeddings1)
    unit_embed2 = embeddings2 / torch.norm(embeddings2)
    similarity = torch.matmul(unit_embed1.unsqueeze(-1), unit_embed2.unsqueeze(0)) / temperature
    return similarity

def unsupervised_train(args, train_dataloader, validation_dataloader, model, loss_fn):
    device = args.device
    learning_rate = args.lr
    epochs = args.epochs
    optimizer = Adam(model.parameters(), lr=learning_rate)

    model.to(device)
    loss_fn.to(device)
    best_val_score = 0
    best_model = None

    for t in range(epochs):
        model.train()
        for i, batch in enumerate(tqdm(train_dataloader)):
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            output1 = model(batch)
            output2 = model(batch)

            temperature = 0.05
            cos_sim = cosine_similarity(output1, output2, temperature)
            if cos_sim.dim() == 0 : 
                labels = torch.tensor(0).to(device)
            else : 
                labels = torch.arange(cos_sim.size(0)).to(device)
            
            loss = loss_fn(cos_sim, labels)
            loss.backward()
            optimizer.step()
            
        if i % 250 == 0 or i == len(train_dataloader) - 1:
            print(f'\n{i}th iteration (train loss): ({loss:.4})')

        model.eval()
        with torch.no_grad():
            val_loss = 0
            val_pred = []
            val_label = []

            for _, val_batch in enumerate(tqdm(validation_dataloader)):
                val_batch = {k: v.to(device) for k, v in val_batch.items()}
                predict = model(val_batch)
                
                if predict.dim() == 0 : 
                    predict = predict.unsqueeze(dim=0)
                loss = loss_fn(predict, val_batch['labels'])
                val_loss += loss.item()
                val_pred.extend(predict.clone().cpu().tolist())
                val_label.extend(val_batch['labels'].clone().cpu().tolist())

            val_score = get_score(np.array(val_pred), np.array(val_label))
            if best_val_score < val_score:
                best_val_score = val_score
                best_model = copy.deepcopy(model)

            print(f"\n{t}th epoch Validation loss / cur_val_score / best_val_score : {val_loss} / {val_score} / {best_val_score}")

    return best_model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='bert-base-cased', type=str)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--seq_max_length', default=512, type=int)
    parser.add_argument('--epochs', default=1, type=int)
    parser.add_argument('--lr', default=5e-5, type=float)
    parser.add_argument('--gpu', default=1, type=int)
    parser.add_argument('--seed', default=4885, type=int)
    parser.add_argument('--task', default="glue_sts", type=str)
    parser.add_argument('--example_text', default=False, type=str)

    args = parser.parse_args()
    setattr(args, 'device', f'cuda:{args.gpu}' if torch.cuda.is_available() and args.gpu >= 0 else 'cpu')
    setattr(args, 'time', datetime.now().strftime('%Y%m%d-%H:%M:%S'))

    print('[List of arguments]')
    for a in args.__dict__:
        print(f'{a}: {args.__dict__[a]}')

    model_name = args.model_name
    task = args.task    
    
    # Do downstream task
    if task == "glue_sts":
        data_labels_num = 1
        tokenizer = BertTokenizer.from_pretrained(model_name)
        bert_model = BertForUnsupervisedSimCSE(model_name, data_labels_num)
        loss_fn = nn.CrossEntropyLoss()
        train_setting(args, bert_model, loss_fn, tokenizer)
    else:
        print(f"There is no such task as {task}")

if __name__ == '__main__':
    main()