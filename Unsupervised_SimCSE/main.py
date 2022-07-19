import argparse
import copy
from datetime import datetime
from scipy import stats
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

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
        _, pooler_out = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, return_dict=False)
        linear_out = self.linear(pooler_out)
        sigmoid_out = self.sigmoid(linear_out)
        return sigmoid_out.squeeze()

# ???
# class wikiDataset(Dataset):
#     def __init__(self, example_test):
#         if example_test:
#             data = [
#             "chocolates are my favourite items.",
#             "The fish dreamed of escaping the fishbowl and into the toilet where he saw his friend go.",
#             "The person box was packed with jelly many dozens of months later.",
#             "white chocolates and dark chocolates are favourites for many people.",
#             "I love chocolates"
#                 ]
#         else :
#             dataset_df = pd.read_csv("Proj-Sentence-Representation/Unsupervised_SimCSE/wiki1m_for_simcse.txt", names=["text"], on_bad_lines='skip')
#             dataset_df.dropna(inplace=True).reset_index(inplace=True)
#             data = list(dataset_df["text"].values)
#         self.data = data
#     def __len__(self): return len(self.data)
#     def __getitem__(self,idx) : return self.data[idx]   

def wikiDataset(example_test):
    if example_test:
        data = [
            "chocolates are my favourite items.",
            "The fish dreamed of escaping the fishbowl and into the toilet where he saw his friend go.",
            "The person box was packed with jelly many dozens of months later.",
            "white chocolates and dark chocolates are favourites for many people.",
            "I love chocolates"
                ]
    else :
        dataset_df = pd.read_csv("Proj-Sentence-Representation/Unsupervised_SimCSE/wiki1m_for_simcse.txt", names=["text"], on_bad_lines='skip')
        dataset_df.dropna(inplace=True)
        data = list(dataset_df["text"].values)
    return data
  
def token_embedding(args, dataset, tokenizer):
    tokens = tokenizer(dataset, truncation=True, padding="max_length", max_length=args.seq_max_length, return_tensors="pt")

    return {
        'input_ids' : torch.cat([tokens['input_ids'],tokens['input_ids']],dim=1),
        'token_type_ids' : torch.cat([tokens['token_type_ids'],tokens['token_type_ids']],dim=1),
        'attention_mask' : torch.cat([tokens['attention_mask'],tokens['attention_mask']],dim=1)
            }
    
def glue_sts(args, model, loss_fn, auxloss_fn, tokenizer):
    seq_max_length = args.seq_max_length
    batch_size = args.batch_size
    set_seed(args.seed)

    train_dataset = wikiDataset(True)
    validation_dataset = load_dataset('glue', 'stsb', split="validation")

    def encode_input(examples):
        encoded = tokenizer(examples['sentence1'], examples['sentence2'], max_length=seq_max_length, truncation=True, padding='max_length')
        encoded['input_ids'] = list(map(float, encoded['input_ids']))
        return encoded

    def format_output(example):
        return {'labels': example['label']}

    # train_dataset = train_dataset.map(encode_input).map(format_output)
    # train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'token_type_ids', 'labels'])
 
    train_dataset = token_embedding(args, train_dataset, tokenizer)
    validation_dataset = validation_dataset.map(encode_input).map(format_output)
    validation_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'token_type_ids', 'labels'])
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size)

    unsupervised_train(args, train_dataloader, validation_dataloader, model, loss_fn, auxloss_fn)

def get_score(output, label):
    score = stats.pearsonr(output, label)[0]
    return score

def cosine_similarity(embeddings, temperature=0.05):
    unit_embed = embeddings / torch.norm(embeddings)
    similarity = torch.matmul(unit_embed, torch.transpose(unit_embed, 0, -1)) / temperature
    return similarity

def unsupervised_train(args, train_dataloader, validation_dataloader, model, loss_fn, auxloss_fn):
    device = args.device
    learning_rate = args.lr
    epochs = args.epochs
    optimizer = Adam(model.parameters(), lr=learning_rate)

    model.to(device)
    loss_fn.to(device)
    auxloss_fn.to(device)
    best_val_score = 0
    best_model = None

    for t in range(epochs):
        model.train()
        for i, batch in enumerate(tqdm(train_dataloader)):
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            predict = model(batch)

            temperature = 0.05
            lmbd = 0
            batch['input_ids'] = batch['input_ids'].type(torch.float)
            cos_sim = cosine_similarity(batch['input_ids'], temperature)
            predict_labels = torch.diag(cos_sim)/torch.sum(cos_sim,-1)
            loss = loss_fn(predict_labels, batch['labels'])
            auxloss = auxloss_fn(predict, batch['labels'])
            total_loss = loss + auxloss * lmbd
            total_loss.backward()
            optimizer.step()
            
        if i % 1000 == 0 or i == len(train_dataloader) - 1:
            print(f'\n{i}th iteration (train loss): ({total_loss:.4})')

        model.eval()
        with torch.no_grad():
            val_loss = 0
            val_pred = []
            val_label = []

            for _, val_batch in enumerate(validation_dataloader):
                val_batch = {k: v.to(device) for k, v in val_batch.items()}
                predict = model(val_batch)
                loss = loss_fn(predict, val_batch['labels'])
                
                val_batch['input_ids'] = val_batch['input_ids'].type(torch.float)
                val_cos_sim = cosine_similarity(val_batch['input_ids'], temperature)
                val_predict_labels = torch.diag(val_cos_sim)/torch.sum(val_cos_sim,-1)
                loss = loss_fn(val_predict_labels, val_batch['labels'])
                auxloss = auxloss_fn(predict, batch['labels'])
                total_loss = loss + auxloss * lmbd

                val_loss += total_loss.item()
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
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--seq_max_length', default=512, type=int)
    parser.add_argument('--epochs', default=1, type=int)
    parser.add_argument('--lr', default=5e-5, type=float)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--seed', default=4885, type=int)
    parser.add_argument('--task', default="glue_sts", type=str)
    parser.add_argument('--example_test', default="False", type=str)

    args = parser.parse_args()
    setattr(args, 'device', f'cuda:{args.gpu}' if torch.cuda.is_available() and args.gpu >= 0 else 'cpu')
    setattr(args, 'time', datetime.now().strftime('%Y%m%d-%H:%M:%S'))

    print('[List of arguments]')
    for a in args.__dict__:
        print(f'{a}: {args.__dict__[a]}')

    model_name = args.model_name
    task = args.task
    args.example_test=True
    
    # Do downstream task
    if task == "glue_sts":
        data_labels_num = 1
        tokenizer = BertTokenizer.from_pretrained(model_name)
        bert_model = BertForUnsupervisedSimCSE(model_name, data_labels_num)
        loss_fn = nn.CrossEntropyLoss()
        auxloss_fn = nn.MSELoss()
        glue_sts(args, bert_model, loss_fn, auxloss_fn, tokenizer)
    else:
        print(f"There is no such task as {task}")

if __name__ == '__main__':
    main()
