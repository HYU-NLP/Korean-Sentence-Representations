import argparse
import os
from datetime import datetime

import pandas as pd
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import set_seed, BertTokenizer, BertModel, BertConfig


# TODO
#  1 - For supervised SimCSE, we train our models for 3 epochs, evaluate the model every 250 training steps on the development set of STS-B and keep the best checkpoint for the final evaluation on test sets.
#  2 - STS tasks and Transfer tasks

class BertForSupervisedSimCse(nn.Module):
    def __init__(self, model_name, temperature, add_mlp_layer):
        super(BertForSupervisedSimCse, self).__init__()

        self.config = BertConfig.from_pretrained(model_name)  # uses default dropout rate = 0.1
        self.bert = BertModel.from_pretrained('bert-base-uncased')

        self.cosine_similarity = nn.CosineSimilarity(dim=-1)
        self.temperature = temperature
        self.add_mlp_layer = add_mlp_layer

        if self.add_mlp_layer:  # paper says, performance is same, with or without mlp layer
            self.mlp_layer = nn.Sequential(
                nn.Linear(self.config.hidden_size, self.config.hidden_size),
                nn.Tanh()
            )

    def forward(self, input_ids, attention_mask, token_type_ids, **kwargs):
        batch_size = input_ids.shape[0]

        # flat, (batch_size * 3, seq_len)
        input_ids = input_ids.view((-1, input_ids.shape[-1]))
        attention_mask = attention_mask.view((-1, input_ids.shape[-1]))
        token_type_ids = token_type_ids.view((-1, input_ids.shape[-1]))

        # encode, (batch_size * 3, hidden_size)
        _, pooler_out = self.bert(input_ids, attention_mask, token_type_ids, return_dict=False)

        if self.add_mlp_layer:
            pooler_out = self.mlp_layer(pooler_out)

        # revert flat, (batch_size, 3, hidden_size)
        pooler_out = pooler_out.view((batch_size, 3, pooler_out.shape[-1]))

        # cos sim
        z1, z2, z3 = pooler_out[:, 0], pooler_out[:, 1], pooler_out[:, 2]
        z1_z2_cos = self.cosine_similarity(z1.unsqueeze(1), z2.unsqueeze(0)) / self.temperature
        z1_z3_cos = self.cosine_similarity(z1.unsqueeze(1), z3.unsqueeze(0)) / self.temperature
        cos_sim = torch.cat([z1_z2_cos, z1_z3_cos], dim=1)
        return cos_sim


class SupervisedSimCseDataset(Dataset):
    def __init__(self, data_frame, tokenizer, max_length, column_names=None):
        if column_names is None:
            column_names = ['sent0', 'sent1', 'hard_neg']

        self.len = len(data_frame)
        self.data_frame = data_frame
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.column_names = column_names

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        premise = self.tokenizer(self.data_frame[self.column_names[0]][idx], padding='max_length', max_length=self.max_length, truncation=True, return_tensors="pt")
        hypothesis = self.tokenizer(self.data_frame[self.column_names[1]][idx], padding='max_length', max_length=self.max_length, truncation=True, return_tensors="pt")
        hard_neg = self.tokenizer(self.data_frame[self.column_names[2]][idx], padding='max_length', max_length=self.max_length, truncation=True, return_tensors="pt")

        return {
            'input_ids': torch.stack((premise['input_ids'].squeeze(), hypothesis['input_ids'].squeeze(), hard_neg['input_ids'].squeeze())),
            'attention_mask': torch.stack((premise['attention_mask'].squeeze(), hypothesis['attention_mask'].squeeze(), hard_neg['attention_mask'].squeeze())),
            'token_type_ids': torch.stack((premise['token_type_ids'].squeeze(), hypothesis['token_type_ids'].squeeze(), hard_neg['token_type_ids'].squeeze())),
        }


def save_model_config(path, model_name, model_state_dict, model_config_dict):
    dirname = os.path.dirname(path)
    os.makedirs(dirname, exist_ok=True)
    torch.save({
        'model_name': model_name,
        'model_state_dict': model_state_dict,
        'model_config_dict': model_config_dict
    }, path)


def pretrain_model(epochs, device, dataloader, model, loss_fn, optimizer, _, model_save_fn):
    model.to(device)
    loss_fn.to(device)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0
        for i, batch in enumerate(tqdm(dataloader)):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels = torch.arange(batch['input_ids'].shape[0]).long().to(device)

            optimizer.zero_grad()
            predict = model(input_ids, attention_mask, token_type_ids)
            loss = loss_fn(predict, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            if i % 1000 == 0 or i == len(dataloader) - 1:
                model_save_fn(model)
                print(f'\n{i}th iteration (train loss): ({train_loss:.4})')
                train_loss = 0


def main():
    # Parser --
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='bert-base-uncased', type=str)  # simcse used bert-base-uncased and roberta-base-cased
    parser.add_argument('--batch_size', default=24, type=int)
    parser.add_argument('--seq_max_length', default=128, type=int)
    parser.add_argument('--epochs', default=1, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--seed', default=4885, type=int)
    parser.add_argument('--task', default='sup_simcse', type=str)
    parser.add_argument('--model_state_name', default='sup_simcse_bert.pt', type=str)  # for downstream tasks, if 'model_state_name' exist, 'model_name' will be ignored
    parser.add_argument('--pretrain_dataset', default='nli_for_simcse.csv', type=str)  # for pretrained task
    parser.add_argument('--temperature', default=0.05, type=float)
    parser.add_argument('--add_mlp_layer', default=True, type=bool)

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
        add_mlp_layer = args.add_mlp_layer

        df_train = pd.read_csv(f'dataset/{pretrain_dataset}', sep=',')

        tokenizer = BertTokenizer.from_pretrained(model_name)
        train_dataset = SupervisedSimCseDataset(df_train, tokenizer, seq_max_length)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
        model = BertForSupervisedSimCse(model_name, temperature, add_mlp_layer)
        optimizer = AdamW(model.parameters(), lr=learning_rate)  # AdamW is used in SimCSE original project
        loss_fn = nn.CrossEntropyLoss()

        def model_save_fn(pretrained_model):
            save_model_config(f'checkpoint/{model_state_name}', model_name, pretrained_model.bert.state_dict(), pretrained_model.config.to_dict())

        pretrain_model(epochs, device, train_dataloader, model, loss_fn, optimizer, None, model_save_fn)

    else:
        ValueError(f"Unknown task: {task}")


if __name__ == "__main__":
    main()
