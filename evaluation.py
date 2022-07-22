import argparse
import logging
import sys
from datetime import datetime

import torch
from prettytable import PrettyTable
from transformers import set_seed, BertConfig, BertModel, BertTokenizer

sys.path.insert(0, './SentEval')  # To SentEval
import senteval

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def load_model_config(path):
    config = torch.load(path, map_location='cpu')
    return config['model_name'], config['model_state_dict'], config['model_config_dict']


def print_table(task_names, scores):
    tb = PrettyTable()
    tb.field_names = task_names
    tb.add_row(scores)
    print(tb)


def print_results(results):
    task_names = []
    scores = []
    for task in ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark', 'SICKRelatedness']:
        task_names.append(task)
        if task in results:
            if task in ['STS12', 'STS13', 'STS14', 'STS15', 'STS16']:
                scores.append("%.2f" % (results[task]['all']['spearman']['all'] * 100))
            else:
                scores.append("%.2f" % (results[task]['test']['spearman'].correlation * 100))
        else:
            scores.append("0.00")
    task_names.append("Avg.")
    scores.append("%.2f" % (sum([float(score) for score in scores]) / len(scores)))
    print_table(task_names, scores)
    task_names = []
    scores = []
    for task in ['MR', 'CR', 'SUBJ', 'MPQA', 'SST2', 'TREC', 'MRPC']:
        task_names.append(task)
        if task in results:
            scores.append("%.2f" % (results[task]['acc']))
        else:
            scores.append("0.00")
    task_names.append("Avg.")
    scores.append("%.2f" % (sum([float(score) for score in scores]) / len(scores)))
    print_table(task_names, scores)


def run_sent_eval(sent_eval_params, tasks, model, tokenizer, device):
    def prepare(params, samples):
        return

    def batcher(params, batch):
        model.to(device)

        # Handle rare token encoding issues in the dataset, copied from simcse
        if len(batch) >= 1 and len(batch[0]) >= 1 and isinstance(batch[0][0], bytes):
            batch = [[word.decode('utf-8') for word in s] for s in batch]

        batch = [' '.join(s) for s in batch]
        batch = tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
        batch = {k: v.to(device) for k, v in batch.items()}

        with torch.no_grad():
            outputs = model(**batch, output_hidden_states=True, return_dict=True)
            last_hidden = outputs.last_hidden_state
            pooler_output = outputs.pooler_output
            hidden_states = outputs.hidden_states

        return pooler_output.cpu()

    results = {}
    se = senteval.engine.SE(sent_eval_params, batcher, prepare)
    for task in tasks:
        result = se.eval(task)
        results[task] = result

    print_results(results)


def main():
    # Parser --
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--seed', default=4885, type=int)

    parser.add_argument('--model_name', default='princeton-nlp/sup-simcse-bert-base-uncased', type=str)  # Should be bert base model, including SimCSE models
    parser.add_argument('--model_path', default='', type=str)  # if exist, model_name will be ignored

    args = parser.parse_known_args()[0]
    setattr(args, 'device', f'cuda:{args.gpu}' if torch.cuda.is_available() and args.gpu >= 0 else 'cpu')
    setattr(args, 'time', datetime.now().strftime('%Y%m%d-%H:%M:%S'))

    logger.info('[List of arguments]')
    for a in args.__dict__:
        logger.info(f'{a}: {args.__dict__[a]}')

    # Device & Seed --
    device = args.device
    set_seed(args.seed)

    # Hyper parameter --
    model_name = args.model_name
    model_path = args.model_path

    sent_eval_params = {  # SimCSE Test mode params
        'task_path': './SentEval/data',
        'classifier': {'nhid': 0, 'optim': 'adam', 'batch_size': 64, 'tenacity': 5, 'epoch_size': 4},
        'usepytorch': True,
        'kfold': 10
    }

    tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark', 'SICKRelatedness']  # TODO Should make it as argument

    # Prepare tokenizer, dataset (+ dataloader), model, loss function, optimizer, etc --
    if model_path:
        model_name, model_state_dict, model_config_dict = load_model_config(model_path)
        config = BertConfig.from_dict(model_config_dict)
        model = BertModel.fromConfig(config)
        model.load_state_dict(model_state_dict)
        tokenizer = BertTokenizer.from_pretrained(model_name)
    else:
        model = BertModel.from_pretrained(model_name)
        tokenizer = BertTokenizer.from_pretrained(model_name)

    run_sent_eval(sent_eval_params, tasks, model, tokenizer, device)


if __name__ == "__main__":
    main()
