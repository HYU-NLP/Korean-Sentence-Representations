import argparse
import argparse
import csv
import logging
import math
import random
import time
from tqdm import tqdm

import numpy as np
import torch
from sentence_transformers import LoggingHandler, InputExample
from sentence_transformers import models
from sentence_transformers.models import Transformer
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator, SimilarityFunction
from torch.utils.data import DataLoader

from loss import Loss
from modules import SentencesDataset, SentenceTransformer
from kobert_tokenizer import KoBERTTokenizer
from datasets import load_dataset


start_time = time.time()

PRETRAINED_MODELS = ['bert-base-nli-cls-token',
                     'bert-base-nli-mean-tokens',
                     'bert-large-nli-cls-token',
                     'bert-large-nli-mean-tokens',
                     'roberta-base-nli-cls-token',
                     'roberta-base-nli-mean-tokens',
                     'roberta-large-nli-cls-token',
                     'roberta-large-nli-mean-tokens']
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--model_name', default='klue/bert-base', type=str) # MBERT : 'bert-base-multilingual-uncased', KoBERT : 'skt/kobert-base-v1'
    parser.add_argument('--pooling', default='cls', type=str) # BERT_T pooling
    parser.add_argument('--pooling2', default='mean', type=str) # BERT_F pooling
    parser.add_argument('--eval_step', default=50, type=int)
    parser.add_argument('--num_epochs', default=1, type=int)
    parser.add_argument('--lr', default=5e-5, type=float)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--T', default=1e-2, type=float)
    parser.add_argument('--eps', default=0.1, type=float)
    parser.add_argument('--lmin', default=0, type=int)
    parser.add_argument('--lmax', default=-1, type=int)
    parser.add_argument('--lamb', default=0.1, type=float)
    parser.add_argument('--es', default=20, type=int) # early stopping patience
    parser.add_argument('--weight_decay', default=0, type=float)
    parser.add_argument('--training', default=True, action='store_true')
    parser.add_argument('--freeze', default=True, action='store_true')
    parser.add_argument('--clone', default=True, action='store_true')
    parser.add_argument('--disable_tqdm', default=False, action='store_true')
    parser.add_argument('--obj', default='SG-OPT', type=str)
    parser.add_argument('--device', default='cuda:1', type=str)
    parser.add_argument('--max_seq_length', default=128, type=int)
    parser.add_argument("--train_data", default ='news',type=str, help="dataset for training")
    parser.add_argument("--dev_test_data", default ='kakao', type=str, help="dev_test pair for dev and evaluation")


    args = parser.parse_args()
    for a in args.__dict__:
        print(f'{a}: {args.__dict__[a]}')

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.random.manual_seed(args.seed)

    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO,
                        handlers=[LoggingHandler()])

    
    model_args = {'output_hidden_states': True, 'output_attentions': True}
    word_embedding_model = Transformer(args.model_name, model_args=model_args, max_seq_length=args.max_seq_length)
    #if args.model_name == 'skt/kobert-base-v1':
        #word_embedding_model.tokenizer = KoBERTTokenizer.from_pretrained(args.model_name)
   # else:
   #     word_embedding_model.tokenizer =

    pooling_model = models.Pooling(
        word_embedding_model.get_word_embedding_dimension(),
        pooling_mode_mean_tokens=args.pooling == 'mean' or args.pooling not in ['cls', 'max'],
        pooling_mode_cls_token=args.pooling == 'cls',
        pooling_mode_max_tokens=args.pooling == 'max')

    modules = [word_embedding_model, pooling_model]
    model = SentenceTransformer(modules=modules, name=args.model_name, device=args.device)

    logging.info(f"Read datasets")

    # dataset path
    train_dataset_path = None
    if args.train_data == "news":
        train_dataset_path = 'data/korean_news_data_1m.txt'
    elif args.train_data == "wiki":
        train_dataset_path = 'data/wiki_corpus_len15_normalized_1m.txt'
    assert args.train_data in ["news", "wiki"]

    dev_dataset_path = None
    test_dataset_path = None
    if args.dev_test_data == "kakao":
        dev_dataset_path = 'data/KorSTS/sts-dev.tsv'
        test_dataset_path = 'data/KorSTS/sts-test.tsv'
    elif args.dev_test_data == "klue":
        pass # gonna fetch from huggingface
    assert args.dev_test_data in ["kakao", "klue"]


    # Read train dataset
    logging.info(f"Read {args.train_data} file")
    train_samples = []
    with open(train_dataset_path, "r") as f:
        while True: # while true is not the appropriate coding style
            line = f.readline()
            if not line:
                break
            train_samples.append(InputExample(texts=[line.strip()]))

    train_dataset = SentencesDataset(train_samples, model=model)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size)
    train_loss = Loss(model, args)



    # Read STSbenchmark dataset and use it as development set
    if args.dev_test_data == "kakao":
        logging.info(f"Read {args.dev_test_data} dev dataset")
        dev_samples = []
        with open(dev_dataset_path, 'rt', encoding='utf8') as f:
            reader = csv.DictReader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
            for row in reader:
                score = float(row['score']) / 5.0 #Normalize score to range 0 ... 1
                dev_samples.append(InputExample(texts=[row['sentence1'], row['sentence2']], label=score))
        dev_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_samples, batch_size=args.batch_size, name='kakao-dev',
                        main_similarity=SimilarityFunction.COSINE)
    elif args.dev_test_data == "klue":
        def format_label(batch):
            return {'score': batch['labels']['label']}
        logging.info(f"Read {args.dev_test_data} dev dataset")
        dev_set = load_dataset('klue', 'sts', split='train[90%:]').map(format_label)
        dev_samples=[]
        for dev in dev_set:
            dev['score'] = float(dev['score'])/5.0
            dev_samples.append(InputExample(texts=[dev['sentence1'], dev['sentence2']], label=dev['score']))
        dev_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_samples, batch_size=args.batch_size, name='klue-dev',
                        main_similarity=SimilarityFunction.COSINE)

    warmup_steps = math.ceil(len(train_dataset) * args.num_epochs / args.batch_size * 0.1) #10% of train data for warm-up
    logging.info("Warmup-steps: {}".format(warmup_steps))

    args_string = args.model_name + '-' + str(args.seed) + '-' + str(args.lr) + '-' + str(args.max_seq_length) + '-' +  str(args.es)
    logging.info(f'args_string: {args_string}')
    model_save_path = f'output/{args_string}'

    model.fit(train_objectives=[(train_dataloader, train_loss)],
            dev_evaluator=dev_evaluator,
            test_evaluator= None,
            epochs=args.num_epochs,
            optimizer_params={'lr': args.lr, 'correct_bias': True, 'weight_decay': args.weight_decay, 'betas': (0.9, 0.9)},
            evaluation_steps=args.eval_step,
            warmup_steps=warmup_steps,
            output_path=model_save_path,
            early_stopping_limit=args.es,
            disable_tqdm=args.disable_tqdm)

    logging.info('Training finished.')

    logging.info("Read test data")
    if test_dataset_path == None: # klue # hf load dataset
        def format_label(batch):
            return {'score': batch['labels']['label']}
        test_set = load_dataset('klue', 'sts', split='validation').map(format_label)
        test_samples=[]
        for test in test_set:
            test['score'] = float(test['score'])/5.0
            test_samples.append(InputExample(texts=[test['sentence1'], test['sentence2']], label=test['score']))
        logging.info(f"Loaded examples from klue_sts_test dataset, total {len(test_samples)} examples")
    else: # kakao
        test_samples=[]
        with open(test_dataset_path, 'rt', encoding='utf8') as f:
            reader = csv.DictReader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
            for row in reader:
                score = float(row['score']) / 5.0 #Normalize score to range 0 ... 1
                test_samples.append(InputExample(texts=[row['sentence1'], row['sentence2']], label=score))

        logging.info(f"Loaded examples from KorSTS_test dataset, total {len(test_samples)} examples")
        
    #test batch size differs from the one in training,,, doesn't it?
    test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_samples, batch_size=args.batch_size, name=f"{args.dev_test_data}",
                        main_similarity=SimilarityFunction.COSINE)
    
    dev_score = dev_evaluator(model, output_path=model_save_path)
    test_score = test_evaluator(model, output_path=model_save_path)
    print(f'dev_score : {dev_score}')
    print(f'test_score : {test_score}')



if __name__ == "__main__":
    main()