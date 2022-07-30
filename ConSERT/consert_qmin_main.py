from ast import arg
from torch.utils.data import DataLoader
import math
from sentence_transformers import models, losses
from sentence_transformers import SentencesDataset, LoggingHandler, SentenceTransformer, util, InputExample
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator, SimilarityFunction
import logging
from datetime import datetime
import sys
import os
import json
import copy
import gzip
import csv
import random
import torch
import numpy as np
import argparse
import shutil
import data_utils, eval
from transformers import set_seed




def set_seed(seed: int, for_multi_gpu: bool):
    """
    Added script to set random seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if for_multi_gpu:
        torch.cuda.manual_seed_all(seed)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no_pair", action="store_true", help="If provided, do not pair two training texts") #일단 냅두자
    parser.add_argument("--seed", type=int, required=True, help="Random seed for reproducing experimental results")
    parser.add_argument("--model_name_or_path", type=str, default="bert-base-uncased", help="The model path or model name of pre-trained model")
    parser.add_argument("--model_save_path", type=str, default=None, help="Custom output dir")
    parser.add_argument("--force_del", action="store_true", help="Delete the existing save_path and do not report an error")
    
    parser.add_argument("--no_dropout", action="store_true", help="only turn on when cutoff used")
    parser.add_argument("--batch_size", type=int, default=96, help="Training mini-batch size")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs") # if consecutive is on, num_epochs = 1
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="The learning rate")
    parser.add_argument("--evaluation_steps", type=int, default=200, help="The steps between every evaluations")
    parser.add_argument("--max_seq_length", type=int, default=128, help="The max sequence length")
    
    parser.add_argument("--cutoff_rate", type=float, default=0.2, help="The rate of cutoff strategy, in (0.0, 1.0)")
    parser.add_argument("--cl_rate", type=float, default=0.15, help="The contrastive loss rate")
    parser.add_argument("--temperature", type=float, default=0.5, help="The temperature for contrastive loss")
    
    # there is a priority in this order -> (none, shuffle, token_cutoff, feature_cutoff )
    parser.add_argument("--data_aug_strategy1", type=str, default=None, help="4 data augmentation strategies for view2 (none, shuffle, token_cutoff, feature_cutoff)")
    parser.add_argument("--data_aug_strategy2", type=str, default=None, help="4 data augmentation strategies for view2 (none, shuffle, token_cutoff, feature_cutoff)")

    parser.add_argument("--patience", default=None, type=int, help="The patience for early stop")

    # should've used apex for batch size
    parser.add_argument("--use_apex_amp", action="store_false", help="Use apex amp or not")
    parser.add_argument("--apex_amp_opt_level", type=str, default="O1", help="The opt_level argument in apex amp")

    ################# ADDED #################
    parser.add_argument('--gpu', default=0, type=int) 
    parser.add_argument('--train_way', default='unsup', type=str, choices=["unsup", "joint", "sup-unsup", "joint-unsup", "sup"])

    args = parser.parse_args()
    setattr(args, 'device', f'cuda:{args.gpu}' if torch.cuda.is_available() and args.gpu >= 0 else 'cpu')
    #setattr(args, 'time', datetime.datetime.now().strftime('%Y%m%d-%H:%M:%S'))

    if args.train_way in ["unsup", "joint-unsup", "sup-unsup"]:
        setattr(args, 'no_pair', True)
        setattr(args, 'no_dropout', True)

    elif args.train_way in ["joint", "sup"]:
        # we have to have the joint model first
        setattr(args, "num_epochs", 1)
    ################# ADDED #################

    logging.info(f"Training arguments: {args.__dict__}")
    set_seed(args.seed, for_multi_gpu=False)

    ############# modified ##############
    # Check if dataset exsist. If not, download and extract  it
    nli_dataset_path = 'datasets/AllNLI.tsv.gz'  # for supervised
    sts_dataset_path = 'datasets/stsbenchmark.tsv.gz'
    if not os.path.exists(nli_dataset_path):
        util.http_get('https://sbert.net/datasets/AllNLI.tsv.gz', nli_dataset_path)
    if not os.path.exists(sts_dataset_path):
        util.http_get('https://sbert.net/datasets/stsbenchmark.tsv.gz', sts_dataset_path)

    # Read the dataset
    train_batch_size = args.batch_size
    model_save_path = args.model_save_path

    if os.path.exists(model_save_path):
        if args.force_del:
            shutil.rmtree(model_save_path)
            os.mkdir(model_save_path)
        else:
            raise ValueError("Existing output_dir for save model")
    else:
        os.mkdir(model_save_path)

    # define model
    if args.train_way == "sup-unsup":
        if not os.path.exists(args.model_name_or_path):
            print("generate sup model fisrt")
            return
        model = SentenceTransformer(args.model_name_or_path, device=args.device)
    elif args.train_way == "joint-unsup":
        if not os.path.exists(args.model_name_or_path):
            print("generate joint model fisrt")
            return
        model = SentenceTransformer(args.model_name_or_path, device=args.device)
    elif args.train_way in ["unsup" ,"joint", "sup"]:
        if args.no_dropout:
            word_embedding_model = models.Transformer(args.model_name_or_path, attention_probs_dropout_prob=0.0, hidden_dropout_prob=0.0)
        else:
            word_embedding_model = models.Transformer(args.model_name_or_path)
        # Apply mean pooling to get one fixed sized sentence vector
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), #just hidden size
                                       pooling_mode_mean_tokens=True,
                                       pooling_mode_cls_token=False,
                                       pooling_mode_max_tokens=False)
        # SentenceTransformer inherits from nn.Sequential
        model = SentenceTransformer(modules=[word_embedding_model, pooling_model], device=args.device)

    model.max_seq_length = args.max_seq_length


    #load_data
    label2int = {"contradiction": 0, "entailment": 1, "neutral": 2}
    if args.train_way in ["joint" ,"sup"]:
        # Read the AllNLI.tsv.gz file and create the training dataset
        logging.info("Read AllNLI train dataset")
        train_samples = []
        with gzip.open(nli_dataset_path, 'rt', encoding='utf8') as fIn:
            reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
            for row in reader:
                if row['split'] == 'train':
                    label_id = label2int[row['label']]
                    if args.no_pair:
                        train_samples.append(InputExample(texts=[row['sentence1']]))
                        train_samples.append(InputExample(texts=[row['sentence2']]))
                    else:
                        train_samples.append(InputExample(texts=[row['sentence1'], row['sentence2']], label=label_id))
    elif args.train_way in ["unsup" ,"joint", "sup-unsup", "joint-unsup"]:
        # Read data/downstream/STS and data/downstream/SICK and create the training dataset
        logging.info("Read STS and SICK train dataset")
        train_samples = data_utils.load_datasets(datasets=["sts12", "sts13", "sts14", "sts15", "sts16", "stsb", "sickr"], need_label=False, use_all_unsupervised_texts=True, no_pair=args.no_pair)

    data_utils.save_samples(train_samples, os.path.join(model_save_path, "train_texts.txt"))

    train_dataset = SentencesDataset(train_samples, model=model)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=train_batch_size, num_workers=8)

    # initiate loss model
    if args.train_way == "sup":
        train_loss = losses.MyLoss(args, model=model, sentence_embedding_dimension=model.get_sentence_embedding_dimension(),
         num_labels=len(label2int))
    else:
        train_loss = losses.MyLoss(args, model=model, sentence_embedding_dimension=model.get_sentence_embedding_dimension(), 
        num_labels=len(label2int), data_aug_strategy1 = args.data_aug_strategy1, data_aug_strategy2 = args.data_aug_strategy2, 
        contrastive_loss_rate=args.cl_rate, temperature=args.temperature)
        

    # Read STSbenchmark dataset and use it as development set
    logging.info("Read STSbenchmark dev dataset")
    dev_samples = []
    with gzip.open(sts_dataset_path, 'rt', encoding='utf8') as fIn:
        reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
        for row in reader:
            if row['split'] == 'dev':
                score = float(row['score']) / 5.0 #Normalize score to range 0 ... 1
                dev_samples.append(InputExample(texts=[row['sentence1'], row['sentence2']], label=score))
    dev_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_samples, batch_size=train_batch_size, name='sts-dev',
                    main_similarity=SimilarityFunction.COSINE)

    # Configure the training
    num_epochs = args.num_epochs

    model.num_steps_total = math.ceil(len(train_dataset) * num_epochs / train_batch_size)
    warmup_steps = math.ceil(len(train_dataset) * num_epochs / train_batch_size * 0.1) # 10% of train data for warm-up
    logging.info("Warmup-steps: {}".format(warmup_steps))

    # Train the model
    model.fit(train_objectives=[(train_dataloader, train_loss)],
              evaluator=dev_evaluator,
              epochs=num_epochs,
              optimizer_params={'lr': args.learning_rate, 'eps': 1e-6, 'correct_bias': False},
              evaluation_steps=args.evaluation_steps,
              warmup_steps=warmup_steps,
              output_path=model_save_path,
              # we could make use of apex
              use_apex_amp=args.use_apex_amp,
              apex_amp_opt_level = args.apex_amp_opt_level,
              early_stop_patience=args.patience)

    # Test
    eval.eval_nli_unsup(model_save_path, main_similarity=SimilarityFunction.COSINE,last2avg=True)


    


if __name__ == "__main__":
    main()


        