"""
The system trains BERT (or any other transformer model like RoBERTa, DistilBERT etc.) on the SNLI + MultiNLI (AllNLI) dataset
with softmax loss function. At every 1000 training steps, the model is evaluated on the
STS benchmark dataset

Usage:
python training_nli.py --seed 1234

OR
python training_nli.py --seed 1234 --model_name_or_path bert-base-uncased
"""
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
from transformers import BertModel, BertTokenizer, GPT2Model, GPT2Tokenizer, set_seed



# class OurSentenceDataset()  => inherits SentenceDataset from sentence_transformer
# class OurSentenceTransformer(SentenceTransformer)
# class OutLoss()  => probably inherits AdvCLSoftmaxLoss


def train(args, model, tokenizer, train_data, val_data):
    #need to add branches for large_model
    if args.train_way == "unsup":
        pass
    elif args.train_way == "joint_unsup":
        pass
    elif args.train_way == "sup-unsup":
        pass
    elif args.train_way == "joint-sup":
        pass
   

    #model = SentenceTransformer
    #model.fit()
    #copy.deepcopy(model),,, 이런식으로 안하고 그냥 save_path로 할까?
    #return best_model or model_save


def evaluate(args, model, tokenizer, test_data):
    pass


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
    
    parser.add_argument("--batch_size", type=int, default=96, help="Training mini-batch size")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of training epochs") # if continue_training is on, use default num_epochs
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="The learning rate")
    parser.add_argument("--evaluation_steps", type=int, default=1000, help="The steps between every evaluations")
    parser.add_argument("--max_seq_length", type=int, default=128, help="The max sequence length")
    parser.add_argument("--loss_rate_scheduler", type=int, default=0, help="The loss rate scheduler, default strategy 0 (i.e. do nothing, see AdvCLSoftmaxLoss for more details)")
    
    parser.add_argument("--concatenation_sent_max_square", action="store_true", help="Concat max-square features of two text representations when training classification")

    #adv 관련 옵션없음
    parser.add_argument("--add_cl", action="store_true", help="Use contrastive loss or not")
    parser.add_argument("--data_augmentation_strategy", type=str, default="adv", choices=["adv", "none", "meanmax", "shuffle", "cutoff", "shuffle-cutoff", "shuffle+cutoff", "shuffle_embeddings"], help="The data augmentation strategy in contrastive learning")
    
    # feature cutoff는 무조건 column자르는거여서 정작 이거 필요없는데
    parser.add_argument("--cutoff_direction", type=str, default=None, help="The direction of cutoff strategy, row, column or random")
    parser.add_argument("--cutoff_rate", type=float, default=None, help="The rate of cutoff strategy, in (0.0, 1.0)")
    parser.add_argument("--cl_loss_only", action="store_true", help="Ignore the main task loss (e.g. the CrossEntropy loss) and use the contrastive loss only")
    parser.add_argument("--cl_rate", type=float, default=0.01, help="The contrastive loss rate")
    parser.add_argument("--regularization_term_rate", type=float, default=0.0, help="The loss rate of regularization term for contrastive learning") 
    parser.add_argument("--cl_type", type=str, default="nt_xent", help="The contrastive loss type, nt_xent or cosine")
    parser.add_argument("--temperature", type=float, default=0.5, help="The temperature for contrastive loss")
    parser.add_argument("--mapping_to_small_space", type=int, default=None, help="Whether to mapping sentence representations to a low dimension space (similar to SimCLR) and give the dimension")
    
    # only for base model?
    parser.add_argument("--da_final_1", type=str, default=None, help="The final 4 data augmentation strategies for view1 (none, shuffle, token_cutoff, feature_cutoff, dropout)")
    parser.add_argument("--da_final_2", type=str, default=None, help="The final 4 data augmentation strategies for view2 (none, shuffle, token_cutoff, feature_cutoff, dropout)")
    parser.add_argument("--cutoff_rate_final_1", type=float, default=None, help="The final cutoff/dropout rate for view1")
    parser.add_argument("--cutoff_rate_final_2", type=float, default=None, help="The final cutoff/dropout rate for view2")
    parser.add_argument("--patience", default=None, type=int, help="The patience for early stop")

    ################# ADDED #################
    parser.add_argument('--gpu', default=0, type=int) 
    parser.add_argument('--train_way', default='unsup', type=str, choices=["unsup", "joint", "sup-unsup", "joint-unsup"])

    args = parser.parse_args()
    setattr(args, 'device', f'cuda:{args.gpu}' if torch.cuda.is_available() and args.gpu >= 0 else 'cpu')
    setattr(args, 'time', datetime.datetime.now().strftime('%Y%m%d-%H:%M:%S'))

    if args.train_way == "unsup":
        setattr(args, '--no_dropout', True)
        setattr(args, '--no_pair', True)
        
    elif args.train_way == "joint":
        setattr(args, '--no_dropout', False)
        
    elif args.train_way == "sup-unsup":
        # we have to have the sup model first
        setattr(args, '--no_dropout', False)
        setattr(args, '--continue_training', True)

    elif args.train_way == "joint-unsup":
        # we have to have the joint model first

        setattr(args, '--no_dropout', False)
        setattr(args, '--continue_training', True)

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
    bert_model_type_str = "base" if "base" in args.model_name_or_path else "large" # this line might not be needed
    adv_loss_rate_str = "" if args.adv_loss_rate == 1.0 else f"-rate{args.adv_loss_rate}"


    model_save_path = args.model_save_path

    if os.path.exists(model_save_path):
        if args.force_del:
            shutil.rmtree(model_save_path)
            os.mkdir(model_save_path)
        else:
            raise ValueError("Existing output_dir for save model")
    else:
        os.mkdir(model_save_path)
    

    with open(os.path.join(model_save_path, "args.json"), "w") as f:
        json.dump(args.__dict__, f, indent=4, ensure_ascii=False)
    with open(os.path.join(model_save_path, "command.txt"), "w") as f:
        CUDA_VISIBLE_DEVICES = os.environ.get("CUDA_VISIBLE_DEVICES")
        f.write(f"CUDA_VISIBLE_DEVICES={CUDA_VISIBLE_DEVICES} python3 {' '.join(sys.argv)}")


    if args.continue_training: # joint-unsup or sup-unsup
        if args.no_dropout:
            sentence_bert_config_path = os.path.join(args.model_name_or_path, "0_Transformer", "sentence_bert_config.json")
            sentence_bert_config_dict = json.load(open(sentence_bert_config_path, "r"))
            # change config
            new_config = copy.deepcopy(sentence_bert_config_dict)
            new_config["attention_probs_dropout_prob"] = 0.0
            new_config["hidden_dropout_prob"] = 0.0
            json.dump(new_config, open(sentence_bert_config_path, "w"), indent=2)
            # load model
            model = SentenceTransformer(args.model_name_or_path)
            # recover config
            json.dump(sentence_bert_config_dict, open(sentence_bert_config_path, "w"), indent=2)
        else:
            model = SentenceTransformer(args.model_name_or_path) #model_name_or_path = bert_base
    else: #only for unsup
        # Use Huggingface/transformers model (like BERT, RoBERTa) for mapping tokens to embeddings
        if args.no_dropout: #unsup일때만 ,bert dropout을 끄나??..
            word_embedding_model = models.Transformer(args.model_name_or_path, attention_probs_dropout_prob=0.0, hidden_dropout_prob=0.0)
        else:
            word_embedding_model = models.Transformer(args.model_name_or_path)
        # Apply mean pooling to get one fixed sized sentence vector
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), #just hidden size
                                       pooling_mode_mean_tokens=True,
                                       pooling_mode_cls_token=False,
                                       pooling_mode_max_tokens=False)
        
        # SentenceTransformer inherits from nn.Sequential
        model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    model.max_seq_length = args.max_seq_length

    # Below are codes for sup
    label2int = {"contradiction": 0, "entailment": 1, "neutral": 2}
    if args.train_data == "nli":
        # Read the AllNLI.tsv.gz file and create the training dataset
        logging.info("Read AllNLI train dataset")
        train_samples = []
        with gzip.open(nli_dataset_path, 'rt', encoding='utf8') as fIn:
            reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
            for row in reader:
                if row['split'] == 'train':
                    label_id = label2int[row['label']]
                    if args.no_pair:
                        assert args.cl_loss_only, "no pair texts only used when contrastive loss only"
                        train_samples.append(InputExample(texts=[row['sentence1']]))
                        train_samples.append(InputExample(texts=[row['sentence2']]))
                    else:
                        train_samples.append(InputExample(texts=[row['sentence1'], row['sentence2']], label=label_id))
    
    elif args.train_data == "stssick":
        # Read data/downstream/STS and data/downstream/SICK and create the training dataset
        logging.info("Read STS and SICK train dataset")
        train_samples = data_utils.load_datasets(datasets=["sts12", "sts13", "sts14", "sts15", "sts16", "stsb", "sickr"], need_label=False, use_all_unsupervised_texts=True, no_pair=args.no_pair)

    
    data_utils.save_samples(train_samples, os.path.join(model_save_path, "train_texts.txt"))
        
    train_dataset = SentencesDataset(train_samples, model=model)
    train_dataloader = DataLoader(train_dataset, shuffle=not args.no_shuffle, batch_size=train_batch_size)

    if args.add_cl:
        train_loss = losses.AdvCLSoftmaxLoss(model=model, sentence_embedding_dimension=model.get_sentence_embedding_dimension(), num_labels=len(label2int), concatenation_sent_max_square=args.concatenation_sent_max_square, use_contrastive_loss=args.add_cl, contrastive_loss_type=args.cl_type, contrastive_loss_rate=args.cl_rate, temperature=args.temperature, contrastive_loss_stop_grad=args.contrastive_loss_stop_grad, mapping_to_small_space=args.mapping_to_small_space, add_contrastive_predictor=args.add_contrastive_predictor, projection_hidden_dim=args.projection_hidden_dim, projection_use_batch_norm=args.projection_use_batch_norm, add_projection=args.add_projection, projection_norm_type=args.projection_norm_type, contrastive_loss_only=args.cl_loss_only, data_augmentation_strategy=args.data_augmentation_strategy, cutoff_direction=args.cutoff_direction, cutoff_rate=args.cutoff_rate, no_pair=args.no_pair, regularization_term_rate=args.regularization_term_rate, loss_rate_scheduler=args.loss_rate_scheduler, data_augmentation_strategy_final_1=args.da_final_1, data_augmentation_strategy_final_2=args.da_final_2, cutoff_rate_final_1=args.cutoff_rate_final_1, cutoff_rate_final_2=args.cutoff_rate_final_2)
    else:
        train_loss = losses.AdvCLSoftmaxLoss(model=model, sentence_embedding_dimension=model.get_sentence_embedding_dimension(), num_labels=len(label2int), concatenation_sent_max_square=args.concatenation_sent_max_square, normal_loss_stop_grad=args.normal_loss_stop_grad)

    # Read STSbenchmark dataset and use it as development set
    logging.info("Read STSbenchmark dev dataset")
    dev_samples = []
    with gzip.open(sts_dataset_path, 'rt', encoding='utf8') as fIn:
        reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
        for row in reader:
            if row['split'] == 'dev':
                score = float(row['score']) / 5.0 #Normalize score to range 0 ... 1
                dev_samples.append(InputExample(texts=[row['sentence1'], row['sentence2']], label=score))
    dev_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_samples, batch_size=train_batch_size, name='sts-dev', main_similarity=SimilarityFunction.COSINE)

    # Configure the training
    num_epochs = args.num_epochs

    model.num_steps_total = math.ceil(len(train_dataset) * num_epochs / train_batch_size)
    warmup_steps = math.ceil(len(train_dataset) * num_epochs / train_batch_size * 0.1) #10% of train data for warm-up
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
              early_stop_patience=args.patience)

    # Test on STS Benchmark
    test_samples = []
    with gzip.open(sts_dataset_path, 'rt', encoding='utf8') as fIn:
        reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
        for row in reader:
            if row['split'] == 'test':
                score = float(row['score']) / 5.0 #Normalize score to range 0 ... 1
                test_samples.append(InputExample(texts=[row['sentence1'], row['sentence2']], label=score))
    
    model = SentenceTransformer(model_save_path) # this becomes the best model
    test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_samples, batch_size=train_batch_size, name='sts-test', main_similarity=SimilarityFunction.COSINE)
    test_evaluator(model, output_path=model_save_path)

    
    # Test on unsupervised dataset (mainly STS related dataset)
    eval.eval_nli_unsup(model_save_path, main_similarity=SimilarityFunction.COSINE)
    eval.eval_nli_unsup(model_save_path, main_similarity=SimilarityFunction.COSINE, last2avg=True)
    


if __name__ == "__main__":
    main()


        