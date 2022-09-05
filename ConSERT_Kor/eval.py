import os
import json
import logging
import argparse
import csv
from sentence_transformers import SentenceTransformer, InputExample, LoggingHandler
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator, SimilarityFunction
from datasets import load_dataset

logging.basicConfig(format='%(asctime)s - %(filename)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="The saved model path for evaluation")
    parser.add_argument("--main_similarity", type=str, choices=["cosine", "euclidean", "manhattan", "dot_product"], default=None, help="The main similarity type")
    parser.add_argument("--last2avg", action="store_true", help="Use last 2 layer average or not")
    parser.add_argument("--firstlastavg", action="store_true", help="Use first and last layers average or not")
    args = parser.parse_args()
    return args

def load_model(model_path: str, last2avg: bool = False, firstlastavg: bool = False, device: str =None):
    model = SentenceTransformer(model_path, device=device)
    if last2avg:
        model[1].pooling_mode_mean_tokens = False
        model[1].pooling_mode_mean_last_2_tokens = True
        model[0].auto_model.config.output_hidden_states = True
    if firstlastavg:
        model[1].pooling_mode_mean_tokens = False
        model[1].pooling_mode_mean_first_last_tokens = True
        model[0].auto_model.config.output_hidden_states = True
    logging.info("Model successfully loaded")
    return model

def eval_KorSTS(model, batch_size=16, output_path="./", main_similarity=None, test_path: str=None):
    logging.info("Evaluation on STSBenchmark dataset")
    if test_path == None: # klue # hf load dataset
        def format_label(batch):
            return {'score': batch['labels']['label']}
        test_set = load_dataset('klue', 'sts', split='validation').map(format_label)
        test_samples=[]
        for test in test_set:
            test['score'] = float(test['score'])/5.0
            test_samples.append(InputExample(texts=[test['sentence1'], test['sentence2']], label=test['score']))
        logging.info(f"Loaded examples from klue_sts_test dataset, total {len(test_samples)} examples")
        evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_samples, batch_size=batch_size, name='klue-test',
                        main_similarity=SimilarityFunction.COSINE)
        best_result = evaluator(model, output_path=output_path)
        logging.info(f"Results on klue_sts_test: {best_result:.6f}")

    else: # kakao
        test_samples=[]
        with open(test_path, 'rt', encoding='utf8') as f:
            reader = csv.DictReader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
            for row in reader:
                score = float(row['score']) / 5.0 #Normalize score to range 0 ... 1
                test_samples.append(InputExample(texts=[row['sentence1'], row['sentence2']], label=score))

        logging.info(f"Loaded examples from KorSTS_test dataset, total {len(test_samples)} examples")
        evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_samples, batch_size=batch_size, name="KorSTS_test", main_similarity=main_similarity)
        best_result = evaluator(model, output_path=output_path)
        logging.info(f"Results on kakako_STS_test: {best_result:.6f}")

    return best_result    

def eval_nli_unsup(model_path, main_similarity=None, last2avg=False, firstlastavg=False, device: str =None, test_path: str =None):
    model = load_model(model_path, last2avg=last2avg, firstlastavg=firstlastavg, device=device)
    if last2avg:
        output_path = os.path.join(model_path, "sts_eval_last2")
    elif firstlastavg:
        output_path = os.path.join(model_path, "sts_eval_first_last")
    else:
        output_path = os.path.join(model_path, "sts_eval")
        
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    score = eval_KorSTS(model, output_path=output_path, main_similarity=main_similarity, test_path=test_path)
    return score


if __name__ == "__main__":
    args = parse_args()
    model_path = args.model_path
    main_similarity = None
    if args.main_similarity == "cosine": 
        main_similarity = SimilarityFunction.COSINE
    elif args.main_similarity == "euclidean":
        main_similarity = SimilarityFunction.EUCLIDEAN
    elif args.main_similarity == "manhattan":
        main_similarity = SimilarityFunction.MANHATTAN
    elif args.main_similarity == "dot_product":
        main_similarity = SimilarityFunction.DOT_PRODUCT
    elif args.main_similarity == None:
        main_similarity = None
    else:
        raise ValueError("Invalid similarity type")
    eval_nli_unsup(model_path, main_similarity, last2avg=args.last2avg, firstlastavg=args.firstlastavg)