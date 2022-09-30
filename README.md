# Code of paper 'Comparison and Analysis of Unsupervised Contrastive Learning Approaches for Korean Sentence Representations'

## How to run code

### ConSERT_Kor

#### Requirements
```
torch=1.10.0
transformers=4.8.1
python=3.9.13
sentencepiece=0.1.96
cudatoolkit=11.3
```
To install apex, run
```
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir ./
```

#### Get Started
Before run korean_ConSERT_main, train,dev,test datasets have to be placed in ConSERT_Kor/data/.
```
python3 -u korean_ConSERT_main.py \
--seed 3 \
--gpu 1 \
--batch_size 96 \
--max_seq_length 64 \
--train_data news \
--dev_test_data klue \
--train_way unsup \
--temperature 0.1 \
--learning_rate 5e-07 \
--data_aug_strategy1 token_cutoff \
--data_aug_strategy2 feature_cutoff \
--cutoff_rate 0.2 \
--model_name_or_path krbert \
--force_del \
--patience 10 \
--model_save_path ./outputs/krbert-ConSERT-token_cutoff+feature_cutoff
```

Below are arguments for details   
model_name_or_path => choose one in [kobert, krbert, klue_bert]   
train_data => choose one in [news, wiki]   
dev_test_data => choose one in [klue, kakao]   
data_aug_strategy1 and data_aug_strategy2 => each of them chooses one in [none, shuffle, token_cutoff, feature_cutoff, dropout]    
model_save_path => your_output_dir    



#### SimCSE_mul

```bash
$ python main.py \
--max_seq_len 32 \
--learning_rate 1e-05 \
--task_mode train-unsup krbert klue \
--train_file ./data/kor/korean_news_data_normalized_1m.txt \
--output_dir your_output_dir
```
