# Code of paper 'Comparison and Analysis of Unsupervised Contrastive Learning Approaches for Korean Sentence Representations'

### How to run code

##### ConSERT_Kor

##### SimCSE_mul

```bash
$ python main.py \
--max_seq_len 32 \
--learning_rate 1e-05 \
--task_mode train-unsup krbert klue \
--train_file ./data/kor/korean_news_data_normalized_1m.txt \
--output_dir your_output_dir
```