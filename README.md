# ConSERT

#### How to run

1. Move to the ConSERT directory `cd ConSERT`
2. Run `cd data && bash get_transfer_data.bash`
3. Download pre-trained BERT from https://huggingface.co/bert-base-uncased     
   **:Note that `bert-base-uncased` has to be located in ConSERT/.**
4. Run bash files to train models `bash commands/unsup-consert-base.sh`

#### Implementation details

SentenceTransformer.py : inheriting two modules ,Transformer and Pooling, and there exist other inner functions such
as `fit` to train the model, `encode` to eval the model    
Transformer.py : fetching the pretrained_model 'BERT' and getting representations      
Pooling.py : averaging the representations from Transformer and they become the sentence representation      
MyLoss.py : implementing contrastive loss for unsupervision, and crossentropy for supervision      
modeling_bert.py : depending on data augmentation strategies, perturbating input embeddings before passing through bert
layers (see codes line 713~965 for details)

#### Result

|          | Model       | STS12 | STS13 | STS14 | STS15 | STS16 | STSb  | SICK-R | Avg.  |
|----------|-------------|-------|-------|-------|-------|-------|-------|--------|-------|
| BASELINE | unsup       | 64.64 | 78.49 | 69.07 | 79.72 | 75.95 | 73.97 | 67.31  | 72.74 |
| re-imple | unsup       | 64.69 | 78.56 | 69.01 | 79.70 | 75.77 | 73.86 | 67.15  | 72.68 |
| BASELINE | sup         | 69.93 | 76.00 | 72.15 | 78.59 | 73.53 | 76.10 | 73.01  | 74.19 |
| re-imple | sup         | 69.16 | 73.38 | 71.06 | 77.77 | 73.79 | 75.81 | 72.20  | 73.31 |
| BASELINE | sup-unsup   | 73.02 | 84.86 | 77.32 | 82.70 | 78.2  | 81.34 | 75.00  | 78.92 |
| re-imple | sup-unsup   | 72.99 | 84.28 | 76.73 | 82.54 | 78.12 | 81.12 | 75.02  | 78.69 |
| BASELINE | joint       | 70.92 | 79.98 | 74.88 | 81.76 | 76.46 | 78.99 | 78.15  | 77.31 |
| re-imple | joint       | 70.47 | 78.73 | 73.80 | 70.94 | 76.03 | 77.75 | 77.70  | 76.49 |
| BASELINE | joint-unsup | 74.46 | 84.19 | 77.08 | 83.77 | 78.55 | 81.37 | 77.01  | 79.49 |
| re-imple | joint-unsup | 73.46 | 83.44 | 76.06 | 83.07 | 78.46 | 80.27 | 75.90  | 78.66 |

data agumentation strategies settings      
joint : token_cutoff(cutoff rate = 0.1), none     
joint-unsup : shuffle, none     
sup-unsup : feature_cutoff(cutoff rate = 0.1), none

Combinations of augmnetation strategies have a effect on the results.      
Just so you know, on joint, joint-unsup and sup-unsup settings, taking feature_cutoff and shuffle strategies lowers the
Avg scores by 2 points compared to the results above

# SimCSE

#### Result

|                  | Model | STS12 | STS13 | STS14 | STS15 | STS16 | STSb  | SICK-R | Avg.  |
|------------------|-------|-------|-------|-------|-------|-------|-------|--------|-------|
| SimCSE-Bert_base | unsup | 68.40 | 82.41 | 74.38 | 80.91 | 78.56 | 76.85 | 72.23  | 76.25 |
| re-imple         | unsup | 67.65 | 81.59 | 74.10 | 81.12 | 75.97 | 77.94 | 70.99  | 76.14 |
| SimCSE-Bert_base | sup   | 75.30 | 84.67 | 80.19 | 85.40 | 80.82 | 84.25 | 80.39  | 81.57 |
| re-imple         | sup   | 75.57 | 81.82 | 78.94 | 85.87 | 81.57 | 84.27 | 80.25  | 81.18 |

Below is important parameters of re-imple:

* max_seq_length: 32
* per_device_train_batch_size: 64
* _n_gpu: 1
* seed: 42

# SimCSE_mul

* random-permute: create positive pair from x
  by `t = x.split(); random.shuffle(t); permuted_examples.append(' '.join(t))`.
* Validated with kor sts-dev.tsv while training.
* Tested with kor sts-test.tsv with validation best score checkpoint while training.
* 'spearman' used for score.
* uniform_loss, align_loss is calculated w/ kor sts-test.tsv.
* m-bert: bert-base-multilingual-uncased.
* Validate every 125 steps.
* `korean_news_data.sample.txt` is a sample of `korean_news_data_1m.txt`, created
  as `$ head -50000 korean_news_data_1m.txt > korean_news_data.sample.txt`.

#### Result

Common parameters for below experiments:

* max_seq_length: 32
* per_device_train_batch_size: 64
* _n_gpu: 1
* seed: 42

| Model                    | Trained with                        | lr      | STS-B (dev)   | STS-B (test)   | uniform-align loss |
|--------------------------|-------------------------------------|---------|---------------|----------------|--------------------|
| m-bert                   | -                                   | -       | 0.3026        | 0.2226         |                    |
| ko-bert                  | -                                   | -       | 0.3413        | 0.2566         |                    |
| ------------------------ | ----------------------------------- | ------- | ------------- | -------------- | ------------------ |
| Sup-SimCSE-m-bert        | snli_1.0_train.ko.tsv               | 1e-05   | 0.7650        | 0.6937         | (-2.0897, 0.2540)  |
| "                        | "                                   | 3e-05   | 0.7644        | 0.6996         | (-2.0793, 0.2434)  |
| "                        | "                                   | 5e-05   | 0.7559        | 0.6929         | (-2.1024, 0.2476)  |
| ------------------------ | ----------------------------------- | ------- | ------------- | -------------- | ------------------ |
| Unsup-SimCSE-bert-base   | wiki1m_for_simcse.txt               | 1e-05   | 0.8191        | 0.7854         | (-1.8364, 0.2158)  |
| " (w/ random-permute)    | "                                   | 1e-05   | 0.8096        | 0.7612         | (-1.8559, 0.2129)  |
| ------------------------ | ----------------------------------- | ------- | ------------- | -------------- | ------------------ |
| Unsup-SimCSE-ko-bert     | korean_news_data.sample.txt         | 1e-05   | 0.6307        | 0.5528         | (-2.1972, 0.4252)  |
| "                        | "                                   | 3e-05   | 0.6599        | 0.5777         | (-2.2318, 0.4037)  |
| "                        | "                                   | 5e-05   | 0.6731        | 0.5974         | (-2.2215, 0.3833)  |
| " (w/ random-permute)    | "                                   | 1e-05   | 0.7171        | 0.6348         | (-1.8739, 0.2740)  |
| " (w/ random-permute)    | "                                   | 3e-05   | 0.7383        | 0.6550         | (-1.9860, 0.2874)  |
| " (w/ random-permute)    | "                                   | 5e-05   | 0.7471        | 0.6520         | (-2.0237, 0.2757)  |
| ------------------------ | ----------------------------------- | ------- | ------------- | -------------- | ------------------ |
| Unsup-SimCSE-ko-bert     | korean_news_data_1m.txt             | 1e-05   | 0.7143        | 0.6453         | ?                  |
| "                        | "                                   | 3e-05   | 0.6820        | 0.6165         | ?                  |
| "                        | "                                   | 5e-05   | 0.6813        | 0.6286         | ?                  |
| "                        | "                                   | 7e-05   | 0.6894        | 0.6207         | ?                  |
| " (w/ random-permute)    | "                                   | 1e-05   | 0.7776        | 0.6952         | ?                  |
| " (w/ random-permute)    | "                                   | 3e-05   | 0.7696        | 0.6962         | ?                  |
| " (w/ random-permute)    | "                                   | 5e-05   | 0.7270        | 0.6426         | ?                  |
| " (w/ random-permute)    | "                                   | 7e-05   | 0.7289        | 0.6415         | ?                  |
| ------------------------ | ----------------------------------- | ------- | ------------- | -------------- | ------------------ |
| Unsup-SimCSE-ko-bert     | korean_news_data_1m.txt (shuffle)   | 1e-05   | 0.6886        | 0.6315         | ?                  |
| "                        | "                                   | 3e-05   | 0.6005        | 0.52688        | ?                  |
| "                        | "                                   | 5e-05   | 0.6182        | 0.5138         | ?                  |
| "                        | "                                   | 7e-05   | 0.5870        | 0.5235         | ?                  |
| " (w/ random-permute)    | "                                   | 1e-05   | 0.7808        | 0.6897         | ?                  |
| " (w/ random-permute)    | "                                   | 3e-05   | 0.7636        | 0.6835         | ?                  |
| " (w/ random-permute)    | "                                   | 5e-05   | 0.7367        | 0.6577         | ?                  |
| " (w/ random-permute)    | "                                   | 7e-05   | 0.7152        | 0.6208         | ?                  |
| ------------------------ | ----------------------------------- | ------- | ------------- | -------------- | ------------------ |
| Unsup-SimCSE-kr-bert     | korean_news_data_1m.txt             | 3e-05   | 0.7347        | 0.6795         | ?                  |
| "                        | "                                   | 5e-05   | 0.7092        | 0.6463         | ?                  |
| "                        | "                                   | 7e-05   | 0.7441        | 0.6742         | ?                  |
| "                        | "                                   | 9e-05   | 0.7269        | 0.6515         | ?                  |
| ------------------------ | ----------------------------------- | ------- | ------------- | -------------- | ------------------ |
| Unsup-SimCSE-kr-bert     | korean_news_data_1m.txt (shuffle)   | 3e-05   | 0.7337        | 0.6722         | ?                  |
| "                        | "                                   | 5e-05   | 0.6804        | 0.6254         | ?                  |
| "                        | "                                   | 7e-05   | 0.6908        | 0.6308         | ?                  |
| "                        | "                                   | 9e-05   | 0.7452        | 0.6850         | ?                  |
| ------------------------ | ----------------------------------- | ------- | ------------- | -------------- | ------------------ |
| Unsup-SimCSE-kr-bert     | wiki_corpus_len15_normalized_1m.txt | 1e-05   | 0.7721        | 0.6925         | ?                  |
| "                        | "                                   | 3e-05   | 0.7967        | 0.7187         | ?                  |
| "                        | "                                   | 5e-05   | 0.7301        | 0.6554         | ?                  |
| "                        | "                                   | 7e-05   | 0.7662        | 0.6985         | ?                  |
| " (w/ random-permute)    | "                                   | 1e-05   | 0.7709        | 0.6904         | ?                  |
| "                        | "                                   | 3e-05   | 0.7721        | 0.7064         | ?                  |
| "                        | "                                   | 5e-05   | 0.7608        | 0.6800         | ?                  |
| "                        | "                                   | 7e-05   | 0.7676        | 0.6899         | ?                  |
| ------------------------ | ----------------------------------- | ------- | ------------- | -------------- | ------------------ |
| Unsup-SimCSE-kr-bert     | wiki_corpus_len30_normalized_1m.txt | 1e-05   | 0.7629        | 0.6944         | ?                  |
| "                        | "                                   | 3e-05   | 0.7782        | 0.7093         | ?                  |
| "                        | "                                   | 5e-05   | 0.7677        | 0.6861         | ?                  |
| "                        | "                                   | 7e-05   | 0.7679        | 0.6822         | ?                  |
| " (w/ random-permute)    | "                                   | 1e-05   | 0.7429        | 0.6613         | ?                  |
| "                        | "                                   | 3e-05   | 0.7793        | 0.6988         | ?                  |
| "                        | "                                   | 5e-05   | 0.7582        | 0.6750         | ?                  |
| "                        | "                                   | 7e-05   | 0.7589        | 0.6713         | ?                  |

Common parameters for below experiments:

* max_seq_length: 128
* per_device_train_batch_size: 64
* _n_gpu: 1
* seed: 42

| Model                 | Trained with                        | lr    | STS-B (dev) | STS-B (test) | uniform-align loss |
|-----------------------|-------------------------------------|-------|-------------|--------------|--------------------|
| Unsup-SimCSE-kr-bert  | wiki_corpus_len15_normalized_1m.txt | 1e-05 | 0.7889      | 0.7162       | (-1.7571, 0.2669)  |
| "                     | "                                   | 3e-05 | 0.7779      | 0.7164       | (-1.8360, 0.2735)  |
| "                     | "                                   | 5e-05 | 0.7769      | 0.7050       | (-1.9990, 0.3155)  |
| " (w/ random-permute) | "                                   | 1e-05 | 0.7987      | 0.7382       | (-1.7547, 0.2516)  |
| "                     | "                                   | 3e-05 | 0.8046      | 0.7411       | (-1.8084, 0.2633)  |
| "                     | "                                   | 5e-05 | 0.7987      | 0.7385       | (-1.8612, 0.2720)  |

# SG-BERT_kor

#### Result

| Model         | Trained with          | lr    | STS-B (dev) | STS-B (test) |
|---------------|-----------------------|-------|-------------|--------------|
| m-bert        | -                     | -     | 0.3026      | 0.2226       |
| Sup-SG-M-BERT | snli_1.0_train.ko.tsv | 1e-05 | 0.7105      | 0.6265       |
| "             | "                     | 3e-05 | 0.7204      | 0.6321       |
| "             | "                     | 5e-05 | 0.7129      | 0.6267       |
| SG-KoBERT     | "                     | 5e-05 | 0.6670      | 0.5529       |
| "             | korean_news_data.txt  | 5e-05 | 0.5856      | 0.4655       |

Sup-SG-M-BERT

* Used multilingual BERT based model (m-bert: bert-base-multilingual-uncased)
* Trained with snli_1.0_train.ko.tsv
* Validated with sts-dev.tsv while training
* Tested with sts-test.tsv with validation best score checkpoint while training
* Used to spearman correlation
* Parameters for above experiments:
    * max_seq_length: 128
    * per_device_train_batch_size: 16
    * _n_gpu: 1
    * seed: 42

SG-KoBERT

* Used KoBERT based model (kobert: skt/kobert-based-v1)
* Trained with snli_1.0_train.ko.tsv and korean_news_data.txt
* Others are same as Sup-SG-M-BERT
