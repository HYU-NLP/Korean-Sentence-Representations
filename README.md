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
* uniform_loss, align_loss is calculated w/
  kakaobrain KorSTS (test) for korean data results,
  eng SentEval sts-b (test) for english data results.
* Validate every 125 steps.

#### Result

##### English data results

Common parameters for below experiments:

* max_seq_length: 32
* per_device_train_batch_size: 64
* _n_gpu: 1
* seed: 42

| Model                             | Trained with          | lr    | STS-B (dev) | STS-B (test) | uniform-align loss |
|-----------------------------------|-----------------------|-------|-------------|--------------|--------------------|
| Unsup-SimCSE-bert-base            | wiki1m_for_simcse.txt | 1e-05 | 0.8191      | 0.7854       | (-1.8364, 0.2158)  |
| "                                 | "                     | 3e-05 | 0.8186      | 0.7511       | ?                  |
| "                                 | "                     | 5e-05 | 0.8211      | 0.7708       | ?                  |
| "                                 | "                     | 7e-05 | 0.8088      | 0.7480       | ?                  |
| " (w/ permute-by-space-sep-token) | "                     | 1e-05 | 0.8096      | 0.7612       | (-1.8559, 0.2129)  |
| " (w/ permute-by-space-sep-token) | "                     | 3e-05 | 0.8127      | 0.7567       | ?                  |
| " (w/ permute-by-space-sep-token) | "                     | 5e-05 | 0.8064      | 0.7622       | ?                  |
| " (w/ permute-by-space-sep-token) | "                     | 7e-05 | 0.7994      | 0.7396       | ?                  |

##### Korean data results

Common parameters for below experiments:

* max_seq_length: 32
* per_device_train_batch_size: 64
* _n_gpu: 1
* seed: 42

| Model                                        | Trained with                        | lr      | STS-B (dev)   | STS-B (test)   | uniform-align loss |
|----------------------------------------------|-------------------------------------|---------|---------------|----------------|--------------------|
| Unsup-SimCSE-ko-bert                         | korean_news_data_1m.txt             | 1e-05   | 0.7143        | 0.6453         | ?                  |
| "                                            | "                                   | 3e-05   | 0.6820        | 0.6165         | ?                  |
| "                                            | "                                   | 5e-05   | 0.6813        | 0.6286         | ?                  |
| "                                            | "                                   | 7e-05   | 0.6894        | 0.6207         | ?                  |
| " (w/ permute-by-space-sep-token)            | "                                   | 1e-05   | 0.7776        | 0.6952         | ?                  |
| " (w/ permute-by-space-sep-token)            | "                                   | 3e-05   | 0.7696        | 0.6962         | ?                  |
| " (w/ permute-by-space-sep-token)            | "                                   | 5e-05   | 0.7270        | 0.6426         | ?                  |
| " (w/ permute-by-space-sep-token)            | "                                   | 7e-05   | 0.7289        | 0.6415         | ?                  |
| -------------------------------------------- | ----------------------------------- | ------- | ------------- | -------------- | ------------------ |
| Unsup-SimCSE-ko-bert                         | wiki_corpus_len15_normalized_1m.txt | 1e-05   | 0.7296        | 0.6722         | ?                  |
| "                                            | "                                   | 3e-05   | 0.7480        | 0.7032         | ?                  |
| "                                            | "                                   | 5e-05   | 0.7182        | 0.6617         | ?                  |
| "                                            | "                                   | 7e-05   | 0.7001        | 0.6257         | ?                  |
| " (w/ permute-by-space-sep-token)            | "                                   | 1e-05   | 0.7817        | 0.6986         | ?                  |
| " (w/ permute-by-space-sep-token)            | "                                   | 3e-05   | 0.7654        | 0.6860         | ?                  |
| " (w/ permute-by-space-sep-token)            | "                                   | 5e-05   | 0.7474        | 0.6585         | ?                  |
| " (w/ permute-by-space-sep-token)            | "                                   | 7e-05   | 0.7558        | 0.6692         | ?                  |
| -------------------------------------------- | ----------------------------------- | ------- | ------------- | -------------- | ------------------ |
| Unsup-SimCSE-kr-bert                         | korean_news_data_1m.txt             | 1e-05   | 0.7064        | 0.6493         | ?                  |
| "                                            | "                                   | 3e-05   | 0.7347        | 0.6795         | ?                  |
| "                                            | "                                   | 5e-05   | 0.7093        | 0.6463         | ?                  |
| "                                            | "                                   | 7e-05   | 0.7446        | 0.6746         | ?                  |
| " (w/ permute-by-space-sep-token wo dropout) | "                                   | 1e-05   | 0.7193        | 0.6233         | ?                  |
| " (w/ permute-by-space-sep-token wo dropout) | "                                   | 3e-05   | 0.7335        | 0.6513         | ?                  |
| " (w/ permute-by-space-sep-token wo dropout) | "                                   | 5e-05   | 0.7499        | 0.6709         | ?                  |
| " (w/ permute-by-space-sep-token wo dropout) | "                                   | 7e-05   | 0.7500        | 0.6711         | ?                  |
| " (w/ permute-by-space-sep-token)            | "                                   | 1e-05   | 0.7504        | 0.6779         | ?                  |
| " (w/ permute-by-space-sep-token)            | "                                   | 3e-05   | 0.7547        | 0.6762         | ?                  |
| " (w/ permute-by-space-sep-token)            | "                                   | 5e-05   | 0.7475        | 0.6592         | ?                  |
| " (w/ permute-by-space-sep-token)            | "                                   | 7e-05   | 0.7413        | 0.6691         | ?                  |
| " (w/ permute-by-tokenizer-token wo dropout) | "                                   | 1e-05   | 0.7344        | 0.6690         | ?                  |
| " (w/ permute-by-tokenizer-token wo dropout) | "                                   | 3e-05   | 0.7586        | 0.6827         | ?                  |
| " (w/ permute-by-tokenizer-token wo dropout) | "                                   | 5e-05   | 0.7625        | 0.6885         | ?                  |
| " (w/ permute-by-tokenizer-token wo dropout) | "                                   | 7e-05   | 0.7111        | 0.6468         | ?                  |
| " (w/ permute-by-tokenizer-token)            | "                                   | 1e-05   | 0.7666        | 0.7074         | ?                  |
| " (w/ permute-by-tokenizer-token)            | "                                   | 3e-05   | 0.7590        | 0.7021         | ?                  |
| " (w/ permute-by-tokenizer-token)            | "                                   | 5e-05   | 0.7577        | 0.6981         | ?                  |
| " (w/ permute-by-tokenizer-token)            | "                                   | 7e-05   | 0.7429        | 0.6747         | ?                  |
| -------------------------------------------- | ----------------------------------- | ------- | ------------- | -------------- | ------------------ |
| Unsup-SimCSE-kr-bert                         | korean_news_data_normalized_1m.txt  | 1e-05   | ?             | ?              | ?                  |
| "                                            | "                                   | 3e-05   | ?             | ?              | ?                  |
| "                                            | "                                   | 5e-05   | ?             | ?              | ?                  |
| "                                            | "                                   | 7e-05   | ?             | ?              | ?                  |
| -------------------------------------------- | ----------------------------------- | ------- | ------------- | -------------- | ------------------ |
| Unsup-SimCSE-kr-bert                         | wiki_corpus_len15_normalized_1m.txt | 1e-05   | 0.7721        | 0.6925         | ?                  |
| "                                            | "                                   | 3e-05   | 0.7967        | 0.7187         | ?                  |
| "                                            | "                                   | 5e-05   | 0.7301        | 0.6554         | ?                  |
| "                                            | "                                   | 7e-05   | 0.7662        | 0.6985         | ?                  |
| " (w/ permute-by-space-sep-token wo dropout) | "                                   | 1e-05   | 0.7282        | 0.6502         | ?                  |
| " (w/ permute-by-space-sep-token wo dropout) | "                                   | 3e-05   | 0.7451        | 0.6590         | ?                  |
| " (w/ permute-by-space-sep-token wo dropout) | "                                   | 5e-05   | 0.7445        | 0.6491         | ?                  |
| " (w/ permute-by-space-sep-token wo dropout) | "                                   | 7e-05   | 0.7376        | 0.6787         | ?                  |
| " (w/ permute-by-space-sep-token)            | "                                   | 1e-05   | 0.7708        | 0.6898         | ?                  |
| " (w/ permute-by-space-sep-token)            | "                                   | 3e-05   | 0.7760        | 0.7101         | ?                  |
| " (w/ permute-by-space-sep-token)            | "                                   | 5e-05   | 0.7588        | 0.6840         | ?                  |
| " (w/ permute-by-space-sep-token)            | "                                   | 7e-05   | 0.7675        | 0.6900         | ?                  |
| " (w/ permute-by-tokenizer-token wo dropout) | "                                   | 1e-05   | 0.7590        | 0.6973         | ?                  |
| " (w/ permute-by-tokenizer-token wo dropout) | "                                   | 3e-05   | 0.7599        | 0.6987         | ?                  |
| " (w/ permute-by-tokenizer-token wo dropout) | "                                   | 5e-05   | 0.7354        | 0.6707         | ?                  |
| " (w/ permute-by-tokenizer-token wo dropout) | "                                   | 7e-05   | 0.7308        | 0.6488         | ?                  |
| " (w/ permute-by-tokenizer-token)            | "                                   | 1e-05   | 0.7774        | 0.7045         | ?                  |
| " (w/ permute-by-tokenizer-token)            | "                                   | 3e-05   | 0.7669        | 0.6971         | ?                  |
| " (w/ permute-by-tokenizer-token)            | "                                   | 5e-05   | 0.7591        | 0.6900         | ?                  |
| " (w/ permute-by-tokenizer-token)            | "                                   | 7e-05   | 0.7620        | 0.6918         | ?                  |

Common parameters for below experiments:

* max_seq_length: 128
* per_device_train_batch_size: 64
* _n_gpu: 1
* seed: 42

| Model                             | Trained with                        | lr    | STS-B (dev) | STS-B (test) | uniform-align loss |
|-----------------------------------|-------------------------------------|-------|-------------|--------------|--------------------|
| Unsup-SimCSE-kr-bert              | wiki_corpus_len15_normalized_1m.txt | 1e-05 | 0.7889      | 0.7162       | (-1.7571, 0.2669)  |
| "                                 | "                                   | 2e-05 | 0.7837      | 0.7157       | -                  |
| "                                 | "                                   | 3e-05 | 0.7779      | 0.7164       | (-1.8360, 0.2735)  |
| "                                 | "                                   | 4e-05 | 0.7617      | 0.6978       | -                  |
| "                                 | "                                   | 5e-05 | 0.7769      | 0.7050       | (-1.9990, 0.3155)  |
| "                                 | "                                   | 6e-05 | 0.7665      | 0.6964       | -                  |
| " (w/ permute-by-space-sep-token) | "                                   | 1e-05 | 0.7987      | 0.7382       | (-1.7547, 0.2516)  |
| " (w/ permute-by-space-sep-token) | "                                   | 2e-05 | 0.8066      | 0.7429       | -                  |
| " (w/ permute-by-space-sep-token) | "                                   | 3e-05 | 0.8046      | 0.7411       | (-1.8084, 0.2633)  |
| " (w/ permute-by-space-sep-token) | "                                   | 4e-05 | 0.8028      | 0.7354       | -                  |
| " (w/ permute-by-space-sep-token) | "                                   | 5e-05 | 0.7987      | 0.7385       | (-1.8612, 0.2720)  |
| " (w/ permute-by-space-sep-token) | "                                   | 6e-05 | 0.7917      | 0.7188       | -                  |

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
