# Notes

* random-permute: create positive pair from x by `t = x.split(); random.shuffle(t); permuted_examples.append(' '.join(t))`.
* Validated with kor sts-dev.tsv while training.
* Tested with kor sts-test.tsv with validation best score checkpoint while training.
* 'spearman' used for score.
* Uniform_loss, Align_loss is calculated w/ kakaobrain KorSTS (test) for korean data results, eng SentEval sts-b (test) for english data results.
* Validate every 125 steps.

# Result

Common parameters for below experiments:
* Learning rate: 1e-05, 3e-05, 5e-05, 7e-05

### English data results

Common parameters for below experiments:

* max_seq_length: 32
* per_device_train_batch_size: 64
* _n_gpu: 1
* seed: 42

| Model                             | Trained with          | STS-B (dev) | STS-B (test) |
|-----------------------------------|-----------------------|-------------|--------------|
| Unsup-SimCSE-bert-base            | wiki1m_for_simcse.txt | 0.8211      | 0.7708       |
| " (w/ permute-by-space-sep-token) | "                     | 0.8127      | 0.7567       |

Common parameters for below experiments:

* max_seq_length: 128
* per_device_train_batch_size: 64
* _n_gpu: 1
* seed: 42

| Model                             | Trained with          | STS-B (dev) | STS-B (test) |
|-----------------------------------|-----------------------|-------------|--------------|
| Unsup-SimCSE-bert-base            | wiki1m_for_simcse.txt | 0.8250      | 0.7807       |
| " (w/ permute-by-space-sep-token) | "                     | 0.8418      | 0.7852       |

### Korean data results

Common parameters for below experiments:

* max_seq_length: 32
* per_device_train_batch_size: 64
* _n_gpu: 1
* seed: 42

| Model                                        | Trained with                        | STS-B (dev)   | STS-B (test)   |
|----------------------------------------------|-------------------------------------|---------------|----------------|
| Unsup-SimCSE-ko-bert                         | korean_news_data_1m.txt             | 0.7143        | 0.6453         |
| " (w/ permute-by-space-sep-token)            | "                                   | 0.7776        | 0.6952         |
| -------------------------------------------- | ----------------------------------- | ------------- | -------------- |
| Unsup-SimCSE-ko-bert                         | wiki_corpus_len15_normalized_1m.txt | 0.7480        | 0.7032         |
| " (w/ permute-by-space-sep-token)            | "                                   | 0.7817        | 0.6986         |
| -------------------------------------------- | ----------------------------------- | ------------- | -------------- |
| Unsup-SimCSE-kr-bert                         | korean_news_data_1m.txt             | 0.7446        | 0.6746         |
| " (w/ permute-by-space-sep-token wo dropout) | "                                   | 0.7500        | 0.6711         |
| " (w/ permute-by-space-sep-token)            | "                                   | 0.7547        | 0.6762         |
| " (w/ permute-by-tokenizer-token wo dropout) | "                                   | 0.7625        | 0.6885         |
| " (w/ permute-by-tokenizer-token)            | "                                   | 0.7666        | 0.7074         |
| -------------------------------------------- | ----------------------------------- | ------------- | -------------- |
| Unsup-SimCSE-kr-bert                         | wiki_corpus_len15_normalized_1m.txt | 0.7967        | 0.7187         |
| " (w/ permute-by-space-sep-token wo dropout) | "                                   | 0.7451        | 0.6590         |
| " (w/ permute-by-space-sep-token)            | "                                   | 0.7760        | 0.7101         |
| " (w/ permute-by-tokenizer-token wo dropout) | "                                   | 0.7599        | 0.6987         |
| " (w/ permute-by-tokenizer-token)            | "                                   | 0.7774        | 0.7045         |

Common parameters for below experiments:

* max_seq_length: 128
* per_device_train_batch_size: 64
* _n_gpu: 1
* seed: 42

| Model                             | Trained with                        | lr    | STS-B (dev) | STS-B (test) |
|-----------------------------------|-------------------------------------|-------|-------------|--------------|
| Unsup-SimCSE-kr-bert              | wiki_corpus_len15_normalized_1m.txt | 1e-05 | 0.7889      | 0.7162       |
| " (w/ permute-by-space-sep-token) | "                                   | 2e-05 | 0.8066      | 0.7429       |