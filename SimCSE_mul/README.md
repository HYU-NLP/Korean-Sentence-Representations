# Notes

* random-permute: create positive pair from x by `t = x.split(); random.shuffle(t); permuted_examples.append(' '.join(t))`.
* Validated with kor sts-dev.tsv while training.
* Tested with kor sts-test.tsv with validation best score checkpoint while training.
* 'spearman' used for score.
* Uniform_loss, Align_loss is calculated w/ kakaobrain KorSTS (test) for korean data results, eng SentEval sts-b (test) for english data results.
* Validate every 125 steps.

# Result

### English data results

Common parameters for below experiments:

* max_seq_length: 32
* per_device_train_batch_size: 64
* _n_gpu: 1
* seed: 42

| Model                             | Trained with          | lr    | STS-B (dev) | STS-B (test) |
|-----------------------------------|-----------------------|-------|-------------|--------------|
| Unsup-SimCSE-bert-base            | wiki1m_for_simcse.txt | 1e-05 | 0.8191      | 0.7854       |
| "                                 | "                     | 3e-05 | 0.8186      | 0.7511       |
| "                                 | "                     | 5e-05 | 0.8211      | 0.7708       |
| "                                 | "                     | 7e-05 | 0.8088      | 0.7480       |
| " (w/ permute-by-space-sep-token) | "                     | 1e-05 | 0.8096      | 0.7612       |
| " (w/ permute-by-space-sep-token) | "                     | 3e-05 | 0.8127      | 0.7567       |
| " (w/ permute-by-space-sep-token) | "                     | 5e-05 | 0.8064      | 0.7622       |
| " (w/ permute-by-space-sep-token) | "                     | 7e-05 | 0.7994      | 0.7396       |

Common parameters for below experiments:

* max_seq_length: 128
* per_device_train_batch_size: 64
* _n_gpu: 1
* seed: 42

| Model                             | Trained with          | lr    | STS-B (dev) | STS-B (test) |
|-----------------------------------|-----------------------|-------|-------------|--------------|
| Unsup-SimCSE-bert-base            | wiki1m_for_simcse.txt | 1e-05 | 0.8016      | 0.7607       |
| "                                 | "                     | 3e-05 | 0.8250      | 0.7807       |
| "                                 | "                     | 5e-05 | 0.8178      | 0.7538       |
| "                                 | "                     | 7e-05 | 0.8025      | 0.7390       |
| " (w/ permute-by-space-sep-token) | "                     | 1e-05 | 0.8418      | 0.7852       |
| " (w/ permute-by-space-sep-token) | "                     | 3e-05 | 0.8385      | 0.7707       |
| " (w/ permute-by-space-sep-token) | "                     | 5e-05 | 0.8343      | 0.7674       |
| " (w/ permute-by-space-sep-token) | "                     | 7e-05 | 0.8308      | 0.7683       |
| " (w/ permute-by-tokenizer-token) | "                     | 1e-05 | ?           | ?            |
| " (w/ permute-by-tokenizer-token) | "                     | 3e-05 | ?           | ?            |
| " (w/ permute-by-tokenizer-token) | "                     | 5e-05 | ?           | ?            |
| " (w/ permute-by-tokenizer-token) | "                     | 7e-05 | ?           | ?            |

### Korean data results

Common parameters for below experiments:

* max_seq_length: 32
* per_device_train_batch_size: 64
* _n_gpu: 1
* seed: 42

| Model                                        | Trained with                        | lr      | STS-B (dev)   | STS-B (test)   |
|----------------------------------------------|-------------------------------------|---------|---------------|----------------|
| Unsup-SimCSE-ko-bert                         | korean_news_data_1m.txt             | 1e-05   | 0.7143        | 0.6453         |
| "                                            | "                                   | 3e-05   | 0.6820        | 0.6165         |
| "                                            | "                                   | 5e-05   | 0.6813        | 0.6286         |
| "                                            | "                                   | 7e-05   | 0.6894        | 0.6207         |
| " (w/ permute-by-space-sep-token)            | "                                   | 1e-05   | 0.7776        | 0.6952         |
| " (w/ permute-by-space-sep-token)            | "                                   | 3e-05   | 0.7696        | 0.6962         |
| " (w/ permute-by-space-sep-token)            | "                                   | 5e-05   | 0.7270        | 0.6426         |
| " (w/ permute-by-space-sep-token)            | "                                   | 7e-05   | 0.7289        | 0.6415         |
| -------------------------------------------- | ----------------------------------- | ------- | ------------- | -------------- |
| Unsup-SimCSE-ko-bert                         | wiki_corpus_len15_normalized_1m.txt | 1e-05   | 0.7296        | 0.6722         |
| "                                            | "                                   | 3e-05   | 0.7480        | 0.7032         |
| "                                            | "                                   | 5e-05   | 0.7182        | 0.6617         |
| "                                            | "                                   | 7e-05   | 0.7001        | 0.6257         |
| " (w/ permute-by-space-sep-token)            | "                                   | 1e-05   | 0.7817        | 0.6986         |
| " (w/ permute-by-space-sep-token)            | "                                   | 3e-05   | 0.7654        | 0.6860         |
| " (w/ permute-by-space-sep-token)            | "                                   | 5e-05   | 0.7474        | 0.6585         |
| " (w/ permute-by-space-sep-token)            | "                                   | 7e-05   | 0.7558        | 0.6692         |
| -------------------------------------------- | ----------------------------------- | ------- | ------------- | -------------- |
| Unsup-SimCSE-kr-bert                         | korean_news_data_1m.txt             | 1e-05   | 0.7064        | 0.6493         |
| "                                            | "                                   | 3e-05   | 0.7347        | 0.6795         |
| "                                            | "                                   | 5e-05   | 0.7093        | 0.6463         |
| "                                            | "                                   | 7e-05   | 0.7446        | 0.6746         |
| " (w/ permute-by-space-sep-token wo dropout) | "                                   | 1e-05   | 0.7193        | 0.6233         |
| " (w/ permute-by-space-sep-token wo dropout) | "                                   | 3e-05   | 0.7335        | 0.6513         |
| " (w/ permute-by-space-sep-token wo dropout) | "                                   | 5e-05   | 0.7499        | 0.6709         |
| " (w/ permute-by-space-sep-token wo dropout) | "                                   | 7e-05   | 0.7500        | 0.6711         |
| " (w/ permute-by-space-sep-token)            | "                                   | 1e-05   | 0.7504        | 0.6779         |
| " (w/ permute-by-space-sep-token)            | "                                   | 3e-05   | 0.7547        | 0.6762         |
| " (w/ permute-by-space-sep-token)            | "                                   | 5e-05   | 0.7475        | 0.6592         |
| " (w/ permute-by-space-sep-token)            | "                                   | 7e-05   | 0.7413        | 0.6691         |
| " (w/ permute-by-tokenizer-token wo dropout) | "                                   | 1e-05   | 0.7344        | 0.6690         |
| " (w/ permute-by-tokenizer-token wo dropout) | "                                   | 3e-05   | 0.7586        | 0.6827         |
| " (w/ permute-by-tokenizer-token wo dropout) | "                                   | 5e-05   | 0.7625        | 0.6885         |
| " (w/ permute-by-tokenizer-token wo dropout) | "                                   | 7e-05   | 0.7111        | 0.6468         |
| " (w/ permute-by-tokenizer-token)            | "                                   | 1e-05   | 0.7666        | 0.7074         |
| " (w/ permute-by-tokenizer-token)            | "                                   | 3e-05   | 0.7590        | 0.7021         |
| " (w/ permute-by-tokenizer-token)            | "                                   | 5e-05   | 0.7577        | 0.6981         |
| " (w/ permute-by-tokenizer-token)            | "                                   | 7e-05   | 0.7429        | 0.6747         |
| -------------------------------------------- | ----------------------------------- | ------- | ------------- | -------------- |
| Unsup-SimCSE-kr-bert                         | wiki_corpus_len15_normalized_1m.txt | 1e-05   | 0.7721        | 0.6925         |
| "                                            | "                                   | 3e-05   | 0.7967        | 0.7187         |
| "                                            | "                                   | 5e-05   | 0.7301        | 0.6554         |
| "                                            | "                                   | 7e-05   | 0.7662        | 0.6985         |
| " (w/ permute-by-space-sep-token wo dropout) | "                                   | 1e-05   | 0.7282        | 0.6502         |
| " (w/ permute-by-space-sep-token wo dropout) | "                                   | 3e-05   | 0.7451        | 0.6590         |
| " (w/ permute-by-space-sep-token wo dropout) | "                                   | 5e-05   | 0.7445        | 0.6491         |
| " (w/ permute-by-space-sep-token wo dropout) | "                                   | 7e-05   | 0.7376        | 0.6787         |
| " (w/ permute-by-space-sep-token)            | "                                   | 1e-05   | 0.7708        | 0.6898         |
| " (w/ permute-by-space-sep-token)            | "                                   | 3e-05   | 0.7760        | 0.7101         |
| " (w/ permute-by-space-sep-token)            | "                                   | 5e-05   | 0.7588        | 0.6840         |
| " (w/ permute-by-space-sep-token)            | "                                   | 7e-05   | 0.7675        | 0.6900         |
| " (w/ permute-by-tokenizer-token wo dropout) | "                                   | 1e-05   | 0.7590        | 0.6973         |
| " (w/ permute-by-tokenizer-token wo dropout) | "                                   | 3e-05   | 0.7599        | 0.6987         |
| " (w/ permute-by-tokenizer-token wo dropout) | "                                   | 5e-05   | 0.7354        | 0.6707         |
| " (w/ permute-by-tokenizer-token wo dropout) | "                                   | 7e-05   | 0.7308        | 0.6488         |
| " (w/ permute-by-tokenizer-token)            | "                                   | 1e-05   | 0.7774        | 0.7045         |
| " (w/ permute-by-tokenizer-token)            | "                                   | 3e-05   | 0.7669        | 0.6971         |
| " (w/ permute-by-tokenizer-token)            | "                                   | 5e-05   | 0.7591        | 0.6900         |
| " (w/ permute-by-tokenizer-token)            | "                                   | 7e-05   | 0.7620        | 0.6918         |

Common parameters for below experiments:

* max_seq_length: 128
* per_device_train_batch_size: 64
* _n_gpu: 1
* seed: 42

| Model                             | Trained with                        | lr    | STS-B (dev) | STS-B (test) |
|-----------------------------------|-------------------------------------|-------|-------------|--------------|
| Unsup-SimCSE-kr-bert              | wiki_corpus_len15_normalized_1m.txt | 1e-05 | 0.7889      | 0.7162       |
| "                                 | "                                   | 2e-05 | 0.7837      | 0.7157       |
| "                                 | "                                   | 3e-05 | 0.7779      | 0.7164       |
| "                                 | "                                   | 4e-05 | 0.7617      | 0.6978       |
| "                                 | "                                   | 5e-05 | 0.7769      | 0.7050       |
| "                                 | "                                   | 6e-05 | 0.7665      | 0.6964       |
| " (w/ permute-by-space-sep-token) | "                                   | 1e-05 | 0.7987      | 0.7382       |
| " (w/ permute-by-space-sep-token) | "                                   | 2e-05 | 0.8066      | 0.7429       |
| " (w/ permute-by-space-sep-token) | "                                   | 3e-05 | 0.8046      | 0.7411       |
| " (w/ permute-by-space-sep-token) | "                                   | 4e-05 | 0.8028      | 0.7354       |
| " (w/ permute-by-space-sep-token) | "                                   | 5e-05 | 0.7987      | 0.7385       |
| " (w/ permute-by-space-sep-token) | "                                   | 6e-05 | 0.7917      | 0.7188       |