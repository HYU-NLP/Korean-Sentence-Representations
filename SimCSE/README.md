# Result

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