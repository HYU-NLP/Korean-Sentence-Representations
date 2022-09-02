### Result

| Model     | Trained with          | lr    | STS-B (dev) | STS-B (test) |
|-----------|-----------------------|-------|-------------|--------------|
| m-bert    | -                     | -     | 0.3026      | 0.2226       |
| SG-M-BERT | snli_1.0_train.ko.tsv | 1e-05 | 0.7105      | 0.6265       |
| "         | "                     | 3e-05 | 0.7204      | 0.6321       |
| "         | "                     | 5e-05 | 0.7129      | 0.6267       |
| SG-KoBERT | "                     | 5e-05 | 0.6670      | 0.5529       |
| "         | korean_news_data.txt  | 5e-05 | 0.5856      | 0.4655       |

SG-M-BERT

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
