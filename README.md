# How to run SentEval for my Model

1. Run `SentEval/data/downstream/download_dataset.sh`
2. Run `evaluation.py` with appropriate arguments.


# ConSERT

#### How to run
1. Move to the ConSERT directory `cd ConSERT`
2. Run `cd data && bash get_transfer_data.bash`
3. Download pre-trained BERT from https://huggingface.co/bert-base-uncased     
**:Note that `bert-base-uncased` has to be located in ConSERT/.**
4. Run bash files to train models `bash commands/unsup-consert-base.sh`    

#### Implementation detalis

SentenceTransformer.py : inheriting two modules ,Transformer and Pooling, and there exist other inner functions such as `fit` to train the model, `encode` to eval the model    
Transformer.py : fetching the pretrained_model 'BERT' and getting representations      
Pooling.py : averaging the representations from Transformer and they become the sentence representation      
MyLoss.py : implementing contrastive loss for unsupervision, and crossentropy for supervision      
modeling_bert.py : depending on data augmentation strategies, perturbating input embeddings before passing through bert layers (see codes line 713~965 for details)     


#### Result

|              | **Model**   | **STS12** | **STS13** | **STS14** | **STS15** | **STS16** | **STSb** | **SICK-R** | **Avg.** |
|--------------|-------------|-----------|-----------|-----------|-----------|-----------|----------|------------|----------|
| **BASELINE** | unsup       | 64.64     | 78.49     | 69.07     | 79.72     | 75.95     | 73.97    | 67.31      | 72.74    |
| **re-imple** | unsup       | 64.69     | 78.56     | 69.01     | 79.70     | 75.77     | 73.86    | 67.15      | 72.68    |
| **BASELINE** | sup         | 69.93     | 76        | 72.15     | 78.59     | 73.53     | 76.1     | 73.01      | 74.19    |
| **re-imple** | sup         | 69.6      | 73.1      | 70.72     | 77.51     | 73.51     | 75.46    | 72.48      | 73.2     |
| **BASELINE** | sup-unsup   | 73.02     | 84.86     | 77.32     | 82.7      | 78.2      | 81.34    | 75         | 78.92    |
| **re-imple** | sup-unsup   | 72.31     | 82.32     | 73.97     | 81.82     | 76.98     | 79.54    | 73.5       | 77.21    |
| **BASELINE** | joint       | 70.92     | 79.98     | 74.88     | 81.76     | 76.46     | 78.99    | 78.15      | 77.31    |
| **re-imple** | joint       | 70.40     | 75.79     | 71.88     | 79.39     | 74.67     | 76.76    | 75.49      | 75.49    |
| **BASELINE** | joint-unsup | 74.46     | 84.19     | 77.08     | 83.77     | 78.55     | 81.37    | 77.01      | 79.49    |
| **re-imple** | joint-unsup | 72.35     | 79.13     | 73.96     | 80.85     | 75.18     | 78.62    | 75.33      | 76.49    |


