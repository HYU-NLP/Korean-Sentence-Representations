# How to run

1. Move to the ConSERT directory `cd ConSERT`
2. Run `cd data && bash get_transfer_data.bash`
3. Download pre-trained BERT from https://huggingface.co/bert-base-uncased **:Note that `bert-base-uncased` has to be located in ConSERT/.**
4. Run bash files to train models `bash commands/unsup-consert-base.sh`

# Implementation details

SentenceTransformer.py : inheriting two modules ,Transformer and Pooling, and there exist other inner functions such as `fit` to train the model, `encode` to eval the model
Transformer.py : fetching the pretrained_model 'BERT' and getting representations      
Pooling.py : averaging the representations from Transformer and they become the sentence representation      
MyLoss.py : implementing contrastive loss for unsupervision, and crossentropy for supervision      
modeling_bert.py : depending on data augmentation strategies, perturbating input embeddings before passing through bert layers (see codes line 713~965 for details)

# Result

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