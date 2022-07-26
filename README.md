# How to run SentEval for my Model

1. Run `SentEval/data/downstream/download_dataset.sh`
2. Run `evaluation.py` with appropriate arguments.


# How to run ConSERT on SentEval

1. Move to the ConSERT directory with `cd ConSERT`
2. Run `cd data && bash get_transfer_data.bash`
3. Download pre-trained BERT from https://huggingface.co/bert-base-uncased     
**:Note that `bert-base-uncased` has to be located in ConSERT/.**
4. Run bash files to train models `bash commands/unsup-consert-base.sh`    

#### Implementation detalis

SentenceTransformer.py : inheriting two modules ,Transformer and Pooling, and there exist other inner functions such as `fit` to train the model, `encode` to eval the model              
Transformer.py : fetching the pretrained_model 'BERT' and getting representations      
Pooling.py : averaging the representations from Transformer and they become the sentence representation      
MyLoss.py : implementing contrastive loss for unsupervision, and crossentropy for supervision      
modeling_bert.py : depending on data augmentation strategies, changing input embeddings before passing through bert layers     

