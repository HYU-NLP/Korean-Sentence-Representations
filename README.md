# How to run SentEval for my Model

1. Run `SentEval/data/downstream/download_dataset.sh`
2. Run `evaluation.py` with appropriate arguments.


# How to run ConSERT on SentEval

1. Move to the ConSERT directory with `cd ConSERT`
2. Run `cd data && bash ConSERT/data/get_transfer_data.bash`
3. Download pre-trained BERT from https://huggingface.co/bert-base-uncased
   :Note that `bert-base-uncased` has to be located in ConSERT/.
4. Run bash files  eg) `bash commands/unsup-consert-base.sh`
