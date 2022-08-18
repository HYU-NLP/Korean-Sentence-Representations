import pandas as pd

mnli = pd.read_csv('multinli.train.ko.tsv', sep='\t')
snli = pd.read_csv('snli_1.0_train.ko.tsv', sep='\t')
pd.concat([snli, mnli]).to_csv('mnli_n_snli.train.ko.tsv', sep='\t')
