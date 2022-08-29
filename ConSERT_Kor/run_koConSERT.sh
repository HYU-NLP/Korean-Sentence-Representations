nohup python3 -u koConSERT_main.py \
    --seed 1 \
    --gpu 0 \
    --batch_size 96 \
    --max_seq_length 64 \
    --train_way unsup \
    --temperature 0.1 \
    --learning_rate 0.0000005 \
    --data_aug_strategy1 shuffle \
    --data_aug_strategy2 dropout \
    --cutoff_rate 0.2 \
    --model_name_or_path skt/kobert-base-v1 \
    --force_del \
    --patience 10 \
    --model_save_path ./outputs/unsup-KoConSERT-dropout+shuffle \
    > unsup-KoConSERT-dropout+shuffle.out &

# change .out and model_save_path name + data_aug_strategy + gpu