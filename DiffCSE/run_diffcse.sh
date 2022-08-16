#!/bin/bash

LR=7e-6
MASK=0.15
LAMBDA=0.005
CUDA_VISIBLE_DEVICES=3 nohup python -u diffcse_main.py \
    --model_name_or_path bert-base-uncased \
    --generator_name distilbert-base-uncased \
    --lambda_weight $LAMBDA \
    --temp 0.05 \
    --masking_ratio $MASK \
    --train_file data/wiki1m_for_simcse.txt \
    --max_seq_length 32 \
    --output_dir ./outputs_seed103 \
    --num_train_epochs 2 \
    --per_device_train_batch_size 64 \
    --learning_rate $LR \
    --evaluation_strategy steps \
    --metric_for_best_model stsb_spearman \
    --load_best_model_at_end \
    --eval_steps 125 \
    --overwrite_output_dir \
    --logging_first_step \
    --logging_dir ./outputs_seed103 \
    --do_train \
    --do_eval \
    --fp16 \
    --seed 103 \
    > ./results/diffcse_seed103.out &
