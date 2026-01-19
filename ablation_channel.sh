#!/bin/bash

# 循环执行命令的bash脚本
for num_heads in {0..20}
do
    echo "正在运行 num_heads=$num_heads"
    
    python main.py \
        --model "BrainVAE" \
        --num_repeat 3 \
        --dataset 'Dementia' \
        --data_dir "../data/Dementia200/Dementia200.npy" \
        --percentage 1. \
        --batch_size 8 \
        --num_epochs 300 \
        --drop_last False \
        --integration "add" \
        --cor_comput "pearson" \
        --d_model 64 \
        --window_size 50 \
        --window_stride 3 \
        --dynamic_length 440 \
        --num_heads "$num_heads" \
        --num_layers 1 \
        --schedule 'cos' \
        --learning_rate 1e-4 \
        --do_train \
        --do_evaluate \
        --do_test
        
    # 可选：添加延时以避免资源冲突
    sleep 3
    
    echo "num_heads=$num_heads 运行完成"
    echo "----------------------------------------"
done

echo "所有实验运行完成！"