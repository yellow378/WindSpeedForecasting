#!/bin/bash

# 创建必要的目录
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi

model_name=LightTime

root_path_name=./dataset/processed
data_path_name=Turb1.csv
model_id_name=Wind_LightTime
data_name=custom

# seq_len patch_len e_layers d_model 
pred_len=36

for seq_len in 36 72 432
do  
  for patch_len in 18 36
  do
    stride=$((patch_len / 2))
    for e_layers in 1 2 3
    do 
      for d_model in 256 512 1024
      do
        for n_heads in 4 8 16 32
        do
          # 构建命令字符串
          cmd="python -u run.py \
            --is_training 1 \
            --root_path $root_path_name \
            --data_path $data_path_name \
            --model_id ${model_id_name}_${seq_len}_${pred_len} \
            --model $model_name \
            --task_name long_term_forecast \
            --data $data_name \
            --features MS \
            --target Wspd \
            --seq_len $seq_len \
            --label_len 18 \
            --pred_len $pred_len \
            --enc_in 10 \
            --dec_in 10 \
            --c_out 1 \
            --d_model $d_model \
            --n_heads $n_heads \
            --e_layers $e_layers \
            --patch_len $patch_len \
            --stride $stride \
            --dropout 0.1 \
            --train_epochs 40 \
            --patience 10 \
            --decomposition 1 \
            --loss mse \
            --use_gpu True \
            --inverse \
            --itr 1 --batch_size 256 --learning_rate 0.001"

          # 定义日志文件路径
          log_file="logs/LongForecasting/${model_name}_${model_id_name}_${seq_len}_${pred_len}_${d_model}_${n_heads}_${e_layers}_${patch_len}.log"
          # 如果日志文件存在，则删除它
          if [ -f "$log_file" ]; then
              rm "$log_file"
              echo "Removed existing log file: $log_file"
          fi

          # 打印分隔符、命令以及再次打印分隔符到日志文件
          echo "========================================================================" >> $log_file
          echo "Executing command at $(date):" >> $log_file
          echo "$cmd" >> $log_file
          echo "------------------------------------------------------------------------" >> $log_file

          # 执行命令并将输出追加到日志文件
          $cmd >> $log_file 2>&1

          # 在日志文件中记录命令结束的时间戳
          echo "Command finished at $(date)" >> $log_file
          echo "========================================================================" >> $log_file
        done
      done
    done
  done
done