#!/bin/bash
# LightST: Light Spatio-Temporal 风电场多风机风速预测
# 三阶段渐进训练: 时序 → 图学习 → 联合微调

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LightST" ]; then
    mkdir ./logs/LightST
fi

model_name=LightST

root_path_name=./dataset/spatial_wind
data_path_name=spatial_wind
model_id_name=LightST_wind

seq_len=432
pred_len=36
label_len=18

for d_model in 512
do
for n_heads in 8
do
for e_layers in 2
do

cmd="python -u run.py \
  --task_name lightst \
  --is_training 1 \
  --root_path $root_path_name \
  --data_path $data_path_name \
  --model_id ${model_id_name}_${seq_len}_${pred_len} \
  --model $model_name \
  --data spatial_wind \
  --features MS \
  --target Wspd \
  --freq 10min \
  --seq_len $seq_len \
  --label_len $label_len \
  --pred_len $pred_len \
  --enc_in 10 \
  --dec_in 10 \
  --c_out 1 \
  --d_model $d_model \
  --n_heads $n_heads \
  --e_layers $e_layers \
  --d_ff 2048 \
  --dropout 0.1 \
  --batch_size 4 \
  --learning_rate 0.001 \
  --train_epochs 40 \
  --patience 10 \
  --loss MSE \
  --use_gpu True \
  --gpu 0 \
  --n_nodes 134 \
  --n_clusters 8 \
  --gnn_layers 2 \
  --stage1_epochs 15 \
  --stage2_epochs 15 \
  --stage3_epochs 10 \
  --static_graph_dir ./dataset/static_graphs \
  --positions_path ./dataset/spatial_wind/positions.npy \
  --inverse"

log_file="logs/LightST/${model_name}_${model_id_name}_${seq_len}_${pred_len}_${d_model}_${n_heads}_${e_layers}.log"
if [ -f "$log_file" ]; then
  rm "$log_file"
fi

echo "========================================================================" >> $log_file
echo "Executing command at $(date):" >> $log_file
echo "$cmd" >> $log_file
echo "------------------------------------------------------------------------" >> $log_file

$cmd >> $log_file 2>&1

echo "Command finished at $(date)" >> $log_file
echo "========================================================================" >> $log_file

done
done
done
