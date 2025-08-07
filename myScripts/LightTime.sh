#LightTime
files=(1)
# 遍历文件列表
for file in "${files[@]}"; do
  # 构造文件路径
  data_path="Turb${file}.csv"
  model_id="Wind_LightTime_432_36_Turb${file}"
  # 打印当前处理的文件
  echo "Processing file: $data_path"
  echo "Model id: ${model_id}"
  # 运行 Python 脚本

  cmd="python -u run.py --is_training 1 --root_path ./dataset/processed --data_path ${data_path} --model_id ${model_id} --model LightTime\
    --task_name long_term_forecast --data custom --features MS --target Wspd --seq_len 432 --label_len 0 --pred_len 36 --enc_in 10 --dec_in 10 --c_out 1\
    --d_model 120 --n_heads 16  --patch_len 108 --strid 54 --dropout 0 --train_epochs 40 --patience 5 --loss mae --use_gpu True --inverse --itr 1 --batch_size 64\
    --learning_rate 0.001 --moving_avg 72 --noEx"

  # 定义日志文件路径
  log_file="logs/LongForecasting/LightTime/${model_id}.log"
  # 如果日志文件存在，则删除它
  if [ -f "$log_file" ]; then
    rm "$log_file"
    echo "Removed existing log file: $log_file"
  fi

  # 打印分隔符、命令以及再次打印分隔符到日志文件
  echo "========================================================================" >>$log_file
  echo "Executing command at $(date):" >>$log_file
  echo "$cmd" >>$log_file
  echo "------------------------------------------------------------------------" >>$log_file

  # 执行命令并将输出追加到日志文件
  $cmd >>$log_file 2>&1

  # 在日志文件中记录命令结束的时间戳
  echo "Command finished at $(date)" >>$log_file
  echo "========================================================================" >>$log_file
done
echo "All files processed successfully!"
