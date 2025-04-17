python -u run.py --is_training 1 --root_path ./dataset/short_term_forecast/m4 --model_id M4_LightTime \
--model LightTime --task_name short_term_forecast --data m4 \
--d_model 80 --n_heads 64  --patch_len 8  --strid 2 --dropout 0 --moving_avg 13 \
--train_epochs 40 --patience 10 --loss mse --use_gpu True \
--itr 1 --batch_size 64 --learning_rate 0.01  --noEx --loss "SMAPE" --enc_in 1 --dec_in 1 --c_out 1 \
--seasonal_patterns 'Yearly' --lradj cosine

#16
# python -u run.py --is_training 1 --root_path ./dataset/short_term_forecast/m4 --model_id M4_LightTime \
# --model LightTime --task_name short_term_forecast --data m4 \
# --d_model 64 --n_heads 32  --patch_len 2  --strid 2 --dropout 0 --moving_avg 17 \
# --train_epochs 40 --patience 5 --loss mse --use_gpu True \
# --itr 1 --batch_size 32 --learning_rate 0.01  --noEx --loss "SMAPE" --enc_in 1 --dec_in 1 --c_out 1 \
# --seasonal_patterns 'Quarterly' --lradj cosine


# python -u run.py --is_training 1 --root_path ./dataset/short_term_forecast/m4 --model_id M4_LightTime \
# --model LightTime --task_name short_term_forecast --data m4 \
# --d_model 64 --n_heads 32  --patch_len 2  --strid 2 --dropout 0 --moving_avg 17 \
# --train_epochs 40 --patience 5 --loss mse --use_gpu True \
# --itr 1 --batch_size 32 --learning_rate 0.01  --noEx --loss "SMAPE" --enc_in 1 --dec_in 1 --c_out 1 \
# --seasonal_patterns 'Monthly' --lradj cosine


# python -u run.py --is_training 1 --root_path ./dataset/short_term_forecast/m4 --model_id M4_LightTime \
# --model LightTime --task_name short_term_forecast --data m4 \
# --d_model 64 --n_heads 32  --patch_len 2  --strid 2 --dropout 0 --moving_avg 17 \
# --train_epochs 40 --patience 5 --loss mse --use_gpu True \
# --itr 1 --batch_size 32 --learning_rate 0.01  --noEx --loss "SMAPE" --enc_in 1 --dec_in 1 --c_out 1 \
# --seasonal_patterns 'Weekly' --lradj cosine

# python -u run.py --is_training 1 --root_path ./dataset/short_term_forecast/m4 --model_id M4_LightTime \
# --model LightTime --task_name short_term_forecast --data m4 \
# --d_model 64 --n_heads 32  --patch_len 2  --strid 2 --dropout 0 --moving_avg 17 \
# --train_epochs 40 --patience 5 --loss mse --use_gpu True \
# --itr 1 --batch_size 32 --learning_rate 0.01  --noEx --loss "SMAPE" --enc_in 1 --dec_in 1 --c_out 1 \
# --seasonal_patterns 'Daily' --lradj cosine

# python -u run.py --is_training 1 --root_path ./dataset/short_term_forecast/m4 --model_id M4_LightTime \
# --model LightTime --task_name short_term_forecast --data m4 \
# --d_model 64 --n_heads 32  --patch_len 2  --strid 2 --dropout 0 --moving_avg 17 \
# --train_epochs 40 --patience 5 --loss mse --use_gpu True \
# --itr 1 --batch_size 32 --learning_rate 0.01  --noEx --loss "SMAPE" --enc_in 1 --dec_in 1 --c_out 1 \
# --seasonal_patterns 'Hourly' --lradj cosine