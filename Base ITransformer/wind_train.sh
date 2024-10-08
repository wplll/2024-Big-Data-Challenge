export CUDA_VISIBLE_DEVICES=1

model_name=iTransformer

python -u train.py 
  --root_path ./dataset/global 
  --data_path wind.npy 
  --model_id v1 
  --model iTransformer 
  --data Meteorology 
  --features MS 
  --seq_len 168 
  --label_len 1 
  --pred_len 24 
  --e_layers 1 
  --enc_in 37 
  --d_model 64 
  --d_ff 64 
  --n_heads 1 
  --des 'global_wind' 
  --learning_rate 0.01 
  --batch_size 65536 
  --train_epochs 4