import os
import numpy as np
import random
import torch
from models import iTransformer
from config import MockPath
import pandas as pd

def smooth_moving_average_centered(data, window_size=3):
    """平滑处理 - 居中滑动平均"""
    df = pd.DataFrame(data)
    smoothed_data = df.rolling(window=window_size, center=True, min_periods=1).mean().to_numpy()
    return smoothed_data

def weighted_moving_average_window_2(data, alpha=0.8):
    """窗口大小为2的加权滑动平均"""
    weighted_data = []
    for i in range(len(data)):
        if i == 0:
            weighted_data.append(data[i])
        else:
            weighted_avg = alpha * data[i] + (1 - alpha) * data[i-1]
            weighted_data.append(weighted_avg)
    return np.array(weighted_data)


def kalman_filter(data, process_variance=1e-5, measurement_variance=0.1**2):
    n_timesteps = len(data)
    posteri_estimate = np.zeros(n_timesteps)
    posteri_error_estimate = np.zeros(n_timesteps)
    priori_estimate = np.zeros(n_timesteps)
    priori_error_estimate = np.zeros(n_timesteps)
    blending_factor = np.zeros(n_timesteps)

    posteri_estimate[0] = data[0]
    posteri_error_estimate[0] = 1.0

    for t in range(1, n_timesteps):
        priori_estimate[t] = posteri_estimate[t-1]
        priori_error_estimate[t] = posteri_error_estimate[t-1] + process_variance
        
        blending_factor[t] = priori_error_estimate[t] / (priori_error_estimate[t] + measurement_variance)
        posteri_estimate[t] = priori_estimate[t] + blending_factor[t] * (data[t] - priori_estimate[t])
        posteri_error_estimate[t] = (1 - blending_factor[t]) * priori_error_estimate[t]
    
    return posteri_estimate

def bayesian_update(prior, likelihood, observation):
    posterior = prior * likelihood
    posterior /= posterior.sum()  # 归一化
    return posterior

def post_process_forecast(data, forecast):
    forecast =  np.concatenate([data.cpu().numpy(), forecast], axis=1)
    smoothed_forecast = np.array([weighted_moving_average_window_2(f) for f in forecast])
    # kalman_forecast = np.array([kalman_filter(f) for f in smoothed_forecast])
    return smoothed_forecast[:,1:,:]

args = {
    'model_id': 'v1',
    'model': 'iTransformer',
    'data': 'Meteorology',
    'features': 'M',
    'checkpoints': './checkpoints/',
    'seq_len': 168,
    'label_len': 1,
    'pred_len': 24,
    'enc_in': 37,
    'd_model': 64,
    'n_heads': 1,
    'e_layers': 1,
    'd_ff': 64,
    'dropout': 0.1,
    'activation': 'gelu',
    'output_attention': False,
    'is_positional_embedding': False,
    'time_embedding': None
}
args_1 = {
    'model_id': 'v1',
    'model': 'iTransformer',
    'data': 'Meteorology',
    'features': 'M',
    'checkpoints': './checkpoints/',
    'seq_len': 168,
    'label_len': 1,
    'pred_len': 24,
    'enc_in': 37,
    'd_model': 64,
    'n_heads': 8,
    'e_layers': 1,
    'd_ff': 512,
    'dropout': 0.1,
    'activation': 'gelu',
    'output_attention': False,
    'is_positional_embedding': False,
    'time_embedding': None
}
args_2 = {
    'model_id': 'v1',
    'model': 'iTransformer',
    'data': 'Meteorology',
    'features': 'M',
    'checkpoints': './checkpoints/',
    'seq_len': 168,
    'label_len': 1,
    'pred_len': 24,
    'enc_in': 37,
    'd_model': 64,
    'n_heads': 8,
    'e_layers': 1,
    'd_ff': 512,
    'dropout': 0.1,
    'activation': 'gelu',
    'output_attention': False,
    'is_positional_embedding': True,
    'time_embedding': None
}
args_3 = {
    'model_id': 'v1',
    'model': 'iTransformer',
    'data': 'Meteorology',
    'features': 'M',
    'checkpoints': './checkpoints/',
    'seq_len': 168,
    'label_len': 1,
    'pred_len': 24,
    'enc_in': 37,
    'd_model': 64,
    'n_heads': 8,
    'e_layers': 2,
    'd_ff': 64,
    'dropout': 0.1,
    'activation': 'gelu',
    'output_attention': False,
    'is_positional_embedding': True,
    'time_embedding': None
}
args_4 = {
    'model_id': 'v1',
    'model': 'iTransformer',
    'data': 'Meteorology',
    'features': 'M',
    'checkpoints': './checkpoints/',
    'seq_len': 168,
    'label_len': 1,
    'pred_len': 24,
    'enc_in': 37,
    'd_model': 64,
    'n_heads': 8,
    'e_layers': 1,
    'd_ff': 512,
    'dropout': 0.1,
    'activation': 'gelu',
    'output_attention': False,
    'is_positional_embedding': True,
    'time_embedding': 'gru'
}
args_5 = {
    'model_id': 'v1',
    'model': 'iTransformer',
    'data': 'Meteorology',
    'features': 'M',
    'checkpoints': './checkpoints/',
    'seq_len': 168,
    'label_len': 1,
    'pred_len': 24,
    'enc_in': 37,
    'd_model': 64,
    'n_heads': 8,
    'e_layers': 1,
    'd_ff': 512,
    'dropout': 0.1,
    'activation': 'gelu',
    'output_attention': False,
    'is_positional_embedding': True,
    'time_embedding': 'lstm'
}
args_6 = {
    'model_id': 'v1',
    'model': 'iTransformer',
    'data': 'Meteorology',
    'features': 'M',
    'checkpoints': './checkpoints/',
    'seq_len': 168,
    'label_len': 1,
    'pred_len': 24,
    'enc_in': 37,
    'd_model': 64,
    'n_heads': 8,
    'e_layers': 1,
    'd_ff': 512,
    'dropout': 0.1,
    'activation': 'gelu',
    'output_attention': False,
    'is_positional_embedding': False,
    'time_embedding': 'lstm'
}


class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)
arg = Struct(**args)
arg1 = Struct(**args_1)
arg2 = Struct(**args_2)
arg3 = Struct(**args_3)
arg4 = Struct(**args_4)
arg5 = Struct(**args_5)
arg6 = Struct(**args_6)

weights = {
    "arg":[2.5,3],
    "arg1":4,
    "arg2":3.5,
    "arg3":3,
    "arg4":1,
    "arg5":2,
}
sum_weights = 2.5+3+4+3.5+3+1+2

def invoke(inputs):
    cwd = os.path.dirname(inputs)
    save_path = '/home/mw/project'

    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)
    

    test_data_root_path = inputs
    batch_size = 8

    def load_models(path):
        models = []
        for k in os.listdir(path):
            print(k)
            model_path = os.path.join(path, k)

            arg_use = globals()[k.split('.')[0].split('_')[-1]]
            model = iTransformer.Model(arg_use).cuda()
            
            model.load_state_dict(torch.load(model_path))
            if  k.split('.')[0].split('_')[-1] == "arg":
                if k.split('.')[0].split('_')[1] == "0":
                    weight = weights[k.split('.')[0].split('_')[-1]][0]/sum_weights
                else:
                    weight = weights[k.split('.')[0].split('_')[-1]][1]/sum_weights
            
            else:
                weight = weights[k.split('.')[0].split('_')[-1]]/sum_weights

            models.append([model,weight])
        return models

    for i in range(2):
        if i == 0:
            data = np.load(os.path.join(test_data_root_path, "temp_lookback.npy")) # (N, L, S, 1)
            path = r"/home/mw/project/checkpoints/temp"
        else:
            data = np.load(os.path.join(test_data_root_path, "wind_lookback.npy"))
            path = r"/home/mw/project/checkpoints/wind"
        
        N, L, S, _ = data.shape # 72, 168, 60
        cenn_era5_data = np.load(os.path.join(test_data_root_path, "cenn_data.npy")) 
        repeat_era5 = np.repeat(cenn_era5_data, 3, axis=1) # (N, L, 4, 9, S)
        C1 = repeat_era5.shape[2] * repeat_era5.shape[3]
        covariate = repeat_era5.reshape(repeat_era5.shape[0], repeat_era5.shape[1], -1, repeat_era5.shape[4]) # (N, L, C1, S)
        data = data.transpose(0, 1, 3, 2) # (N, L, 1, S)
        C = C1 + 1
        data = np.concatenate([covariate, data], axis=2) # (N, L, C, S)
        data = data.transpose(0, 3, 1, 2) # (N, S, L, C)
        data = data.reshape(N * S, L, C)
        data = torch.tensor(data).float().cuda() # (N * S, L, C)

        models = load_models(path)

        num_batches = (N * S + batch_size - 1) // batch_size
        outputs = None

        for b in range(num_batches):
            batch_data = data[b * batch_size: (b + 1) * batch_size]
            batch_output = None
            for j, (model,weight) in enumerate(models):
                model_output = model(batch_data) * weight
                model_output = model_output[:, :, -1:].detach().cpu().numpy()
                if batch_output is None:
                    batch_output = model_output
                else:
                    batch_output += model_output
            # batch_output /= len(models)
            if outputs is None:
                outputs = batch_output
            else:
                outputs = np.concatenate((outputs, batch_output), axis=0)

            # Free up GPU memory
            del batch_data
            torch.cuda.empty_cache()

        outputs = post_process_forecast(data[:,-1:,-1:], outputs)
        P = outputs.shape[1]
        forecast = outputs.reshape(N, S, P, 1) # (71, 60, 24, 1)
        # for j in range(N):
        #     for k in range(S):
        #         forecast[j, k, : , : ] = smooth_moving_average(forecast[j, k, : ], window_size=3)
        forecast = forecast.transpose(0, 2, 1, 3) # (N, P, S, 1)
        
        if i == 0:
            np.save(os.path.join(save_path, "temp_predict.npy"), forecast)
        else:
            np.save(os.path.join(save_path, "wind_predict.npy"), forecast)
