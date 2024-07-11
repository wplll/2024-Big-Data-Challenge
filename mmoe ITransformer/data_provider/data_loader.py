import os
import numpy as np
from torch.utils.data import Dataset
import warnings

warnings.filterwarnings('ignore')

def reduce_mem_usage_np(arr, verbose=False):
    start_mem = arr.nbytes / 1024**2  # 计算初始内存使用情况
    
    if arr.dtype in [np.int16, np.int32, np.int64, np.float16, np.float32, np.float64]:
        c_min = arr.min()
        c_max = arr.max()
        
        if np.issubdtype(arr.dtype, np.integer):
            if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                arr = arr.astype(np.int8)
            elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                arr = arr.astype(np.int16)
            elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                arr = arr.astype(np.int32)
            elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                arr = arr.astype(np.int64)
        else:
            if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                arr = arr.astype(np.float16)
            elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                arr = arr.astype(np.float32)
            else:
                arr = arr.astype(np.float64)

    end_mem = arr.nbytes / 1024**2  # 计算最终内存使用情况
    if verbose: 
        print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return arr

from tqdm import tqdm

class Dataset_Meteorology(Dataset):
    def __init__(self, root_path, data_path, size=None, features='MS'):
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]

        self.features = features
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()
        self.stations_num = self.full_data.shape[-1]
        self.tot_len = len(self.full_data) - self.seq_len - self.pred_len + 1

    def __read_data__(self):
        data_temp = np.load(os.path.join(self.root_path, 'temp.npy')) # (T, S, 1)    
        data_temp = np.squeeze(data_temp) # (T, S)
        data_wind = np.load(os.path.join(self.root_path, 'wind.npy')) # (T, S, 1)    
        data_wind = np.squeeze(data_wind) # (T, S)
        era5 = np.load(os.path.join(self.root_path, 'global_data.npy')) 
        repeat_era5 = np.repeat(era5, 3, axis=0)[:len(data_temp), :, :, :] # (T, 4, 9, S)
        repeat_era5 = repeat_era5.reshape(repeat_era5.shape[0], -1, repeat_era5.shape[3]) # (T, 36, S)
        data_temp = reduce_mem_usage_np(data_temp)
        data_wind = reduce_mem_usage_np(data_wind)
        repeat_era5 = reduce_mem_usage_np(repeat_era5)
        # Pre-concatenate covariates with data
        self.full_data = []
        for station_id in tqdm(range(data_temp.shape[1]),desc="Pre-concatenating data"):
            station_data_y_temp = data_temp[:, station_id:station_id+1]
            station_data_y_wind = data_wind[:, station_id:station_id+1]
            station_covariate = repeat_era5[:, :, station_id:station_id+1].squeeze()

            concatenated = np.concatenate([station_covariate, station_data_y_temp, station_data_y_wind], axis=1)

            self.full_data.append(concatenated)
        
        self.full_data = np.stack(self.full_data, axis=2) # (T, 38, S)

    def __getitem__(self, index):
        station_id = index // self.tot_len
        s_begin = index % self.tot_len
        
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        
        seq_x = self.full_data[s_begin:s_end, :, station_id] # (seq_len, 38)
        seq_y = self.full_data[r_begin:r_end, :, station_id] # (label_len + pred_len, 38)
        
        return seq_x, seq_y

    def __len__(self):
        return (len(self.full_data) - self.seq_len - self.pred_len + 1) * self.stations_num