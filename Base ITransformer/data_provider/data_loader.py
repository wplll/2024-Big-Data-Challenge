# import os
# import numpy as np
# from torch.utils.data import Dataset
# import warnings

# warnings.filterwarnings('ignore')

# class Dataset_Meteorology(Dataset):
#     def __init__(self, root_path, data_path, size=None, features='MS'):
#         self.seq_len = size[0]
#         self.label_len = size[1]
#         self.pred_len = size[2]

#         self.features = features
#         self.root_path = root_path
#         self.data_path = data_path
#         self.__read_data__()
#         self.stations_num = self.data_x.shape[-1]
#         self.tot_len = len(self.data_x) - self.seq_len - self.pred_len + 1

#     def __read_data__(self):
#         data = np.load(os.path.join(self.root_path, self.data_path)) # (T, S, 1)    
#         data = np.squeeze(data) # (T S)
#         era5 = np.load(os.path.join(self.root_path, 'global_data.npy')) 
#         repeat_era5 = np.repeat(era5, 3, axis=0)[:len(data), :, :, :] # (T, 4, 9, S)
#         repeat_era5 = repeat_era5.reshape(repeat_era5.shape[0], -1, repeat_era5.shape[3]) # (T, 36, S)

#         self.data_x = data
#         self.data_y = data
#         self.covariate = repeat_era5

#     def __getitem__(self, index):
#         station_id = index // self.tot_len
#         s_begin = index % self.tot_len
        
#         s_end = s_begin + self.seq_len
#         r_begin = s_end - self.label_len
#         r_end = r_begin + self.label_len + self.pred_len
#         seq_x = self.data_x[s_begin:s_end, station_id:station_id+1]
#         seq_y = self.data_y[r_begin:r_end, station_id:station_id+1] # (L 1)
#         t1 = self.covariate[s_begin:s_end, :, station_id:station_id+1].squeeze()
#         t2 = self.covariate[r_begin:r_end, :, station_id:station_id+1].squeeze()
#         seq_x = np.concatenate([t1, seq_x], axis=1) # (L 37)
#         seq_y = np.concatenate([t2, seq_y], axis=1) # (L 37)
#         return seq_x, seq_y

#     def __len__(self):
#         return (len(self.data_x) - self.seq_len - self.pred_len + 1) * self.stations_num

# import os
# import numpy as np
# from torch.utils.data import Dataset
# import warnings
# from tqdm import tqdm
# warnings.filterwarnings('ignore')

# class Dataset_Meteorology(Dataset):
#     def __init__(self, root_path, data_path, size=None, features='MS', debug=False):
#         self.seq_len = size[0]
#         self.label_len = size[1]
#         self.pred_len = size[2]

#         self.features = features
#         self.root_path = root_path
#         self.data_path = data_path
#         self.__read_data__(debug)
#         self.stations_num = self.data_x.shape[-1]
#         self.tot_len = len(self.data_x) - self.seq_len - self.pred_len + 1

#     def __read_data__(self,debug):
#         data = np.load(os.path.join(self.root_path, self.data_path)) # (T, S, 1)    
#         data = np.squeeze(data) # (T, S)
#         era5 = np.load(os.path.join(self.root_path, 'global_data.npy')) 
#         repeat_era5 = np.repeat(era5, 3, axis=0)[:len(data), :, :, :] # (T, 4, 9, S)
#         repeat_era5 = repeat_era5.reshape(repeat_era5.shape[0], -1, repeat_era5.shape[3]) # (T, 36, S)

#         self.data_x = data if not debug else data[:,0:2]
#         self.data_y = data if not debug else data[:,0:2]
#         self.covariate = repeat_era5 if not debug else repeat_era5[:,:,0:2]

#         # Pre-concatenate covariates with data
#         self.full_data_x = []
#         self.full_data_y = []
#         for station_id in tqdm(range(self.data_x.shape[1]),desc="Pre-concatenating data"):
#             station_data_x = self.data_x[:, station_id:station_id+1]
#             station_data_y = self.data_y[:, station_id:station_id+1]
#             station_covariate = self.covariate[:, :, station_id:station_id+1].squeeze()

#             concatenated_x = np.concatenate([station_covariate, station_data_x], axis=1)
#             concatenated_y = np.concatenate([station_covariate, station_data_y], axis=1)

#             self.full_data_x.append(concatenated_x)
#             self.full_data_y.append(concatenated_y)
        
#         self.full_data_x = np.stack(self.full_data_x, axis=2) # (T, 37, S)
#         self.full_data_y = np.stack(self.full_data_y, axis=2) # (T, 37, S)

#     def __getitem__(self, index):
#         station_id = index // self.tot_len
#         s_begin = index % self.tot_len
        
#         s_end = s_begin + self.seq_len
#         r_begin = s_end - self.label_len
#         r_end = r_begin + self.label_len + self.pred_len
        
#         seq_x = self.full_data_x[s_begin:s_end, :, station_id] # (seq_len, 37)
#         seq_y = self.full_data_y[r_begin:r_end, :, station_id] # (label_len + pred_len, 37)
        
#         return seq_x, seq_y

#     def __len__(self):
#         return (len(self.data_x) - self.seq_len - self.pred_len + 1) * self.stations_num

# import os
# import numpy as np
# from torch.utils.data import Dataset
# import warnings
# from tqdm import tqdm
# warnings.filterwarnings('ignore')

# class Dataset_Meteorology(Dataset):
#     def __init__(self, root_path, data_path, selected_stations, size=None, features='MS', debug=False):
#         self.seq_len = size[0]
#         self.label_len = size[1]
#         self.pred_len = size[2]

#         self.features = features
#         self.root_path = root_path
#         self.data_path = data_path
#         self.selected_stations = selected_stations
#         self.__read_data__(debug)
#         self.stations_num = len(self.selected_stations)
#         self.tot_len = len(self.full_data_x) - self.seq_len - self.pred_len + 1

#     def __read_data__(self, debug):
#         data = np.load(os.path.join(self.root_path, self.data_path)) # (T, S, 1)    
#         data = np.squeeze(data) # (T, S)
#         era5 = np.load(os.path.join(self.root_path, 'global_data.npy')) 
#         repeat_era5 = np.repeat(era5, 3, axis=0)[:len(data), :, :, :] # (T, 4, 9, S)
#         repeat_era5 = repeat_era5.reshape(repeat_era5.shape[0], -1, repeat_era5.shape[3]) # (T, 36, S)

#         selected_data_x = data[:, self.selected_stations]
#         selected_data_y = data[:, self.selected_stations]
#         selected_covariate = repeat_era5[:, :, self.selected_stations]

#         if debug:
#             selected_data_x = selected_data_x[:, :2]
#             selected_data_y = selected_data_y[:, :2]
#             selected_covariate = selected_covariate[:, :, :2]

#         self.full_data_x = []
#         self.full_data_y = []
#         for station_id in tqdm(range(len(self.selected_stations)), desc="Pre-concatenating data", leave=False):
#             station_data_x = selected_data_x[:, station_id:station_id+1]
#             station_data_y = selected_data_y[:, station_id:station_id+1]
#             station_covariate = selected_covariate[:, :, station_id:station_id+1].squeeze()

#             concatenated_x = np.concatenate([station_covariate, station_data_x], axis=1)
#             concatenated_y = np.concatenate([station_covariate, station_data_y], axis=1)

#             self.full_data_x.append(concatenated_x)
#             self.full_data_y.append(concatenated_y)

#         self.full_data_x = np.stack(self.full_data_x, axis=2) # (T, 37, S)
#         self.full_data_y = np.stack(self.full_data_y, axis=2) # (T, 37, S)

#     def __getitem__(self, index):
#         station_id = index // self.tot_len
#         s_begin = index % self.tot_len
        
#         s_end = s_begin + self.seq_len
#         r_begin = s_end - self.label_len
#         r_end = r_begin + self.label_len + self.pred_len
        
#         seq_x = self.full_data_x[s_begin:s_end, :, station_id] # (seq_len, 37)
#         seq_y = self.full_data_y[r_begin:r_end, :, station_id] # (label_len + pred_len, 37)
        
#         return seq_x, seq_y

#     def __len__(self):
#         return (len(self.full_data_x) - self.seq_len - self.pred_len + 1) * self.stations_num

import os
import numpy as np
from torch.utils.data import Dataset
import random
import warnings
from tqdm import tqdm
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

class Dataset_Meteorology(Dataset):
    def __init__(self, root_path, data_path, selected_stations, size=None, features='MS', debug=False, is_augment=False, augment_prob=1e-3):
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        self.is_augment = is_augment
        self.augment_prob = augment_prob
        print("is_augment:",self.is_augment)
        if self.is_augment:
            print("augment_prob:",self.augment_prob)

        self.features = features
        self.root_path = root_path
        self.data_path = data_path
        self.selected_stations = selected_stations
        self.__read_data__(debug)
        self.stations_num = len(self.selected_stations)
        self.tot_len = len(self.full_data_x) - self.seq_len - self.pred_len + 1

    def __read_data__(self, debug):
        data = np.load(os.path.join(self.root_path, self.data_path)) # (T, S, 1)
        data = np.squeeze(data) # (T, S)
        era5 = np.load(os.path.join(self.root_path, 'global_data.npy')) 
        repeat_era5 = np.repeat(era5, 3, axis=0)[:len(data), :, :, :] # (T, 4, 9, S)
        repeat_era5 = repeat_era5.reshape(repeat_era5.shape[0], -1, repeat_era5.shape[3]) # (T, 36, S)
        data = reduce_mem_usage_np(data)
        repeat_era5 = reduce_mem_usage_np(repeat_era5)
        selected_data_x = data[:, self.selected_stations]
        selected_data_y = data[:, self.selected_stations]
        selected_covariate = repeat_era5[:, :, self.selected_stations]
        
        if debug:
            selected_data_x = selected_data_x[:, :5]
            selected_data_y = selected_data_y[:, :5]
            selected_covariate = selected_covariate[:, :, :5]
            self.selected_stations = self.selected_stations[:5]

        self.full_data_x = []
        self.full_data_y = []
        for station_id in tqdm(range(len(self.selected_stations)), desc="Pre-concatenating data", leave=False):
            station_data_x = selected_data_x[:, station_id:station_id+1]
            station_data_y = selected_data_y[:, station_id:station_id+1]
            station_covariate = selected_covariate[:, :, station_id:station_id+1].squeeze()

            concatenated_x = np.concatenate([station_covariate, station_data_x], axis=1)
            concatenated_y = np.concatenate([station_covariate, station_data_y], axis=1)

            self.full_data_x.append(concatenated_x)
            self.full_data_y.append(concatenated_y)

        self.full_data_x = np.stack(self.full_data_x, axis=2) # (T, 37, S)
        self.full_data_y = np.stack(self.full_data_y, axis=2) # (T, 37, S)

    def data_augmentation(self, data):
        """根据概率选择数据增强方法"""
        augment_methods = [
            self.add_noise,
            self.time_shift,
            self.scale,
            self.time_masking,
            self.reverse,
            self.time_jitter,
            self.random_crop,
            self.smoothing,
            self.random_drop,
            self.mixup
        ]

        if np.random.rand() < self.augment_prob:
            data = random.choice(augment_methods)(data)

        return data

    def add_noise(self, data, noise_level=0.01):
        """加噪声"""
        noise = np.random.normal(0, noise_level, data.shape)
        return data + noise

    def time_shift(self, data, shift=5):
        """时间偏移"""
        shift = np.random.randint(-shift, shift)
        return np.roll(data, shift, axis=0)

    def scale(self, data, factor_range=(0.9, 1.1)):
        """缩放"""
        factor = np.random.uniform(*factor_range)
        return data * factor

    def time_masking(self, data, mask_ratio=0.1):
        """时间遮掩"""
        mask = np.random.binomial(1, mask_ratio, size=data.shape)
        return data * (1 - mask)

    def reverse(self, data):
        """反转"""
        return data[::-1].copy()  # 使用copy()避免负步长问题

    def time_jitter(self, data, jitter=0.01):
        """时间抖动"""
        jitter = np.random.normal(0, jitter, data.shape)
        return data + jitter

    def random_crop(self, data, crop_size=150):
        """随机裁剪"""
        if data.shape[0] > crop_size:
            start = np.random.randint(0, data.shape[0] - crop_size + 1)
            cropped_data = np.pad(data[start:start+crop_size], ((0, data.shape[0] - crop_size), (0, 0)), 'edge')
            return cropped_data
        else:
            return data

    def smoothing(self, data, window_size=5):
        """平滑"""
        window = np.ones(window_size) / window_size
        return np.apply_along_axis(lambda m: np.convolve(m, window, mode='same'), axis=0, arr=data)

    def random_drop(self, data, drop_ratio=0.1):
        """随机删除"""
        mask = np.random.binomial(1, drop_ratio, size=data.shape)
        data[mask == 1] = 0
        return np.nan_to_num(data)

    def mixup(self, data, alpha=0.2):
        """噪声混合"""
        lam = np.random.beta(alpha, alpha)
        index = np.random.permutation(data.shape[0])
        mixed_data = lam * data + (1 - lam) * data[index]
        return mixed_data

    def __getitem__(self, index):
        station_id = index // self.tot_len
        s_begin = index % self.tot_len
        
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        
        seq_x = self.full_data_x[s_begin:s_end, :, station_id] # (seq_len, 37)
        seq_y = self.full_data_y[r_begin:r_end, :, station_id] # (label_len + pred_len, 37)
        if self.is_augment:
            seq_x = self.data_augmentation(seq_x)
            seq_y = self.data_augmentation(seq_y)
        
        return seq_x, seq_y

    def __len__(self):
        return (len(self.full_data_x) - self.seq_len - self.pred_len + 1) * self.stations_num
