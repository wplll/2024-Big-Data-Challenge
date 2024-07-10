# from data_provider.data_loader import Dataset_Meteorology
# from torch.utils.data import DataLoader

# data_dict = {
#     'Meteorology' : Dataset_Meteorology
# }


# def data_provider(args):
#     Data = data_dict[args.data]

#     shuffle_flag = True
#     drop_last = False
#     batch_size = args.batch_size 

#     data_set = Data(
#         root_path=args.root_path,
#         data_path=args.data_path,
#         size=[args.seq_len, args.label_len, args.pred_len],
#         features=args.features,
#         debug=args.debug,
#     )
#     data_loader = DataLoader(
#         data_set,
#         batch_size=batch_size,
#         shuffle=shuffle_flag,
#         num_workers=args.num_workers,
#         drop_last=drop_last)
#     return data_set, data_loader

import gc
from data_provider.data_loader import Dataset_Meteorology
from torch.utils.data import DataLoader
import random

data_dict = {
    'Meteorology' : Dataset_Meteorology
}

def split_stations(station_num, val_ratio=0.05, seed=42):
    random.seed(seed)  
    stations = list(range(station_num))
    val_size = int(station_num * val_ratio)
    val_stations = random.sample(stations, val_size)
    train_stations = [s for s in stations if s not in val_stations]
    return train_stations, val_stations

def data_provider(args):
    Data = data_dict[args.data]

    shuffle_flag = True
    drop_last = False
    batch_size = args.batch_size 
    if args.val_ratio > 0.0:
        train_stations, val_stations = split_stations(3850, val_ratio=args.val_ratio, seed=args.seed)
    else:
        train_stations = list(range(3850))
        val_stations = None
    print("train dataset: ")
    train_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        debug=args.debug,
        selected_stations=train_stations,
        is_augment=True,
        augment_prob=args.augment_prob
    )

    if val_stations is not None:
        print("val dataset: ")
        val_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            debug=args.debug,
            selected_stations=val_stations,
            is_augment=False
        )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)

    if val_stations is not None:
        val_loader = DataLoader(
            val_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            drop_last=drop_last)
    print('>>>>>>>start has load data >>>>>>>>>>>>>>>>>>>>>>>>>>')
    if val_stations is not None:
        return train_set, train_loader, val_set, val_loader
    else:
        return train_set, train_loader
