from collections import OrderedDict
import os
from models import iTransformer
import torch
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
    'd_model': 128,
    'n_heads': 8,
    'e_layers': 1,
    'd_ff': 512,
    'dropout': 0.1,
    'activation': 'gelu',
    'output_attention': False,
    'is_positional_embedding': False,
    'time_embedding': 'lstm',
    'mode':0,
    'use_mem':1,
    'mem_size':512
}


class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)
arg = Struct(**args)

if __name__ == '__main__':
    task = 'temp' # temp/wind
    checkpath = r'D:\CODE\game\大数据竞赛\code\baseline\baseline\checkpoints'
    path = fr'v1_iTransformer_Meteorology_ftMS_sl168_ll1_pl24_dm128_nh8_el1_df512_global_{task}'
    path = os.path.join(checkpath,path)
    model = iTransformer.Model(arg).cuda()
 
    model_paths = [os.path.join(path,i) for i in os.listdir(path) if int(i.split('_')[1]) > 1]
    if model_paths:
        bone_dict = model.state_dict()
        new_state_dict = OrderedDict()
        data_len=len(model_paths)
        for model_path in model_paths:
            state_dict = torch.load(model_path)
 
            for k, v in state_dict.items():
                head = k[:7]
                if head == 'module.':
                    tmp_name = k[7:]  # remove `module.`
                else:
                    tmp_name = k  # continue
                need_v = bone_dict[tmp_name]
 
                if tmp_name in new_state_dict:
                    new_state_dict[tmp_name] += v/data_len
                else:
                    new_state_dict[tmp_name] = v/data_len
        model.load_state_dict(new_state_dict, strict=False)
        torch.save(model.state_dict(), os.path.join(path,"finalweight.pth"))
 