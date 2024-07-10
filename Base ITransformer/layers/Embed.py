import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import math

class DataEmbedding_inverted_own_BiLSTM(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.1, is_positional_embedding=True):
        super(DataEmbedding_inverted_own_BiLSTM, self).__init__()
        self.lstm = nn.LSTM(c_in, d_model // 2, batch_first=True, bidirectional=True)  # 使用双向LSTM
        self.dropout = nn.Dropout(p=dropout)
        self.is_positional_embedding = is_positional_embedding
        if is_positional_embedding:
            self.position_embedding = nn.Parameter(torch.zeros(9, d_model))
            self.position_weight = nn.Parameter(torch.ones(1, 37, d_model))  # 可训练权重
            self._init_position_embedding(d_model)

    def _init_position_embedding(self, d_model):
        index_order = [
            [2, 1, 2],
            [1, 0, 1],
            [2, 1, 2]
        ]
        with torch.no_grad():
            for i in range(3):
                for j in range(3):
                    pos_val = index_order[i][j] * d_model // 2 + 1
                    pos = i * 3 + j
                    for k in range(d_model // 2):
                        self.position_embedding[pos, 2 * k] = math.sin(pos_val / 10000 ** (2 * k / d_model))
                        self.position_embedding[pos, 2 * k + 1] = math.cos(pos_val / 10000 ** (2 * k / d_model))

    def forward(self, x, x_mark):
        x = x.permute(0, 2, 1)  # [Batch, Variate, Time]
        if x_mark is not None:
            x = torch.cat([x, x_mark.permute(0, 2, 1)], 1)

        x, _ = self.lstm(x)  

        if self.is_positional_embedding:
            # Add position encoding to the embedding
            pos_embedding = self.position_embedding.unsqueeze(0).repeat(x.size(0), 1, 1).to(x.device)
            pos_embedding = torch.cat((pos_embedding, pos_embedding, pos_embedding, pos_embedding, pos_embedding[:, 4, :].unsqueeze(1)), dim=1)
            
            # Apply position weights
            pos_embedding = pos_embedding * self.position_weight

            x = x + pos_embedding

        return self.dropout(x)


class DataEmbedding_inverted_own_BiGRU(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.1, is_positional_embedding=True):
        super(DataEmbedding_inverted_own_BiGRU, self).__init__()
        self.gru = nn.GRU(c_in, d_model // 2, batch_first=True, bidirectional=True)  # 使用双向GRU
        self.dropout = nn.Dropout(p=dropout)
        self.is_positional_embedding = is_positional_embedding
        if is_positional_embedding:
            self.position_embedding = nn.Parameter(torch.zeros(9, d_model))
            self.position_weight = nn.Parameter(torch.ones(1, 37, d_model))  # 可训练权重
            self._init_position_embedding(d_model)

    def _init_position_embedding(self, d_model):
        index_order = [
            [2, 1, 2],
            [1, 0, 1],
            [2, 1, 2]
        ]
        with torch.no_grad():
            for i in range(3):
                for j in range(3):
                    pos_val = index_order[i][j] * d_model // 2 + 1
                    pos = i * 3 + j
                    for k in range(d_model // 2):
                        self.position_embedding[pos, 2 * k] = math.sin(pos_val / 10000 ** (2 * k / d_model))
                        self.position_embedding[pos, 2 * k + 1] = math.cos(pos_val / 10000 ** (2 * k / d_model))

    def forward(self, x, x_mark):
        x = x.permute(0, 2, 1)  # [Batch, Variate, Time]
        if x_mark is not None:
            x = torch.cat([x, x_mark.permute(0, 2, 1)], 1)

        x, _ = self.gru(x)  
        if self.is_positional_embedding:
            # Add position encoding to the embedding
            pos_embedding = self.position_embedding.unsqueeze(0).repeat(x.size(0), 1, 1).to(x.device)
            pos_embedding = torch.cat((pos_embedding, pos_embedding, pos_embedding, pos_embedding, pos_embedding[:, 4, :].unsqueeze(1)), dim=1)

            # Apply position weights
            pos_embedding = pos_embedding * self.position_weight
            
            x = x + pos_embedding

        return self.dropout(x)


class DataEmbedding_inverted_own(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.1):
        super(DataEmbedding_inverted_own, self).__init__()
        self.value_embedding = nn.Linear(c_in, d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.position_embedding = nn.Parameter(torch.zeros(9, d_model))
        self._init_position_embedding(d_model)

    def _init_position_embedding(self, d_model):
        index_order = [
            [2, 1, 2],
            [1, 0, 1],
            [2, 1, 2]
        ]
        with torch.no_grad():
            for i in range(3):
                for j in range(3):
                    pos_val = index_order[i][j]*d_model//2 + 1
                    pos = i*3 + j
                    for k in range(d_model // 2):
                        self.position_embedding[pos, 2 * k] = math.sin(pos_val / 10000 ** (2 * k / d_model))
                        self.position_embedding[pos, 2 * k + 1] = math.cos(pos_val / 10000 ** (2 * k / d_model))

    def forward(self, x, x_mark):
        x = x.permute(0, 2, 1)  # [Batch, Variate, Time]
        if x_mark is not None:
            x = torch.cat([x, x_mark.permute(0, 2, 1)], 1)
        x = self.value_embedding(x)  # [Batch, Variate, d_model]

        # Add position encoding to the embedding
        pos_embedding = self.position_embedding.unsqueeze(0).repeat(x.size(0), 1, 1).to(x.device)
        pos_embedding = torch.cat((pos_embedding,pos_embedding,pos_embedding,pos_embedding,pos_embedding[:,4,:].unsqueeze(1)),dim=1)
        x = x + pos_embedding

        return self.dropout(x)
    
class DataEmbedding_inverted(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.1, positional_encoding=False):
        super(DataEmbedding_inverted, self).__init__()
        self.value_embedding = nn.Linear(c_in, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = x.permute(0, 2, 1)
        # x: [Batch Variate Time]
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            x = self.value_embedding(torch.cat([x, x_mark.permute(0, 2, 1)], 1))
        # x: [Batch Variate d_model]
        return self.dropout(x)
