from math import sqrt
import torch
import torch.nn as nn
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import *


class Model(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2310.06625
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.mode = configs.mode
        # Embedding
        self.time_embedding = False
        if configs.time_embedding in ["lstm", "gru"]:
            self.time_embedding = True
            if configs.time_embedding == "lstm":
                self.enc_embedding = DataEmbedding_inverted_own_BiLSTM(configs.seq_len, configs.d_model, configs.dropout, configs.is_positional_embedding)
            else:
                self.enc_embedding = DataEmbedding_inverted_own_BiGRU(configs.seq_len, configs.d_model, configs.dropout, configs.is_positional_embedding)
        else:
            if configs.is_positional_embedding:
                self.enc_embedding = DataEmbedding_inverted_own(configs.seq_len, configs.d_model, configs.dropout)
            else:
                self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.dropout)
        
        
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        # memory
        self.use_mem = False
        if configs.use_mem:
            self.use_mem = True
            self.mem = nn.Parameter(torch.FloatTensor(1, configs.d_model, configs.mem_size).normal_(0.0, 1.0))
            print(f"Using memory block ,the memory size is :{configs.mem_size}")

        # Decoder
        self.time_decode = False
        if configs.time_decode in ["lstm", "gru"]:
            self.time_decode = True
            if configs.time_embedding == "lstm":
                self.projection = nn.LSTM(configs.d_model, configs.pred_len, batch_first=True, bidirectional=False)
            else:
                self.projection = nn.GRU(configs.d_model, configs.pred_len, batch_first=True, bidirectional=False)
            if self.mode:
                self.projection_feature = nn.Linear(configs.enc_in, configs.label_len, bias=True)
        else:
            self.projection = nn.Linear(configs.d_model, configs.pred_len, bias=True)

    def forecast(self, x_enc):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        _, _, N = x_enc.shape

        # Embedding
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None) # bs,enc_in,d_model

        # mem block
        if self.use_mem:
            mem = self.mem.repeat(x_enc.shape[0], 1, 1) # (bs,d_model,mem_size)
            m_key = mem.transpose(1, 2) # (bs,mem_size,d_model)
            enc_out = enc_out.transpose(1, 2)
            logits = torch.bmm(m_key, enc_out) / sqrt(enc_out.shape[2])
            enc_out = torch.bmm(m_key.transpose(1, 2), F.softmax(logits, dim=1)) # bs,enc_in,d_model
            enc_out = enc_out.transpose(1, 2)

        # muti task 
        if self.mode:
            fusion = self.projection_feature(enc_out.permute(0, 2, 1))
            if self.time_embedding:
                att,_ = self.projection(fusion.permute(0, 2, 1))
            else:
                att = self.projection(fusion.permute(0, 2, 1))
            att = att * (stdev[:, 0, -1:].unsqueeze(2).repeat(1, 1, self.pred_len))
            att = att + (means[:, 0, -1:].unsqueeze(2).repeat(1, 1, self.pred_len))

        
        if self.time_decode:
            dec_out,_ = self.projection(enc_out)
        else:
            dec_out = self.projection(enc_out)
        dec_out = dec_out.permute(0, 2, 1)[:, :, :N]
        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        if self.mode:
            return dec_out,att
        else:
            return dec_out

    def forward(self, x_enc):
        if self.mode:
            dec_out,att_feature = self.forecast(x_enc)
            return dec_out[:, -self.pred_len:, :], att_feature[:, -self.pred_len:, :] # [B, L, C]
        else:
            dec_out = self.forecast(x_enc)
            return dec_out[:, -self.pred_len:, :]