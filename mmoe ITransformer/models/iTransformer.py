import torch
import torch.nn as nn
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted


class Model(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2310.06625
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        # Embedding
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
        self.norm_layer = torch.nn.LayerNorm(configs.d_model)
        self.temp_encode = nn.GRU(configs.d_model, configs.d_model//2, batch_first=True, bidirectional=True, dropout=configs.dropout)
        self.wind_encode = nn.GRU(configs.d_model, configs.d_model//2, batch_first=True, bidirectional=True, dropout=configs.dropout)
        # Decoder
        self.temp_projection = nn.Linear(configs.d_model, configs.d_model, bias=True)
        self.wind_projection = nn.Linear(configs.d_model, configs.d_model, bias=True)
        self.temp_decoder = nn.GRU(configs.d_model, configs.pred_len, batch_first=True, bidirectional=False)
        self.wind_decoder = nn.GRU(configs.d_model, configs.pred_len, batch_first=True, bidirectional=False)
        

    def forecast(self, x_enc):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        _, _, N = x_enc.shape

        # Embedding
        enc_out = self.enc_embedding(x_enc, None)

        # Encoder
        temp,_ = self.temp_encode(enc_out[:,-2:-1,:])
        wind,_ = self.wind_encode(enc_out[:,-1:,:])
        temp = self.norm_layer(temp)
        wind = self.norm_layer(wind)

        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        enc_out[:,-2:-1,:] += temp
        enc_out[:,-1:,:] += wind
        
        # Decode
        dec_out_temp = self.temp_projection(enc_out)
        dec_out_wind = self.wind_projection(enc_out)
        out_temp = self.temp_decoder(dec_out_temp)[0].permute(0, 2, 1)[:, :, :N]
        out_wind = self.wind_decoder(dec_out_wind)[0].permute(0, 2, 1)[:, :, :N]

        # De-Normalization from Non-stationary Transformer
        out_temp = out_temp * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        out_temp = out_temp + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        out_wind = out_wind * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        out_wind = out_wind + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return out_temp,out_wind

    def forward(self, x_enc):
        temp,wind = self.forecast(x_enc)
        return temp[:, -self.pred_len:, -2:-1],wind[:, -self.pred_len:, -1:]  # [B, L, C]