import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import pandas as pd


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len

        self.d_model = configs.d_model
        self.ex_model = configs.d_model // 4
        self.model = (
            (configs.d_model * 2)
            if configs.noEx
            else (configs.d_model * 2 + self.ex_model)
        )
        #print(f"model: {self.model}")

        self.enc_in = configs.enc_in
        self.c_out = configs.c_out

        self.dropout = configs.dropout
        self.d_ff = configs.d_ff
        self.n_heads = configs.n_heads

        self.noEx = configs.noEx

        self.stride = configs.stride
        self.patch_len = configs.patch_len
        self.padding = self.stride
        self.patch_num = (
            self.seq_len + 1 + self.stride * 2 - self.patch_len
        ) // self.stride

        if (
            self.task_name == "classification"
            or self.task_name == "anomaly_detection"
            or self.task_name == "imputation"
        ):
            self.pred_len = configs.seq_len
            raise Exception("暂未支持 分类/异常值检测/插值 等功能")
        else:
            self.pred_len = configs.pred_len

        # 趋势分解
        self.decompsition = series_decomp(configs.moving_avg)

        # Patch
        self.trend_embedding = PatchEmbedding(
            self.d_model, self.patch_len, self.stride, self.padding, self.dropout
        )
        self.en_patch_embedding = PatchEmbedding(
            self.d_model, self.patch_len, self.stride, self.padding, self.dropout
        )
        self.trend_linear = nn.Linear(self.patch_num, self.n_heads)
        self.seasaonal_linear = nn.Linear(self.patch_num, self.n_heads)
        self.en_drop = nn.Dropout(self.dropout)

        if not self.noEx:
            self.ex_patch_embedding = PatchEmbedding(
                self.ex_model,
                self.patch_len,
                self.stride,
                self.padding,
                self.dropout,
            )
            self.ex_projection = nn.Linear(
                self.ex_model * (self.enc_in - 1), self.ex_model
            )
            self.ex_drop = nn.Dropout(self.dropout)
            self.ex_linear = nn.Linear(self.patch_num, self.n_heads)

        # 线性运算
        self.linear2 = nn.Linear(self.model, self.d_model)
        # self.linearBlocks = nn.Sequential(
        #     *[LinearBlock(self.d_model,self.n_heads,self.dropout) for _ in range(2)]
        # )

        # Decoder
        self.decoder_TimeExpend = nn.Linear(self.n_heads, self.pred_len)
        self.decoder_varShrink = nn.Linear((self.d_model), self.c_out)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if (
            self.task_name == "long_term_forecast"
            or self.task_name == "short_term_forecast"
        ):
            dec_out = self.forecast(x_enc)
            return dec_out[:, -self.pred_len :, :]  # [B, L, D]
        else:
            raise Exception("暂未支持 分类/异常值检测/插值 等功能")

    def forecast(self, x_enc):
        # [batch_size, seq_len, enc_in]

        # Step1: Normalization
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # Step2: Encoder
        enc_out = self.encoder(x_enc)

        # Step3: Decoder
        dec_out = self.decoder(enc_out)
        # [batch_size, pred_len, c_out]

        # Step4: De-Normalization from Non-stationary Transformer
        # print(stdev[:, 0, 0].unsqueeze(1).unsqueeze(2).repeat(1,self.pred_len,1))#.unsqueeze(1).unsqueeze(2).repeat(1,self.pred_len,1).shape)
        dec_out = dec_out * (
            stdev[:, 0, -1].unsqueeze(1).unsqueeze(2).repeat(1, self.pred_len, 1)
        )
        dec_out = dec_out + (
            means[:, 0, -1].unsqueeze(1).unsqueeze(2).repeat(1, self.pred_len, 1)
        )
        return dec_out

    def encoder(self, x_enc):
        # Step1: 区分External和Endogenous变量
        # [batch_size, enc_in, seq_len]
        x_en = x_enc[:, :, -1:]
        x_ex = x_enc[:, :, 0:-1]

        # Step2: 处理External变量
        # [batch_size,self.d_model*n_vars, patch_num]
        ex_out = None
        if not self.noEx:
            ex_out, n_vars = self.x_ex_encoder(x_ex)

        # Step3: 处理Endogenous变量
        # [batch_size, d_model, patch_num]
        en_out = self.x_en_encoder(x_en)

        # Step4: External和Endogenous融合
        if not self.noEx:
            enc_out = torch.cat([ex_out, en_out], dim=-1)
        else:
            enc_out = en_out

        # Step5: 线性运算
        enc_out = self.linear2(enc_out)
        enc_out = F.sigmoid(enc_out)
        enc_out - self.en_drop(enc_out)
        # [batch_size, n_heads, d_model]
        # enc_out = self.linearBlocks(enc_out)
        # [batch_size, n_heads, d_model]
        return enc_out

    def x_ex_encoder(self, x_ex):
        # [batch_size, patch_len, enc_in-1]
        x_ex = x_ex.permute(0, 2, 1)
        ex_out, n_vars = self.ex_patch_embedding(x_ex)
        # [batch_size*n_vars, patch_num, d_model]
        batch_size = x_ex.size(0)
        ex_out = ex_out.reshape([batch_size, n_vars, -1, self.ex_model]).permute(
            0, 2, 3, 1
        )
        ex_out = ex_out.reshape([batch_size, -1, self.ex_model * n_vars]).permute(
            0, 2, 1
        )
        ex_out = self.ex_linear(ex_out)
        ex_out = F.sigmoid(ex_out)
        # [batch_size,self.d_model*n_vars, n_heads]
        ex_out = self.ex_projection(ex_out.permute(0, 2, 1))
        ex_out = F.sigmoid(ex_out)
        ex_out = self.ex_drop(ex_out)
        return ex_out, self.ex_model * n_vars
    def x_en_encoder(self, x_en):
        # [batch_size, seq_len, 1]
        # Patch 和 Embeding
        original_sequence = x_en  # 保存分解之前的序列
        seasonal_init, trend_init = self.decompsition(x_en)  # 分解之后的两个序列
        # # 保存十组数据
        # if not hasattr(self, 'all_data'):
        #     self.all_data = {
        #         'original': [],
        #         'seasonal': [],
        #         'trend': []
        #     }

        # if len(self.all_data['trend'])  == 0:
        #     print(original_sequence[0:2].cpu().numpy().reshape(-1).tolist())
        #     self.all_data['original'].extend(original_sequence[0:1].cpu().numpy().reshape(-1).tolist())
        #     self.all_data['seasonal'].extend(seasonal_init[0:1].cpu().numpy().reshape(-1).tolist())
        #     self.all_data['trend'].extend(trend_init[0:1].cpu().numpy().reshape(-1).tolist())
        #     df = pd.DataFrame({
        #         'original': self.all_data['original'],
        #         'seasonal': self.all_data['seasonal'],
        #         'trend': self.all_data['trend']
        #     })
        #     df.to_csv('series.csv', index=False)
        #     print("Data saved successfully")

        seasonal_init, trend_init = (
            seasonal_init.permute(0, 2, 1),
            trend_init.permute(0, 2, 1),
        )

        trend_out, n_vars = self.trend_embedding(trend_init)
        trend_out = F.sigmoid(self.trend_linear(trend_out.permute(0, 2, 1))).permute(
            0, 2, 1
        )
        x_en_out, n_vars = self.en_patch_embedding(seasonal_init)
        x_en_out = F.sigmoid(self.seasaonal_linear(x_en_out.permute(0, 2, 1))).permute(
            0, 2, 1
        )
        x_en_out = torch.cat([trend_out, x_en_out], dim=-1)
        x_en_out = self.en_drop(x_en_out)

        return x_en_out

    def decoder(self, enc_out):
        # [batch_size, n_heads, d_model]
        dec_out = enc_out.permute(0, 2, 1)
        dec_out = self.decoder_TimeExpend(dec_out)
        dec_out = F.dropout(dec_out, self.dropout)
        dec_out = F.sigmoid(dec_out)
        dec_out = dec_out.permute(0, 2, 1)
        # [batch_size, pred_len, d_model]
        dec_out = dec_out  # + self.de_weights
        dec_out = self.decoder_varShrink(dec_out)
        # [batch_size, pred_len, c_out]
        return dec_out


class LinearBlock(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super(LinearBlock, self).__init__()
        self.linear1 = nn.Linear(d_model, d_model)
        self.linear2 = nn.Linear(n_heads, n_heads)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        out = self.linear1(x)
        out = F.sigmoid(out).permute(0, 2, 1)
        out = self.drop(out)
        out = self.linear2(out)
        out = F.sigmoid(out).permute(0, 2, 1)
        out = self.drop(out)
        out = self.norm(out)
        return out + x


class PatchEmbedding(nn.Module):
    def __init__(self, d_model, patch_len, stride, padding, dropout):
        super(PatchEmbedding, self).__init__()
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch_layer = nn.ReplicationPad1d((0, padding))
        self.dropout = dropout

        # Backbone, Input encoding: projection of feature vectors onto a d-dim vector space
        self.value_embedding1 = nn.Linear(patch_len, d_model, bias=True)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        n_vars = x.shape[1]
        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        # Input encoding
        x = self.value_embedding1(x)
        x = F.sigmoid(x)
        x = self.drop(x)
        return x, n_vars


class series_decomp(nn.Module):
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class moving_avg(nn.Module):
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        ).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return self.pe[:, : x.size(1)]


class LearnablePositionalEmbedding(nn.Module):
    def __init__(self, d_model):
        super(LearnablePositionalEmbedding, self).__init__()
        self.pe = nn.Parameter(torch.zeros(1, d_model))
        torch.nn.init.xavier_uniform_(self.pe)

    def forward(self, x):
        # x: [batch_size, d_model, n]
        return x + self.pe.unsqueeze(2).expand_as(x)
