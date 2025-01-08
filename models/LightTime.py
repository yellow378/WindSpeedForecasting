import torch
import torch.nn as nn
import torch.nn.functional as F

# from layers.Embed import PatchEmbedding
from layers.Autoformer_EncDec import series_decomp
import math


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.d_model = configs.d_model
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
            self.seq_len + self.stride * 2 - self.patch_len
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

        if not self.noEx:
            self.ex_conv = nn.Conv1d(9, 9, 9, 1, 4)

        # 趋势分解
        self.decompsition = series_decomp(configs.moving_avg)

        # Patch
        self.trend_embedding = PatchEmbedding(
            self.d_model, self.patch_len, self.stride, self.padding, self.dropout
        )
        self.en_patch_embedding = PatchEmbedding(
            self.d_model, self.patch_len, self.stride, self.padding, self.dropout
        )
        if not self.noEx:
            self.ex_patch_embedding = PatchEmbedding(
                self.d_model + 64,
                self.patch_len,
                self.stride,
                self.padding,
                self.dropout,
            )

        self.global_position_embedding = LearnablePositionalEmbedding(self.patch_num)
        # 线性运算
        if self.noEx:
            self.linear1 = nn.Linear(self.patch_num, self.n_heads)
        else:
            self.linear1 = nn.Linear(
                self.patch_num + (self.enc_in - 1) * 2, self.n_heads
            )

        # Decoder
        self.decoder_TimeExpend = nn.Linear(self.n_heads, self.pred_len)
        # self.decoder_batchNormal = nn.BatchNorm1d(self.d_model)
        self.decoder_varShrink = nn.Linear((self.d_model + self.d_model), self.c_out)

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

        # region Step1: Normalization
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev
        # endregion

        # Step2: Encoder
        enc_out = self.encoder(x_enc)

        # Step3: Decoder
        dec_out = self.decoder(enc_out)
        # [batch_size, pred_len, c_out]

        # region Step4: De-Normalization from Non-stationary Transformer
        # print(stdev[:, 0, 0].unsqueeze(1).unsqueeze(2).repeat(1,self.pred_len,1))#.unsqueeze(1).unsqueeze(2).repeat(1,self.pred_len,1).shape)
        dec_out = dec_out * (
            stdev[:, 0, -1].unsqueeze(1).unsqueeze(2).repeat(1, self.pred_len, 1)
        )
        dec_out = dec_out + (
            means[:, 0, -1].unsqueeze(1).unsqueeze(2).repeat(1, self.pred_len, 1)
        )
        # endregion
        return dec_out

    def encoder(self, x_enc):
        # Step1: 区分External和Endogenous变量
        x_en = x_enc[:, :, -1:]
        x_ex = x_enc[:, :, 0:-1]

        # Step1: 处理External变量
        if not self.noEx:
            ex_out, n_vars = self.x_ex_encoder(x_ex)
        # [batch_size, n_heads, d_model]

        # Step2: 处理Endogenous变量
        en_out = self.x_en_encoder(x_en)

        # Step3: External和Endogenous融合
        if not self.noEx:
            enc_out = torch.cat([ex_out, en_out], dim=-1)
            # [batch_size, d_model, patch_num+enc_in-1]
        else:
            enc_out = en_out
            # [batch_size, d_model, patch_num]
        # enc_out = enc_out.permute(0,2,1)

        enc_out = self.linear1(enc_out)
        enc_out = F.sigmoid(enc_out)
        # [batch_size, n_heads, d_model]
        return enc_out

    def x_ex_encoder(self, x_ex):
        # [batch_size, patch_len, enc_in-1]
        # [batch_size, patch_len, enc_in-1]
        x_ex = x_ex.permute(0, 2, 1)
        x_ex = self.ex_conv(x_ex)
        x_ex = x_ex[:, :, -self.patch_len :]
        ex_out, n_vars = self.ex_patch_embedding(x_ex)
        # [batch_size*n_vars, d_model, 1]
        batch_size = x_ex.size(0)
        ex_out = ex_out.reshape([batch_size, self.d_model + 64, n_vars * 2])
        # [batch_size,d_model,n_vars]
        return ex_out, n_vars

    def x_en_encoder(self, x_en):
        # [batch_size, seq_len, 1]
        # Patch 和 Embeding
        seasonal_init, trend_init = self.decompsition(x_en)
        seasonal_init, trend_init = (
            seasonal_init.permute(0, 2, 1),
            trend_init.permute(0, 2, 1),
        )

        trend_out, n_vars = self.trend_embedding(trend_init)
        trend_out = trend_out.permute(0, 2, 1)
        # [batch_size, 16, patch_num]

        x_en_out, n_vars = self.en_patch_embedding(seasonal_init)
        x_en_out = x_en_out.permute(0, 2, 1)
        # [batch_size, d_model, patch_num]

        x_en_out = torch.cat([x_en_out, trend_out], dim=1)
        x_en_out = x_en_out + self.global_position_embedding(x_en_out)

        return x_en_out

    def decoder(self, enc_out):
        # [batch_size, n_heads, d_model]
        # dec_out = enc_out.permute(0,2,1)
        dec_out = self.decoder_TimeExpend(enc_out)
        dec_out = F.dropout(dec_out, self.dropout)
        dec_out = F.sigmoid(dec_out)
        dec_out = dec_out.permute(0, 2, 1)
        # [batch_size, pred_len, d_model]
        dec_out = self.decoder_varShrink(dec_out)
        # [batch_size, pred_len, c_out]
        return dec_out


class PatchEmbedding(nn.Module):
    def __init__(self, d_model, patch_len, stride, padding, dropout):
        super(PatchEmbedding, self).__init__()
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch_layer = nn.ReplicationPad1d((0, padding))

        # Backbone, Input encoding: projection of feature vectors onto a d-dim vector space
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)

        # Positional embedding
        # self.position_embedding = PositionalEmbedding(d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # do patching
        n_vars = x.shape[1]
        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        # Input encoding
        x = self.value_embedding(x)  # + self.position_embedding(x)
        x = F.sigmoid(x)
        x = self.dropout(x)
        return x, n_vars


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
    def __init__(self, d_model, max_len=300):
        super(LearnablePositionalEmbedding, self).__init__()
        self.pe = nn.Parameter(torch.zeros(max_len, d_model))

    def forward(self, x):
        return self.pe[: x.size(1), :].unsqueeze(0)
