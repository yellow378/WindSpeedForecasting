import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Embed import PatchEmbedding
from layers.Autoformer_EncDec import series_decomp

class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.d_model = configs.d_model
        self.enc_in = configs.enc_in
        self.c_out = configs.c_out

        self.stride = configs.stride
        self.patch_len = configs.patch_len
        self.padding = self.stride
        self.patch_num = (self.seq_len+self.stride*2 - self.patch_len) // self.stride

        if self.task_name == 'classification' or self.task_name == 'anomaly_detection' or self.task_name == 'imputation':
            self.pred_len = configs.seq_len
            raise Exception("暂未支持 分类/异常值检测/插值 等功能")
        else:
            self.pred_len = configs.pred_len
        
        # 趋势分解
        self.decompsition = series_decomp(configs.moving_avg)


        # Patch前增加一个一维卷积
        self.conv = nn.Conv1d(1,1,9,1,4)
        self.conv_trend = nn.Conv1d(1,1,9,1,4)

        # Patch
        self.patch_embedding = PatchEmbedding(self.d_model, self.patch_len, self.stride, self.padding, configs.dropout)
        self.patch_embedding_trend = PatchEmbedding(self.d_model, self.patch_len, self.stride, self.padding, configs.dropout)
        

        # TODO: 外生变量的运算

        
        # 线性运算
        self.Linear_Time = nn.ModuleList()
        # [batch_size * enc_in, patch_num, d_model]
        # permute [batch_size * enc_in, d_model, patch_num]
        self.Linear_Time.append(nn.Linear(self.patch_num,configs.n_heads))
        # [batch_size * enc_in, d_model, n_heads]
        self.Linear_Time.append(nn.LeakyReLU())
        self.Linear_Time.append(nn.Linear(configs.n_heads,self.pred_len))
        # [batch_size * enc_in, d_model, pred_len]
        self.Linear_Time.append(nn.BatchNorm1d(self.d_model))
        self.Linear_Time.append(nn.Dropout(configs.dropout))
        self.Linear_Time.append(nn.LeakyReLU())
    
        
        self.Linear_Decoder = nn.ModuleList()
        # [batch_size, pred_len, d_model]
        self.Linear_Decoder.append(nn.Linear(self.d_model, self.c_out))
        #self.Linear_Decoder.append(nn.BatchNorm2d(self.pred_len))

    def encoder(self, x):
        # [batch_size * enc_in, patch_num, d_model]
        out = x.permute(0, 2, 1)
        # print(out.shape)
        # print(self.patch_num)
        for layer in self.Linear_Time:
            out = layer(out)
        out = out.permute(0, 2, 1)

        # [batch_size * enc_in, pred_len, d_model]
        return out


    def decoder(self, x):
        # [batch_size, pred_len, 2]
        for layer in self.Linear_Decoder:
            #print(x.shape)
            x = layer(x)

        # [batch_size, pred_len, c_out]
        return x

    def forecast(self, x_enc):
        # [batch_size, seq_len, enc_in]
        
        

        # 区分External和Endogenous变量
        x_en = x_enc[:,:,0:1]
        x_ex = x_enc[:,:,1:]

        # Normalization from Non-stationary Transformer
        means = x_en.mean(1, keepdim=True).detach()
        x_en = x_en - means
        stdev = torch.sqrt(
            torch.var(x_en, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_en /= stdev

        # 趋势分解
        x_enc, trend = self.decompsition(x_en)
        
         # do patching and embedding
        trend = trend.permute(0,2,1)
        trend_out = self.conv_trend(trend)
        trend_out,n_vars = self.patch_embedding_trend(trend_out)

        x_enc = x_enc.permute(0, 2, 1)
        enc_out = self.conv(x_enc)
        enc_out, n_vars = self.patch_embedding(enc_out)
        # [batch_sizes * enc_in, patch_num, d_model]

        enc_out = enc_out + trend_out

        # Encoder
        enc_out = self.encoder(enc_out)
        # [batch_size * enc_in, pred_len, d_model]
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        # [batch_size, nvars, pred_len, d_model]]

        # Decoder
        dec_out = self.decoder(enc_out)
        # [batch_size, nvars, pred_len, c_out]]

        dec_out = dec_out.reshape(dec_out.shape[0],dec_out.shape[1],dec_out.shape[2])
        dec_out = dec_out.permute(0,2,1)
        
        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        else: 
            raise Exception("暂未支持 分类/异常值检测/插值 等功能")
        return None
