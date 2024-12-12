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
        self.dropout = configs.dropout
        self.d_ff = configs.d_ff
        self.n_heads = configs.n_heads

        self.stride = configs.stride
        self.patch_len = configs.patch_len
        self.padding = self.stride
        self.patch_num = (self.seq_len+self.stride*2 - self.patch_len) // self.stride

        self.short_len = 36

        if self.task_name == 'classification' or self.task_name == 'anomaly_detection' or self.task_name == 'imputation':
            self.pred_len = configs.seq_len
            raise Exception("暂未支持 分类/异常值检测/插值 等功能")
        else:
            self.pred_len = configs.pred_len
        
        # 趋势分解
        self.decompsition = series_decomp(configs.moving_avg)

        self.trend_seasonal_weight = nn.Parameter(torch.ones([2])*0.5)

        # Patch前增加一个一维卷积
        self.seasonal_conv = nn.Conv1d(1,1,9,1,4)
        
        self.trend_conv = nn.Conv1d(1,1,9,1,4)

        # Patch
        self.seasonal_patch_embedding = PatchEmbedding(self.d_model // 4 * 3, self.patch_len, self.stride, self.padding, configs.dropout)
        
        self.trend_patch_embedding= PatchEmbedding(self.d_model // 4 * 3, self.patch_len, self.stride, self.padding, configs.dropout)
        

        # 外生变量的运算
        self.ex_varMix = nn.Linear(self.enc_in-1,self.d_ff)

        self.ex_timeShrink = nn.Linear(self.short_len//3, self.n_heads)

        self.ex_dModel = nn.Linear(self.d_ff,self.d_model // 4)

        self.ex_batchNormal = nn.BatchNorm1d(self.n_heads)

        

        
        # 线性运算
        self.Linear_Time = nn.ModuleList()

        self.Linear_Time.append(nn.Linear(self.patch_num,configs.n_heads))

        self.Linear_Time.append(nn.LeakyReLU())


        self.Linear_Time.append(nn.BatchNorm1d(self.d_model // 4 * 3))

        self.Linear_Time.append(nn.Dropout(configs.dropout))
        
        self.Linear_Time.append(nn.LeakyReLU())
    
        # Decoder
        self.decoder_TimeExpend = nn.Linear(self.n_heads,self.pred_len)

        self.decoder_batchNormal = nn.BatchNorm1d(self.d_model)

        self.decoder_varShrink = nn.Linear(self.d_model, self.c_out)

    def x_ex_encoder(self,x_ex):
        # [batch_size, seq_len, enc_in-1]

        x_ex = x_ex[:,-self.short_len:,:]
        #[batch_size, short_len, enc_in-1]

        ex_out = self.ex_varMix(x_ex)
        ex_out = F.leaky_relu(ex_out)
        #[batch_size, short_len, d_ff]

        ex_out = ex_out[:,::3,:].permute(0,2,1)
        ex_out = self.ex_timeShrink(ex_out)
        ex_out = F.leaky_relu(ex_out)
        ex_out = ex_out.permute(0,2,1)
        #[batch_size, n_heads, d_ff]

        ex_out = self.ex_dModel(ex_out)
        ex_out = self.ex_batchNormal(ex_out)
        ex_out = F.dropout(ex_out,self.dropout)
        ex_out = F.leaky_relu(ex_out)
        #[batch_size, n_heads, d_model / 4]

        return ex_out

    def x_en_encoder(self,x_en):
        # [batch_size, seq_len, 1]

        # 趋势分解
        seasonal, trend = self.decompsition(x_en)
        
        # Patch 和 Embeding
        trend_out = trend.permute(0,2,1)
        trend_out = self.trend_conv(trend_out)
        trend_out,n_vars = self.trend_patch_embedding(trend_out)
        # [batch_size, patch_num, d_model / 4 * 3]

        seasonal_out = seasonal.permute(0, 2, 1)
        seasonal_out = self.seasonal_conv(seasonal_out)
        seasonal_out, n_vars = self.seasonal_patch_embedding(seasonal_out)
        # [batch_size, patch_num, d_model / 4 * 3]

        enc_out = trend_out * self.trend_seasonal_weight[0] + seasonal_out * self.trend_seasonal_weight[1]

        enc_out = enc_out.permute(0, 2, 1)
        # [batch_size, d_model / 4 * 3, patch_num]

        for layer in self.Linear_Time:
            enc_out = layer(enc_out)
        enc_out = enc_out.permute(0, 2, 1)
        # [batch_size, n_heads, d_model / 4 * 3]
        
        return enc_out

    def encoder(self, x):
        # Step1: 区分External和Endogenous变量
        x_en = x[:,:,0:1]
        x_ex = x[:,:,1:]

        # Step2: 处理External变量
        ex_out = self.x_ex_encoder(x_ex)
        
        # Step3: 处理Endogenous变量
        en_out = self.x_en_encoder(x_en)

        # Step4: External和Endogenous交互
        enc_out = torch.cat([en_out,ex_out],dim=-1)
        # [batch_size, n_heads, d_model]
        
        return enc_out


    def decoder(self, enc_out):
        # [batch_size, n_heads, d_model]
        dec_out = enc_out.permute(0,2,1)
        dec_out = self.decoder_TimeExpend(dec_out)
        dec_out = self.decoder_batchNormal(dec_out)
        dec_out = F.dropout(dec_out,self.dropout)
        dec_out = F.leaky_relu(dec_out)
        dec_out = dec_out.permute(0,2,1)
        # [batch_size, pred_len, d_model]

        dec_out = self.decoder_varShrink(dec_out)

        # [batch_size, pred_len, c_out]
        return dec_out

    def forecast(self, x_enc):


        # region Step1: Normalization
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev
        # endregion
        # [batch_size, seq_len, enc_in]

        # Step2: Encoder
        enc_out = self.encoder(x_enc)

        # Step3: Decoder
        dec_out = self.decoder(enc_out)
        # [batch_size, nvars, pred_len, c_out]]
        
        # region Step4: De-Normalization from Non-stationary Transformer
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        #endregion
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        else: 
            raise Exception("暂未支持 分类/异常值检测/插值 等功能")
