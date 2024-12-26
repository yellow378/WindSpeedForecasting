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
        self.isEx = configs.isEx

        self.stride = configs.stride
        self.patch_len = configs.patch_len
        self.padding = self.stride
        self.patch_num = (self.seq_len+self.stride*2 - self.patch_len) // self.stride

        if self.task_name == 'classification' or self.task_name == 'anomaly_detection' or self.task_name == 'imputation':
            self.pred_len = configs.seq_len
            raise Exception("暂未支持 分类/异常值检测/插值 等功能")
        else:
            self.pred_len = configs.pred_len
        
        # region 对于内生变量
        # 趋势分解
        self.decompsition = series_decomp(configs.moving_avg)

        # Patch前增加一个一维卷积
        # self.seasonal_conv = nn.Conv2d(1,1,[5,5],[1,1],[2,2])
        # self.trend_conv = nn.Conv2d(1,1,[5,5],[1,1],[2,2])

        # Patch
        self.seasonal_patch_embedding = PatchEmbedding(self.d_model, self.patch_len, self.stride, self.padding, configs.dropout)
        self.trend_patch_embedding= PatchEmbedding(self.d_model, self.patch_len, self.stride, self.padding, configs.dropout)
        
        # 线性运算
        self.en_Linear_Time = nn.ModuleList()
        self.en_Linear_Time.append(nn.Linear(self.patch_num,configs.n_heads))
        self.en_Linear_Time.append(nn.BatchNorm1d(self.d_model))
        self.en_Linear_Time.append(nn.Dropout(configs.dropout))
        self.en_Linear_Time.append(nn.Sigmoid())
        # endregion



        # region 对于外生变量
        if(self.isEx):
            self.ex_batchNormal = nn.BatchNorm1d(self.enc_in-1)
            self.ex_varMix = nn.Linear(self.enc_in-1,self.n_heads)
            self.ex_timeShrink = nn.Linear(self.seq_len, self.n_heads)
        # endregion

        # Decoder
        self.decoder_TimeExpend = nn.Linear(self.n_heads,self.pred_len)
        self.decoder_batchNormal = nn.BatchNorm1d(self.d_model)
        self.decoder_varShrink = nn.Linear(self.d_model, self.c_out)

    def x_ex_encoder(self,x_ex):
        # [batch_size, seq_len, enc_in-1]

        x_ex = x_ex.permute(0,2,1)
        x_ex = self.ex_batchNormal(x_ex).permute(0,2,1)
        ex_out = self.ex_varMix(x_ex)
        ex_out = F.sigmoid(ex_out)
        #[batch_size, seq_len, h_heads]

        ex_out = ex_out.permute(0,2,1)
        ex_out = self.ex_timeShrink(ex_out)
        ex_out = F.dropout(ex_out,self.dropout)
        ex_out = F.sigmoid(ex_out)
        ex_out = ex_out.permute(0,2,1)
        #[batch_size, n_heads, h_heads]

        return ex_out

    def x_en_encoder(self,x_en):
        # [batch_size, seq_len, 1]

        # 趋势分解
        seasonal, trend = self.decompsition(x_en)
        
        # Patch 和 Embeding
        trend_out = trend.permute(0,2,1)
        trend_out,n_vars = self.trend_patch_embedding(trend_out)
        #trend_out = trend_out.unsqueeze(1)
        #trend_out = self.trend_conv(trend_out).squeeze(1)
        # [batch_size, patch_num, d_model]

        seasonal_out = seasonal.permute(0, 2, 1)
        seasonal_out, n_vars = self.seasonal_patch_embedding(seasonal_out)
        #seasonal_out = seasonal_out.unsqueeze(1)
        #seasonal_out = self.seasonal_conv(seasonal_out).squeeze(1)
        # [batch_size, patch_num, d_model]
        enc_out = trend_out  + seasonal_out

        enc_out = enc_out.permute(0, 2, 1)
        # [batch_size, d_model, patch_num]

        for layer in self.en_Linear_Time:
            enc_out = layer(enc_out)
        enc_out = enc_out.permute(0, 2, 1)
        # [batch_size, n_heads, d_model]
        
        return enc_out

    def encoder(self, x_en, x_ex):

        # Step1: 处理External变量
        if self.isEx:
            ex_out = self.x_ex_encoder(x_ex)
        # [batch_size, n_heads, d_model]
        
        # Step2: 处理Endogenous变量
        en_out = self.x_en_encoder(x_en)

        # Step3: External和Endogenous融合
        if self.isEx:
            enc_out = torch.bmm(en_out.permute(0,2,1),ex_out) + en_out.permute(0,2,1)
            enc_out = enc_out.permute(0,2,1)
        else:
            enc_out = en_out
        # [batch_size, n_heads, d_model]
        
        return enc_out


    def decoder(self, enc_out):
        # [batch_size, n_heads, d_model]
        dec_out = enc_out.permute(0,2,1)
        dec_out = self.decoder_TimeExpend(dec_out)
        dec_out = self.decoder_batchNormal(dec_out)
        dec_out = F.dropout(dec_out,self.dropout)
        dec_out = F.sigmoid(dec_out)
        dec_out = dec_out.permute(0,2,1)
        # [batch_size, pred_len, d_model]
        dec_out = self.decoder_varShrink(dec_out)

        # [batch_size, pred_len, c_out]
        return dec_out

    def forecast(self, x_enc):
        # [batch_size, seq_len, enc_in]

        # region Step2: Normalization
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev
        # endregion

        # Step1: 区分External和Endogenous变量
        x_en = x_enc[:,:,-1:]
        x_ex = x_enc[:,:,0:-1]

        # Step2: Encoder
        enc_out = self.encoder(x_en, x_ex)

        # Step3: Decoder
        dec_out = self.decoder(enc_out)
        # [batch_size, pred_len, c_out]
        # region Step4: De-Normalization from Non-stationary Transformer
        #print(stdev[:, 0, 0].unsqueeze(1).unsqueeze(2).repeat(1,self.pred_len,1))#.unsqueeze(1).unsqueeze(2).repeat(1,self.pred_len,1).shape)
        dec_out = dec_out * \
                  (stdev[:, 0, -1].unsqueeze(1).unsqueeze(2).repeat(1,self.pred_len,1))
        dec_out = dec_out + \
                  (means[:, 0, -1].unsqueeze(1).unsqueeze(2).repeat(1,self.pred_len,1))
        #endregion
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        else: 
            raise Exception("暂未支持 分类/异常值检测/插值 等功能")
