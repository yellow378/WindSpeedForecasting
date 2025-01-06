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
        self.noEx = configs.noEx

        self.stride = configs.stride
        self.patch_len = configs.patch_len
        self.padding = self.stride
        self.patch_num = (self.seq_len+self.stride*2 - self.patch_len) // self.stride

        if self.task_name == 'classification' or self.task_name == 'anomaly_detection' or self.task_name == 'imputation':
            self.pred_len = configs.seq_len
            raise Exception("暂未支持 分类/异常值检测/插值 等功能")
        else:
            self.pred_len = configs.pred_len

        self.en_conv = nn.Conv1d(1,1,9,1,4)
        self.ex_conv = nn.Conv1d(9,9,9,1,4)
        # Patch
        self.en_patch_embedding = PatchEmbedding(self.d_model, self.patch_len, self.stride, self.padding, self.dropout)
        if(not self.noEx):
            self.ex_patch_embedding = PatchEmbedding(self.d_model, self.patch_len, self.stride, self.padding, self.dropout)

        # 线性运算
        if self.noEx:
            self.linear1 = nn.Linear(self.patch_num, self.n_heads)
        else:
            self.linear1 = nn.Linear(self.patch_num + (self.enc_in - 1)*2, self.n_heads)
        self.linear2 = nn.Linear(self.n_heads, self.n_heads)
        #self.en_Linear_Time.append(nn.Dropout(self.dropout))

        # Decoder
        self.decoder_TimeExpend = nn.Linear(self.n_heads,self.pred_len)
        #self.decoder_batchNormal = nn.BatchNorm1d(self.d_model)
        self.decoder_varShrink = nn.Linear(self.d_model, self.c_out)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        else: 
            raise Exception("暂未支持 分类/异常值检测/插值 等功能")

    def forecast(self, x_enc):
        # [batch_size, seq_len, enc_in]

        # region Step2: Normalization
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev
        # endregion

        # Step2: Encoder
        enc_out = self.encoder(x_enc)

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
    def encoder(self, x_enc):
        # Step1: 区分External和Endogenous变量
        x_en = x_enc[:,:,-1:]
        x_ex = x_enc[:,:,0:-1]

        # Step1: 处理External变量
        if not self.noEx:
            ex_out,n_vars = self.x_ex_encoder(x_ex)
        # [batch_size, n_heads, d_model]
        
        # Step2: 处理Endogenous变量
        en_out = self.x_en_encoder(x_en)

        # Step3: External和Endogenous融合
        if not self.noEx:
            enc_out = torch.cat([ex_out,en_out],dim=-1)
            # [batch_size, d_model, patch_num+enc_in-1]
        else:
            enc_out = en_out
             # [batch_size, d_model, patch_num]
        #enc_out = enc_out.permute(0,2,1)

        enc_out = self.linear1(enc_out)
        enc_out = F.relu(enc_out)
        enc_out = self.linear2(enc_out)
        enc_out = F.dropout(enc_out,self.dropout)
        enc_out = F.relu(enc_out)
        # [batch_size, n_heads, d_model]
        return enc_out

    def x_ex_encoder(self,x_ex):
        # [batch_size, patch_len, enc_in-1]
        # [batch_size, patch_len, enc_in-1]
        x_ex = x_ex.permute(0,2,1)
        x_ex  = self.ex_conv(x_ex)
        x_ex = x_ex[:,:,-self.patch_len:]
        ex_out,n_vars = self.ex_patch_embedding(x_ex)
        #[batch_size*n_vars, d_model, 1]
        batch_size = x_ex.size(0)
        ex_out = ex_out.reshape([batch_size,self.d_model,n_vars*2])
        # [batch_size,d_model,n_vars]
        return ex_out, n_vars

    def x_en_encoder(self,x_en):
        # [batch_size, seq_len, 1]
        # Patch 和 Embeding
        x_en = x_en.permute(0,2,1)
        x_en_out = self.en_conv(x_en)
        x_en_out,n_vars = self.en_patch_embedding(x_en)
        x_en_out = x_en_out.permute(0, 2, 1)
        # [batch_size, d_model, patch_num]  
        return x_en_out

    def decoder(self, enc_out):
        # [batch_size, n_heads, d_model]
        #dec_out = enc_out.permute(0,2,1)
        dec_out = self.decoder_TimeExpend(enc_out)
        dec_out = F.dropout(dec_out,self.dropout)
        dec_out = F.relu(dec_out)
        dec_out = dec_out.permute(0,2,1)
        # [batch_size, pred_len, d_model]
        dec_out = self.decoder_varShrink(dec_out)
        # [batch_size, pred_len, c_out]
        return dec_out