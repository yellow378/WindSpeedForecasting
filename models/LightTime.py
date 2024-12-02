import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Autoformer_EncDec import series_decomp


class Model(nn.Module):

    def __init__(self, configs):
        """
        individual: Bool, whether shared model among different variates.
        """
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.d_model = configs.d_model
        if self.task_name == 'classification' or self.task_name == 'anomaly_detection' or self.task_name == 'imputation':
            self.pred_len = configs.seq_len
            raise Exception("暂未支持 分类/异常值检测/插值 等功能")
        else:
            self.pred_len = configs.pred_len
        # Series decomposition block from Autoformer
        # self.decompsition = series_decomp(configs.moving_avg)
        self.enc_in = configs.enc_in
        self.c_out = configs.c_out

        self.Linear_MultiVariable = nn.ModuleList()
        self.Linear_Time = nn.ModuleList()
        self.Linear_Cross = nn.ModuleList()
        self.Linear_Decoder = nn.ModuleList()

        # [batch_size, seq_len, enc_in]
        self.Linear_MultiVariable.append(nn.Linear(self.enc_in,self.d_model))
        # [[batch_size, seq_len, d_model]]
        self.Linear_MultiVariable.append(nn.BatchNorm1d(self.seq_len))
        self.Linear_MultiVariable.append(nn.LeakyReLU())
        

        # permute [batch_size, d_model, seq_len]
        self.Linear_Time.append(nn.Linear(self.seq_len,configs.n_heads//4))
        # [batch_size, d_model, configs.n_heads//4]
        self.Linear_Time.append(nn.LeakyReLU())
        self.Linear_Time.append(nn.Linear(configs.n_heads//4,configs.n_heads))
        # [batch_size, d_model, configs.n_heads]
        self.Linear_Time.append(nn.BatchNorm1d(self.d_model))
        self.Linear_Time.append(nn.Dropout(configs.dropout))
        self.Linear_Time.append(nn.LeakyReLU())
    
        # [batch_size, d_model, configs.n_heads]
        # reshape [batch_size, d_model*configs.n_heads]
        self.Linear_Cross.append(nn.Linear(self.d_model*configs.n_heads, self.pred_len*2))
        # [batch_size, self.pred_len*2]
        self.Linear_Cross.append(nn.Dropout(configs.dropout))
        self.Linear_Cross.append(nn.LeakyReLU())
        
        # [batch_size, pred_len, d_model]
        self.Linear_Decoder.append(nn.Linear(2, self.c_out))
        self.Linear_Decoder.append(nn.BatchNorm1d(self.pred_len))

    def encoder(self, x):
        # [batch_size, seq_len, enc_in]
        for layer in self.Linear_MultiVariable:
            x = layer(x)

        # [batch_size, seq_len, d_model]
        out = x.permute(0, 2, 1)
        for layer in self.Linear_Time:
            out = layer(out)
        out = out.permute(0, 2, 1)

        # [batch_size, d_model, d_model]
        originShape = out.shape
        out = out.reshape(out.shape[0], -1)
        for layer in self.Linear_Cross:
            out = layer(out)
        out = out.reshape(originShape[0],self.pred_len, 2)

        # [batch_size, pred_len, 2]
        return out

    def decoder(self, x):
        # [batch_size, pred_len, 2]
        for layer in self.Linear_Decoder:
            x = layer(x)

        # [batch_size, pred_len, c_out]
        return x

    def forecast(self, x_enc):
        # Encoder
        out = self.encoder(x_enc)
        # Decoder
        out = self.decoder(out)
        return out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        else: 
            raise Exception("暂未支持 分类/异常值检测/插值 等功能")
        return None
