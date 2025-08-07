import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from models import LightTime

class GAT(torch.nn.Module):
    """
    时空图注意力网络，结合LightTime的编码器和解码器
    """
    def __init__(self, configs):
        super(GAT, self).__init__()
        
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.n_nodes = configs.n_nodes
        self.heads = configs.heads
        self.dropout = configs.dropout
        
        # 初始化LightTime模型
        from models import LightTime
        self.lightTime = LightTime.Model(configs).float()
        
        # GAT的输入通道数来自LightTime编码器的输出
        self.in_channels = self.lightTime.n_heads * self.lightTime.d_model  # 保留完整时间信息
        
        # GAT层 - 用于空间建模
        self.gat = GATConv(
            in_channels=self.in_channels, 
            out_channels=self.in_channels,
            heads=self.heads, 
            dropout=0, 
            concat=False
        )
        
        # 额外的线性层用于GAT后的处理
        self.gat_norm = torch.nn.LayerNorm(self.in_channels)
        self.gat_dropout = torch.nn.Dropout(self.dropout)
        
        # 时间注意力池化层（可选）
        self.temporal_attention = torch.nn.MultiheadAttention(
            embed_dim=self.lightTime.d_model,
            num_heads=4,
            dropout=self.dropout,
            batch_first=True
        )
        
        # 时间卷积层（可选）
        self.temporal_conv = torch.nn.Conv1d(
            in_channels=self.lightTime.d_model,
            out_channels=self.lightTime.d_model,
            kernel_size=3,
            padding=1
        )
        
        # 用于将GAT输出转换为decoder输入的投影层
        self.spatial_projection = torch.nn.Linear(self.in_channels, self.lightTime.n_heads * self.lightTime.d_model)

    def forward(self, data, device=None):
        """
        前向传播
        :param data: 包含x和edge_index的数据对象
                    x: [batch_size, n_nodes, seq_len, enc_in]
                    edge_index: [2, num_edges]
        :param device: 设备
        """
        x, edge_index = data.x, data.edge_index
        batch_size, n_nodes, seq_len, enc_in = x.shape
        
        # 重塑输入以适应LightTime编码器
        # [batch_size*n_nodes, seq_len, enc_in]
        x_reshaped = x.reshape(batch_size * n_nodes, seq_len, enc_in)
        
        # Step 1: 使用LightTime的时间编码器处理每个节点
        temporal_features = self.encode_temporal_features(x_reshaped)
        # [batch_size*n_nodes, d_model]
        
        # Step 2: 为GAT准备批处理
        # 创建批处理边索引
        batch_edge_index = self.create_batch_edge_index(edge_index, batch_size, n_nodes, device or x.device)
        
        # Step 3: GAT空间建模
        spatial_features = self.gat(temporal_features, batch_edge_index)
        spatial_features = self.gat_norm(spatial_features)
        spatial_features = self.gat_dropout(spatial_features)
        
        # Step 4: 重新整形并投影到decoder输入维度
        # [batch_size, n_nodes, n_heads * d_model]
        spatial_features = spatial_features.reshape(batch_size, n_nodes, self.in_channels)
        
        # 在节点维度上聚合(可以选择不同的聚合方式)
        # 这里使用平均聚合，你也可以使用求和、最大值等
        aggregated_features = spatial_features.mean(dim=1)  # [batch_size, n_heads * d_model]
        
        # 投影到decoder期望的维度并重塑
        decoder_input = self.spatial_projection(aggregated_features)  # [batch_size, n_heads * d_model]
        decoder_input = decoder_input.reshape(batch_size, self.lightTime.n_heads, self.lightTime.d_model)
        
        # Step 5: 使用LightTime的解码器
        prediction = self.lightTime.decoder(decoder_input)
        
        return prediction

    def encode_temporal_features(self, x):
        """
        使用LightTime编码器提取时间特征
        :param x: [batch_size*n_nodes, seq_len, enc_in]
        :return: [batch_size*n_nodes, n_heads * d_model] 保留完整时间信息
        """
        # Step1: 标准化
        means = x.mean(1, keepdim=True).detach()
        x_normalized = x - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_normalized /= stdev
        
        # Step2: 使用LightTime编码器
        enc_out = self.lightTime.encoder(x_normalized)
        # enc_out: [batch_size*n_nodes, n_heads, d_model]
        
        # 保留完整时间信息的几种方案:
        
        # 方案1: 展平保留所有信息
        temporal_features = enc_out.reshape(enc_out.shape[0], -1)  # [batch_size*n_nodes, n_heads * d_model]
        
        # 方案2: 使用可学习的时间注意力机制
        # temporal_features = self.temporal_attention_pooling(enc_out)
        
        # 方案3: 使用时间卷积进一步处理
        # temporal_features = self.temporal_conv(enc_out)
        
        return temporal_features

    def create_batch_edge_index(self, edge_index, batch_size, n_nodes, device):
        """
        为批处理创建边索引
        :param edge_index: [2, num_edges] 单个图的边索引
        :param batch_size: 批处理大小
        :param n_nodes: 节点数量
        :param device: 设备
        :return: 批处理的边索引
        """
        batch_edge_indices = []
        
        for i in range(batch_size):
            # 为每个批次添加节点偏移
            offset_edge_index = edge_index + i * n_nodes
            batch_edge_indices.append(offset_edge_index)
        
        # 合并所有批次的边索引
        batch_edge_index = torch.cat(batch_edge_indices, dim=1)
        return batch_edge_index.to(device)