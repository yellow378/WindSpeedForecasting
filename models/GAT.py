import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from models import LightTime


class GAT(torch.nn.Module):
    """
    时空图注意力网络，结合LightTime的编码器和解码器，支持多维边特征
    """
    def __init__(self, configs):
        super(GAT, self).__init__()
        
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.n_nodes = getattr(configs, 'n_nodes', 5)  # 默认值
        self.heads = getattr(configs, 'n_heads', 8)
        self.dropout = configs.dropout
        
        # 边特征维度配置
        self.static_edge_dim = getattr(configs, 'static_edge_dim', None)
        self.dynamic_edge_dim = 13 if configs.seq_len >= 3 else 11
        self.total_edge_dim = self.static_edge_dim + self.dynamic_edge_dim if self.static_edge_dim is not None else self.dynamic_edge_dim
        self.use_edge_features = getattr(configs, 'use_edge_features', True)
        if self.use_edge_features is None or self.use_edge_features is False:
            self.total_edge_dim = 0
        # 初始化LightTime模型
        self.lightTime = LightTime.Model(configs).float()
        
        # GAT的输入通道数来自LightTime编码器的输出
        self.in_channels = self.lightTime.n_heads * self.lightTime.d_model

        # 边特征预处理层
        if self.use_edge_features and self.total_edge_dim > 0:
            self.edge_preprocessing = torch.nn.Sequential(
                torch.nn.Linear(self.total_edge_dim, self.total_edge_dim),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.1),
                torch.nn.Linear(self.total_edge_dim, self.total_edge_dim // 2)
            )
        
        # 使用原生GATConv，支持多维边特征
        print(f"in_channels:{self.in_channels}, out_channels:{self.in_channels}, heads:{self.heads}")
        self.gat = GATConv(
            in_channels=self.in_channels, 
            out_channels=self.in_channels,
            heads=self.heads, 
            dropout=0, 
            concat=False,
            #edge_dim=self.total_edge_dim // 2  # 指定边特征维度
        )

        print("parameters number of GATConv:")
        print(sum(p.numel() for p in self.gat.parameters()))
        
        # 额外的线性层用于GAT后的处理
        self.gat_norm = torch.nn.LayerNorm(self.in_channels)
        self.gat_dropout = torch.nn.Dropout(self.dropout)
        
        # 用于将GAT输出转换为decoder输入的投影层
        self.spatial_projection = torch.nn.Linear(
            self.in_channels, 
            self.lightTime.n_heads * self.lightTime.d_model
        )
        
        # 最终输出投影层
        self.output_projection = torch.nn.Linear(
            self.lightTime.n_heads * self.lightTime.d_model,
            self.pred_len
        )
        
        print(f"动态GAT模型初始化完成:")
        print(f"  - 节点数: {self.n_nodes}")
        print(f"  - 注意力头数: {self.heads}")
        print(f"  - 静态边特征维度: {self.static_edge_dim}")
        print(f"  - 动态边特征维度: {self.dynamic_edge_dim}")
        print(f"  - 总边特征维度: {self.total_edge_dim}")

    def forward(self, data, device=None):
        """
        前向传播
        :param data: 包含x, edge_index和edge_attr的数据对象
                    x: [batch_size*n_nodes, seq_len, enc_in]
                    edge_index: [2, num_edges] - 边索引
                    edge_attr: [num_edges, static_edge_dim] - 多维边特征
        :param device: 设备
        """
        x, edge_index = data.x, data.edge_index
        edge_attr = data.edge_attr

        # 预处理边特征
        processed_edge_attr = self.edge_preprocessing(edge_attr)

        batch_size_nnodes, seq_len, enc_in = x.shape
        
        # Step 1: 使用LightTime的时间编码器处理每个节点
        temporal_features = self.encode_temporal_features(x)
        # [batch_size*n_nodes, n_heads * d_model]

        # Step 2: GAT空间建模（使用多维边特征）
        if self.use_edge_features and processed_edge_attr is not None:
            # 使用多维边特征
            spatial_features = self.gat(temporal_features, edge_index, edge_attr=processed_edge_attr)
        else:
            # 不使用边特征，只基于节点特征
            spatial_features = self.gat(temporal_features, edge_index)
        
        spatial_features = self.gat_norm(spatial_features)
        spatial_features = self.gat_dropout(spatial_features)
        
        # Step 3: 重新整形
        spatial_features = spatial_features.reshape(batch_size_nnodes, self.in_channels)
        
        # Step 4: 空间特征到时间预测的转换
        decoder_input = self.spatial_projection(spatial_features)
        decoder_input = decoder_input.reshape(batch_size_nnodes, self.lightTime.n_heads, self.lightTime.d_model)
        
        # Step 5: 使用LightTime的解码器
        node_pred = self.lightTime.decoder(decoder_input)
        if node_pred.shape[-1] == 1:
            node_pred = node_pred.squeeze(-1)

        return node_pred

    def encode_temporal_features(self, x):
        """
        使用LightTime编码器提取时间特征
        :param x: [batch_size*n_nodes, seq_len, enc_in]
        :return: [batch_size*n_nodes, n_heads * d_model]
        """
        # Step1: 标准化
        means = x.mean(1, keepdim=True).detach()
        x_normalized = x - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_normalized /= stdev
        
        # Step2: 使用LightTime编码器
        enc_out = self.lightTime.encoder(x_normalized)
        
        # 展平保留所有信息
        temporal_features = enc_out.reshape(enc_out.shape[0], -1)
        
        return temporal_features


class Model(GAT):
    """
    包装类，兼容现有框架的命名约定
    """
    def __init__(self, configs):
        super(Model, self).__init__(configs)