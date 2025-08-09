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
        self.n_nodes = getattr(configs, 'n_nodes', 5)  # 默认值
        self.heads = getattr(configs, 'n_heads', 8)
        self.dropout = configs.dropout
        
        # 初始化LightTime模型
        self.lightTime = LightTime.Model(configs).float()
        
        # GAT的输入通道数来自LightTime编码器的输出
        self.in_channels = self.lightTime.n_heads * self.lightTime.d_model
        
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

    def forward(self, data, device=None):
        """
        前向传播
        :param data: 包含x和edge_index的数据对象
                    x: [batch_size, n_nodes, seq_len, enc_in]
                    edge_index: [2, num_edges] - 单个图的边索引
        :param device: 设备
        """
        x, edge_index = data.x, data.edge_index
        batch_size, n_nodes, seq_len, enc_in = x.shape
        
        # 重塑输入以适应LightTime编码器
        # [batch_size*n_nodes, seq_len, enc_in]
        x_reshaped = x.reshape(batch_size * n_nodes, seq_len, enc_in)
        
        # Step 1: 使用LightTime的时间编码器处理每个节点
        temporal_features = self.encode_temporal_features(x_reshaped)
        # [batch_size*n_nodes, n_heads * d_model]
        
        # Step 2: 为GAT准备批处理 - 修复这里的边索引计算
        batch_edge_index = self.create_batch_edge_index_fixed(
            edge_index, batch_size, n_nodes, device or x.device
        )
        
        # Step 3: GAT空间建模
        spatial_features = self.gat(temporal_features, batch_edge_index)
        spatial_features = self.gat_norm(spatial_features)
        spatial_features = self.gat_dropout(spatial_features)
        
        # Step 4: 重新整形
        # [batch_size, n_nodes, in_channels]
        spatial_features = spatial_features.reshape(batch_size, n_nodes, self.in_channels)
        
        # Step 5: 空间特征到时间预测的转换
        # 方式1: 节点级别的预测（每个节点独立预测）
        node_predictions = []
        for node_idx in range(n_nodes):
            node_feature = spatial_features[:, node_idx, :]  # [batch_size, in_channels]
            
            # 投影到decoder输入维度
            decoder_input = self.spatial_projection(node_feature)  # [batch_size, n_heads * d_model]
            decoder_input = decoder_input.reshape(batch_size, self.lightTime.n_heads, self.lightTime.d_model)
            
            # 使用LightTime的解码器
            node_pred = self.lightTime.decoder(decoder_input)  # [batch_size, pred_len, c_out]
            node_predictions.append(node_pred)
        
        # 合并所有节点的预测
        # [batch_size, n_nodes, pred_len, c_out] -> [batch_size, n_nodes, pred_len]
        prediction = torch.stack(node_predictions, dim=1)
        if prediction.shape[-1] == 1:
            prediction = prediction.squeeze(-1)  # [batch_size, n_nodes, pred_len]
        
        return prediction

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
        # enc_out: [batch_size*n_nodes, n_heads, d_model]
        
        # 展平保留所有信息
        temporal_features = enc_out.reshape(enc_out.shape[0], -1)  # [batch_size*n_nodes, n_heads * d_model]
        
        return temporal_features

    def create_batch_edge_index_fixed(self, edge_index, batch_size, n_nodes, device):
        """
        修复版本：为批处理创建边索引
        :param edge_index: [2, num_edges] 单个图的边索引
        :param batch_size: 批处理大小
        :param n_nodes: 单个图中的节点数量
        :param device: 设备
        :return: 批处理的边索引
        """
        # 验证原始边索引的有效性
        if edge_index.max() >= n_nodes:
            print(f"Warning: edge_index contains invalid node indices. Max index: {edge_index.max()}, n_nodes: {n_nodes}")
            # 过滤无效的边
            valid_mask = (edge_index[0] < n_nodes) & (edge_index[1] < n_nodes)
            edge_index = edge_index[:, valid_mask]
            print(f"Filtered edge_index shape: {edge_index.shape}")
        
        if edge_index.shape[1] == 0:
            print("Warning: No valid edges found, creating self-loops")
            # 创建自环
            self_loops = torch.arange(n_nodes, device=device)
            edge_index = torch.stack([self_loops, self_loops], dim=0)
        
        batch_edge_indices = []
        
        for i in range(batch_size):
            # 为每个批次添加节点偏移
            offset_edge_index = edge_index + i * n_nodes
            batch_edge_indices.append(offset_edge_index)
        
        # 合并所有批次的边索引
        batch_edge_index = torch.cat(batch_edge_indices, dim=1)
        
        # 验证批处理边索引的有效性
        total_nodes = batch_size * n_nodes
        if batch_edge_index.max() >= total_nodes:
            print(f"Error: batch_edge_index max ({batch_edge_index.max()}) >= total_nodes ({total_nodes})")
            raise ValueError(f"Invalid batch edge index: max={batch_edge_index.max()}, total_nodes={total_nodes}")
        
        print(f"Created batch edge index: shape={batch_edge_index.shape}, max_idx={batch_edge_index.max()}, total_nodes={total_nodes}")
        return batch_edge_index.to(device)

    def create_batch_edge_index(self, edge_index, batch_size, n_nodes, device):
        """
        原始版本（保留作为备用）
        """
        batch_edge_indices = []
        
        for i in range(batch_size):
            # 为每个批次添加节点偏移
            offset_edge_index = edge_index + i * n_nodes
            batch_edge_indices.append(offset_edge_index)
        
        # 合并所有批次的边索引
        batch_edge_index = torch.cat(batch_edge_indices, dim=1)
        return batch_edge_index.to(device)


class Model(GAT):
    """
    包装类，兼容现有框架的命名约定
    """
    def __init__(self, configs):
        super(Model, self).__init__(configs)