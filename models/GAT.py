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
        # 初始化LightTime模型
        self.lightTime = LightTime.Model(configs).float()
        
        # GAT的输入通道数来自LightTime编码器的输出
        self.in_channels = self.lightTime.n_heads * self.lightTime.d_model

        # 边特征预处理层
        self.edge_preprocessing = torch.nn.Sequential(
            torch.nn.Linear(self.total_edge_dim, self.total_edge_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(self.total_edge_dim, self.total_edge_dim // 2)
        )
        
        # 使用原生GATConv，支持多维边特征
        self.gat = GATConv(
            in_channels=self.in_channels, 
            out_channels=self.in_channels,
            heads=self.heads, 
            dropout=0, 
            concat=False,
            edge_dim=self.total_edge_dim // 2  # 指定边特征维度
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
        static_edge_attr = data.edge_attr #[num_edges, static_edge_dim]
        # 计算动态边特征并合并
        combined_edge_attr = self.compute_dynamic_edge_features(
            x, edge_index, static_edge_attr, device
        )

        # 预处理边特征
        processed_edge_attr = self.edge_preprocessing(combined_edge_attr)

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
    
    def compute_dynamic_edge_features(self, x, edge_index, static_edge_attr, device):
        """
        batch_size_nnodes, seq_len, feature_dim = x.shape
        num_edges = edge_index.shape[1]
        
        # 根据数据集的实际特征列确定索引
        # 从TimeSeriesGraphDataset可以看出特征列除了date外包含: Wspd, Wdir, Etmp, Itmp, Ndir, Pab1, Pab2, Pab3, Prtv, Patv
        # 由于date列在预处理时被排除，所以特征顺序为: [Wspd, Wdir, Etmp, Itmp, Ndir, Pab1, Pab2, Pab3, Prtv, Patv]
        动态边特征说明（总计13维）:
        1. wdir_diff_norm: 风向差异归一化
        2. wdir_consistency: 风向一致性（cos相似度）
        3. wspd_diff_norm: 风速差异归一化  
        4. wspd_ratio: 风速比率
        5. wspd_level: 平均风速等级
        6. wake_effect: 尾流效应强度
        7. temp_diff_norm: 温度差异归一化
        8. power_diff_norm: 功率差异归一化
        9. power_correlation: 功率相关性
        10. ndir_diff_norm: 机舱方向差异
        11. ndir_wind_align: 机舱方向与风向一致性
        12. trend_consistency: 风速变化趋势一致性（seq_len>=3时）
        13. trend_strength_diff: 趋势强度差异（seq_len>=3时）

        特征优势:
        - 考虑风电场特有的尾流效应
        - 结合风向、风速的时变特性
        - 包含设备状态（机舱方向、功率）的动态交互
        - 引入时序趋势分析，捕获短期变化模式
        - 与静态图特征互补，提供完整的时空建模能力
        """
        n_nodes = self.n_nodes
        batch_size = x.shape[0] // n_nodes

        # 使用默认索引（基于常见的风电数据格式）
        wspd_idx, wdir_idx = 0, 1  # 风速、风向
        etmp_idx, itmp_idx = 2, 3  # 环境温度、内部温度
        ndir_idx = 4  # 机舱方向
        patv_idx = 9   # 有功功率（假设是第10列，索引9）
        
        # 取最后一个时间步的特征用于计算动态边特征
        current_features = x[:, -1, :]  # [batch_size*n_nodes, feature_dim]

        # 为每个批次计算动态边特征
        batch_dynamic_features = []
        for b in range(batch_size):
            # 获取当前批次的节点特征
            batch_start = b * n_nodes
            batch_end = (b + 1) * n_nodes
            batch_features = current_features[batch_start:batch_end]  # [n_nodes, feature_dim]

                    
            # 获取边的源节点和目标节点索引（针对当前批次）
            source_idx = edge_index[b,0]  # [num_edges]
            target_idx = edge_index[b,1]  # [num_edges]
            
            # 提取源节点和目标节点的特征
            source_features = batch_features[source_idx]  # [num_edges, feature_dim]
            target_features = batch_features[target_idx]  # [num_edges, feature_dim]
            
            dynamic_features = []
            
            # 1. 风向相关的动态特征
            source_wdir = source_features[:, wdir_idx]
            target_wdir = target_features[:, wdir_idx]
            
            # 风向差异（考虑角度的周期性）
            wdir_diff = torch.abs(source_wdir - target_wdir)
            wdir_diff = torch.min(wdir_diff, 360 - wdir_diff)
            wdir_diff_norm = wdir_diff / 180.0
            dynamic_features.append(wdir_diff_norm.unsqueeze(1))
            
            # 风向一致性（cos相似度）
            source_wdir_rad = torch.deg2rad(source_wdir)
            target_wdir_rad = torch.deg2rad(target_wdir)
            wdir_consistency = torch.cos(source_wdir_rad - target_wdir_rad)
            dynamic_features.append(wdir_consistency.unsqueeze(1))
           
            
            # 2. 风速相关的动态特征
            source_wspd = source_features[:, wspd_idx]
            target_wspd = target_features[:, wspd_idx]
            
            # 风速差异
            wspd_diff = torch.abs(source_wspd - target_wspd)
            wspd_diff_norm = torch.clamp(wspd_diff / 25.0, 0, 1)
            dynamic_features.append(wspd_diff_norm.unsqueeze(1))
            
            # 风速比率
            wspd_ratio = torch.clamp(torch.min(source_wspd, target_wspd) / 
                            (torch.max(source_wspd, target_wspd) + 1e-5), 0, 1)
            dynamic_features.append(wspd_ratio.unsqueeze(1))
            
            # 平均风速等级
            avg_wspd = (source_wspd + target_wspd) / 2
            wspd_level = torch.clamp(avg_wspd / 25.0, 0, 1)
            dynamic_features.append(wspd_level.unsqueeze(1))
            
            # 从静态特征中提取距离和方向信息
            batch_static_edge_attr = static_edge_attr[b]  # [num_edges, 21] 获取当前批次的静态边特征
            euclidean_dist = batch_static_edge_attr[:, 0]  # [num_edges]
            bearing_angle = batch_static_edge_attr[:, 4]   # [num_edges]

            # 计算尾流影响强度
            source_to_target_angle = bearing_angle
            wake_alignment = torch.cos(torch.deg2rad(source_wdir - source_to_target_angle))
            wake_alignment = torch.clamp(wake_alignment, 0, 1)
            
            # 距离衰减因子
            distance_decay = torch.exp(-euclidean_dist / 500.0)
            
            # 综合尾流强度
            wake_effect = wake_alignment * distance_decay * wspd_level.squeeze()
            dynamic_features.append(wake_effect.unsqueeze(1))
            
            source_etmp = source_features[:, etmp_idx]
            target_etmp = target_features[:, etmp_idx]
            
            temp_diff = torch.abs(source_etmp - target_etmp)
            temp_diff_norm = torch.clamp(temp_diff / 50.0, 0, 1)
            dynamic_features.append(temp_diff_norm.unsqueeze(1))
            
            # 5. 功率相关特征
            source_patv = source_features[:, patv_idx]
            target_patv = target_features[:, patv_idx]
            
            power_diff = torch.abs(source_patv - target_patv)
            power_diff_norm = torch.clamp(power_diff / 2000.0, 0, 1)
            dynamic_features.append(power_diff_norm.unsqueeze(1))
            
            power_correlation = torch.clamp(torch.min(source_patv, target_patv) / 
                                    (torch.max(source_patv, target_patv) + 1e-5), 0, 1)
            dynamic_features.append(power_correlation.unsqueeze(1))
            
            
            source_ndir = source_features[:, ndir_idx]
            target_ndir = target_features[:, ndir_idx]
            
            ndir_diff = torch.abs(source_ndir - target_ndir)
            ndir_diff = torch.min(ndir_diff, 360 - ndir_diff)
            ndir_diff_norm = ndir_diff / 180.0
            dynamic_features.append(ndir_diff_norm.unsqueeze(1))
            
            source_ndir_wind_align = torch.cos(torch.deg2rad(source_ndir - source_wdir))
            target_ndir_wind_align = torch.cos(torch.deg2rad(target_ndir - target_wdir))
            avg_alignment = (source_ndir_wind_align + target_ndir_wind_align) / 2
            dynamic_features.append(avg_alignment.unsqueeze(1))
            
            # 7. 时序稳定性特征（基于近期变化趋势）
            # 获取当前批次的历史风速数据
            batch_recent_wspd = x[batch_start:batch_end, -3:, wspd_idx]  # [n_nodes, 3]
            wspd_trend = batch_recent_wspd[:, -1] - batch_recent_wspd[:, 0]
            
            source_trend = wspd_trend[source_idx]
            target_trend = wspd_trend[target_idx]
            
            # 创建与source_trend相同设备和dtype的tensor
            ones_tensor = torch.ones_like(source_trend)

            trend_consistency = torch.cos(torch.atan2(source_trend, ones_tensor) - torch.atan2(target_trend, ones_tensor))
            dynamic_features.append(trend_consistency.unsqueeze(1))
            
            trend_strength_diff = torch.abs(torch.abs(source_trend) - torch.abs(target_trend))
            trend_strength_diff_norm = torch.clamp(trend_strength_diff / 10.0, 0, 1)
            dynamic_features.append(trend_strength_diff_norm.unsqueeze(1))
            
            # 合并当前批次的动态特征
            batch_dynamic_edge_attr = torch.cat(dynamic_features, dim=1)  # [num_edges, dynamic_dim]
            batch_dynamic_features.append(batch_dynamic_edge_attr)
        
        batch_static_features = []
        for b in range(batch_size):
            batch_static_features.append(static_edge_attr[b])  # [num_edges, 21]

        all_static_features = torch.stack(batch_static_features, dim=0)  # [batch_size, num_edges, 21]

        # 合并所有批次的动态特征
        all_dynamic_features = torch.stack(batch_dynamic_features, dim=0)  # [batch_size, num_edges, dynamic_dim]
        # 结合静态和动态边特征
        combined_edge_attr = torch.cat([all_static_features, all_dynamic_features], dim=2)
        return combined_edge_attr



class Model(GAT):
    """
    包装类，兼容现有框架的命名约定
    """
    def __init__(self, configs):
        super(Model, self).__init__(configs)