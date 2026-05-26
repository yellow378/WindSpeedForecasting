"""
LightST: Light Spatio-Temporal 风电场多风机风速预测模型
基于动静态特征稀疏图注意力的风电场多风机风速预测

核心组件 (对应论文各节):
- 4.2.2 基于LightTemporal的层次化编解码器
- 4.2.3 基于风向感知的条件化静态图构建
- 4.2.4 基于动态感知与动静融合的自适应边建模
- 4.2.5 边增强型多头图注意力网络 (E-MGAT)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from models import LightTime
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax as pyg_softmax


# ============================================================
#  4.2.2 基于LightTemporal的层次化编解码器
# ============================================================

class TurbineAdapter(nn.Module):
    """
    个体风机适配器 (T-Adapter): 轻量级瓶颈MLP
    为每台风机提供个体化的残差修正,保留风机独特动态特性

    采用瓶颈结构: d_model → bottleneck_dim → d_model
    以控制参数量 (134台风机 × 瓶颈MLP参数)
    """
    def __init__(self, d_model, bottleneck_dim=None):
        super().__init__()
        if bottleneck_dim is None:
            bottleneck_dim = max(d_model // 4, 16)
        self.down = nn.Linear(d_model, bottleneck_dim)
        self.up = nn.Linear(bottleneck_dim, d_model)

    def forward(self, h):
        """
        Args:
            h: [*, d_model] 共享编码器的输出
        Returns:
            [*, d_model] 个体残差
        """
        return self.up(F.silu(self.down(h)))


class ClusterAdapter(nn.Module):
    """
    聚类级别适配器 (C-Adapter): 同一隐空间聚类的风机群组共享参数

    三级参数共享架构的第二级:
    (1) 全局共享层: LT-Encoder
    (2) 聚类适配层: C-Adapter  ← 本类
    (3) 个体适配层: T-Adapter
    """
    def __init__(self, d_model, bottleneck_dim=None):
        super().__init__()
        if bottleneck_dim is None:
            bottleneck_dim = max(d_model // 4, 16)
        self.down = nn.Linear(d_model, bottleneck_dim)
        self.up = nn.Linear(bottleneck_dim, d_model)

    def forward(self, h):
        return self.up(F.silu(self.down(h)))


class HierarchicalEncoder(nn.Module):
    """
    层次化编码器: 全局共享LT-Encoder + 聚类C-Adapter + 个体T-Adapter

    编码公式:
        h_i = LT_Encoder(x_i) + C_Adapter_{c(i)}(h_shared) + T_Adapter_i(h_shared)

    其中 c(i) 为风机i的聚类分配, 通过隐空间聚类确定
    """
    def __init__(self, configs, n_nodes, n_clusters=8):
        super().__init__()
        self.n_nodes = n_nodes
        self.n_clusters = n_clusters
        self.d_model = configs.d_model
        self.n_heads = configs.heads

        # 全局共享的LightTemporal编码器
        self.lightTime = LightTime.Model(configs).float()
        # 编码器输出维度: n_heads * d_model
        self.d_h = self.n_heads * self.d_model

        # 聚类级别适配器: n_clusters个, 同一聚类内的风机共享
        self.cluster_adapters = nn.ModuleList([
            ClusterAdapter(self.d_h) for _ in range(n_clusters)
        ])

        # 个体级别适配器: n_nodes个, 每台风机独有
        self.turbine_adapters = nn.ModuleList([
            TurbineAdapter(self.d_h) for _ in range(n_nodes)
        ])

        # 聚类分配: 可通过隐空间聚类更新
        # 初始化为均匀分配, 后续通过聚类算法更新
        self.register_buffer(
            'cluster_assignment',
            torch.arange(n_nodes) % n_clusters  # 初始均匀分配
        )

    def update_clustering(self, x_all):
        """
        隐空间聚类更新: 基于编码器输出的K-means聚类

        Args:
            x_all: [n_nodes, seq_len, n_features] 所有风机的输入数据
                   (使用训练集的统计特征进行聚类)
        """
        from sklearn.cluster import KMeans

        self.lightTime.eval()
        with torch.no_grad():
            # 编码所有风机
            h_list = []
            for i in range(self.n_nodes):
                x_i = x_all[i:i+1]  # [1, seq_len, n_features]
                # 归一化
                means = x_i.mean(1, keepdim=True).detach()
                x_i = x_i - means
                stdev = torch.sqrt(
                    torch.var(x_i, dim=1, keepdim=True, unbiased=False) + 1e-5
                )
                x_i /= stdev
                # 编码
                enc_out = self.lightTime.encoder(x_i)
                h_i = enc_out.reshape(1, -1)  # [1, d_h]
                h_list.append(h_i)
            H = torch.cat(h_list, dim=0).cpu().numpy()  # [n_nodes, d_h]

        # K-means聚类
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(H)

        self.cluster_assignment = torch.tensor(
            labels, dtype=torch.long, device=self.cluster_assignment.device
        )
        print(f"聚类更新完成, 各聚类风机数: "
              f"{[int((labels == k).sum()) for k in range(self.n_clusters)]}")

    def encode(self, x, node_indices=None):
        """
        层次化编码

        Args:
            x: [batch_size, seq_len, enc_in] 单个风机的输入
            node_indices: 风机编号列表, 用于选择对应的适配器
        Returns:
            h: [batch_size, d_h] 层次化编码结果
        """
        # Step1: 归一化 (与LightTime一致)
        means = x.mean(1, keepdim=True).detach()
        x_norm = x - means
        stdev = torch.sqrt(
            torch.var(x_norm, dim=1, keepdim=True, unbiased=False) + 1e-5
        )
        x_norm /= stdev

        # Step2: 全局共享编码
        enc_out = self.lightTime.encoder(x_norm)  # [bs, n_heads, d_model]
        h_shared = enc_out.reshape(enc_out.size(0), -1)  # [bs, d_h]

        # Step3: 聚类适配器 (如果提供了风机编号)
        if node_indices is not None:
            # 收集每个风机对应的聚类适配器输出
            h_cluster = torch.zeros_like(h_shared)
            for idx, node_id in enumerate(node_indices):
                c = self.cluster_assignment[node_id].item()
                h_cluster[idx] = self.cluster_adapters[c](h_shared[idx:idx+1]).squeeze(0)

            # Step4: 个体适配器
            h_individual = torch.zeros_like(h_shared)
            for idx, node_id in enumerate(node_indices):
                h_individual[idx] = self.turbine_adapters[node_id](
                    h_shared[idx:idx+1]
                ).squeeze(0)
        else:
            # 批量模式: 所有节点使用相同的适配器 (推理时简化)
            h_cluster = torch.zeros_like(h_shared)
            h_individual = torch.zeros_like(h_shared)

        # Step5: 三级融合 h_i = h_shared + h_cluster + h_individual
        h = h_shared + h_cluster + h_individual
        return h, means, stdev


# ============================================================
#  4.2.3 基于风向感知的条件化静态图构建
# ============================================================

def compute_dominant_wind_direction(wspd, wdir):
    """
    算法1: 计算风电场主导风向

    将风电场所有风机的风向按风速加权平均, 得到整体主导风向

    Args:
        wspd: [s] 各风机风速 (s为风机数量)
        wdir: [s] 各风机风向角度 (弧度制, 0=北, 顺时针)
    Returns:
        dominant_dir: 主导风向角度 (弧度制)
        consistency: 一致性度量 (越接近1表示风向越一致)
    """
    # 过滤低风速风机 (风速 > 0.1 m/s 才纳入计算)
    valid_mask = wspd > 0.1
    if valid_mask.sum() == 0:
        return torch.tensor(0.0), torch.tensor(0.0)

    wspd_valid = wspd[valid_mask]
    wdir_valid = wdir[valid_mask]

    # 风速加权向量分解
    u_x = (wspd_valid * torch.cos(wdir_valid)).sum()
    u_y = (wspd_valid * torch.sin(wdir_valid)).sum()
    total_speed = wspd_valid.sum()

    # 加权平均向量
    u_x_bar = u_x / total_speed
    u_y_bar = u_y / total_speed

    # 主导风向 (使用atan2避免角度歧义)
    dominant_dir = torch.atan2(u_y_bar, u_x_bar)

    # 一致性度量: 加权平均向量的模长 (0~1之间, 越大越一致)
    consistency = torch.sqrt(u_x_bar ** 2 + u_y_bar ** 2)

    return dominant_dir, consistency


class StaticGraphManager:
    """
    静态图管理器: 管理16个风向扇区的预计算稀疏有向静态图

    每个扇区包含:
    - edge_index: [2, n_edges] 有向边索引
    - adj_weights: [n_edges] 邻接权重 (物理先验 + 统计相关)
    - static_edge_attr: [n_edges, 11] 11维静态边特征

    离线构建, 在线查询
    """
    N_SECTORS = 16  # 风向扇区数量
    SECTOR_ANGLE = 2 * math.pi / 16  # 每扇区角度 (22.5°)

    def __init__(self, static_graphs):
        """
        Args:
            static_graphs: 字典, key为扇区编号(0-15), value为包含
                'edge_index', 'adj_weights', 'static_edge_attr'的字典
        """
        self.static_graphs = static_graphs

    @classmethod
    def from_files(cls, graph_dir, device='cpu'):
        """从预计算文件加载静态图"""
        static_graphs = {}
        for z in range(cls.N_SECTORS):
            data = {}
            data['edge_index'] = torch.tensor(
                np.load(f"{graph_dir}/sector_{z}_edge_index.npy"),
                dtype=torch.long
            ).to(device)
            data['adj_weights'] = torch.tensor(
                np.load(f"{graph_dir}/sector_{z}_adj_weights.npy"),
                dtype=torch.float32
            ).to(device)
            data['static_edge_attr'] = torch.tensor(
                np.load(f"{graph_dir}/sector_{z}_static_edge_attr.npy"),
                dtype=torch.float32
            ).to(device)
            static_graphs[z] = data
        return cls(static_graphs)

    def get_sector(self, dominant_dir):
        """
        根据主导风向确定扇区编号

        扇区划分: 0号扇区以正北为起始(0°), 顺时针编号至15
        z = floor(Θ / 22.5°) mod 16

        Args:
            dominant_dir: 主导风向 (弧度制, 0=北/东, 取决于坐标系)
        Returns:
            z: 扇区编号 (0-15)
        """
        # 将角度归一化到 [0, 2π)
        angle = dominant_dir % (2 * math.pi)
        z = int(angle / self.SECTOR_ANGLE) % self.N_SECTORS
        return z

    def get_graph(self, sector_id):
        """
        获取指定扇区的静态图

        Args:
            sector_id: 扇区编号 (0-15)
        Returns:
            edge_index, adj_weights, static_edge_attr
        """
        graph = self.static_graphs[sector_id]
        return (graph['edge_index'],
                graph['adj_weights'],
                graph['static_edge_attr'])


# ============================================================
#  4.2.4 基于动态感知与动静融合的自适应边建模
# ============================================================

class DynamicEdgeFeatureComputer:
    """
    动态边特征计算模块: 从近期时间窗口提取14维动态边特征

    动态特征涵盖:
    - 环境状态特征: 风向差、风向余弦一致性、风速差、风速比
    - 尾流效应特征: wake_effect (风向对其·距离衰减·风速等级)
    - 控制响应特征: 温度差、机舱方向差、机舱-风向对齐度
    - 时序相关性特征: 趋势一致性、趋势强度差、风速相关性、
                       差分风速相关性、风向相关性
    """

    def __init__(self, positions, rotor_radius=1.0, d_max_factor=10.0):
        """
        Args:
            positions: [n_nodes, 2] 风机坐标
            rotor_radius: 叶轮半径 R
            d_max_factor: 最大尾流距离系数 (D_max = d_max_factor * R)
        """
        self.positions = positions  # [n_nodes, 2]
        self.rotor_radius = rotor_radius
        self.d_max = d_max_factor * rotor_radius  # D_max = 10R
        self.L0 = 5.0 * rotor_radius  # 距离衰减尺度 L0 = 5R

        # 预计算风机间距离矩阵和方位角
        n = positions.shape[0]
        diff = positions.unsqueeze(1) - positions.unsqueeze(0)  # [n, n, 2]
        self.dist_matrix = torch.norm(diff, dim=-1)  # [n, n]
        # 方位角: 从风机i指向风机j的地理方位角 (弧度制)
        self.bearing_matrix = torch.atan2(diff[:, :, 1], diff[:, :, 0])  # [n, n]

    def compute_wake_effect(self, wdir_i, d_ij, v_avg):
        """
        动态尾流效应估计 (论文公式 4.2.4)

        wake_effect_ij(t) = max(0, cos(θ_wdir^(i)(t) - θ_ij))
                            × exp(-d_ij / L0) × v_avg(t)

        Args:
            wdir_i: [n_edges] 上游风机i的实测风向 (弧度)
            d_ij: [n_edges] 风机i到j的欧氏距离
            v_avg: [n_edges] 两风机的平均风速
        Returns:
            [n_edges] 尾流效应估计值
        """
        # 风向对其程度: 仅当风向大致指向下游时为正
        alignment = torch.clamp(torch.cos(wdir_i - self.bearing_for_edges), min=0.0)
        # 距离衰减因子
        distance_decay = torch.exp(-d_ij / self.L0)
        return alignment * distance_decay * v_avg

    def compute(self, x, edge_index):
        """
        计算所有有向边的14维动态特征

        Args:
            x: [n_nodes, seq_len, n_features] 各风机的近期时序数据
               特征顺序: [Wspd, Wdir, Etmp, Itmp, Ndir, Pab1, Pab2, Pab3, Prtv, Patv]
            edge_index: [2, n_edges] 有向边索引 (j→i, 即j为上游, i为下游)
        Returns:
            dynamic_features: [n_edges, 14]
        """
        src, tgt = edge_index  # src=j(上游), tgt=i(下游)
        n_edges = src.shape[0]

        # 提取各特征列 (假设特征顺序与SDWPF数据集一致)
        wspd = x[:, -1, 0]   # [n_nodes] 最近时刻风速
        wdir = x[:, -1, 1]   # [n_nodes] 最近时刻风向
        etmp = x[:, -1, 2]   # [n_nodes] 环境温度
        itmp = x[:, -1, 3]   # [n_nodes] 机舱温度
        ndir = x[:, -1, 4]   # [n_nodes] 机舱方向

        # 上游和下游风机特征
        wspd_j, wspd_i = wspd[src], wspd[tgt]
        wdir_j, wdir_i = wdir[src], wdir[tgt]
        etmp_j, etmp_i = etmp[src], etmp[tgt]
        ndir_j, ndir_i = ndir[src], ndir[tgt]

        # 距离和方位角
        d_ij = self.dist_matrix[src, tgt]
        bearing_ij = self.bearing_matrix[src, tgt]

        # ---- 14维动态特征 ----

        # 1. 风向差 (归一化到[-1,1])
        wdir_diff = torch.cos(wdir_j - wdir_i)

        # 2. 风向余弦一致性
        wdir_cos_consistency = (torch.cos(wdir_j) * torch.cos(wdir_i) +
                                torch.sin(wdir_j) * torch.sin(wdir_i))

        # 3. 风速差 (截断到合理范围)
        wspd_diff = torch.clamp(wspd_j - wspd_i, min=-5.0, max=5.0) / 5.0

        # 4. 风速比 (下游/上游, 避免除零)
        wspd_ratio = wspd_i / (wspd_j + 1e-6)

        # 5. 平均风速等级
        v_avg = (wspd_j + wspd_i) / 2.0

        # 6. 动态尾流效应估计
        # 预存当前边的方位角信息
        self.bearing_for_edges = bearing_ij
        wake_effect = self.compute_wake_effect(wdir_j, d_ij, v_avg)

        # 7. 温度差 (归一化)
        etmp_diff = torch.clamp(etmp_j - etmp_i, min=-10.0, max=10.0) / 10.0

        # 8. 机舱方向差 (余弦相似度)
        ndir_diff = torch.cos(ndir_j - ndir_i)

        # 9. 机舱-风向对齐度均值
        nacelle_wind_align_j = torch.cos(ndir_j - wdir_j)
        nacelle_wind_align_i = torch.cos(ndir_i - wdir_i)
        nacelle_wind_avg = (nacelle_wind_align_j + nacelle_wind_align_i) / 2.0

        # ---- 基于时序窗口的特征 (10-14) ----
        # 使用最近3个时间步计算趋势和相关性
        window = min(3, x.shape[1])

        # 10. 趋势一致性 (风速变化方向是否一致)
        if window >= 2:
            trend_j = x[src, -1, 0] - x[src, -window, 0]
            trend_i = x[tgt, -1, 0] - x[tgt, -window, 0]
            trend_consistency = torch.sign(trend_j) * torch.sign(trend_i)
        else:
            trend_consistency = torch.zeros(n_edges, device=x.device)

        # 11. 趋势强度差
        if window >= 2:
            trend_strength_j = torch.abs(trend_j)
            trend_strength_i = torch.abs(trend_i)
            trend_strength_diff = torch.clamp(
                trend_strength_j - trend_strength_i, min=-3.0, max=3.0
            ) / 3.0
        else:
            trend_strength_diff = torch.zeros(n_edges, device=x.device)

        # 12-14. 短时相关性特征 (基于窗口内序列)
        if window >= 3:
            seq_j = x[src, -window:, 0]  # [n_edges, window]
            seq_i = x[tgt, -window:, 0]

            # 12. 风速相关性 (Pearson)
            speed_corr = self._batch_pearson(seq_j, seq_i)

            # 13. 差分风速相关性
            diff_j = seq_j[:, 1:] - seq_j[:, :-1]
            diff_i = seq_i[:, 1:] - seq_i[:, :-1]
            diff_corr = self._batch_pearson(diff_j, diff_i)

            # 14. 风向相关性
            wdir_seq_j = x[src, -window:, 1]
            wdir_seq_i = x[tgt, -window:, 1]
            # 风向使用cos/sin分解计算循环相关
            wdir_cos_corr = self._batch_pearson(
                torch.cos(wdir_seq_j), torch.cos(wdir_seq_i)
            )
            wdir_sin_corr = self._batch_pearson(
                torch.sin(wdir_seq_j), torch.sin(wdir_seq_i)
            )
            wdir_corr = (wdir_cos_corr + wdir_sin_corr) / 2.0
        else:
            speed_corr = torch.zeros(n_edges, device=x.device)
            diff_corr = torch.zeros(n_edges, device=x.device)
            wdir_corr = torch.zeros(n_edges, device=x.device)

        # 拼接14维动态特征
        dynamic_features = torch.stack([
            wdir_diff,                    # 1. 风向差
            wdir_cos_consistency,         # 2. 风向余弦一致性
            wspd_diff,                    # 3. 风速差
            wspd_ratio,                   # 4. 风速比
            v_avg / 15.0,                 # 5. 平均风速等级 (归一化)
            wake_effect,                  # 6. 动态尾流效应
            etmp_diff,                    # 7. 温度差
            ndir_diff,                    # 8. 机舱方向差
            nacelle_wind_avg,             # 9. 机舱-风向对齐度
            trend_consistency,            # 10. 趋势一致性
            trend_strength_diff,          # 11. 趋势强度差
            speed_corr,                   # 12. 风速相关性
            diff_corr,                    # 13. 差分风速相关性
            wdir_corr,                    # 14. 风向相关性
        ], dim=-1)  # [n_edges, 14]

        return dynamic_features

    @staticmethod
    def _batch_pearson(x, y):
        """批量计算Pearson相关系数"""
        x_mean = x.mean(dim=-1, keepdim=True)
        y_mean = y.mean(dim=-1, keepdim=True)
        x_centered = x - x_mean
        y_centered = y - y_mean
        cov = (x_centered * y_centered).sum(dim=-1)
        std_x = torch.sqrt((x_centered ** 2).sum(dim=-1) + 1e-8)
        std_y = torch.sqrt((y_centered ** 2).sum(dim=-1) + 1e-8)
        corr = cov / (std_x * std_y + 1e-8)
        return torch.clamp(corr, -1.0, 1.0)


class EdgeWeightAdjuster(nn.Module):
    """
    边权重动态调整模块 (论文公式 4.2.4)

    基于融合边特征的MLP评分函数, 实时调整有向边权重:
        w_ij^(t) = α_ij^static × σ(MLP_θ(e_ij))

    其中:
    - α_ij^static: 静态图构建时的物理先验权重
    - e_ij: 25维复合边特征 (11静态 + 14动态)
    - σ: sigmoid激活, 映射到(0,1)
    """
    def __init__(self, edge_feat_dim=25, hidden_dim=32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(edge_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, static_weights, edge_features):
        """
        Args:
            static_weights: [n_edges] 静态物理先验权重 α^static
            edge_features: [n_edges, 25] 融合边特征
        Returns:
            adjusted_weights: [n_edges] 动态调整后的边权重
        """
        # MLP评分 + sigmoid → (0, 1)
        dynamic_score = torch.sigmoid(self.mlp(edge_features)).squeeze(-1)
        # 与静态权重相乘
        adjusted_weights = static_weights * dynamic_score
        return adjusted_weights


# ============================================================
#  4.2.5 边增强型多头图注意力网络 (E-MGAT)
# ============================================================

class EMGATLayer(MessagePassing):
    """
    边增强型多头图注意力层 (Edge-enhanced Multi-head Graph Attention Layer)

    核心公式:
        score_ji^(m) = LeakyReLU(a^(m)^T [q_i^(m) || k_j^(m) || e_ji^(m)]) × w_ji^(t)
        α_ji^(m) = softmax_{j∈N_in(i)}(score_ji^(m))
        z_i^(m) = Σ_{j∈N_in(i)} α_ji^(m) × v_j^(m)

    三重融合:
    1. 节点语义匹配: q_i与k_j的交互 → 风机运行状态相似性
    2. 边关系建模: e_ji编码空间几何、统计相关与瞬时交互特性
    3. 物理先验约束: w_ji^(t)确保注意力权重受尾流物理规律引导

    消息流向: 仅从入邻居聚合 (上游→下游), 符合尾流物理传播规律
    """
    def __init__(self, d_h, d_k, n_heads, edge_feat_dim=25, dropout=0.1):
        """
        Args:
            d_h: 节点隐层维度
            d_k: 每个注意力头的维度
            n_heads: 注意力头数 M
            edge_feat_dim: 边特征维度 (默认25: 11静态 + 14动态)
            dropout: dropout率
        """
        super().__init__(aggr='add', flow='source_to_target', node_dim=0)

        self.d_h = d_h
        self.d_k = d_k
        self.n_heads = n_heads
        self.dropout = dropout

        # Q, K, V 投影矩阵 (每个头的参数独立)
        self.W_Q = nn.Linear(d_h, n_heads * d_k, bias=False)
        self.W_K = nn.Linear(d_h, n_heads * d_k, bias=False)
        self.W_V = nn.Linear(d_h, n_heads * d_k, bias=False)

        # 边特征投影矩阵 (每个头独立)
        self.W_E = nn.Linear(edge_feat_dim, n_heads * d_k, bias=False)

        # 注意力参数向量 a ∈ R^{3d_k} (每个头独立)
        self.a = nn.Parameter(torch.randn(n_heads, 3 * d_k) * 0.01)

        # 输出投影
        self.W_O = nn.Linear(n_heads * d_k, d_h, bias=False)

        # LayerNorm + 残差
        self.layer_norm = nn.LayerNorm(d_h)

    def forward(self, H, edge_index, edge_attr, edge_weights):
        """
        Args:
            H: [n_nodes, d_h] 节点隐层表示
            edge_index: [2, n_edges] 有向边 (j→i, j=source=上游, i=target=下游)
            edge_attr: [n_edges, 25] 复合边特征 (静态11维 + 动态14维)
            edge_weights: [n_edges] 物理先验边权重 w_ji^(t)
        Returns:
            H_out: [n_nodes, d_h] 更新后的节点表示
        """
        # 计算 Q, K, V
        Q = self.W_Q(H).view(-1, self.n_heads, self.d_k)  # [n_nodes, M, d_k]
        K = self.W_K(H).view(-1, self.n_heads, self.d_k)
        V = self.W_V(H).view(-1, self.n_heads, self.d_k)

        # 边特征投影
        E = self.W_E(edge_attr).view(-1, self.n_heads, self.d_k)  # [n_edges, M, d_k]

        # 消息传递
        out = self.propagate(
            edge_index, Q=Q, K=K, V=V, E=E,
            edge_weights=edge_weights, size=None
        )  # [n_nodes, M * d_k]

        # 输出投影 + 残差 + LayerNorm
        out = self.W_O(out)
        H_out = self.layer_norm(H + out)

        return H_out

    def message(self, Q_i, K_j, V_j, E, edge_weights, index):
        """
        计算边增强型注意力消息

        Args:
            Q_i: [n_edges, M, d_k] 目标节点查询向量
            K_j: [n_edges, M, d_k] 源节点键向量
            V_j: [n_edges, M, d_k] 源节点值向量
            E: [n_edges, M, d_k] 边特征投影
            edge_weights: [n_edges] 物理先验权重
            index: [n_edges] 目标节点索引 (PyG自动提供, 用于softmax分组)
        Returns:
            msg: [n_edges, M * d_k] 注意力加权后的值向量
        """
        # 拼接 [q_i || k_j || e_ji] → [n_edges, M, 3*d_k]
        QKE = torch.cat([Q_i, K_j, E], dim=-1)

        # 计算注意力得分: a^(m)^T × [q_i || k_j || e_ji]
        score = (QKE * self.a.unsqueeze(0)).sum(dim=-1)  # [n_edges, M]
        score = F.leaky_relu(score, negative_slope=0.2)

        # 乘以物理先验权重 w_ji^(t)
        score = score * edge_weights.unsqueeze(-1)  # [n_edges, M]

        # 在入邻居集合上softmax归一化 (按目标节点分组)
        alpha = pyg_softmax(score, index)  # [n_edges, M]
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        # 加权聚合值向量
        msg = alpha.unsqueeze(-1) * V_j  # [n_edges, M, d_k]
        msg = msg.reshape(msg.size(0), -1)  # [n_edges, M * d_k]

        return msg

    def update(self, aggr_out):
        """聚合后的输出 (已在forward中处理残差和LayerNorm)"""
        return aggr_out


class EMGAT(nn.Module):
    """
    边增强型多头图注意力网络 (多层堆叠)

    堆叠 L_g 层 E-MGAT, 逐步扩大感受野, 实现多跳信息传播
    经过 L_g 层后, 每个节点可接收到 L_g 跳范围内上游风机的信息
    """
    def __init__(self, d_h, d_k, n_heads, n_layers, edge_feat_dim=25, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            EMGATLayer(d_h, d_k, n_heads, edge_feat_dim, dropout)
            for _ in range(n_layers)
        ])

    def forward(self, H, edge_index, edge_attr, edge_weights):
        """
        Args:
            H: [n_nodes, d_h] 初始节点表示 (来自层次化编码器)
            edge_index: [2, n_edges] 有向边索引
            edge_attr: [n_edges, 25] 边特征
            edge_weights: [n_edges] 边权重
        Returns:
            H_D: [n_nodes, d_h] 空间融合后的节点表示
        """
        for layer in self.layers:
            H = layer(H, edge_index, edge_attr, edge_weights)
        return H


# ============================================================
#  LightST 主模型
# ============================================================

class LightST(nn.Module):
    """
    LightST: Light Spatio-Temporal 风电场多风机风速预测模型

    整体流程:
    1. 层次化编码: x_i → LT_Encoder + C_Adapter + T_Adapter → h_i
    2. 风向感知静态图选择: 根据主导风向选择扇区子图 G_z^static
    3. 动态边特征计算: 从近期窗口提取14维动态特征
    4. 动静融合与权重调整: 拼接25维边特征, MLP调整权重 → G^(t)
    5. 边增强图注意力: E-MGAT空间信息融合 → H_D
    6. 共享解码: LT_Decoder → 预测风速 P ∈ R^{s×T}
    """
    def __init__(self, configs):
        super().__init__()

        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.n_nodes = getattr(configs, 'n_nodes', 134)
        self.n_clusters = getattr(configs, 'n_clusters', 8)
        self.n_sectors = getattr(configs, 'n_sectors', 16)
        self.gnn_layers = getattr(configs, 'gnn_layers', 2)
        self.dropout = configs.dropout

        # ---- 4.2.2 层次化编解码器 ----
        self.hierarchical_encoder = HierarchicalEncoder(
            configs, self.n_nodes, self.n_clusters
        )
        self.d_h = self.hierarchical_encoder.d_h  # n_heads * d_model

        # 共享解码器 (LightTemporal Decoder)
        self.decoder = self.hierarchical_encoder.lightTime.decoder

        # ---- 4.2.3 静态图管理 (需在外部加载) ----
        self.static_graph_manager = None
        self.positions = None  # [n_nodes, 2] 风机坐标

        # ---- 4.2.4 动态边特征 + 权重调整 ----
        self.dynamic_feature_computer = None  # 延迟初始化 (需要positions)
        self.edge_weight_adjuster = EdgeWeightAdjuster(
            edge_feat_dim=25,  # 11静态 + 14动态
            hidden_dim=32
        )

        # ---- 4.2.5 边增强型多头图注意力网络 ----
        d_k = self.hierarchical_encoder.d_model  # 每个头的维度 = d_model
        n_heads = self.hierarchical_encoder.n_heads
        self.emgat = EMGAT(
            d_h=self.d_h,
            d_k=d_k,
            n_heads=n_heads,
            n_layers=self.gnn_layers,
            edge_feat_dim=25,
            dropout=self.dropout
        )

    def load_static_graphs(self, graph_dir, positions, device='cpu'):
        """
        加载预计算的静态图和风机坐标

        Args:
            graph_dir: 静态图文件目录 (包含16个扇区的npy文件)
            positions: [n_nodes, 2] 风机坐标 (numpy或tensor)
            device: 计算设备
        """
        if isinstance(positions, np.ndarray):
            positions = torch.tensor(positions, dtype=torch.float32)
        self.positions = positions.to(device)

        # 加载静态图
        self.static_graph_manager = StaticGraphManager.from_files(graph_dir, device)

        # 初始化动态特征计算器 (需要positions)
        self.dynamic_feature_computer = DynamicEdgeFeatureComputer(
            positions=self.positions,
            rotor_radius=getattr(self, 'rotor_radius', 1.0),
            d_max_factor=getattr(self, 'd_max_factor', 10.0)
        )

    def forward(self, data, device=None):
        """
        前向传播

        Args:
            data: 数据对象, 包含:
                - x: [bs*n_nodes, seq_len, n_features] 节点时序输入
                - edge_index: [2, n_edges] 边索引 (可选, 若无则从静态图获取)
                - edge_attr: [n_edges, static_dim] 静态边特征 (可选)
            device: 计算设备
        Returns:
            pred: [bs*n_nodes, pred_len] 各风机预测风速
        """
        x = data.x  # [bs*n_nodes, seq_len, n_features]
        total_nodes = x.shape[0]
        bs = total_nodes // self.n_nodes

        # 保存归一化参数用于反归一化
        means = x.mean(1, keepdim=True).detach()
        stdev = torch.sqrt(
            torch.var(x - means, dim=1, keepdim=True, unbiased=False) + 1e-5
        )

        # ---- Step 1: 层次化编码 ----
        # 对每台风机分别编码 (共享编码器 + 个体/聚类适配器)
        h_list = []
        for node_id in range(self.n_nodes):
            # 提取该风机在所有batch样本中的数据
            x_i = x[node_id::self.n_nodes]  # [bs, seq_len, n_features]
            h_i, _, _ = self.hierarchical_encoder.encode(
                x_i, node_indices=[node_id] * bs
            )
            h_list.append(h_i)

        H = torch.cat(h_list, dim=0)  # [bs*n_nodes, d_h]

        # ---- Step 2: 风向感知静态图选择 ----
        # 计算第一个样本的主导风向 (同一batch内风向相近)
        # 取所有风机的最近时刻特征
        x_reshaped = x.reshape(bs, self.n_nodes, self.seq_len, -1)
        wspd_all = x_reshaped[0, :, -1, 0]  # [n_nodes] 最近时刻风速
        wdir_all = x_reshaped[0, :, -1, 1]  # [n_nodes] 最近时刻风向

        dominant_dir, consistency = compute_dominant_wind_direction(wspd_all, wdir_all)
        sector_id = self.static_graph_manager.get_sector(dominant_dir)

        # 获取该扇区的静态图
        edge_index, static_weights, static_edge_attr = \
            self.static_graph_manager.get_graph(sector_id)

        # 为batch中的每个样本复制边 (PyG批处理)
        if bs > 1:
            edge_index_list = []
            for b in range(bs):
                edge_index_list.append(edge_index + b * self.n_nodes)
            edge_index = torch.cat(edge_index_list, dim=1)
            static_weights = static_weights.repeat(bs)
            static_edge_attr = static_edge_attr.repeat(bs, 1)

        # ---- Step 3: 动态边特征计算 ----
        # 从最近时间窗口提取14维动态特征
        dynamic_edge_attr = self.dynamic_feature_computer.compute(x, edge_index)

        # ---- Step 4: 动静融合与权重调整 ----
        # 拼接: e_ij = [e_static || e_dynamic] ∈ R^25
        edge_attr = torch.cat([static_edge_attr, dynamic_edge_attr], dim=-1)

        # MLP动态调整: w_ij^(t) = α^static × σ(MLP(e_ij))
        adjusted_weights = self.edge_weight_adjuster(static_weights, edge_attr)

        # ---- Step 5: 边增强型多头图注意力 ----
        H_D = self.emgat(H, edge_index, edge_attr, adjusted_weights)

        # ---- Step 6: 共享解码 ----
        # 重塑为LightTime解码器的输入格式: [bs*n_nodes, n_heads, d_model]
        decoder_input = H_D.reshape(-1, self.hierarchical_encoder.n_heads,
                                    self.hierarchical_encoder.d_model)
        dec_out = self.decoder(decoder_input)  # [bs*n_nodes, pred_len, c_out]

        # 反归一化 (Non-stationary Transformer策略)
        if dec_out.shape[-1] == 1:
            pred = dec_out.squeeze(-1)  # [bs*n_nodes, pred_len]
        else:
            pred = dec_out[:, :, -1]  # 取最后一个特征(风速)

        # 使用目标特征的归一化参数进行反归一化
        pred = pred * stdev[:, 0, -1].unsqueeze(-1).repeat(1, self.pred_len)
        pred = pred + means[:, 0, -1].unsqueeze(-1).repeat(1, self.pred_len)

        return pred

    def forward_stage1(self, x):
        """
        第一阶段前向: 仅时序编解码 (无图结构)

        用于三阶段渐进训练的第一阶段:
        只训练 LT-Encoder + Adapters + LT-Decoder

        Args:
            x: [bs*n_nodes, seq_len, n_features]
        Returns:
            pred: [bs*n_nodes, pred_len]
        """
        total_nodes = x.shape[0]
        bs = total_nodes // self.n_nodes

        # 归一化
        means = x.mean(1, keepdim=True).detach()
        stdev = torch.sqrt(
            torch.var(x - means, dim=1, keepdim=True, unbiased=False) + 1e-5
        )

        # 层次化编码
        h_list = []
        for node_id in range(self.n_nodes):
            x_i = x[node_id::self.n_nodes]
            h_i, _, _ = self.hierarchical_encoder.encode(
                x_i, node_indices=[node_id] * bs
            )
            h_list.append(h_i)
        H = torch.cat(h_list, dim=0)

        # 直接解码 (不经过E-MGAT)
        decoder_input = H.reshape(-1, self.hierarchical_encoder.n_heads,
                                  self.hierarchical_encoder.d_model)
        dec_out = self.decoder(decoder_input)

        if dec_out.shape[-1] == 1:
            pred = dec_out.squeeze(-1)
        else:
            pred = dec_out[:, :, -1]

        pred = pred * stdev[:, 0, -1].unsqueeze(-1).repeat(1, self.pred_len)
        pred = pred + means[:, 0, -1].unsqueeze(-1).repeat(1, self.pred_len)

        return pred


class Model(LightST):
    """
    框架兼容性包装类
    遵循项目约定: 每个模型文件导出名为 Model 的类
    """
    def __init__(self, configs):
        super().__init__(configs)
