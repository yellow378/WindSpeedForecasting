import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm
import warnings
import numpy as np
warnings.filterwarnings("ignore")

# 设置中文字体
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


class EdgeFeatureGraphBuilder:
    """
    构建带有丰富边特征的图
    每条边包含多个特征维度：距离、相关性、方向等
    """

    def __init__(self, location_file, correlation_matrix=None, wind_speed_files_pattern=None):
        """
        初始化边特征图构建器
        
        Parameters:
        location_file: 风机位置文件
        correlation_matrix: 预计算的相关性矩阵 (可选)
        wind_speed_files_pattern: 风速数据文件模式，如 "dated_Turb*.csv" (可选)
        """
        # 加载位置数据
        self.location_df = pd.read_csv(location_file)
        self.coords = self.location_df[["x", "y"]].values
        self.num_nodes = len(self.coords)
        
        # 计算距离矩阵
        self.distance_matrix = squareform(pdist(self.coords, "euclidean"))
        
        # 存储相关性矩阵
        self.correlation_matrix = correlation_matrix
        self.wind_speed_files_pattern = wind_speed_files_pattern
        print(f"初始化完成: {self.num_nodes} 个节点")
    
    def load_wind_speed_data(self, train_ratio=0.7, feature_col="Wspd"):
        """
        从CSV文件加载风速数据
        
        Parameters:
        train_ratio: 训练集比例
        feature_col: 特征列名
        
        Returns:
        numpy array: 风速数据矩阵 [num_turbines, timesteps]
        """
        if self.wind_speed_files_pattern is None:
            raise ValueError("未提供风速文件模式，请在初始化时设置 wind_speed_files_pattern")
        
        import glob
        data_files = glob.glob(self.wind_speed_files_pattern)
        data_files.sort()
        
        if len(data_files) == 0:
            raise ValueError(f"未找到匹配的风速数据文件: {self.wind_speed_files_pattern}")
        
        print(f"找到 {len(data_files)} 个风速数据文件")
        
        # 检查文件数量与节点数量是否匹配
        if len(data_files) != self.num_nodes:
            print(f"警告: 风速文件数量 ({len(data_files)}) 与节点数量 ({self.num_nodes}) 不匹配")
            # 取较小值
            num_files = min(len(data_files), self.num_nodes)
            data_files = data_files[:num_files]
        else:
            num_files = len(data_files)
        
        # 读取第一个文件确定数据长度
        sample_df = pd.read_csv(data_files[0])
        total_timesteps = len(sample_df)
        train_split = int(train_ratio * total_timesteps)
        
        print(f"总时间步数: {total_timesteps}, 训练集时间步数: {train_split}")
        
        # 检查特征列是否存在
        if feature_col not in sample_df.columns:
            print(f"特征列 '{feature_col}' 不存在，可用列: {list(sample_df.columns)}")
            raise ValueError(f"特征列 '{feature_col}' 不存在")
        
        # 初始化数组
        wind_speeds = np.zeros((num_files, train_split))
        problematic_turbines = []
        
        # 加载数据
        for i, file in enumerate(tqdm(data_files, desc="加载风速数据")):
            try:
                df = pd.read_csv(file)
                
                # 检查数据长度
                if len(df) != total_timesteps:
                    print(f"警告: 文件 {file} 的数据长度不一致")
                    min_length = min(len(df), train_split)
                    wind_speeds[i, :min_length] = df[feature_col].iloc[:min_length].values
                else:
                    wind_speeds[i, :] = df[feature_col].iloc[:train_split].values
                
                # 数据质量检查
                turbine_data = wind_speeds[i, :]
                nan_count = np.isnan(turbine_data).sum()
                zero_count = (turbine_data == 0).sum()
                valid_ratio = (len(turbine_data) - nan_count - zero_count) / len(turbine_data)
                
                if valid_ratio < 0.5:
                    print(f"警告: 风机 {i} 有效数据比例过低 ({valid_ratio:.2%})")
                    problematic_turbines.append(i)
                
                # 处理缺失值
                if np.isnan(turbine_data).any():
                    if valid_ratio > 0.1:
                        mean_val = np.nanmean(turbine_data)
                        if not np.isnan(mean_val):
                            mask = np.isnan(turbine_data)
                            wind_speeds[i, mask] = mean_val
                        else:
                            global_mean = np.nanmean(wind_speeds)
                            wind_speeds[i, :] = global_mean
                            problematic_turbines.append(i)
                    else:
                        problematic_turbines.append(i)
                        
            except Exception as e:
                print(f"读取文件 {file} 时出错: {e}")
                problematic_turbines.append(i)
        
        # 处理问题风机
        if problematic_turbines:
            print(f"发现 {len(problematic_turbines)} 个问题风机: {problematic_turbines}")
            global_mean = np.nanmean(
                wind_speeds[~np.isin(range(num_files), problematic_turbines), :]
            )
            for i in problematic_turbines:
                wind_speeds[i, :] = global_mean
            print(f"已用全局均值 {global_mean:.2f} 填充问题风机数据")
        
        self.wind_speed_data = wind_speeds
        self.problematic_turbines = problematic_turbines
        print(f"风速数据加载完成: {wind_speeds.shape}")
        
        return wind_speeds
    
    def compute_correlation_matrix(self, method="pearson", use_abs=False):
        """
        计算风机间的相关性矩阵
        
        Parameters:
        method: 相关性计算方法 ('pearson', 'spearman')
        use_abs: 是否使用绝对值相关性
        
        Returns:
        numpy array: 相关性矩阵
        """
        if self.wind_speed_data is None:
            if self.wind_speed_files_pattern is not None:
                print("风速数据未加载，正在自动加载...")
                self.load_wind_speed_data()
            else:
                raise ValueError("无风速数据，无法计算相关性矩阵")
        
        print(f"正在计算相关性矩阵 (方法: {method})...")
        
        # 数据质量检查
        print("数据质量检查:")
        zero_variance_turbines = []
        for i in range(self.wind_speed_data.shape[0]):
            data = self.wind_speed_data[i, :]
            std_val = np.std(data)
            if std_val == 0:
                print(f"警告: 风机 {i} 的风速数据方差为0 (均值: {np.mean(data):.2f})")
                zero_variance_turbines.append(i)
        
        # 计算相关性矩阵
        if method == "pearson":
            correlation_matrix = np.corrcoef(self.wind_speed_data)
        elif method == "spearman":
            from scipy.stats import spearmanr
            correlation_matrix, _ = spearmanr(self.wind_speed_data, axis=1)
        else:
            raise ValueError(f"不支持的相关性计算方法: {method}")
        
        # 处理NaN值
        nan_mask = np.isnan(correlation_matrix)
        if np.any(nan_mask):
            print("发现相关性矩阵中有NaN值，进行处理...")
            correlation_matrix = np.nan_to_num(correlation_matrix, nan=0.0)
            # 重新设置对角线为1
            np.fill_diagonal(correlation_matrix, 1.0)
        
        # 使用绝对值相关性
        if use_abs:
            correlation_matrix = np.abs(correlation_matrix)
            print("使用绝对值相关性")
        
        # 设置对角线为0（用于图构建）
        np.fill_diagonal(correlation_matrix, 0)
        
        # 更新实例变量
        self.correlation_matrix = correlation_matrix
        
        # 打印统计信息
        self._print_correlation_stats(correlation_matrix)
        
        return correlation_matrix
    
    def _print_correlation_stats(self, correlation_matrix):
        """打印相关性统计信息"""
        # 获取上三角矩阵的相关性值（排除对角线）
        corr_values = correlation_matrix[np.triu_indices_from(correlation_matrix, k=1)]
        
        print("\n=== 相关性统计信息 ===")
        print(f"相关系数范围: [{np.min(corr_values):.3f}, {np.max(corr_values):.3f}]")
        print(f"平均相关系数: {np.mean(corr_values):.3f}")
        print(f"相关系数标准差: {np.std(corr_values):.3f}")
        print(f"中位数相关系数: {np.median(corr_values):.3f}")
        
        # 相关性分布统计
        strong_corr = np.sum(np.abs(corr_values) > 0.7)
        medium_corr = np.sum((np.abs(corr_values) > 0.4) & (np.abs(corr_values) <= 0.7))
        weak_corr = np.sum((np.abs(corr_values) > 0.2) & (np.abs(corr_values) <= 0.4))
        very_weak_corr = np.sum(np.abs(corr_values) <= 0.2)
        
        total_pairs = len(corr_values)
        print(f"\n相关性强度分布:")
        print(f"  强相关 (>0.7): {strong_corr}/{total_pairs} ({strong_corr/total_pairs*100:.1f}%)")
        print(f"  中等相关 (0.4-0.7): {medium_corr}/{total_pairs} ({medium_corr/total_pairs*100:.1f}%)")
        print(f"  弱相关 (0.2-0.4): {weak_corr}/{total_pairs} ({weak_corr/total_pairs*100:.1f}%)")
        print(f"  很弱相关 (≤0.2): {very_weak_corr}/{total_pairs} ({very_weak_corr/total_pairs*100:.1f}%)")
        
        # 不同阈值下的连接统计
        thresholds = [0.3, 0.5, 0.7, 0.8, 0.9]
        print(f"\n不同阈值下的连接统计:")
        for thresh in thresholds:
            connections = np.sum(np.abs(correlation_matrix) > thresh) // 2  # 除以2因为对称
            total_possible = self.num_nodes * (self.num_nodes - 1) // 2
            ratio = connections / total_possible * 100 if total_possible > 0 else 0
            print(f"  阈值 {thresh:.1f}: {connections}/{total_possible} ({ratio:.1f}%) 连接")
    
    def visualize_correlation_matrix(self, save_path="correlation_matrix.png", figsize=(12, 10)):
        """
        可视化相关性矩阵
        
        Parameters:
        save_path: 保存路径
        figsize: 图像大小
        """
        if self.correlation_matrix is None:
            raise ValueError("相关性矩阵不存在，请先调用 compute_correlation_matrix()")
        
        plt.figure(figsize=figsize)
        
        # 创建相关性矩阵的副本用于显示（恢复对角线为1）
        display_matrix = self.correlation_matrix.copy()
        np.fill_diagonal(display_matrix, 1.0)
        
        # 绘制热力图
        im = plt.imshow(display_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
        
        # 标记问题风机
        if hasattr(self, 'problematic_turbines') and self.problematic_turbines:
            for i in self.problematic_turbines:
                plt.axhline(y=i-0.5, color='red', linewidth=2, alpha=0.7)
                plt.axvline(x=i-0.5, color='red', linewidth=2, alpha=0.7)
        
        plt.colorbar(im, label='相关系数')
        plt.title('风速相关性矩阵')
        plt.xlabel('风机ID')
        plt.ylabel('风机ID')
        
        # 添加问题风机说明
        if hasattr(self, 'problematic_turbines') and self.problematic_turbines:
            plt.text(0.02, 0.98, f'红线标记问题风机: {self.problematic_turbines}',
                    transform=plt.gca().transAxes, fontsize=10, 
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"相关性矩阵图已保存为 {save_path}")
    
    def compute_spatial_features(self):
        """计算空间相关的特征"""
        print("计算空间特征...")
        
        features = {}
        
        # 1. 欧几里得距离
        features['euclidean_distance'] = self.distance_matrix.copy()
        
        # 2. 标准化距离 (0-1)
        max_dist = np.max(self.distance_matrix)
        features['normalized_distance'] = self.distance_matrix / max_dist
        
        # 3. 距离倒数 (距离越近权重越大)
        inv_dist = np.zeros_like(self.distance_matrix)
        mask = self.distance_matrix > 0
        inv_dist[mask] = 1.0 / self.distance_matrix[mask]
        features['inverse_distance'] = inv_dist
        
        # 4. 高斯距离核
        sigma = np.mean(self.distance_matrix[self.distance_matrix > 0]) / 3
        features['gaussian_distance'] = np.exp(-(self.distance_matrix ** 2) / (2 * sigma ** 2))
        
        # 5. 相对方位角 (弧度)
        angles = np.zeros((self.num_nodes, self.num_nodes))
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                if i != j:
                    dx = self.coords[j, 0] - self.coords[i, 0]
                    dy = self.coords[j, 1] - self.coords[i, 1]
                    angles[i, j] = np.arctan2(dy, dx)
        features['bearing_angle'] = angles
        
        # 6. 方位角的sin和cos分量 (便于神经网络处理)
        features['bearing_sin'] = np.sin(angles)
        features['bearing_cos'] = np.cos(angles)
        
        # 7. 坐标差值特征
        coord_diff_x = np.zeros((self.num_nodes, self.num_nodes))
        coord_diff_y = np.zeros((self.num_nodes, self.num_nodes))
        
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                coord_diff_x[i, j] = self.coords[j, 0] - self.coords[i, 0]
                coord_diff_y[i, j] = self.coords[j, 1] - self.coords[i, 1]
        
        features['coord_diff_x'] = coord_diff_x
        features['coord_diff_y'] = coord_diff_y
        
        return features
    
    def compute_correlation_features(self):
        """计算相关性相关的特征"""
        if self.correlation_matrix is None:
            print("未提供相关性矩阵，跳过相关性特征计算")
            return {}
            
        print("计算相关性特征...")
        
        features = {}
        
        # 1. 原始相关性
        features['pearson_correlation'] = self.correlation_matrix.copy()
        
        # 2. 绝对相关性
        features['abs_correlation'] = np.abs(self.correlation_matrix)
        
        # 3. 相关性平方 (强调强相关)
        features['correlation_squared'] = self.correlation_matrix ** 2
        
        # 4. 相关性符号 (正相关=1, 负相关=-1)
        corr_sign = np.zeros_like(self.correlation_matrix)
        corr_sign[self.correlation_matrix > 0] = 1
        corr_sign[self.correlation_matrix < 0] = -1
        features['correlation_sign'] = corr_sign
        
        # 5. 相关性等级 (分为强、中、弱相关)
        corr_level = np.zeros_like(self.correlation_matrix)
        abs_corr = np.abs(self.correlation_matrix)
        corr_level[abs_corr > 0.7] = 3  # 强相关
        corr_level[(abs_corr > 0.4) & (abs_corr <= 0.7)] = 2  # 中等相关
        corr_level[(abs_corr > 0.2) & (abs_corr <= 0.4)] = 1  # 弱相关
        features['correlation_level'] = corr_level
        
        return features
    
    def compute_statistical_features(self):
        """计算统计相关的特征"""
        if self.wind_speed_data is None:
            print("未提供风速数据，跳过统计特征计算")
            return {}
            
        print("计算统计特征...")
        features = {}
        
        # 计算每对风机的统计特征
        cross_correlation = np.zeros((self.num_nodes, self.num_nodes))
        variance_ratio = np.zeros((self.num_nodes, self.num_nodes))
        mean_diff = np.zeros((self.num_nodes, self.num_nodes))
        
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                if i != j:
                    data_i = self.wind_speed_data[i, :]
                    data_j = self.wind_speed_data[j, :]
                    
                    # 交叉相关 (滞后0)
                    if np.std(data_i) > 0 and np.std(data_j) > 0:
                        cross_correlation[i, j] = np.corrcoef(data_i, data_j)[0, 1]
                    
                    # 方差比
                    var_i, var_j = np.var(data_i), np.var(data_j)
                    if var_j > 0:
                        variance_ratio[i, j] = var_i / var_j
                    
                    # 均值差
                    mean_diff[i, j] = np.mean(data_i) - np.mean(data_j)
        
        features['cross_correlation'] = cross_correlation
        features['variance_ratio'] = variance_ratio
        features['mean_difference'] = mean_diff
        
        return features
    
    def compute_topological_features(self, k_neighbors=5):
        """计算拓扑相关的特征"""
        print("计算拓扑特征...")
        
        features = {}
        
        # 1. KNN邻接关系
        knn_graph = kneighbors_graph(
            self.coords, k_neighbors, mode='connectivity', include_self=False
        ).toarray()
        features['knn_connectivity'] = knn_graph
        
        # 2. 相互KNN (双向KNN邻居)
        mutual_knn = knn_graph * knn_graph.T
        features['mutual_knn'] = mutual_knn
        
        # 3. KNN距离排名
        knn_rank = np.zeros((self.num_nodes, self.num_nodes))
        for i in range(self.num_nodes):
            distances = self.distance_matrix[i, :]
            sorted_indices = np.argsort(distances)
            for rank, j in enumerate(sorted_indices):
                if i != j:
                    knn_rank[i, j] = rank
        features['knn_rank'] = knn_rank
        
        # 4. 局部密度 (每个节点周围的邻居密度)
        local_density = np.zeros((self.num_nodes, self.num_nodes))
        for i in range(self.num_nodes):
            # 计算节点i的k邻居的平均距离
            neighbors = np.where(knn_graph[i, :] == 1)[0]
            if len(neighbors) > 0:
                avg_neighbor_dist = np.mean(self.distance_matrix[i, neighbors])
                for j in range(self.num_nodes):
                    if i != j:
                        local_density[i, j] = 1.0 / (1.0 + avg_neighbor_dist)
        features['local_density'] = local_density
        
        return features
    
    def build_edge_features(self, feature_types=['spatial', 'correlation', 'statistical', 'topological']):
        """
        构建边特征矩阵
        
        Parameters:
        feature_types: 要包含的特征类型列表
        
        Returns:
        dict: 包含所有特征的字典
        """
        all_features = {}
        
        if 'spatial' in feature_types:
            spatial_features = self.compute_spatial_features()
            all_features.update(spatial_features)
        
        if 'correlation' in feature_types:
            correlation_features = self.compute_correlation_features()
            all_features.update(correlation_features)
        
        if 'statistical' in feature_types:
            statistical_features = self.compute_statistical_features()
            all_features.update(statistical_features)
        
        if 'topological' in feature_types:
            topological_features = self.compute_topological_features()
            all_features.update(topological_features)
        
        self.edge_features = all_features
        return all_features
    
    def create_edge_feature_tensor(self, adjacency_matrix=None, normalize_features=True):
        """
        创建用于GNN的边特征张量
        
        Parameters:
        adjacency_matrix: 邻接矩阵，定义哪些边存在
        normalize_features: 是否标准化特征
        
        Returns:
        edge_index: [2, num_edges] 边索引
        edge_attr: [num_edges, num_features] 边特征
        feature_names: 特征名称列表
        """
        if not hasattr(self, 'edge_features'):
            raise ValueError("请先调用 build_edge_features() 构建边特征")
        
        # 如果没有提供邻接矩阵，使用全连接图
        if adjacency_matrix is None:
            adjacency_matrix = np.ones((self.num_nodes, self.num_nodes)) - np.eye(self.num_nodes)
        
        # 找到所有存在的边
        edge_indices = np.where(adjacency_matrix > 0)
        edge_index = np.stack([edge_indices[0], edge_indices[1]], axis=0)
        num_edges = edge_index.shape[1]
        
        # 收集所有特征
        feature_names = list(self.edge_features.keys())
        num_features = len(feature_names)
        
        # 创建边特征矩阵
        edge_attr = np.zeros((num_edges, num_features))
        
        for feat_idx, feature_name in enumerate(feature_names):
            feature_matrix = self.edge_features[feature_name]
            # 提取对应边的特征值
            edge_attr[:, feat_idx] = feature_matrix[edge_indices[0], edge_indices[1]]
        
        # 标准化特征
        if normalize_features:
            scaler = StandardScaler()
            edge_attr = scaler.fit_transform(edge_attr)
            
            # 保存scaler用于后续使用
            self.feature_scaler = scaler
        
        print(f"边特征张量构建完成:")
        print(f"  边数量: {num_edges}")
        print(f"  特征维度: {num_features}")
        print(f"  特征名称: {feature_names}")
        
        return edge_index, edge_attr, feature_names
    
    def visualize_edge_features(self, feature_names=None, save_path="edge_features_analysis.png"):
        """可视化边特征分布"""
        if not hasattr(self, 'edge_features'):
            raise ValueError("请先调用 build_edge_features() 构建边特征")
        
        if feature_names is None:
            feature_names = list(self.edge_features.keys())[:9]  # 最多显示9个特征
        
        n_features = len(feature_names)
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_rows == 1 else axes
        
        for i, feature_name in enumerate(feature_names):
            feature_matrix = self.edge_features[feature_name]
            
            # 只考虑非对角线元素
            mask = ~np.eye(self.num_nodes, dtype=bool)
            feature_values = feature_matrix[mask]
            
            # 移除无效值
            feature_values = feature_values[np.isfinite(feature_values)]
            
            if len(feature_values) > 0:
                axes[i].hist(feature_values, bins=50, alpha=0.7, edgecolor='black')
                axes[i].set_title(f'{feature_name}\n均值: {np.mean(feature_values):.3f}')
                axes[i].set_xlabel('特征值')
                axes[i].set_ylabel('频数')
                axes[i].grid(True, alpha=0.3)
        
        # 隐藏多余的子图
        for i in range(n_features, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"特征分布图已保存为 {save_path}")
    
    def create_graph_with_features(self, graph_type='knn', k=5, threshold=0.3):
        """
        创建带有边特征的图
        
        Parameters:
        graph_type: 'knn', 'threshold', 'full'
        k: KNN的k值
        threshold: 阈值图的阈值
        
        Returns:
        NetworkX图对象，包含边特征
        """
        if not hasattr(self, 'edge_features'):
            raise ValueError("请先调用 build_edge_features() 构建边特征")
        
        # 创建邻接矩阵
        if graph_type == 'knn':
            adj_matrix = kneighbors_graph(
                self.coords, k, mode='connectivity', include_self=False
            ).toarray()
            # 确保对称性
            adj_matrix = (adj_matrix + adj_matrix.T > 0).astype(int)
            
        elif graph_type == 'threshold':
            if self.correlation_matrix is not None:
                adj_matrix = (np.abs(self.correlation_matrix) > threshold).astype(int)
            else:
                # 使用距离阈值
                dist_threshold = np.percentile(self.distance_matrix[self.distance_matrix > 0], 20)
                adj_matrix = (self.distance_matrix < dist_threshold).astype(int)
                np.fill_diagonal(adj_matrix, 0)
                
        elif graph_type == 'full':
            adj_matrix = np.ones((self.num_nodes, self.num_nodes)) - np.eye(self.num_nodes)
        
        # 创建NetworkX图
        G = nx.from_numpy_array(adj_matrix)
        
        # 为图添加位置信息
        pos = dict(zip(range(self.num_nodes), self.coords))
        nx.set_node_attributes(G, pos, 'pos')
        
        # 为边添加特征
        for u, v in G.edges():
            edge_features = {}
            for feature_name, feature_matrix in self.edge_features.items():
                edge_features[feature_name] = feature_matrix[u, v]
            G[u][v].update(edge_features)
        
        return G, adj_matrix
    
    def save_edge_features_for_pytorch_geometric(self, adj_matrix, save_prefix="graph_data"):
        """
        保存为PyTorch Geometric格式
        """
        edge_index, edge_attr, feature_names = self.create_edge_feature_tensor(adj_matrix)
        
        # 保存数据
        np.save(f"{save_prefix}_edge_index.npy", edge_index)
        np.save(f"{save_prefix}_edge_attr.npy", edge_attr)
        np.save(f"{save_prefix}_node_coords.npy", self.coords)
        
        # 保存特征名称
        with open(f"{save_prefix}_feature_names.txt", "w") as f:
            for name in feature_names:
                f.write(f"{name}\n")
        
        print(f"PyTorch Geometric数据已保存:")
        print(f"  {save_prefix}_edge_index.npy - 边索引 [{edge_index.shape}]")
        print(f"  {save_prefix}_edge_attr.npy - 边特征 [{edge_attr.shape}]")
        print(f"  {save_prefix}_node_coords.npy - 节点坐标 [{self.coords.shape}]")
        print(f"  {save_prefix}_feature_names.txt - 特征名称")
        
        return edge_index, edge_attr, feature_names


def create_comprehensive_edge_feature_graph(
    location_file,
    correlation_matrix_path=None,
    wind_speed_files_pattern=None,
    graph_type='knn',
    k=5,
    compute_correlation=True
):
    """
    完整的边特征图构建工作流程
    Parameters:
    location_file: 位置文件路径
    correlation_matrix_path: 预计算的相关性矩阵路径
    wind_speed_files_pattern: 风速CSV文件模式，如 "dated_Turb*.csv"
    graph_type: 图类型 ('knn', 'threshold', 'full')
    k: KNN的k值
    compute_correlation: 是否自动计算相关性矩阵
    """
    print("=== 边特征图构建工作流程 ===\n")
    
    # 1. 初始化
    print("1. 初始化构建器...")
    correlation_matrix = None
    
    # 加载预计算的相关性矩阵
    if correlation_matrix_path:
        try:
            correlation_matrix = np.load(correlation_matrix_path)
            print(f"   已加载相关性矩阵: {correlation_matrix.shape}")
        except Exception as e:
            print(f"   加载相关性矩阵失败: {e}")
    
    # 初始化构建器
    builder = EdgeFeatureGraphBuilder(
        location_file, 
        correlation_matrix, 
        wind_speed_files_pattern
    )
    
    # 2. 处理相关性矩阵
    if correlation_matrix is None and compute_correlation:
        print("\n2. 从CSV文件加载风速数据并计算相关性...")
        try:
            # 加载风速数据
            wind_speed_data = builder.load_wind_speed_data()
            
            # 计算相关性矩阵
            correlation_matrix = builder.compute_correlation_matrix(method="pearson")
            
            # 可视化相关性矩阵
            builder.visualize_correlation_matrix()
            
            # 保存相关性矩阵
            np.save("computed_correlation_matrix.npy", correlation_matrix)
            print("   相关性矩阵已保存为 computed_correlation_matrix.npy")
            
        except Exception as e:
            print(f"   计算相关性矩阵失败: {e}")
            print("   将仅使用空间特征")
    
    # 3. 构建边特征
    print(f"\n3. 构建边特征...")
    feature_types = ['spatial']  # 空间特征始终包含
    
    if builder.correlation_matrix is not None:
        feature_types.append('correlation')
        print("   包含相关性特征")
    
    if builder.wind_speed_data is not None:
        feature_types.extend(['statistical'])
        print("   包含统计特征")
    
    feature_types.append('topological')
    print("   包含拓扑特征")
    
    edge_features = builder.build_edge_features(feature_types)
    print(f"   构建了 {len(edge_features)} 种边特征")
    
    # 4. 可视化特征分布
    print(f"\n4. 可视化特征分布...")
    builder.visualize_edge_features()
    
    # 5. 创建图
    print(f"\n5. 创建{graph_type}图...")
    G, adj_matrix = builder.create_graph_with_features(graph_type=graph_type, k=k)
    
    # 6. 创建PyTorch Geometric格式数据
    print(f"\n6. 创建PyTorch Geometric数据...")
    edge_index, edge_attr, feature_names = builder.save_edge_features_for_pytorch_geometric(
        adj_matrix, f"edge_feature_{graph_type}"
    )
    
    print(f"\n=== 构建完成 ===")
    print(f"图统计:")
    print(f"  节点数: {G.number_of_nodes()}")
    print(f"  边数: {G.number_of_edges()}")
    print(f"  边特征维度: {len(feature_names)}")
    print(f"  特征列表: {feature_names}")
    
    # 使用建议
    print(f"\n使用建议:")
    print(f"在PyTorch Geometric中使用:")
    print(f"```python")
    print(f"import torch")
    print(f"import numpy as np")
    print(f"from torch_geometric.nn import GATConv")
    print(f"")
    print(f"# 加载数据")
    print(f"edge_index = torch.from_numpy(np.load('edge_feature_{graph_type}_edge_index.npy')).long()")
    print(f"edge_attr = torch.from_numpy(np.load('edge_feature_{graph_type}_edge_attr.npy')).float()")
    print(f"node_coords = torch.from_numpy(np.load('edge_feature_{graph_type}_node_coords.npy')).float()")
    print(f"")
    print(f"# 在GATConv中使用边特征")
    print(f"conv = GATConv(in_channels, out_channels, edge_dim={len(feature_names)})")
    print(f"out = conv(x, edge_index, edge_attr)")
    print(f"```")
    
    return builder, G, edge_index, edge_attr, feature_names


# 使用示例
if __name__ == "__main__":
    # 示例：构建KNN图的边特征
    builder, G, edge_index, edge_attr, feature_names = create_comprehensive_edge_feature_graph(
        location_file="sdwpf_baidukddcup2022_turb_location.CSV",
        correlation_matrix_path="computed_correlation_matrix.npy",  # 如果有的话
        graph_type='knn',
        k=5,
        wind_speed_files_pattern="dated_Turb*.csv",  # 风速数据文件模式
        compute_correlation=True
    )
    
    # 查看边特征示例
    print(f"\n边特征示例 (前5条边):")
    print(f"Edge Index: {edge_index[:, :5]}")
    print(f"Edge Features Shape: {edge_attr[:5, :].shape}")
    print(f"Feature Names: {feature_names}")