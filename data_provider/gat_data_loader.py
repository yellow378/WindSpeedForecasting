import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics.pairwise import pairwise_distances
import warnings
warnings.filterwarnings('ignore')


class GraphData:
    """
    简单的图数据类，替代PyTorch Geometric的Data
    """
    def __init__(self, x=None, edge_index=None, y=None, **kwargs):
        self.x = x
        self.edge_index = edge_index
        self.y = y
        
        # 存储额外的属性
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def to(self, device):
        """将数据移动到指定设备"""
        if self.x is not None:
            self.x = self.x.to(device)
        if self.edge_index is not None:
            self.edge_index = self.edge_index.to(device)
        if self.y is not None:
            self.y = self.y.to(device)
        return self


class GraphBatch:
    """
    简单的图批处理类，替代PyTorch Geometric的Batch
    """
    def __init__(self, batch_data):
        """
        :param batch_data: GraphData对象列表
        """
        self.batch_size = len(batch_data)
        self.n_nodes = batch_data[0].x.shape[0] if batch_data[0].x is not None else 0
        
        # 合并x
        if batch_data[0].x is not None:
            # [batch_size, n_nodes, seq_len, n_features]
            self.x = torch.stack([data.x for data in batch_data], dim=0)
        else:
            self.x = None
        
        # 合并y
        if batch_data[0].y is not None:
            # [batch_size, n_nodes, pred_len] or [batch_size * n_nodes, pred_len]
            if len(batch_data[0].y.shape) == 2:  # [n_nodes, pred_len]
                self.y = torch.stack([data.y for data in batch_data], dim=0)
            else:  # [pred_len] - single node
                self.y = torch.stack([data.y for data in batch_data], dim=0)
        else:
            self.y = None
        
        # 合并edge_index (假设所有图具有相同的结构)
        if batch_data[0].edge_index is not None:
            # 为每个批次创建独立的边索引
            batch_edge_indices = []
            for i, data in enumerate(batch_data):
                # 添加节点偏移
                offset_edges = data.edge_index + i * self.n_nodes
                batch_edge_indices.append(offset_edges)
            
            self.edge_index = torch.cat(batch_edge_indices, dim=1)
            
            # 创建batch标识符
            self.batch = torch.cat([torch.full((self.n_nodes,), i, dtype=torch.long) 
                                   for i in range(self.batch_size)])
        else:
            self.edge_index = None
            self.batch = None
    
    def to(self, device):
        """将数据移动到指定设备"""
        if self.x is not None:
            self.x = self.x.to(device)
        if self.edge_index is not None:
            self.edge_index = self.edge_index.to(device)
        if self.y is not None:
            self.y = self.y.to(device)
        if self.batch is not None:
            self.batch = self.batch.to(device)
        return self


class TimeSeriesGraphDataset(Dataset):
    """
    时间序列图数据集 - 不依赖PyTorch Geometric
    """
    def __init__(self, data_dir, seq_len=96, pred_len=24, target_col='Wspd', 
                 scaler_type='standard', edge_file=None, flag='train'):
        """
        初始化数据集
        :param data_dir: 数据文件目录
        :param seq_len: 输入序列长度
        :param pred_len: 预测长度
        :param target_col: 目标列名
        :param scaler_type: 标准化类型 ('standard', 'minmax', None)
        :param edge_file: 边文件路径，如果为None则自动生成
        :param flag: 数据集类型 ('train', 'val', 'test')
        """
        self.data_dir = data_dir
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.target_col = target_col
        self.scaler_type = scaler_type
        self.flag = flag
        
        print(f"Loading data files for {flag} set...")
        # 加载所有数据文件
        self.data_files = self._get_data_files()
        self.raw_data = self._load_all_data()
        
        print("Preprocessing data...")
        # 预处理数据
        self.processed_data, self.scalers = self._preprocess_data()
        
        print("Generating/loading graph edges...")
        # 生成或加载边信息
        if edge_file and os.path.exists(edge_file):
            self.edge_index = self._load_edge_index(edge_file)
        else:
            self.edge_index = self._generate_edges()
            if edge_file:
                self._save_edge_index(self.edge_index, edge_file)
        
        print(f"Creating samples for {flag} set...")
        # 创建样本索引
        self.samples = self._create_samples()
        
        print(f"{flag.upper()} dataset initialized: {len(self.samples)} samples, {len(self.processed_data)} nodes")
        
    def _get_data_files(self):
        """获取所有CSV数据文件"""
        files = []
        for file in os.listdir(self.data_dir):
            if file.endswith('.csv'):
                files.append(os.path.join(self.data_dir, file))
        files = sorted(files)
        #print(f"Found {len(files)} data files")
        return files
    
    def _load_all_data(self):
        """加载所有数据文件"""
        all_data = {}
        
        for file_path in self.data_files:
            node_name = os.path.basename(file_path).replace('.csv', '')
            try:
                df = pd.read_csv(file_path)
                
                # 解析日期时间
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                    df = df.sort_values('date').reset_index(drop=True)
                
                # 存储数据
                all_data[node_name] = df
                #print(f"Loaded {node_name}: {len(df)} records")
                
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue
                
        return all_data
    
    def _preprocess_data(self):
        """预处理数据：标准化、对齐时间等"""
        processed_data = {}
        scalers = {}
        
        # 找到所有节点的共同时间范围
        common_time_range = self._get_common_time_range()
        
        for node_name, df in self.raw_data.items():
            try:
                # 过滤到共同时间范围
                if 'date' in df.columns and common_time_range[0] is not None:
                    mask = (df['date'] >= common_time_range[0]) & (df['date'] <= common_time_range[1])
                    df_filtered = df[mask].copy()
                else:
                    df_filtered = df.copy()
                
                # 选择特征列（除了date列）
                feature_cols = [col for col in df_filtered.columns if col != 'date']
                features = df_filtered[feature_cols].values
                
                # 处理NaN值
                if np.isnan(features).any():
                    print(f"Warning: NaN values found in {node_name}, filling with forward fill")
                    features_df = pd.DataFrame(features, columns=feature_cols)
                    features_df = features_df.fillna(method='ffill').fillna(method='bfill')
                    features = features_df.values
                
                # 数据标准化 - 注意：验证集和测试集应该使用训练集的标准化参数
                if self.scaler_type == 'standard':
                    scaler = StandardScaler()
                elif self.scaler_type == 'minmax':
                    scaler = MinMaxScaler()
                else:
                    scaler = None
                
                if scaler is not None:
                    if self.flag == 'train':
                        # 训练集：拟合并转换
                        features_scaled = scaler.fit_transform(features)
                        scalers[node_name] = scaler
                    else:
                        # 验证集和测试集：只转换（需要预先拟合的scaler）
                        # 这里暂时用fit_transform，实际使用时需要传入训练集的scaler
                        features_scaled = scaler.fit_transform(features)
                        scalers[node_name] = scaler
                        print(f"Warning: {self.flag} set should use scaler fitted on train set")
                else:
                    features_scaled = features
                    scalers[node_name] = None
                
                processed_data[node_name] = {
                    'features': features_scaled,
                    'raw_features': features,
                    'columns': feature_cols,
                    'dates': df_filtered['date'].values if 'date' in df_filtered.columns else None
                }
                
                print(f"Processed {node_name}: {features_scaled.shape}")
                
            except Exception as e:
                print(f"Error preprocessing {node_name}: {e}")
                continue
        
        return processed_data, scalers
    
    def _get_common_time_range(self):
        """获取所有节点的共同时间范围"""
        if not self.raw_data:
            return None, None
            
        first_df = list(self.raw_data.values())[0]
        if 'date' not in first_df.columns:
            return None, None
            
        min_start = None
        max_end = None
        
        for df in self.raw_data.values():
            if 'date' in df.columns:
                start_time = df['date'].min()
                end_time = df['date'].max()
                
                if min_start is None or start_time > min_start:
                    min_start = start_time
                if max_end is None or end_time < max_end:
                    max_end = end_time
        
        print(f"Common time range: {min_start} to {max_end}")
        return min_start, max_end
    
    def _generate_edges(self):
        """生成图的边，基于多种相似性度量"""
        print("Generating graph edges based on correlation and similarity...")
        
        node_names = list(self.processed_data.keys())
        n_nodes = len(node_names)
        
        if n_nodes < 2:
            print("Warning: Only one node found, creating self-loop")
            return torch.tensor([[0], [0]], dtype=torch.long)
        
        # 提取目标变量的时间序列用于相似性计算
        target_series = []
        for node_name in node_names:
            data = self.processed_data[node_name]
            if self.target_col in data['columns']:
                target_idx = data['columns'].index(self.target_col)
                target_series.append(data['features'][:, target_idx])
            else:
                # 如果没有目标列，使用第一列
                print(f"Warning: {self.target_col} not found in {node_name}, using first column")
                target_series.append(data['features'][:, 0])
        
        # 确保所有序列长度一致
        min_length = min(len(series) for series in target_series)
        target_series = [series[:min_length] for series in target_series]
        target_series = np.array(target_series)  # [n_nodes, time_steps]
        
        print(f"Computing similarities for {n_nodes} nodes with {min_length} time steps")
        
        # 计算多种相似性度量
        similarity_matrices = {}
        
        # 1. 皮尔逊相关系数
        similarity_matrices['pearson'] = self._compute_pearson_correlation(target_series)
        
        # 2. 斯皮尔曼相关系数  
        similarity_matrices['spearman'] = self._compute_spearman_correlation(target_series)
        
        # 3. 余弦相似性
        similarity_matrices['cosine'] = self._compute_cosine_similarity(target_series)
        
        # 4. 欧氏距离相似性
        similarity_matrices['euclidean'] = self._compute_euclidean_similarity(target_series)
        
        # 组合多种相似性度量
        edge_index = self._combine_similarities(similarity_matrices, node_names)
        
        return edge_index
    
    def _compute_pearson_correlation(self, series):
        """计算皮尔逊相关系数矩阵"""
        n_nodes = series.shape[0]
        corr_matrix = np.zeros((n_nodes, n_nodes))
        
        for i in range(n_nodes):
            for j in range(n_nodes):
                if i == j:
                    corr_matrix[i, j] = 1.0
                else:
                    try:
                        corr, _ = pearsonr(series[i], series[j])
                        corr_matrix[i, j] = abs(corr) if not np.isnan(corr) else 0.0
                    except:
                        corr_matrix[i, j] = 0.0
        
        return corr_matrix
    
    def _compute_spearman_correlation(self, series):
        """计算斯皮尔曼相关系数矩阵"""
        n_nodes = series.shape[0]
        corr_matrix = np.zeros((n_nodes, n_nodes))
        
        for i in range(n_nodes):
            for j in range(n_nodes):
                if i == j:
                    corr_matrix[i, j] = 1.0
                else:
                    try:
                        corr, _ = spearmanr(series[i], series[j])
                        corr_matrix[i, j] = abs(corr) if not np.isnan(corr) else 0.0
                    except:
                        corr_matrix[i, j] = 0.0
        
        return corr_matrix
    
    def _compute_euclidean_similarity(self, series):
        """计算欧氏距离相似性"""
        try:
            distances = pairwise_distances(series, metric='euclidean')
            max_dist = np.max(distances)
            if max_dist > 0:
                similarity_matrix = 1 - (distances / max_dist)
            else:
                similarity_matrix = np.ones_like(distances)
            return similarity_matrix
        except:
            n_nodes = series.shape[0]
            return np.eye(n_nodes)
    
    def _compute_cosine_similarity(self, series):
        """计算余弦相似性"""
        try:
            from sklearn.metrics.pairwise import cosine_similarity
            similarity_matrix = cosine_similarity(series)
            return np.abs(similarity_matrix)  # 取绝对值
        except:
            n_nodes = series.shape[0]
            return np.eye(n_nodes)
    
    def _combine_similarities(self, similarity_matrices, node_names, threshold=0.3, top_k=3):
        """组合多种相似性度量生成边"""
        n_nodes = len(node_names)
        
        # 权重组合不同的相似性度量
        weights = {
            'pearson': 0.4,
            'spearman': 0.3,
            'euclidean': 0.15,
            'cosine': 0.15
        }
        
        # 计算加权平均相似性
        combined_similarity = np.zeros((n_nodes, n_nodes))
        for metric, weight in weights.items():
            if metric in similarity_matrices:
                combined_similarity += weight * similarity_matrices[metric]
        
        # 生成边的方法：每个节点的top-k连接 + 阈值过滤
        edges = set()
        
        # 方法1: Top-K连接
        for i in range(n_nodes):
            similarities = combined_similarity[i].copy()
            similarities[i] = -1  # 排除自连接
            
            # 获取top-k个最相似的节点
            if np.max(similarities) > 0:
                top_indices = np.argsort(similarities)[-top_k:]
                for j in top_indices:
                    if similarities[j] > threshold:
                        edges.add((i, j))
                        edges.add((j, i))  # 无向图
        
        # 方法2: 如果边太少，降低阈值
        if len(edges) < n_nodes:
            print(f"Too few edges ({len(edges)}), lowering threshold")
            lower_threshold = threshold * 0.5
            for i in range(n_nodes):
                for j in range(n_nodes):
                    if i != j and combined_similarity[i, j] > lower_threshold:
                        edges.add((i, j))
                        edges.add((j, i))
        
        # 方法3: 如果还是没有足够的边，创建基于距离的连接
        if len(edges) < n_nodes:
            print("Creating distance-based connections")
            for i in range(n_nodes):
                # 至少连接到一个最相似的节点
                similarities = combined_similarity[i].copy()
                similarities[i] = -1
                if np.max(similarities) > 0:
                    best_j = np.argmax(similarities)
                    edges.add((i, best_j))
                    edges.add((best_j, i))
        
        # 转换为tensor
        if len(edges) == 0:
            # 创建自环作为最后的备选
            edges = [(i, i) for i in range(n_nodes)]
        
        edges_list = list(edges)
        edge_index = torch.tensor(edges_list, dtype=torch.long).t().contiguous()
        
        print(f"Generated {edge_index.shape[1]} edges for {n_nodes} nodes")
        return edge_index
    
    def _create_samples(self):
        """创建训练样本，按照flag进行7:2:1划分"""
        samples = []
        node_names = list(self.processed_data.keys())
        
        if not node_names:
            raise ValueError("No valid data found")
        
        # 找到最短的时间序列长度
        min_length = min(len(data['features']) for data in self.processed_data.values())
        max_start_idx = min_length - self.seq_len - self.pred_len
        
        if max_start_idx <= 0:
            raise ValueError(f"数据长度不足，需要至少 {self.seq_len + self.pred_len} 个时间点, 当前最短长度: {min_length}")
        
        # 创建所有可能的样本索引
        all_samples = []
        for start_idx in range(0, max_start_idx, 1):  # 步长为1
            sample = {
                'start_idx': start_idx,
                'end_idx': start_idx + self.seq_len,
                'pred_start_idx': start_idx + self.seq_len,
                'pred_end_idx': start_idx + self.seq_len + self.pred_len,
                'node_names': node_names
            }
            all_samples.append(sample)
        
        # 按照7:2:1的比例划分数据集
        total_samples = len(all_samples)
        train_end = int(total_samples * 0.7)
        val_end = int(total_samples * 0.9)  # 0.7 + 0.2 = 0.9
        
        if self.flag == 'train':
            samples = all_samples[:train_end]
        elif self.flag == 'val':
            samples = all_samples[train_end:val_end]
        elif self.flag == 'test':
            samples = all_samples[val_end:]
        else:
            raise ValueError(f"Invalid flag: {self.flag}. Must be 'train', 'val', or 'test'")
        
        print(f"Created {len(samples)} {self.flag} samples (seq_len={self.seq_len}, pred_len={self.pred_len})")
        print(f"Split info - Total: {total_samples}, Train: {train_end}, Val: {val_end-train_end}, Test: {total_samples-val_end}")
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 构建图数据
        node_features = []
        target_values = []
        
        for node_name in sample['node_names']:
            data = self.processed_data[node_name]
            
            # 输入特征 [seq_len, n_features]
            node_feature = data['features'][sample['start_idx']:sample['end_idx']]
            node_features.append(node_feature)
            
            # 目标值（预测Wspd）[pred_len]
            if self.target_col in data['columns']:
                target_idx = data['columns'].index(self.target_col)
                target_val = data['features'][sample['pred_start_idx']:sample['pred_end_idx'], target_idx]
            else:
                # 如果没有目标列，使用第一列
                target_val = data['features'][sample['pred_start_idx']:sample['pred_end_idx'], 0]
            
            target_values.append(target_val)
        
        # 转换为tensor
        x = torch.FloatTensor(np.array(node_features))  # [n_nodes, seq_len, n_features]
        y = torch.FloatTensor(np.array(target_values))   # [n_nodes, pred_len]
        
        # 创建图数据对象
        graph_data = GraphData(
            x=x,
            edge_index=self.edge_index,
            y=y
        )
        
        return graph_data
    
    def _save_edge_index(self, edge_index, file_path):
        """保存边索引到文件"""
        edges_df = pd.DataFrame({
            'source': edge_index[0].numpy(),
            'target': edge_index[1].numpy()
        })
        edges_df.to_csv(file_path, index=False)
        print(f"Edges saved to {file_path}")
    
    def _load_edge_index(self, file_path):
        """从文件加载边索引"""
        edges_df = pd.read_csv(file_path)
        edge_index = torch.tensor([edges_df['source'].values, edges_df['target'].values], dtype=torch.long)
        print(f"Loaded {edge_index.shape[1]} edges from {file_path}")
        return edge_index
    
    def get_scaler(self, node_name):
        """获取特定节点的标准化器"""
        return self.scalers.get(node_name)
    
    def inverse_transform(self, data, node_name, target_col_only=True):
        """反标准化数据"""
        scaler = self.scalers.get(node_name)
        if scaler is None:
            return data
        
        if target_col_only:
            # 只反标准化目标列
            node_data = self.processed_data[node_name]
            if self.target_col in node_data['columns']:
                target_idx = node_data['columns'].index(self.target_col)
                # 创建完整的特征矩阵用于反标准化
                if len(data.shape) == 1:
                    data = data.reshape(-1, 1)
                full_data = np.zeros((data.shape[0], len(node_data['columns'])))
                full_data[:, target_idx] = data.flatten()
                full_data_inv = scaler.inverse_transform(full_data)
                return full_data_inv[:, target_idx].reshape(data.shape)
        else:
            return scaler.inverse_transform(data)
        
        return data


def collate_fn(batch):
    """自定义批处理函数"""
    return GraphBatch(batch)


def create_dataloader(args, flag):
    """创建数据加载器"""
    shuffle_flag = flag == 'train'  # 只有训练集需要shuffle
    drop_last = flag == 'train'     # 只有训练集需要drop_last
    batch_size = args.batch_size

    # 根据args构建参数
    dataset_args = {
        'data_dir': args.root_path,
        'seq_len': args.seq_len,
        'pred_len': args.pred_len,
        'target_col': args.target,
        'scaler_type': "standard",
        'flag': flag
    }
    
    # 添加边文件路径（如果提供）
    if hasattr(args, 'edge_file') and args.edge_file:
        dataset_args['edge_file'] = args.edge_file
    else:
        # 自动生成边文件路径
        import os
        edge_file = os.path.join(args.root_path, 'generated_edges.csv')
        dataset_args['edge_file'] = edge_file

    dataset = TimeSeriesGraphDataset(**dataset_args)
    print(f"{flag} dataset size: {len(dataset)}")
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=0,  # 设为0避免多进程问题
        pin_memory=True,
        drop_last=drop_last,
        collate_fn=collate_fn
    )
    
    return dataloader, dataset  # 注意：返回顺序是 (dataloader, dataset)
    
    return dataloader, dataset


# 使用示例
def create_all_dataloaders(args):
    """创建训练、验证和测试数据加载器"""
    train_loader, train_dataset = create_dataloader(args, 'train')
    val_loader, val_dataset = create_dataloader(args, 'val')
    test_loader, test_dataset = create_dataloader(args, 'test')
    
    return {
        'train': (train_loader, train_dataset),
        'val': (val_loader, val_dataset),
        'test': (test_loader, test_dataset)
    }


# 测试函数
def test_dataloader():
    """测试数据加载器的功能"""
    
    # 创建示例数据
    def create_sample_data(data_dir='./sample_data', n_files=5, n_timesteps=500):
        """创建示例数据文件用于测试"""
        os.makedirs(data_dir, exist_ok=True)
        
        np.random.seed(42)
        
        for i in range(n_files):
            # 生成时间序列数据
            dates = pd.date_range('2024-01-01', periods=n_timesteps, freq='10T')
            
            # 生成相关的时间序列
            base_trend = np.sin(np.linspace(0, 4*np.pi, n_timesteps)) * 10 + 50
            noise = np.random.normal(0, 2, n_timesteps)
            
            # Wspd - 风速（目标变量）
            wspd = np.maximum(0, base_trend + noise + np.random.normal(0, 1, n_timesteps))
            
            # 其他相关变量
            wdir = np.random.uniform(0, 360, n_timesteps)  # 风向
            etmp = base_trend * 0.5 + np.random.normal(25, 3, n_timesteps)  # 环境温度
            itmp = etmp + np.random.normal(3, 1, n_timesteps)  # 内部温度
            ndir = np.random.uniform(0, 360, n_timesteps)  # 机舱方向
            
            # 功率相关变量
            pab1 = np.random.choice([0, 1], n_timesteps, p=[0.1, 0.9])
            pab2 = np.random.choice([0, 1], n_timesteps, p=[0.1, 0.9])
            pab3 = np.random.choice([0, 1], n_timesteps, p=[0.1, 0.9])
            
            # 功率输出（与风速相关）
            prtv = wspd * 0.05 + np.random.normal(0, 0.3, n_timesteps)
            patv = np.maximum(0, wspd * 5 + np.random.normal(0, 30, n_timesteps))
            
            # 创建DataFrame
            df = pd.DataFrame({
                'date': dates,
                'Wspd': wspd,
                'Wdir': wdir,
                'Etmp': etmp,
                'Itmp': itmp,
                'Ndir': ndir,
                'Pab1': pab1,
                'Pab2': pab2,
                'Pab3': pab3,
                'Prtv': prtv,
                'Patv': patv
            })
            
            # 保存文件
            filename = f'turbine_{i+1:02d}.csv'
            df.to_csv(os.path.join(data_dir, filename), index=False)
            
        print(f"Created {n_files} sample data files in {data_dir}")
    
    # 如果没有数据目录，创建示例数据
    data_dir = './sample_data'
    if not os.path.exists(data_dir) or len(os.listdir(data_dir)) == 0:
        create_sample_data(data_dir)
    
    # 测试数据加载器
    print("\n=== Testing DataLoader ===")
    try:
        dataloader, dataset = create_dataloader(
            data_dir=data_dir,
            batch_size=4,
            seq_len=96,
            pred_len=24,
            target_col='Wspd',
            scaler_type='standard',
            edge_file='generated_edges.csv',
            shuffle=True
        )
        
        print(f"Dataset size: {len(dataset)}")
        print(f"Number of nodes: {len(dataset.processed_data)}")
        print(f"Number of edges: {dataset.edge_index.shape[1]}")
        
        # 测试几个批次
        for i, batch in enumerate(dataloader):
            print(f"\nBatch {i}:")
            print(f"  Input shape (x): {batch.x.shape}")
            print(f"  Target shape (y): {batch.y.shape}")
            print(f"  Edge index shape: {batch.edge_index.shape}")
            print(f"  Batch tensor shape: {batch.batch.shape if batch.batch is not None else 'None'}")
            
            if i >= 2:  # 只看前3个批次
                break
        
        # 测试反标准化
        print("\n=== Testing Inverse Transform ===")
        node_names = list(dataset.processed_data.keys())
        if len(node_names) > 0:
            sample_data = batch.y[0, 0, :].numpy()  # 第一个样本，第一个节点
            original_data = dataset.inverse_transform(
                sample_data, 
                node_names[0], 
                target_col_only=True
            )
            print(f"Transformed data range: [{sample_data.min():.3f}, {sample_data.max():.3f}]")
            print(f"Original data range: [{original_data.min():.3f}, {original_data.max():.3f}]")
        
        print("\n=== Test Completed Successfully! ===")
        return dataloader, dataset
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()


# 配置类
class DataConfig:
    """数据配置类，兼容GAT模型"""
    def __init__(self):
        # 数据路径配置
        self.root_path = '../dataset/spatial_wind'  # 数据目录
        
        # 序列长度配置
        self.seq_len = 96
        self.pred_len = 24
        
        # 目标变量配置
        self.target = 'Wspd'
        self.enc_in = 10  # 输入特征数量
        self.c_out = 1    # 输出特征数量
        self.out_channels = 1  # GAT输出通道数
        
        # 数据预处理配置
        self.scaler_type = 'standard'  # 'standard', 'minmax', None
        
        # 批处理配置
        self.batch_size = 32
        self.num_workers = 0
        self.shuffle = True
        
        # 图配置
        self.n_nodes = None  # 将在运行时确定
        self.heads = 8
        self.dropout = 0.1
        
        # 边生成配置
        self.correlation_threshold = 0.3
        self.top_k_edges = 3
        
        # 模型配置（与LightTime兼容）
        self.d_model = 512
        self.d_ff = 2048
        self.n_heads = 8
        self.moving_avg = 25
        self.noEx = True
        self.stride = 8
        self.patch_len = 16
        self.task_name = 'long_term_forecast'


# 高级边生成器（独立版本）
class AdvancedEdgeGenerator:
    """高级边生成器，不依赖PyTorch Geometric"""
    
    def __init__(self, data_dir, target_col='Wspd'):
        self.data_dir = data_dir
        self.target_col = target_col
        self.data = self._load_data()
        
    def _load_data(self):
        """加载数据"""
        data = {}
        for file in os.listdir(self.data_dir):
            if file.endswith('.csv'):
                node_name = file.replace('.csv', '')
                try:
                    df = pd.read_csv(os.path.join(self.data_dir, file))
                    data[node_name] = df
                    print(f"Loaded {node_name}: {len(df)} records")
                except Exception as e:
                    print(f"Error loading {file}: {e}")
        return data
    
    def generate_edges(self, methods=['correlation'], output_file='edges.csv', 
                      threshold=0.3, top_k=3, visualization=False):
        """
        生成边文件
        :param methods: 使用的方法 ['correlation', 'mutual_info', 'granger']
        :param output_file: 输出文件
        :param threshold: 相似性阈值
        :param top_k: 每个节点保持的最高连接数
        :param visualization: 是否可视化
        """
        if not self.data:
            print("No data loaded!")
            return []
            
        node_names = list(self.data.keys())
        n_nodes = len(node_names)
        
        print(f"Generating edges for {n_nodes} nodes using methods: {methods}")
        
        # 提取目标变量序列
        target_series = self._extract_target_series(node_names)
        
        if not target_series:
            print("No valid target series found!")
            return []
        
        # 计算相似性
        edge_weights = {}
        
        if 'correlation' in methods:
            edge_weights.update(self._correlation_analysis(target_series, node_names))
        
        if 'mutual_info' in methods:
            edge_weights.update(self._mutual_info_analysis(target_series, node_names))
            
        if 'granger' in methods:
            edge_weights.update(self._granger_analysis(target_series, node_names))
        
        # 生成最终边列表
        edges = self._combine_edge_weights(edge_weights, threshold, top_k)
        
        # 保存边文件
        self._save_edges(edges, node_names, output_file)
        
        # 可视化
        if visualization:
            self._visualize_graph(edges, node_names)
        
        return edges
    
    def _extract_target_series(self, node_names):
        """提取目标变量序列"""
        target_series = {}
        
        for node_name in node_names:
            df = self.data[node_name]
            
            if self.target_col in df.columns:
                series = df[self.target_col].dropna().values
                if len(series) > 0:
                    target_series[node_name] = series
            else:
                # 使用第一个数值列
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    series = df[numeric_cols[0]].dropna().values
                    if len(series) > 0:
                        target_series[node_name] = series
                        
        print(f"Extracted target series for {len(target_series)} nodes")
        return target_series
    
    def _correlation_analysis(self, target_series, node_names):
        """相关性分析"""
        edges = {}
        
        for i, node1 in enumerate(node_names):
            if node1 not in target_series:
                continue
                
            for j, node2 in enumerate(node_names):
                if node2 not in target_series or i == j:
                    continue
                
                try:
                    # 确保序列长度一致
                    series1 = target_series[node1]
                    series2 = target_series[node2]
                    min_len = min(len(series1), len(series2))
                    
                    if min_len < 10:  # 至少需要10个数据点
                        continue
                    
                    s1 = series1[:min_len]
                    s2 = series2[:min_len]
                    
                    # 皮尔逊相关系数
                    corr, p_value = pearsonr(s1, s2)
                    
                    if not np.isnan(corr) and p_value < 0.05:
                        edges[(i, j)] = abs(corr)
                        
                except Exception as e:
                    continue
        
        print(f"Correlation analysis: found {len(edges)} significant correlations")
        return edges
    
    def _mutual_info_analysis(self, target_series, node_names):
        """互信息分析（并行加速版）"""
        try:
            from sklearn.feature_selection import mutual_info_regression
            from tqdm.notebook import tqdm  # Jupyter专用进度条
            from joblib import Parallel, delayed
            import numpy as np

            edges = {}
            valid_pairs = []

            # 预筛选有效节点对（减少重复计算）
            for i, node1 in enumerate(node_names):
                if node1 not in target_series:
                    continue
                for j, node2 in enumerate(node_names):
                    if node2 in target_series and i != j and j > i:  # 避免重复计算 (i,j)和(j,i)
                        valid_pairs.append((i, j, node1, node2))

            # 并行计算函数
            def _calculate_mi(i, j, node1, node2):
                try:
                    series1 = target_series[node1]
                    series2 = target_series[node2]
                    min_len = min(len(series1), len(series2))
                    
                    if min_len < 10:
                        return None
                    
                    X = series1[:min_len].reshape(-1, 1)
                    y = series2[:min_len]
                    mi = mutual_info_regression(X, y, random_state=42)[0]
                    return (i, j, mi) if mi > 0.1 else None  # 阈值过滤
                except:
                    return None

            # 并行计算（n_jobs=-1使用所有CPU核心）
            results = Parallel(n_jobs=-1)(
                delayed(_calculate_mi)(i, j, node1, node2)
                for i, j, node1, node2 in tqdm(valid_pairs, desc="Calculating MI")
            )

            # 收集结果
            edges = { (i,j): mi for i, j, mi in results if mi is not None }
            print(f"Mutual info analysis: found {len(edges)} significant connections")
            return edges
        except ImportError as e:
            print(f"Required package not found: {e}")
            return {}
    
    def _granger_analysis(self, target_series, node_names):
        """格兰杰因果分析"""
        try:
            from statsmodels.tsa.stattools import grangercausalitytests
            
            edges = {}
            
            for i, node1 in enumerate(node_names):
                if node1 not in target_series:
                    continue
                    
                for j, node2 in enumerate(node_names):
                    if node2 not in target_series or i == j:
                        continue
                    
                    try:
                        series1 = target_series[node1]
                        series2 = target_series[node2]
                        min_len = min(len(series1), len(series2))
                        
                        if min_len < 50:  # 格兰杰检验需要更多数据点
                            continue
                        
                        s1 = series1[:min_len]
                        s2 = series2[:min_len]
                        
                        # 构建数据矩阵 [s2, s1] - 测试s1是否Granger导致s2
                        data = np.column_stack([s2, s1])
                        
                        # 进行格兰杰因果检验
                        max_lags = min(10, min_len // 20)
                        if max_lags < 1:
                            continue
                            
                        result = grangercausalitytests(data, maxlag=max_lags, verbose=False)
                        
                        # 获取最小p值
                        min_p_value = 1.0
                        for lag in range(1, max_lags + 1):
                            if lag in result:
                                p_val = result[lag][0]['ssr_ftest'][1]
                                min_p_value = min(min_p_value, p_val)
                        
                        if min_p_value < 0.05:
                            edges[(i, j)] = 1 - min_p_value
                            
                    except Exception as e:
                        continue
            
            print(f"Granger causality: found {len(edges)} causal relationships")
            return edges
            
        except ImportError:
            print("statsmodels not available for Granger causality")
            return {}
    
    def _combine_edge_weights(self, edge_weights, threshold, top_k):
        """组合边权重"""
        # 统计每条边的权重
        edge_scores = {}
        for (i, j), weight in edge_weights.items():
            if (i, j) not in edge_scores:
                edge_scores[(i, j)] = []
            edge_scores[(i, j)].append(weight)
        
        # 计算平均权重
        edges = []
        for (i, j), weights in edge_scores.items():
            avg_weight = np.mean(weights)
            if avg_weight > threshold:
                edges.append((i, j, avg_weight))
        
        # 如果边太少，选择每个节点的top-k连接
        n_nodes = len(set([i for i, j, w in edges] + [j for i, j, w in edges]))
        if len(edges) < n_nodes * 0.5:  # 如果边数量太少
            print("Adding top-k connections per node")
            
            # 为每个节点添加top-k连接
            node_connections = {}
            for (i, j), weights in edge_scores.items():
                if i not in node_connections:
                    node_connections[i] = []
                node_connections[i].append((j, np.mean(weights)))
            
            for i, connections in node_connections.items():
                # 按权重排序
                connections.sort(key=lambda x: x[1], reverse=True)
                
                # 添加top-k连接
                for j, weight in connections[:top_k]:
                    edge_tuple = (i, j, weight)
                    if edge_tuple not in edges:
                        edges.append(edge_tuple)
        
        print(f"Final edge count: {len(edges)}")
        return edges
    
    def _save_edges(self, edges, node_names, output_file):
        """保存边到文件"""
        if not edges:
            print("No edges to save!")
            return
        
        edge_data = []
        for source_idx, target_idx, weight in edges:
            if source_idx < len(node_names) and target_idx < len(node_names):
                edge_data.append({
                    'source': source_idx,
                    'target': target_idx,
                    'source_name': node_names[source_idx],
                    'target_name': node_names[target_idx],
                    'weight': weight
                })
        
        if edge_data:
            df = pd.DataFrame(edge_data)
            df.to_csv(output_file, index=False)
            print(f"Saved {len(edge_data)} edges to {output_file}")
        else:
            print("No valid edges to save!")
    
    def _visualize_graph(self, edges, node_names):
        print("可视化图结构...")
        """可视化图结构"""
        try:
            import matplotlib.pyplot as plt
            import networkx as nx
            
            G = nx.DiGraph()
            
            # 添加节点
            for i, name in enumerate(node_names):
                G.add_node(i, label=name)
            
            # 添加边（假设edges是(source, target, weight)的元组列表）
            for source, target, weight in edges:
                G.add_edge(source, target, weight=weight)
            
            # 绘制图
            plt.figure(figsize=(12, 8))
            pos = nx.spring_layout(G, k=1.5, iterations=50)
            
            # 绘制节点
            nx.draw_networkx_nodes(
                G, pos, 
                node_color='lightblue', 
                node_size=1000, 
                alpha=0.7
            )
            
            # 绘制边（带权重）
            if edges:
                weights = [d['weight'] for (u, v, d) in G.edges(data=True)]
                max_weight = max(weights) if weights else 1
                normalized_weights = [w/max_weight * 3 for w in weights]
                
                nx.draw_networkx_edges(
                    G, pos,
                    width=normalized_weights,
                    alpha=0.6, 
                    edge_color='gray',
                    arrows=True
                )
            
            # 添加标签
            labels = {i: name[:8] for i, name in enumerate(node_names)}  # 截短标签
            nx.draw_networkx_labels(
                G, pos, 
                labels, 
                font_size=8
            )
            
            plt.title(f"Graph Structure ({len(node_names)} nodes, {len(edges)} edges)")
            plt.axis('off')
            plt.tight_layout()
            plt.savefig('graph_structure.png', dpi=300, bbox_inches='tight')
            plt.show()
            print("Graph visualization saved as 'graph_structure.png'")
            
        except ImportError:
            print("matplotlib or networkx not available for visualization")
        except Exception as e:
            print(f"Error creating visualization: {e}")

def main():
    """主函数 - 完整的使用流程"""
    
    print("=== Standalone Time Series Graph DataLoader ===")
    
    # 配置参数
    config = DataConfig()
    config.data_dir = './sample_data'  # 你的数据目录
    config.edge_file = './edges.csv'   # 边文件路径
    
    # 如果没有数据，创建示例数据
    if not os.path.exists(config.data_dir):
        print("Creating sample data...")
        test_dataloader()
    
    # 步骤1: 生成边文件（如果不存在）
    if not os.path.exists(config.edge_file):
        print("\nStep 1: Generating graph edges...")
        edge_generator = AdvancedEdgeGenerator(config.data_dir, target_col=config.target_col)
        
        edges = edge_generator.generate_edges(
            methods=['correlation', 'mutual_info'],
            output_file=config.edge_file,
            threshold=config.correlation_threshold,
            top_k=config.top_k_edges,
            visualization=True
        )
    else:
        print(f"Using existing edge file: {config.edge_file}")
    
    # 步骤2: 创建数据加载器
    print("\nStep 2: Creating dataloader...")
    try:
        train_loader, train_dataset = create_dataloader(
            data_dir=config.data_dir,
            batch_size=config.batch_size,
            seq_len=config.seq_len,
            pred_len=config.pred_len,
            target_col=config.target_col,
            scaler_type=config.scaler_type,
            edge_file=config.edge_file,
            shuffle=config.shuffle
        )
        
        # 更新配置
        config.n_nodes = len(train_dataset.processed_data)
        config.enc_in = list(train_dataset.processed_data.values())[0]['features'].shape[1]
        
        print(f"Training dataset: {len(train_dataset)} samples")
        print(f"Number of nodes: {config.n_nodes}")
        print(f"Features per node: {config.enc_in}")
        
        # 步骤3: 测试数据加载
        print("\nStep 3: Testing data loading...")
        for i, batch in enumerate(train_loader):
            print(f"Batch {i}: x={batch.x.shape}, y={batch.y.shape}, edges={batch.edge_index.shape[1]}")
            
            if i >= 2:  # 只测试前3个批次
                break
        
        print("\n=== Setup Complete ===")
        print("DataLoader is ready to use with your ST-GAT model!")
        print("\nUsage example:")
        print("```python")
        print("for batch in train_loader:")
        print("    # batch.x: [batch_size, n_nodes, seq_len, n_features]")
        print("    # batch.y: [batch_size, n_nodes, pred_len]")  
        print("    # batch.edge_index: [2, num_edges]")
        print("    predictions = model(batch)")
        print("```")
        
        return train_loader, train_dataset, config
        
    except Exception as e:
        print(f"Error creating dataloader: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


if __name__ == "__main__":
    # 运行完整流程
    main()