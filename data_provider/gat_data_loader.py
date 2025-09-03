import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings
import re
import glob
from torch_geometric.data import Data, Batch
from tqdm import tqdm
import logging
warnings.filterwarnings('ignore')



class TimeSeriesGraphDataset(Dataset):
    """
    时间序列图数据集 - 支持PyTorch Geometric或自定义类
    """
    def __init__(self, data_dir, seq_len=432, pred_len=36, target_col='Wspd', 
                 scaler_type='standard', edge_index='edge_index.npy', 
                 use_edge_features=True, edge_attr='attr.npy', node_index = "node_index.npy", 
                 flag='train', file_pattern="dated_Turb*.csv", shared_scalers=None):
        """
        初始化数据集
        :param data_dir: 数据文件目录
        :param seq_len: 输入序列长度
        :param pred_len: 预测长度
        :param target_col: 目标列名
        :param scaler_type: 标准化类型 ('standard', 'minmax', None)
        :param edge_index: 边index路径
        :param use_edge_features: 是否使用边特征
        :param edge_attr: 边特征文件
        :param node_index: 节点index文件
        :param flag: 数据集类型 ('train', 'val', 'test')
        :param file_pattern: 文件过滤模式
        :param shared_scalers: 共享的scaler（用于val/test集）
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.data_dir = data_dir
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.target_col = target_col
        self.scaler_type = scaler_type
        self.flag = flag
        self.shared_scalers = shared_scalers
        self.file_pattern = file_pattern

        self.logger.info(f"初始化{flag}数据集: seq_len={seq_len}, pred_len={pred_len}, target_col={target_col}")
        self.logger.info("开始加载边数据...")
        self.edge_index = np.load(edge_index, allow_pickle=True)
        self.node_index = np.load(node_index, allow_pickle=True)
        if use_edge_features:
            self.logger.debug("使用并加载边特征...")
            self.edge_attr = np.load(edge_attr, allow_pickle=True)
            self.static_edge_dim = self.edge_attr.shape[-1]
            self.logger.debug(f"边特征shape:{edge_attr.shape}")

        """
        加载所有风机数据及预处理
        """
        self.logger.info(f"===========开始加载并处理所有数据:{data_dir}...{self.flag}=================")
        self.data_files = self._get_data_files()
        self.logger.debug(f"加载了{len(self.data_files)} 个数据文件")
        self.logger.debug(f"开始过滤节点:{self.node_index}")
        #过滤节点
        self.file_list = [self.file_pattern.replace('*', str(file_name)) for file_name in self.node_index+1]
        self.data_files = [f for f in self.data_files if os.path.basename(f) in self.file_list]
        self.logger.info(f"使用提供的 {len(self.file_list)} 个风速数据文件")
        self.logger.debug(f"过滤后的文件为：{self.data_files}")

        self.raw_data = self._load_all_data()
        # 预处理数据， 包含正确处理scaler共享的逻辑
        self.processed_data, self.scalers = self._preprocess_data()

        self.logger.info(f"开始创建样本索引：{flag}...")
        # 创建样本索引
        self.samples = self._create_samples()

        self.logger.info(f"{flag.upper()} dataset initialized: {len(self.samples)} samples, {len(self.processed_data)} nodes")
        # 验证数据集的完整性
        self._validate_dataset()
        
    def _validate_dataset(self):
        """验证数据集的完整性"""
        n_nodes = len(self.processed_data)
        self.n_nodes = n_nodes
        
        # 验证边索引
        if self.edge_index is not None:
            max_edge_idx = self.edge_index.max().item()
            if max_edge_idx >= n_nodes:
                print(f"ERROR: edge_index contains invalid node indices!")
                print(f"Max edge index: {max_edge_idx}, n_nodes: {n_nodes}")
                # 修复：过滤无效边
                valid_mask = (self.edge_index[0] < n_nodes) & (self.edge_index[1] < n_nodes)
                self.edge_index = self.edge_index[:, valid_mask]
                print(f"Fixed: Filtered to {self.edge_index.shape[1]} valid edges")
        
        print(f"Dataset validation passed: {n_nodes} nodes, {self.edge_index.shape[1] if self.edge_index is not None else 0} edges")
    
    def _get_data_files(self):
        """获取所有CSV数据文件"""
        all_files = glob.glob(os.path.join(self.data_dir, self.file_pattern))
        # 按照文件名中的数字进行排序
        def extract_number(filename):
            match = re.search(r'dated_Turb(\d+)\.csv', os.path.basename(filename))
            return int(match.group(1)) if match else 0
        
        all_files.sort(key=extract_number)
        return all_files
    
    def _load_all_data(self):
        """加载所有数据文件"""
        all_data = {}
        for file_path in tqdm(self.data_files, desc="加载风机CSV文件数据..."):
            node_name = os.path.basename(file_path).replace('.csv', '').replace('dated_Turb', '')
            try:
                df = pd.read_csv(file_path)
                # 解析日期时间
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                    df = df.sort_values('date').reset_index(drop=True)
                
                # 存储数据
                all_data[int(node_name)-1] = df  # node index starts from 0
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue
        return all_data
    
    def _preprocess_data(self):
        processed_data = {}
        scalers = {}
        # 找到所有节点的共同时间范围
        common_time_range = self._get_common_time_range()
        
        for node_name, df in tqdm(self.raw_data.items(), desc="处理数据中..."):
            try:
                # 过滤到共同时间范围
                if 'date' in df.columns and common_time_range[0] is not None:
                    mask = (df['date'] >= common_time_range[0]) & (df['date'] <= common_time_range[1])
                    df_filtered = df[mask].copy()
                else:
                    df_filtered = df.copy()
                
                # 选择特征列（除了date列）
                feature_cols = [col for col in df_filtered.columns if col != 'date']
                # 将Target_col放到最后一列
                feature_cols.remove(self.target_col)
                feature_cols.append(self.target_col)
                features = df_filtered[feature_cols].values
                
                # 处理NaN值
                if np.isnan(features).any():
                    print(f"Warning: NaN values found in {node_name}, filling with forward fill")
                    features_df = pd.DataFrame(features, columns=feature_cols)
                    features_df = features_df.fillna(method='ffill').fillna(method='bfill')
                    features = features_df.values
                
                # 数据标准化 - 正确处理scaler共享
                if self.scaler_type == 'standard':
                    scaler = StandardScaler()
                elif self.scaler_type == 'minmax':
                    scaler = MinMaxScaler()
                else:
                    scaler = None

                if scaler is not None:
                    if self.flag == 'train' or self.shared_scalers is None:
                        # 训练集：拟合并转换
                        features_scaled = scaler.fit_transform(features)
                        scalers[node_name] = scaler
                    else:
                        # 验证集和测试集：使用训练集的scaler
                        if node_name in self.shared_scalers:
                            shared_scaler = self.shared_scalers[node_name]
                            if shared_scaler is not None:
                                features_scaled = shared_scaler.transform(features)
                                scalers[node_name] = shared_scaler
                            else:
                                features_scaled = features
                                scalers[node_name] = None
                        else:
                            print(f"Warning: No shared scaler found for {node_name}, fitting new one")
                            features_scaled = scaler.fit_transform(features)
                            scalers[node_name] = scaler
                else:
                    features_scaled = features
                    scalers[node_name] = None
                
                processed_data[node_name] = {
                    'features': features_scaled,
                    'raw_features': features,
                    'columns': feature_cols,
                    'dates': df_filtered['date'].values if 'date' in df_filtered.columns else None
                }
                self.logger.debug(f"columns: {feature_cols}")
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
    
    def _create_samples(self):
        """创建训练样本，按照flag进行7:2:1划分"""
        samples = []
        node_names = sorted(list(self.processed_data.keys()))
        self.logger.debug(f"节点列表: {node_names}")
        
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
                # 如果没有目标列，使用最后一列
                target_val = data['features'][sample['pred_start_idx']:sample['pred_end_idx'], -1]

            target_values.append(target_val)
        
        # 转换为tensor
        x = torch.FloatTensor(np.array(node_features))  # [n_nodes, seq_len, n_features]
        y = torch.FloatTensor(np.array(target_values))   # [n_nodes, pred_len]
        
        # TODO: 计算index和attr
        edge_index, edge_attr = self.compute_dynamic_edge_features(x,edge_index,self.edge_attr)
        edge_index = torch.LongTensor(np.array(edge_index))
        edge_attr = torch.FloatTensor(np.array(edge_attr))

        # 创建图数据对象
        graph_data = Data(
            x=x,
            edge_index=edge_index,
            y=y,
            edge_attr=edge_attr
        )
        
        return graph_data

    def compute_dynamic_edge_features(self, x, edge_index, static_edge_attr):
        """
        nodes, seq_len, feature_dim = x.shape
        目前 edge_index为全连接图
        
        # 根据数据集的实际特征列确定索引
        # 从TimeSeriesGraphDataset可以看出特征列除了date外包含: Wdir, Etmp, Itmp, Ndir, Pab1, Pab2, Pab3, Prtv, Patv, Wspd
        # 由于date列在预处理时被排除，所以特征顺序为: [Wdir, Etmp, Itmp, Ndir, Pab1, Pab2, Pab3, Prtv, Patv, Wspd]
        
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
        # x [n_nodes, seq_len, feature_dim]
        # edge_index [2, num_edges]
        # static_edge_attr [num_edges, static_edge_dim]
        static_edge_attr = torch.tensor(static_edge_attr)
        n_nodes = self.n_nodes

        # 使用默认索引（基于常见的风电数据格式）
        wspd_idx, wdir_idx = -1, 0  # 风速、风向
        etmp_idx, itmp_idx = 1, 2  # 环境温度、内部温度
        ndir_idx = 3  # 机舱方向
        patv_idx = 8   # 有功功率
        
        # 取最后三个时间步的特征的平均值用于计算动态边特征
        current_features = x[:, -3:, :].mean(dim=1).view(n_nodes, -1)   # [n_nodes, feature_dim]
        source_indices = edge_index[0]  # [batch_size,num_edges]
        target_indices = edge_index[1]  # [batch_size, num_edges]
        

        # 批量索引操作
        source_features = current_features[source_indices]  # [num_edges, feature_dim]
        target_features = current_features[target_indices]  # [num_edges, feature_dim]

        dynamic_features = []
        
        # 1. 风向相关的动态特征
        source_wdir = source_features[:,  wdir_idx]
        target_wdir = target_features[:,  wdir_idx]

        # 风向差异（考虑角度的周期性）
        wdir_diff = torch.abs(source_wdir - target_wdir)
        wdir_diff = torch.min(wdir_diff, 360 - wdir_diff)
        wdir_diff_norm = wdir_diff / 180.0
        dynamic_features.append(wdir_diff_norm. unsqueeze(-1))
        
        # 风向一致性（cos相似度）
        source_wdir_rad = torch.deg2rad(source_wdir)
        target_wdir_rad = torch.deg2rad(target_wdir)
        wdir_consistency = torch.cos(source_wdir_rad - target_wdir_rad)
        dynamic_features.append(wdir_consistency. unsqueeze(-1))
        
        
        # 2. 风速相关的动态特征
        source_wspd = source_features[:,  wspd_idx]
        target_wspd = target_features[:,  wspd_idx]

        # 风速差异
        wspd_diff = torch.abs(source_wspd - target_wspd)
        wspd_diff_norm = torch.clamp(wspd_diff / 25.0, 0, 1)
        dynamic_features.append(wspd_diff_norm. unsqueeze(-1))
        
        # 风速比率
        wspd_ratio = torch.clamp(torch.min(source_wspd, target_wspd) / 
                        (torch.max(source_wspd, target_wspd) + 1e-5), 0, 1)
        dynamic_features.append(wspd_ratio. unsqueeze(-1))
        
        # 平均风速等级
        avg_wspd = (source_wspd + target_wspd) / 2
        wspd_level = torch.clamp(avg_wspd / 25.0, 0, 1)
        dynamic_features.append(wspd_level.unsqueeze(-1))

        # 从静态特征中提取距离和方向信息
        euclidean_dist = static_edge_attr[:, 0]  # [num_edges]
        bearing_angle = static_edge_attr[:, 2]   # [num_edges] [-pi, pi]

        # TODO: 计算尾流影响强度
        source_to_target_angle = bearing_angle
        wake_alignment = torch.cos(torch.deg2rad(source_wdir - source_to_target_angle))
        wake_alignment = torch.clamp(wake_alignment, 0, 1)
        
        # 距离衰减因子
        distance_decay = torch.exp(-euclidean_dist / 500.0)
        
        # 综合尾流强度
        wake_effect = wake_alignment * distance_decay * wspd_level.squeeze()
        dynamic_features.append(wake_effect. unsqueeze(-1))
        
        source_etmp = source_features[:,  etmp_idx]
        target_etmp = target_features[:,  etmp_idx]
        
        temp_diff = torch.abs(source_etmp - target_etmp)
        temp_diff_norm = torch.clamp(temp_diff / 50.0, 0, 1)
        dynamic_features.append(temp_diff_norm. unsqueeze(-1))
        
        # 5. 功率相关特征
        source_patv = source_features[:,  patv_idx]
        target_patv = target_features[:,  patv_idx]
        
        power_diff = torch.abs(source_patv - target_patv)
        power_diff_norm = torch.clamp(power_diff / 2000.0, 0, 1)
        dynamic_features.append(power_diff_norm. unsqueeze(-1))
        
        power_correlation = torch.clamp(torch.min(source_patv, target_patv) / 
                                (torch.max(source_patv, target_patv) + 1e-5), 0, 1)
        dynamic_features.append(power_correlation. unsqueeze(-1))
        
        
        source_ndir = source_features[:, ndir_idx]
        target_ndir = target_features[:, ndir_idx]
        
        ndir_diff = torch.abs(source_ndir - target_ndir)
        ndir_diff = torch.min(ndir_diff, 360 - ndir_diff)
        ndir_diff_norm = ndir_diff / 180.0
        dynamic_features.append(ndir_diff_norm. unsqueeze(-1))
        
        source_ndir_wind_align = torch.cos(torch.deg2rad(source_ndir - source_wdir))
        target_ndir_wind_align = torch.cos(torch.deg2rad(target_ndir - target_wdir))
        avg_alignment = (source_ndir_wind_align + target_ndir_wind_align) / 2
        dynamic_features.append(avg_alignment. unsqueeze(-1))
        
        # 7. 时序稳定性特征（基于近期变化趋势）
        # 获取当前批次的历史风速数据
        recent_wspd = x[:, -3:, wspd_idx]  # [n_nodes, 3]
        wspd_trend = recent_wspd[:, -1] - recent_wspd[:, 0]  # [n_nodes]

        source_trend = wspd_trend[source_indices]  # [num_edges]
        target_trend = wspd_trend[target_indices]  # [num_edges]


        # 创建与source_trend相同设备和dtype的tensor
        ones_tensor = torch.ones_like(source_trend)

        trend_consistency = torch.cos(torch.atan2(source_trend, ones_tensor) - torch.atan2(target_trend, ones_tensor))
        dynamic_features.append(trend_consistency. unsqueeze(-1))
        
        trend_strength_diff = torch.abs(torch.abs(source_trend) - torch.abs(target_trend))
        trend_strength_diff_norm = torch.clamp(trend_strength_diff / 10.0, 0, 1)
        dynamic_features.append(trend_strength_diff_norm.unsqueeze(-1))
        
        # 合并动态特征
        all_dynamic_features = torch.cat(dynamic_features, dim=-1)  # [num_edges, dynamic_dim]

        # 合并静态和动态特征
        combined_edge_attr = torch.cat([static_edge_attr, all_dynamic_features], dim=-1)
        return combined_edge_attr
    
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
    return Batch.from_data_list(batch)


def create_dataloader(args, flag):
    """创建数据加载器，正确处理scaler共享"""
    shuffle_flag = flag == 'train'  # 只有训练集需要shuffle
    drop_last = True
    batch_size = args.batch_size

    # 根据args构建参数
    dataset_args = {
        'data_dir': args.root_path,
        'seq_len': args.seq_len,
        'pred_len': args.pred_len,
        'target_col': args.target,
        'scaler_type': "standard",
        'edge_index': args.edge_index,
        'edge_attr': args.edge_attr,
        'flag': flag
    }

    # 处理scaler共享
    shared_scalers = None
    if flag in ['val', 'test'] and hasattr(args, 'train_scalers'):
        shared_scalers = args.train_scalers
        dataset_args['shared_scalers'] = shared_scalers
    
    dataset = TimeSeriesGraphDataset(**dataset_args)
    print(f"{flag} dataset size: {len(dataset)}")
    
    # 如果是训练集，保存scalers供后续使用
    if flag == 'train':
        args.train_scalers = dataset.scalers
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=0,  # 设为0避免多进程问题
        pin_memory=True,
        drop_last=drop_last,
        collate_fn=collate_fn
    )
    
    return dataloader, dataset