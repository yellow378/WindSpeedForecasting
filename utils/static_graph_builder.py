"""
静态图构建工具: 离线构建16个风向扇区的稀疏有向静态图

对应论文4.2.3: 基于风向感知的条件化静态图构建

输出: 对于每个扇区 z (0~15), 生成:
- sector_{z}_edge_index.npy: [2, n_edges] 有向边索引
- sector_{z}_adj_weights.npy: [n_edges] 邻接权重 (物理先验 + 统计相关)
- sector_{z}_static_edge_attr.npy: [n_edges, 11] 11维静态边特征

双重物理约束:
(1) 尾流方向约束: Δθ ≤ Δφ (半扇区宽度 = 11.25°)
(2) 尾流作用距离约束: d_ij < D_max (= 10R)

混合边权重:
    w_ij = exp(-d_ij/(5R)) · cos²(Δθ/2) · max_lag_corr
"""

import numpy as np
import os
from scipy.stats import pearsonr


class StaticGraphBuilder:
    """
    离线静态图构建器

    为每个主导风向扇区构建一个稀疏有向图, 仅保留满足尾流物理约束的边
    """

    N_SECTORS = 16           # 风向扇区数量
    SECTOR_ANGLE = 22.5      # 每扇区角度 (度)
    HALF_SECTOR = 11.25      # 半扇区宽度 (度)

    def __init__(self, positions, historical_data,
                 rotor_radius=1.0, d_max_factor=10.0):
        """
        Args:
            positions: [n_nodes, 2] 风机坐标 (经纬度或投影坐标)
            historical_data: [n_nodes, seq_len, n_features] 历史时序数据
                            特征顺序: [Wspd, Wdir, Etmp, Itmp, Ndir, ...]
            rotor_radius: 叶轮半径 R (与positions同单位)
            d_max_factor: 最大尾流距离系数 (D_max = d_max_factor × R)
        """
        self.positions = np.array(positions)
        self.n_nodes = positions.shape[0]
        self.historical_data = np.array(historical_data)
        self.rotor_radius = rotor_radius
        self.D_max = d_max_factor * rotor_radius  # D_max = 10R
        self.L0 = 5.0 * rotor_radius              # 距离衰减尺度 L0 = 5R

        # 预计算风机间距离和方位角矩阵
        self._precompute_geometry()

        # 预计算运行统计特征
        self._precompute_statistics()

    def _precompute_geometry(self):
        """预计算风机间的空间几何关系"""
        n = self.n_nodes

        # 距离矩阵
        diff = self.positions[np.newaxis, :, :] - self.positions[:, np.newaxis, :]
        self.dist_matrix = np.sqrt((diff ** 2).sum(axis=-1))  # [n, n]

        # 方位角矩阵: 从风机i指向风机j的地理方位角 (度, 正北=0, 顺时针)
        # 使用arctan2计算, 注意坐标轴方向
        self.bearing_matrix = np.degrees(
            np.arctan2(diff[:, :, 1], diff[:, :, 0])
        ) % 360  # [n, n]

        # 距离归一化参数
        self.d_min = self.dist_matrix[self.dist_matrix > 0].min()
        self.d_max_actual = self.dist_matrix.max()

        # 高斯距离带宽参数σ = 所有风机间距离均值的三分之一
        self.gauss_sigma = self.dist_matrix[self.dist_matrix > 0].mean() / 3.0

    def _precompute_statistics(self):
        """预计算基于历史数据的运行统计特征"""
        n = self.n_nodes
        wspd = self.historical_data[:, :, 0]  # [n, seq_len] 风速
        wdir = self.historical_data[:, :, 1]  # [n, seq_len] 风向

        # 风速相关性矩阵
        self.speed_corr = np.zeros((n, n))
        self.speed_diff_corr = np.zeros((n, n))
        self.direction_corr = np.zeros((n, n))
        self.lag_corr = np.zeros((n, n))
        self.variance_ratio = np.zeros((n, n))
        self.mean_difference = np.zeros((n, n))

        # 风速一阶差分
        wspd_diff = np.diff(wspd, axis=1)  # [n, seq_len-1]

        for i in range(n):
            for j in range(n):
                if i == j:
                    self.speed_corr[i, j] = 1.0
                    self.speed_diff_corr[i, j] = 1.0
                    self.direction_corr[i, j] = 1.0
                    self.lag_corr[i, j] = 1.0
                    self.variance_ratio[i, j] = 1.0
                    self.mean_difference[i, j] = 0.0
                    continue

                # 风速相关性
                r, _ = pearsonr(wspd[i], wspd[j])
                self.speed_corr[i, j] = r if not np.isnan(r) else 0.0

                # 风速一阶差分相关性
                r_diff, _ = pearsonr(wspd_diff[i], wspd_diff[j])
                self.speed_diff_corr[i, j] = r_diff if not np.isnan(r_diff) else 0.0

                # 风向相关性 (使用sin/cos分解)
                cos_corr, _ = pearsonr(np.cos(np.radians(wdir[i])),
                                       np.cos(np.radians(wdir[j])))
                sin_corr, _ = pearsonr(np.sin(np.radians(wdir[i])),
                                       np.sin(np.radians(wdir[j])))
                self.direction_corr[i, j] = (cos_corr + sin_corr) / 2.0 \
                    if not (np.isnan(cos_corr) or np.isnan(sin_corr)) else 0.0

                # 滞后相关性 (lag=1,2)
                max_lag_corr = 0.0
                for lag in [1, 2]:
                    if wspd.shape[1] > lag:
                        r_lag, _ = pearsonr(wspd[i, lag:], wspd[j, :-lag])
                        if not np.isnan(r_lag):
                            max_lag_corr = max(max_lag_corr, abs(r_lag))
                self.lag_corr[i, j] = max_lag_corr

                # 方差比
                var_i = np.var(wspd[i])
                var_j = np.var(wspd[j])
                self.variance_ratio[i, j] = var_i / (var_j + 1e-8)

                # 均值差异
                self.mean_difference[i, j] = abs(np.mean(wspd[i]) - np.mean(wspd[j]))

    def _check_wake_constraints(self, i, j, sector_id):
        """
        检查风机对(i→j)是否满足尾流物理约束

        双重约束:
        (1) 尾流方向约束: 最小角度差 Δθ ≤ Δφ (11.25°)
        (2) 尾流作用距离约束: d_ij < D_max

        Args:
            i: 上游风机编号
            j: 下游风机编号
            sector_id: 风向扇区编号
        Returns:
            (satisfies, delta_theta): 是否满足约束, 角度差
        """
        # 扇区中心方向 (度, 正北=0, 顺时针)
        sector_center = sector_id * self.SECTOR_ANGLE + self.SECTOR_ANGLE / 2.0

        # 从风机i指向风机j的方位角
        bearing = self.bearing_matrix[i, j]

        # 最小角度差 (考虑360°周期性)
        delta_theta = abs(bearing - sector_center)
        delta_theta = min(delta_theta, 360.0 - delta_theta)

        # 约束1: 尾流方向约束
        direction_ok = delta_theta <= self.HALF_SECTOR

        # 约束2: 尾流作用距离约束
        distance_ok = self.dist_matrix[i, j] < self.D_max

        return direction_ok and distance_ok, delta_theta

    def _compute_edge_weight(self, i, j, delta_theta):
        """
        计算混合边权重 (论文公式 4.2.3)

        w_ij = exp(-d_ij/(5R)) × cos²(Δθ/2) × max_lag_corr

        Args:
            i: 上游风机编号
            j: 下游风机编号
            delta_theta: 角度差 (度)
        Returns:
            w_ij: 混合边权重
        """
        d_ij = self.dist_matrix[i, j]

        # 物理先验部分
        distance_decay = np.exp(-d_ij / self.L0)  # exp(-d_ij/(5R))
        direction_factor = np.cos(np.radians(delta_theta / 2.0)) ** 2  # cos²(Δθ/2)

        # 统计相关部分
        stat_corr = self.lag_corr[i, j]  # max_lag_corr

        w_ij = distance_decay * direction_factor * stat_corr
        return w_ij

    def _compute_static_edge_features(self, i, j, delta_theta):
        """
        计算11维静态边特征 (论文表 4.2.3)

        空间几何特征(5维):
        1. normalized_distance: 归一化距离
        2. gauss_distance: 高斯距离
        3. bearing_angle: 方位角
        4. bearing_sin: 方位角正弦分量
        5. bearing_cos: 方位角余弦分量

        运行统计特征(6维):
        6. speed_correlation: 风速相关性
        7. speed_diff_correlation: 风速一阶差分相关性
        8. direction_correlation: 风向相关性
        9. lag_correlation: 滞后相关性
        10. variance_ratio: 方差比
        11. mean_difference: 均值差异

        Args:
            i: 上游风机编号
            j: 下游风机编号
            delta_theta: 角度差 (度)
        Returns:
            features: [11] 静态边特征向量
        """
        d_ij = self.dist_matrix[i, j]
        bearing = self.bearing_matrix[i, j]

        # 空间几何特征
        normalized_distance = (d_ij - self.d_min) / (self.d_max_actual - self.d_min + 1e-8)
        gauss_distance = np.exp(-d_ij ** 2 / (2 * self.gauss_sigma ** 2))
        bearing_sin = np.sin(np.radians(bearing))
        bearing_cos = np.cos(np.radians(bearing))

        # 运行统计特征
        features = np.array([
            normalized_distance,         # 1. 归一化距离
            gauss_distance,              # 2. 高斯距离
            bearing,                     # 3. 方位角 (度)
            bearing_sin,                 # 4. 方位角正弦分量
            bearing_cos,                 # 5. 方位角余弦分量
            self.speed_corr[i, j],       # 6. 风速相关性
            self.speed_diff_corr[i, j],  # 7. 风速一阶差分相关性
            self.direction_corr[i, j],   # 8. 风向相关性
            self.lag_corr[i, j],         # 9. 滞后相关性
            self.variance_ratio[i, j],   # 10. 方差比
            self.mean_difference[i, j],  # 11. 均值差异
        ], dtype=np.float32)

        return features

    def build_sector_graph(self, sector_id):
        """
        构建单个风向扇区的稀疏有向静态图

        Args:
            sector_id: 扇区编号 (0-15)
        Returns:
            edge_index: [2, n_edges] 有向边索引
            adj_weights: [n_edges] 邻接权重
            static_edge_attr: [n_edges, 11] 静态边特征
        """
        edge_src = []
        edge_tgt = []
        weights = []
        features = []

        for i in range(self.n_nodes):
            for j in range(self.n_nodes):
                if i == j:
                    continue  # 不含自环

                # 检查尾流物理约束
                satisfies, delta_theta = self._check_wake_constraints(i, j, sector_id)
                if not satisfies:
                    continue

                # 计算混合边权重
                w_ij = self._compute_edge_weight(i, j, delta_theta)
                if w_ij < 1e-6:
                    continue  # 过滤极小权重

                # 计算11维静态边特征
                feat = self._compute_static_edge_features(i, j, delta_theta)

                edge_src.append(i)
                edge_tgt.append(j)
                weights.append(w_ij)
                features.append(feat)

        if len(edge_src) == 0:
            # 空图: 创建一条虚拟边避免错误
            edge_index = np.zeros((2, 0), dtype=np.int64)
            adj_weights = np.zeros((0,), dtype=np.float32)
            static_edge_attr = np.zeros((0, 11), dtype=np.float32)
        else:
            edge_index = np.array([edge_src, edge_tgt], dtype=np.int64)
            adj_weights = np.array(weights, dtype=np.float32)
            static_edge_attr = np.array(features, dtype=np.float32)

        return edge_index, adj_weights, static_edge_attr

    def build_all_sectors(self, output_dir):
        """
        构建所有16个扇区的静态图并保存

        Args:
            output_dir: 输出目录
        """
        os.makedirs(output_dir, exist_ok=True)

        print("开始构建16个风向扇区的静态图...")
        for z in range(self.N_SECTORS):
            edge_index, adj_weights, static_edge_attr = self.build_sector_graph(z)

            # 保存
            np.save(f"{output_dir}/sector_{z}_edge_index.npy", edge_index)
            np.save(f"{output_dir}/sector_{z}_adj_weights.npy", adj_weights)
            np.save(f"{output_dir}/sector_{z}_static_edge_attr.npy", static_edge_attr)

            n_edges = edge_index.shape[1]
            print(f"  扇区 {z:2d} (中心方向 {z*22.5+11.25:6.1f}°): "
                  f"{n_edges:4d} 条有向边, "
                  f"平均权重 {adj_weights.mean():.4f}" if n_edges > 0 else
                  f"  扇区 {z:2d} (中心方向 {z*22.5+11.25:6.1f}°): 0 条有向边")

        # 同时保存风机坐标和距离矩阵 (供动态特征计算使用)
        np.save(f"{output_dir}/positions.npy", self.positions)
        np.save(f"{output_dir}/dist_matrix.npy", self.dist_matrix)
        np.save(f"{output_dir}/bearing_matrix.npy", self.bearing_matrix)

        print(f"\n静态图构建完成, 文件保存在: {output_dir}")
        self._print_summary(output_dir)

    def _print_summary(self, output_dir):
        """打印构建摘要"""
        total_edges = 0
        for z in range(self.N_SECTORS):
            ei = np.load(f"{output_dir}/sector_{z}_edge_index.npy")
            total_edges += ei.shape[1]

        n_possible = self.n_nodes * (self.n_nodes - 1)
        avg_edges = total_edges / self.N_SECTORS
        sparsity = 1.0 - avg_edges / n_possible

        print(f"\n=== 静态图构建摘要 ===")
        print(f"  风机数量: {self.n_nodes}")
        print(f"  扇区数量: {self.N_SECTORS}")
        print(f"  全连接边数: {n_possible}")
        print(f"  平均每扇区边数: {avg_edges:.0f}")
        print(f"  稀疏度: {sparsity:.4f}")
        print(f"  尾流距离约束 D_max: {self.D_max:.1f}")
        print(f"  距离衰减尺度 L0: {self.L0:.1f}")


def build_from_dataset(data_dir, positions_file, output_dir,
                       rotor_radius=1.0, d_max_factor=10.0):
    """
    从数据集目录构建静态图

    Args:
        data_dir: 数据目录 (包含各风机CSV文件)
        positions_file: 风机坐标文件 (.npy)
        output_dir: 静态图输出目录
        rotor_radius: 叶轮半径
        d_max_factor: 最大尾流距离系数
    """
    import pandas as pd
    import glob

    # 加载风机坐标
    positions = np.load(positions_file)
    n_nodes = positions.shape[0]

    # 加载历史数据 (用于计算统计特征)
    data_files = sorted(glob.glob(f"{data_dir}/dated_Turb*.csv"))
    historical_list = []
    for f in data_files[:n_nodes]:
        df = pd.read_csv(f)
        # 提取风速和风向列
        wspd = df['Wspd'].values
        wdir = df['Wdir'].values
        # 拼接为 [seq_len, 2]
        data = np.stack([wspd, wdir], axis=-1)
        historical_list.append(data)

    historical_data = np.array(historical_list)  # [n_nodes, seq_len, 2]

    # 构建静态图
    builder = StaticGraphBuilder(
        positions=positions,
        historical_data=historical_data,
        rotor_radius=rotor_radius,
        d_max_factor=d_max_factor
    )
    builder.build_all_sectors(output_dir)


if __name__ == "__main__":
    # 示例用法
    import argparse
    parser = argparse.ArgumentParser(description="构建LightST静态图")
    parser.add_argument('--data_dir', type=str, default='./dataset/spatial_wind')
    parser.add_argument('--positions_file', type=str, default='./dataset/spatial_wind/positions.npy')
    parser.add_argument('--output_dir', type=str, default='./dataset/static_graphs')
    parser.add_argument('--rotor_radius', type=float, default=1.0)
    parser.add_argument('--d_max_factor', type=float, default=10.0)
    args = parser.parse_args()

    build_from_dataset(
        args.data_dir, args.positions_file, args.output_dir,
        args.rotor_radius, args.d_max_factor
    )
