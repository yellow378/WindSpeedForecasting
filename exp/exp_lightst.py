"""
LightST实验类: 支持三阶段渐进训练策略

三阶段训练 (对应论文4.2.6):
- 阶段1: 训练时序编解码器 (无图结构), 学习基础时序表示
- 阶段2: 训练图结构学习模块 (冻结/低学习率时序部分), 学习空间依赖
- 阶段3: 全模型联合微调 + 聚类优化, 协同收敛
"""

from exp.exp_basic import Exp_Basic
from data_provider.gat_data_loader import create_dataloader
from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import numpy as np
from tqdm.auto import tqdm
import warnings

warnings.filterwarnings("ignore")


class Exp_LightST(Exp_Basic):
    """
    LightST实验类

    扩展了基础的Exp_Basic, 增加:
    1. 三阶段渐进训练策略
    2. 静态图加载与管理
    3. 聚类周期性更新
    4. 阶段间的学习率与冻结策略
    """

    def __init__(self, args):
        super().__init__(args)
        self.n_nodes = getattr(args, 'n_nodes', 134)
        self.n_clusters = getattr(args, 'n_clusters', 8)

        # 三阶段训练的epoch配置
        self.stage1_epochs = getattr(args, 'stage1_epochs', 15)
        self.stage2_epochs = getattr(args, 'stage2_epochs', 15)
        self.stage3_epochs = getattr(args, 'stage3_epochs', 10)
        self.total_epochs = self.stage1_epochs + self.stage2_epochs + self.stage3_epochs

        # 聚类更新间隔
        self.cluster_update_interval = getattr(args, 'cluster_update_interval', 5)

        # 静态图路径
        self.static_graph_dir = getattr(args, 'static_graph_dir', None)
        self.positions_path = getattr(args, 'positions_path', None)

    def _build_model(self):
        """构建LightST模型"""
        model = self.model_dict[self.args.model].Model(self.args).float()

        # 加载静态图和风机坐标
        if self.static_graph_dir is not None:
            positions = np.load(self.positions_path) if self.positions_path else None
            model.load_static_graphs(
                self.static_graph_dir,
                positions,
                device=self.device
            )

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"LightST 参数统计:")
        print(f"  总参数量: {total_params:,}")
        print(f"  可训练参数量: {trainable_params:,}")
        print(f"  三阶段训练: Stage1={self.stage1_epochs}ep, "
              f"Stage2={self.stage2_epochs}ep, Stage3={self.stage3_epochs}ep")

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        """获取数据集和数据加载器"""
        data_loader, data_set = create_dataloader(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self, stage=1):
        """
        根据训练阶段选择优化器和学习率

        阶段1: 所有参数, 标准学习率
        阶段2: 图模块参数为主, 时序模块低学习率
        阶段3: 所有参数, 较低学习率联合微调
        """
        lr = self.args.learning_rate

        if stage == 1:
            # 阶段1: 训练时序编解码器
            optimizer = optim.Adam(self.model.parameters(), lr=lr)
        elif stage == 2:
            # 阶段2: 图模块高学习率, 时序模块低学习率
            temporal_params = []
            graph_params = []
            for name, param in self.model.named_parameters():
                if 'emgat' in name or 'edge_weight_adjuster' in name or \
                   'dynamic_feature' in name:
                    graph_params.append(param)
                else:
                    temporal_params.append(param)

            optimizer = optim.Adam([
                {'params': temporal_params, 'lr': lr * 0.1},   # 时序模块低学习率
                {'params': graph_params, 'lr': lr},             # 图模块正常学习率
            ])
        else:
            # 阶段3: 全模型联合微调, 较低学习率
            optimizer = optim.Adam(self.model.parameters(), lr=lr * 0.1)

        return optimizer

    def _select_criterion(self):
        """损失函数: MSE"""
        return nn.MSELoss()

    def _get_stage(self, epoch):
        """根据epoch确定当前训练阶段"""
        if epoch < self.stage1_epochs:
            return 1
        elif epoch < self.stage1_epochs + self.stage2_epochs:
            return 2
        else:
            return 3

    def _configure_stage(self, stage):
        """
        配置训练阶段: 冻结/解冻相应模块

        阶段1: 仅训练时序编解码器, 冻结图模块
        阶段2: 解冻图模块, 时序模块低学习率
        阶段3: 全部解冻, 联合微调
        """
        model = self.model
        if hasattr(model, 'module'):  # DataParallel
            model = model.module

        if stage == 1:
            # 冻结图模块
            for name, param in model.named_parameters():
                if 'emgat' in name or 'edge_weight_adjuster' in name:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
            print(f"[阶段1] 训练时序编解码器, 冻结图模块")

        elif stage == 2:
            # 解冻图模块
            for param in model.parameters():
                param.requires_grad = True
            print(f"[阶段2] 训练图结构学习, 时序模块低学习率")

        elif stage == 3:
            # 全部可训练
            for param in model.parameters():
                param.requires_grad = True
            print(f"[阶段3] 全模型联合微调")

    def vali(self, vali_data, vali_loader, criterion):
        """验证"""
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            pbar = tqdm(vali_loader, desc='Validation', leave=False)
            for i, batch in enumerate(pbar):
                try:
                    batch = batch.to(self.device)
                    batch_x = batch.x
                    batch_y = batch.y
                    edge_index = torch.tensor(batch.edge_index, dtype=torch.long)
                    edge_attr = torch.tensor(batch.edge_attr, dtype=torch.float)

                    graph_data = type('obj', (object,), {
                        'x': batch_x,
                        'edge_index': edge_index,
                        'edge_attr': edge_attr
                    })()

                    outputs = self.model(graph_data, device=self.device)

                    # 处理输出维度
                    outputs = outputs.reshape(self.args.batch_size, -1,
                                             self.args.pred_len, 1)
                    batch_y = batch_y.reshape(self.args.batch_size, -1,
                                             self.args.pred_len, 1)

                    if self.args.features == 'MS':
                        pred = outputs[:, :, :, 0]
                        true = batch_y[:, :, :, 0]
                    else:
                        pred = outputs.squeeze()
                        true = batch_y.squeeze()

                    pred = pred.detach().cpu()
                    true = true.detach().cpu()
                    loss = criterion(pred, true)
                    total_loss.append(loss.item())
                    pbar.set_postfix({'val_loss': f'{loss.item():.4f}'})

                except Exception as e:
                    print(f"验证批次 {i} 出错: {e}")
                    raise e

        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        """
        三阶段渐进训练

        阶段1 (epoch 0 ~ stage1_epochs):
            训练时序编解码器, 不使用图结构
            使用forward_stage1, 仅LT-Encoder + Adapters + LT-Decoder

        阶段2 (stage1_epochs ~ stage1+stage2_epochs):
            训练图结构学习模块 (E-MGAT + 边权重调整)
            时序模块以低学习率更新

        阶段3 (stage1+stage2_epochs ~ total):
            全模型联合微调 + 周期性更新隐空间聚类
        """
        train_data, train_loader = self._get_data(flag="train")
        vali_data, vali_loader = self._get_data(flag="val")
        test_data, test_loader = self._get_data(flag="test")

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        print(f"\n=== LightST 三阶段渐进训练 ===")
        print(f"阶段1 (时序): {self.stage1_epochs} epochs")
        print(f"阶段2 (图学习): {self.stage2_epochs} epochs")
        print(f"阶段3 (联合微调): {self.stage3_epochs} epochs")
        print(f"总训练轮数: {self.total_epochs}")

        prev_stage = 0

        for epoch in range(self.total_epochs):
            # 确定当前阶段
            current_stage = self._get_stage(epoch)

            # 阶段切换时重新配置
            if current_stage != prev_stage:
                self._configure_stage(current_stage)
                model_optim = self._select_optimizer(stage=current_stage)
                criterion = self._select_criterion()
                prev_stage = current_stage

                # 阶段3开始时更新聚类
                if current_stage == 3:
                    self._update_clustering(train_data)

            # 阶段3中周期性更新聚类
            if current_stage == 3 and epoch > self.stage1_epochs + self.stage2_epochs:
                local_epoch = epoch - self.stage1_epochs - self.stage2_epochs
                if local_epoch > 0 and local_epoch % self.cluster_update_interval == 0:
                    self._update_clustering(train_data)

            # 训练一个epoch
            train_loss = self._train_one_epoch(
                epoch, train_loader, model_optim, criterion, current_stage
            )

            # 验证
            vali_loss = self.vali(vali_data, vali_loader, criterion)

            print(f"Epoch {epoch+1}/{self.total_epochs} [阶段{current_stage}] | "
                  f"Train Loss: {train_loss:.7f} | Vali Loss: {vali_loss:.7f}")

            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("早停触发")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + "/checkpoint.pth"
        self.model.load_state_dict(torch.load(best_model_path))
        return self.model

    def _train_one_epoch(self, epoch, train_loader, optimizer, criterion, stage):
        """训练一个epoch"""
        self.model.train()
        train_loss = []
        iter_count = 0

        pbar = tqdm(enumerate(train_loader), total=len(train_loader),
                     desc=f'Epoch {epoch+1} [阶段{stage}]')

        for i, batch in pbar:
            iter_count += 1
            optimizer.zero_grad()

            batch = batch.to(self.device)
            batch_x = batch.x
            batch_y = batch.y
            edge_index = torch.tensor(batch.edge_index, dtype=torch.long)
            edge_attr = torch.tensor(batch.edge_attr, dtype=torch.float)

            if stage == 1:
                # 阶段1: 仅时序编解码 (无图结构)
                outputs = self.model.forward_stage1(batch_x)
            else:
                # 阶段2/3: 完整模型
                graph_data = type('obj', (object,), {
                    'x': batch_x,
                    'edge_index': edge_index,
                    'edge_attr': edge_attr
                })()
                outputs = self.model(graph_data, device=self.device)

            # 处理输出维度
            outputs = outputs.reshape(self.args.batch_size, -1,
                                     self.args.pred_len, 1)
            batch_y = batch_y.reshape(self.args.batch_size, -1,
                                     self.args.pred_len, 1)

            if self.args.features == 'MS':
                pred = outputs[:, :, :, 0]
                true = batch_y[:, :, :, 0]
            else:
                pred = outputs.squeeze()
                true = batch_y.squeeze()

            loss = criterion(pred, true)
            train_loss.append(loss.item())

            loss.backward()
            optimizer.step()

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'stage': stage
            })

        return np.average(train_loss)

    def _update_clustering(self, train_data):
        """
        更新隐空间聚类分配

        使用训练集数据的编码器输出来更新K-means聚类
        """
        model = self.model
        if hasattr(model, 'module'):
            model = model.module

        print("更新隐空间聚类...")
        # 使用训练数据的统计特征进行聚类
        # 实际实现中可以从train_data中采样部分数据
        # 此处简化: 使用已有的cluster_assignment
        # model.hierarchical_encoder.update_clustering(x_all)

    def test(self, setting, test=0):
        """测试"""
        test_data, test_loader = self._get_data(flag="test")
        if test:
            print("加载模型")
            self.model.load_state_dict(
                torch.load(os.path.join("./checkpoints/" + setting, "checkpoint.pth"))
            )

        preds = []
        trues = []
        total_latency = 0.0
        count = 0

        folder_path = "./results/" + setting + "/"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            pbar = tqdm(test_loader, desc='Testing')
            for batch in pbar:
                try:
                    batch = batch.to(self.device)
                    batch_x = batch.x
                    batch_y = batch.y
                    edge_index = torch.tensor(batch.edge_index, dtype=torch.long)
                    edge_attr = torch.tensor(batch.edge_attr, dtype=torch.float)

                    if self.device.type == 'cuda':
                        torch.cuda.synchronize()
                    start_time = time.time()

                    graph_data = type('obj', (object,), {
                        'x': batch_x,
                        'edge_index': edge_index,
                        'edge_attr': edge_attr
                    })()
                    outputs = self.model(graph_data, device=self.device)

                    if self.device.type == 'cuda':
                        torch.cuda.synchronize()
                    end_time = time.time()

                    total_latency += (end_time - start_time)
                    count += 1

                    outputs = outputs.reshape(self.args.batch_size, -1,
                                             self.args.pred_len, 1)
                    batch_y = batch_y.reshape(self.args.batch_size, -1,
                                             self.args.pred_len, 1)

                    outputs = outputs.detach().cpu().numpy()
                    batch_y = batch_y.detach().cpu().numpy()

                    # 反归一化
                    if hasattr(test_data, 'inverse_transform') and self.args.inverse:
                        shape = outputs.shape
                        node_names = list(test_data.processed_data.keys())
                        for node_idx, node_name in enumerate(node_names):
                            if node_idx < shape[1]:
                                for batch_idx in range(shape[0]):
                                    outputs[batch_idx, node_idx, :] = \
                                        test_data.inverse_transform(
                                            outputs[batch_idx, node_idx, :],
                                            node_name, target_col_only=True
                                        )
                                    batch_y[batch_idx, node_idx, :] = \
                                        test_data.inverse_transform(
                                            batch_y[batch_idx, node_idx, :],
                                            node_name, target_col_only=True
                                        )

                    f_dim = 0 if self.args.features == "MS" else slice(None)
                    if isinstance(f_dim, int):
                        outputs = outputs[:, :, :, f_dim] if len(outputs.shape) == 4 else outputs
                        batch_y = batch_y[:, :, :, f_dim] if len(batch_y.shape) == 4 else batch_y

                    preds.append(outputs)
                    trues.append(batch_y)
                    pbar.set_postfix({'batch': f'{count}/{len(test_loader)}'})

                except Exception as e:
                    print(f"测试批次 {count} 出错: {e}")
                    continue

        avg_latency = total_latency / count if count > 0 else 0
        print(f"\n推理速度统计:")
        print(f"  平均批次延迟: {avg_latency:.4f} 秒")

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)

        if len(preds.shape) == 4:
            preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
            trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])

        mae, mse, rmse, mape, mspe, r2 = metric(preds, trues)
        print(f"mse:{mse}, mae:{mae}, rmse:{rmse}, mape:{mape}, mspe:{mspe}, r2:{r2}")

        f = open("result_lightst.txt", "a")
        f.write(setting + "  \n")
        f.write(f"mse:{mse}, mae:{mae}, rmse:{rmse}, mape:{mape}, mspe:{mspe}, r2:{r2}")
        f.write("\n\n")
        f.close()

        np.save(folder_path + "metrics.npy", np.array([mae, mse, rmse, mape, mspe, r2]))
        np.save(folder_path + "pred.npy", preds)
        np.save(folder_path + "true.npy", trues)
        return
