from data_provider.gat_data_loader import *
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
from utils.soft_dtw import SoftDTW
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")


class Exp_Spatial_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Spatial_Long_Term_Forecast, self).__init__(args)
        # 添加图相关属性
        self.edge_index = None
        self.n_nodes = None

    def _build_model(self):
        # 确保模型参数正确设置
        if not hasattr(self.args, 'n_nodes') or self.args.n_nodes is None:
            # 临时加载数据来获取节点数
            temp_data, _ = self._get_data(flag='train')
            self.args.n_nodes = len(temp_data.processed_data)
            self.args.enc_in = list(temp_data.processed_data.values())[0]['features'].shape[1]
            print(f"Auto-detected: n_nodes={self.args.n_nodes}, enc_in={self.args.enc_in}")

        model = self.model_dict[self.args.model].Model(self.args).float()

        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total number of parameters: {total_params}")

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_loader, data_set = create_dataloader(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        if hasattr(self.args, 'loss') and self.args.loss == 'MSE':
            criterion = nn.MSELoss()
        else:
            criterion = nn.L1Loss()  # MAE Loss
        # criterion = SoftDTW(gamma=1.0, normalize=True)
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            # 添加进度条
            pbar = tqdm(vali_loader, desc='Validation', leave=False)
            for i, batch in enumerate(pbar):
                # 修复：使用GraphBatch格式
                batch = batch.to(self.device)
                batch_x = batch.x  # [batch_size, n_nodes, seq_len, n_features]
                batch_y = batch.y  # [batch_size, n_nodes, pred_len]
                edge_index = batch.edge_index

                # GAT模型预测
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        # GAT模型接受GraphData格式
                        graph_data = type('obj', (object,), {'x': batch_x, 'edge_index': edge_index})()
                        outputs = self.model(graph_data)
                else:
                    graph_data = type('obj', (object,), {'x': batch_x, 'edge_index': edge_index})()
                    outputs = self.model(graph_data)

                # 处理输出维度
                if len(outputs.shape) == 2:
                    batch_size = batch_x.shape[0]
                    n_nodes = batch_x.shape[1]
                    outputs = outputs.reshape(batch_size, n_nodes, -1)
                
                if len(outputs.shape) == 4:
                    outputs = outputs.squeeze(-1)
                
                # 计算损失
                if self.args.features == 'MS':
                    pred = outputs[:, :, :, 0] if len(outputs.shape) == 4 else outputs
                    true = batch_y[:, :, :, 0] if len(batch_y.shape) == 4 else batch_y
                else:
                    pred = outputs.squeeze() if len(outputs.shape) > 3 else outputs
                    true = batch_y.squeeze() if len(batch_y.shape) > 3 else batch_y

                pred = pred.detach().cpu()
                true = true.detach().cpu()

                loss = criterion(pred, true)
                total_loss.append(loss.item())
                
                # 更新进度条描述
                pbar.set_postfix({'val_loss': f'{loss.item():.4f}'})

        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag="train")
        vali_data, vali_loader = self._get_data(flag="val")
        test_data, test_loader = self._get_data(flag="test")

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            
            # 添加训练进度条
            pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f'Epoch {epoch+1}/{self.args.train_epochs}')
            for i, batch in pbar:
                iter_count += 1
                model_optim.zero_grad()
                
                batch = batch.to(self.device)
                batch_x = batch.x
                batch_y = batch.y
                edge_index = batch.edge_index

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        graph_data = type('obj', (object,), {'x': batch_x, 'edge_index': edge_index})()
                        outputs = self.model(graph_data)

                        if len(outputs.shape) == 2:
                            batch_size = batch_x.shape[0]
                            n_nodes = batch_x.shape[1]
                            outputs = outputs.reshape(batch_size, n_nodes, -1)
                        
                        if len(outputs.shape) == 4:
                            outputs = outputs.squeeze(-1)

                        if self.args.features == 'MS':
                            pred = outputs[:, :, :, 0] if len(outputs.shape) == 4 else outputs
                            true = batch_y[:, :, :, 0] if len(batch_y.shape) == 4 else batch_y
                        else:
                            pred = outputs.squeeze() if len(outputs.shape) > 3 else outputs
                            true = batch_y.squeeze() if len(batch_y.shape) > 3 else batch_y

                        loss = criterion(pred, true)
                        train_loss.append(loss.item())
                else:
                    graph_data = type('obj', (object,), {'x': batch_x, 'edge_index': edge_index})()
                    outputs = self.model(graph_data)

                    if len(outputs.shape) == 2:
                        batch_size = batch_x.shape[0]
                        n_nodes = batch_x.shape[1]
                        outputs = outputs.reshape(batch_size, n_nodes, -1)
                    
                    if len(outputs.shape) == 4:
                        outputs = outputs.squeeze(-1)

                    if self.args.features == 'MS':
                        pred = outputs[:, :, :, 0] if len(outputs.shape) == 4 else outputs
                        true = batch_y[:, :, :, 0] if len(batch_y.shape) == 4 else batch_y
                    else:
                        pred = outputs.squeeze() if len(outputs.shape) > 3 else outputs
                        true = batch_y.squeeze() if len(batch_y.shape) > 3 else batch_y

                    loss = criterion(pred, true)
                    train_loss.append(loss.item())

                # 更新进度条
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'lr': model_optim.param_groups[0]['lr']
                })

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss, test_loss
                )
            )
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + "/" + "checkpoint.pth"
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag="test")
        if test:
            print("loading model")
            self.model.load_state_dict(
                torch.load(os.path.join("./checkpoints/" + setting, "checkpoint.pth"))
            )

        preds = []
        trues = []
        total_latency = 0.0
        count = 0

        folder_path = "./spatial_test_results/" + setting + "/"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            # 添加测试进度条
            pbar = tqdm(test_loader, desc='Testing')
            for batch in pbar:
                batch = batch.to(self.device)
                batch_x = batch.x
                batch_y = batch.y
                edge_index = batch.edge_index

                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                start_time = time.time()

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        graph_data = type('obj', (object,), {'x': batch_x, 'edge_index': edge_index})()
                        outputs = self.model(graph_data)
                else:
                    graph_data = type('obj', (object,), {'x': batch_x, 'edge_index': edge_index})()
                    outputs = self.model(graph_data)

                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                end_time = time.time()
                
                total_latency += (end_time - start_time)
                count += 1

                if len(outputs.shape) == 2:
                    batch_size = batch_x.shape[0]
                    n_nodes = batch_x.shape[1]
                    outputs = outputs.reshape(batch_size, n_nodes, -1)
                
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                if hasattr(test_data, 'inverse_transform') and self.args.inverse:
                    shape = outputs.shape
                    node_names = list(test_data.processed_data.keys())
                    for node_idx, node_name in enumerate(node_names):
                        if node_idx < shape[1]:
                            for batch_idx in range(shape[0]):
                                outputs[batch_idx, node_idx, :] = test_data.inverse_transform(
                                    outputs[batch_idx, node_idx, :], 
                                    node_name, 
                                    target_col_only=True
                                )
                                batch_y[batch_idx, node_idx, :] = test_data.inverse_transform(
                                    batch_y[batch_idx, node_idx, :], 
                                    node_name, 
                                    target_col_only=True
                                )

                f_dim = 0 if self.args.features == "MS" else slice(None)
                if isinstance(f_dim, int):
                    outputs = outputs[:, :, :, f_dim] if len(outputs.shape) == 4 else outputs
                    batch_y = batch_y[:, :, :, f_dim] if len(batch_y.shape) == 4 else batch_y

                preds.append(outputs)
                trues.append(batch_y)
                
                # 更新进度条
                pbar.set_postfix({'batch': f'{count}/{len(test_loader)}'})

        avg_latency = total_latency / count
        print(f"\nInference Speed Summary:")
        print(f"- Total batches: {count}")
        print(f"- Average latency per batch: {avg_latency:.4f} seconds")
        print(f"- Throughput: {len(test_loader.dataset)/total_latency:.2f} samples/s")

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print("test shape:", preds.shape, trues.shape)

        if len(preds.shape) == 4:
            preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
            trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        
        print("reshaped test shape:", preds.shape, trues.shape)

        folder_path = "./results/" + setting + "/"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe, r2 = metric(preds, trues)
        print("mse:{}, mae:{}, rmse:{}, mape:{}, mspe:{}, r2:{}".format(
            mse, mae, rmse, mape, mspe, r2))
        
        f = open("result_spatial_long_term_forecast.txt", "a")
        f.write(setting + "  \n")
        f.write("mse:{}, mae:{}, rmse:{}, mape:{}, mspe:{}, r2:{}".format(
            mse, mae, rmse, mape, mspe, r2))
        f.write("\n")
        f.write("\n")
        f.close()

        np.save(folder_path + "metrics.npy", np.array([mae, mse, rmse, mape, mspe, r2]))
        np.save(folder_path + "pred.npy", preds)
        np.save(folder_path + "true.npy", trues)
        return