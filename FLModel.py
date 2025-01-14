# Federated Learning Model in PyTorch
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from utils import gaussian_noise
from rdp_analysis import calibrating_sampled_gaussian

from MLModel import *

import numpy as np
import copy


class FLClient(nn.Module):
    """ Client of Federated Learning framework.
        1. Receive global model from server
        2. Perform local training (compute gradients)
        3. Return local model (gradients) to server
    """

    def __init__(self, model, output_size, data, lr, E, batch_size, q, clip, sigma, device=None):
        """
        :param model: ML model's training process should be implemented
        :param data: (tuple) dataset, all data in client side is used as training data
        :param lr: learning rate
        :param E: epoch of local update
        """
        super(FLClient, self).__init__()
        self.device = device
        self.BATCH_SIZE = batch_size
        self.torch_dataset = TensorDataset(torch.tensor(data[0]),
                                           torch.tensor(data[1]))
        self.data_size = len(self.torch_dataset)
        self.data_loader = DataLoader(
            dataset=self.torch_dataset,
            batch_size=self.BATCH_SIZE,
            shuffle=True
        )
        self.sigma = sigma  # DP noise level
        self.lr = lr
        self.E = E
        self.clip = clip
        self.q = q
        if model == 'scatter':
            self.model = ScatterLinear(81, (7, 7), input_norm="GroupNorm", num_groups=27).to(self.device)
        else:
            self.model = model(data[0].shape[1], output_size).to(self.device)

    def recv(self, model_param):
        """receive global model from aggregator (server)"""
        self.model.load_state_dict(copy.deepcopy(model_param))

    def update(self):
        """local model update"""
        self.model.train()
        criterion = nn.CrossEntropyLoss(reduction='none')
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)
        # optimizer = torch.optim.Adam(self.model.parameters())

        for e in range(self.E):
            # randomly select q fraction samples from data
            # according to the privacy analysis of moments accountant
            # training "Lots" are sampled by poisson sampling
            idx = np.where(np.random.rand(len(self.torch_dataset[:][0])) < self.q)[0]
            sampled_dataset = TensorDataset(self.torch_dataset[idx][0], self.torch_dataset[idx][1])
            """训练样本为q采样后(q为L/N)的样本"""
            sample_data_loader = DataLoader(
                dataset=sampled_dataset,
                batch_size=self.BATCH_SIZE,
                shuffle=True
            )

            optimizer.zero_grad()
            """清空梯度信息"""
            clipped_grads = {name: torch.zeros_like(param) for name, param in self.model.named_parameters()}
            for batch_x, batch_y in sample_data_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                pred_y = self.model(batch_x.float())
                loss = criterion(pred_y, batch_y.long())
                """以上是进行损失计算的过程"""
                # bound l2 sensitivity (gradient clipping)
                # clip each of the gradient in the "Lot"
                for i in range(loss.size()[0]):
                    """每一个参数的损失进行遍历"""
                    loss[i].backward(retain_graph=True)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.clip)
                    """梯度裁剪"""
                    for name, param in self.model.named_parameters():
                        clipped_grads[name] += param.grad
                    self.model.zero_grad()

            # add Gaussian noise
            for name, param in self.model.named_parameters():
                clipped_grads[name] += gaussian_noise(clipped_grads[name].shape, self.clip, self.sigma,
                                                      device=self.device)

            # scale back
            for name, param in self.model.named_parameters():
                clipped_grads[name] /= (self.data_size * self.q)
            """将采样的梯度计算放到整体上"""
            for name, param in self.model.named_parameters():
                param.grad = clipped_grads[name]
            """对model的参数进行梯度传递"""

            # update local model
            optimizer.step()


class FLServer(nn.Module):
    """ Server of Federated Learning
        1. Receive model (or gradients) from clients
        2. Aggregate local models (or gradients)
        3. Compute global model, broadcast global model to clients
    """

    def __init__(self, fl_param):
        super(FLServer, self).__init__()
        self.device = fl_param['device']
        # 设备位置
        self.client_num = fl_param['client_num']
        # 用户数目
        self.C = fl_param['C']  # (float) C in [0, 1]
        # C是DP_SGD的裁剪参数吗
        self.clip = fl_param['clip']
        """总共与客户端交互轮数"""
        self.T = fl_param['tot_T']  # total number of global iterations (communication rounds)

        self.data = []
        self.target = []

        for sample in fl_param['data'][self.client_num:]:
            self.data += [torch.tensor(sample[0]).to(self.device)]  # test set
            self.target += [torch.tensor(sample[1]).to(self.device)]  # target label

        self.input_size = int(self.data[0].shape[1])
        self.lr = fl_param['lr']

        # compute noise using moments accountant
        # self.sigma = compute_noise(1, fl_param['q'], fl_param['eps'], fl_param['E']*fl_param['tot_T'], fl_param['delta'], 1e-5)

        # calibration with subsampeld Gaussian mechanism under composition
        self.sigma = 0.78
        # self.sigma = calibrating_sampled_gaussian(fl_param['q'], fl_param['eps'], fl_param['delta'], iters=fl_param['E']*fl_param['tot_T'], err=1e-3)
        print("noise scale =", self.sigma)

        self.clients = [FLClient(fl_param['model'],
                                 fl_param['output_size'],
                                 fl_param['data'][i],
                                 fl_param['lr'],
                                 fl_param['E'],
                                 fl_param['batch_size'],
                                 fl_param['q'],
                                 fl_param['clip'],
                                 self.sigma,
                                 self.device)
                        for i in range(self.client_num)]
        self.global_model = fl_param['model'](self.input_size, fl_param['output_size']).to(self.device)
        self.weight = np.array([client.data_size * 1.0 for client in self.clients])
        self.broadcast(self.global_model.state_dict())

    def aggregated(self, idxs_users):
        """FedAvg"""
        model_par = [self.clients[idx].model.state_dict() for idx in idxs_users]
        new_par = copy.deepcopy(model_par[0])
        for name in new_par:
            new_par[name] = torch.zeros(new_par[name].shape).to(self.device)
        for idx, par in enumerate(model_par):
            w = self.weight[idxs_users[idx]] / np.sum(self.weight[:])
            for name in new_par:
                # new_par[name] += par[name] * (self.weight[idxs_users[idx]] / np.sum(self.weight[idxs_users]))
                new_par[name] += par[name] * (w / self.C)
        self.global_model.load_state_dict(copy.deepcopy(new_par))
        return self.global_model.state_dict().copy()

    """下发给全局变量"""

    def broadcast(self, new_par):
        """Send aggregated model to all clients"""
        for client in self.clients:
            client.recv(new_par.copy())

    def test_acc(self):
        self.global_model.eval()
        correct = 0
        tot_sample = 0
        for i in range(len(self.data)):
            t_pred_y = self.global_model(self.data[i])
            _, predicted = torch.max(t_pred_y, 1)
            correct += (predicted == self.target[i]).sum().item()
            tot_sample += self.target[i].size(0)
        acc = correct / tot_sample
        return acc

    """以下是实现FedSwap算法，将权重进行交换以实现对noniid数据集的"""

    def fedswap_global_update(self, t, k1, k2):
        idxs_users = np.sort(np.random.choice(range(len(self.clients)), int(self.C * len(self.clients)), replace=False))
        for idx in idxs_users:
            self.clients[idx].update()
        # 以上是自我训练的
        print("输出t：" + str(t) + "k1:" + str(k1) + "k2:" + str(k2))

        if t % k1 == 0 and t % (k1 * k2) != 0:
            for idx in idxs_users:
                wr = self.fedswap(idx)
                self.clients[idx].recv(wr)
                print("swap" + str(t))
        if t % (k1 * k2) == 0:
            self.broadcast(self.aggregated(idxs_users))
            print("fedavg" + str(t))
        acc = self.test_acc()
        torch.cuda.empty_cache()
        return acc

    def fedswap(self, idx):
        idxs_users = np.sort(np.random.choice(range(len(self.clients)), int(self.C * len(self.clients)), replace=False))
        random_user = np.random.choice(len(idxs_users))
        wt = self.clients[random_user].model.state_dict()
        self.clients[random_user].recv(self.clients[idx].model.state_dict())
        return wt

    def global_update(self):
        # idxs_users = np.random.choice(range(len(self.clients)), int(self.C * len(self.clients)), replace=False)
        idxs_users = np.sort(np.random.choice(range(len(self.clients)), int(self.C * len(self.clients)), replace=False))
        for idx in idxs_users:
            self.clients[idx].update()
        # 以上是自我训练的
        self.broadcast(self.aggregated(idxs_users))
        acc = self.test_acc()
        torch.cuda.empty_cache()
        return acc

    def set_lr(self, lr):
        for c in self.clients:
            c.lr = lr

