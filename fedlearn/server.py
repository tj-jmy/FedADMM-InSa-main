import copy
import torch
import numpy as np
from numpy.ma.core import zeros_like

from utils.model import init_model


class Server(object):
    """Server side operations."""

    def __init__(self, cfg, alg):
        self.cfg = cfg
        self.model = init_model(cfg)  # model z_k # 初始化服务器模型 z_k
        self.state = self.model.state_dict()  # state z_k # 保存模型参数状态 z_k
        self.alg = alg
        self.fh_hmul_pm = None
        self.fh_nul = None

    def select_clients(self, frac):
        """Randomly select a subset of clients."""
        num = max(np.ceil(frac * self.cfg.m).astype(int), 1)  # number of clients to sample # 计算要选择的客户端数量，确保至少选择一个客户端
        self.active_clients = np.random.choice(range(self.cfg.m), num, replace=False)
        return self.active_clients  # 返回选择的客户端索引

    def aggregate(self, res_clients: dict, add_noise=False):
        """Server aggregation process."""
        alpha = self.cfg.alpha  # 每个客户端的权重
        model_u = res_clients["models"]  # list of clients' models # 获取客户端的模型列表
        model_z = copy.deepcopy(model_u[0])  # init model_server_new # 初始化聚合后的服务器模型
        m = len(model_u)  # number of clients to aggregate # 要聚合的客户端数量
        # Iterate over model layers for aggregation.

        theta_m = copy.deepcopy(model_u)
        for i in range(m):
            for key in theta_m[i].keys():
                theta_m[i][key] = torch.zeros_like(theta_m[i][key])
        theta_mean, theta_std = [], []

        sigma = 0.1

        # 遍历模型的每一层，进行参数聚合
        for key in model_z.keys():
            model_z[key].zero_()  # reset model parameters # 重置模型参数
            if "num_batches_tracked" in key:  # for BN batches count # 如果是 BN 层的批次统计量，则跳过
                continue
            elif "running" in key:  # for BN running mean/var # 如果是 BN 层的运行均值/方差
                for i in range(m):
                    model_z[key] += alpha[i] * model_u[i][key]
            else:  # for other layers # 其他层的参数聚合
                # FedADMM server aggregation. # 对于 FedADMM 算法的聚合过程
                if self.alg in ["admm", "admm_in", "admm_insa"]:
                    beta, lamda = res_clients["beta"], res_clients["lambda"]
                    for i in range(m):  # iterate over clients
                        # model_z[key] += alpha[i] * (beta[i] * model_u[i][key] - lamda[i][key])
                        # model_z[key] += alpha[i] * (beta[i] * model_u[i][key] + lamda[i][key])
                        model_z[key] += (beta[i] * model_u[i][key] + lamda[i][key])
                        theta_m[i][key] = (beta[i] * model_u[i][key] + lamda[i][key])
                    # tmp = [alpha[i] * beta[i] for i in range(m)]
                    tmp = [beta[i] for i in range(m)]
                    sigma = sum(tmp)
                    model_z[key] = torch.div(model_z[key], sum(tmp))
                # FedAvg server aggregation. # 对于 FedAvg 算法的聚合过程
                elif self.alg in ["fedavg"]:
                    for i in range(m):  # iterate over clients
                        model_z[key] += alpha[i] * model_u[i][key]
                        theta_m[i][key] = model_u[i][key]
                else:
                    raise ValueError(f"Invalid algorithm.")
        # Server aggregation with memory  # 如果是 加权ADMM 算法，进行带记忆的聚合，先直接聚合，再和上一轮的全局模型加权聚合
        if self.alg in ["admm_in", "admm_insa"]:
            model_z = self._admm_memory(model_z)

        for i in range(m):
            theta_mean.append(torch.mean(torch.cat([param.view(-1) for param in theta_m[i].values()])))
            theta_std.append(torch.std(torch.cat([param.view(-1) for param in theta_m[i].values()])))

        if add_noise:
            noise = copy.deepcopy(model_z)
            for idx, key in enumerate(noise.keys()):
                noise[key].zero_()
                for i in range(m):
                    noise[key] += (self.fh_hmul_pm[i].real - theta_std[i]) * (theta_m[i][key] - theta_mean[i]) / \
                                  theta_std[i]
                noise[key] += self.fh_nul[idx].real
                noise[key] = torch.div(noise[key], sigma)

            for key in model_z.keys():
                model_z[key] += noise[key]

        # 更新服务器的模型
        self.model.load_state_dict(model_z)  # update server's model

    def _admm_memory(self, model_new):
        """Server aggregation with memory."""
        delta = self.cfg.delta  # 记忆系数
        cof1 = 1 / (1 + delta)
        cof2 = delta / (1 + delta)
        model = copy.deepcopy(model_new)
        # 遍历模型的每一层
        for key in model.keys():  # iterate over model layers
            model[key].zero_()  # reset model parameters # 重置模型参数
            model[key] = cof1 * model_new[key] + cof2 * self.state[key]  # 带记忆的聚合：综合当前和上轮模型的参数
        return model

    def load_noise_args(self, fh_hmul_pm, fh_nul):
        self.fh_hmul_pm = fh_hmul_pm
        self.fh_nul = fh_nul
