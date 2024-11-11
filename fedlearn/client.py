import copy
import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from utils.dataset import DatasetSplit
from utils.model import init_model


class Clients(object):
    """Clients' local update processes."""

    def __init__(self, cfg, server_model, dataset, alg):
        # 初始化函数，用于设置客户端的配置和参数
        self.cfg = cfg  # scenario config
        self.device = cfg.device  # gpu or cpu
        self.datasets = dataset  # clients' datasets
        # initialize parameters# 调用内部方法初始化参数，包括拉格朗日乘子和惩罚参数
        self.alg = alg  # algorithm name
        self._init_params(cfg, server_model)
        self.alpha = self.cfg.alpha  # 每个客户端的权重


    def local_update(self, model_server, selected_clients, k):
        # model_server：传入的服务器模型，用于初始化客户端模型，使所有客户端从相同的参数开始本地更新。
        # selected_clients：一个列表或集合，包含在当前轮次中选中的客户端索引。仅在此列表中的客户端会执行本地更新。
        # k：当前的训练轮次编号，通常用于控制特定轮次的操作，或记录客户端的训练步数。
        """Clients' local update processes."""

        # 将服务器模型和当前轮次编号k赋给实例变量
        self.model_server = model_server
        self.k = k

        # 遍历所有客户端
        for self.i in range(self.cfg.m):
            # 如果客户端i在本轮次中被选中，则执行本地更新
            if self.i in selected_clients:
                # Load client i's local settings
                # 深拷贝服务器模型并将其设置为训练模式
                self.model_i = copy.deepcopy(self.model_server).train()
                # 加载客户端i的数据集切片
                self.dataset_i = DatasetSplit(self.datasets, self.i)
                # 初始化数据加载器和优化器
                data_loader = init_data_loader(self.cfg, self.dataset_i)
                optimizer = init_optimizer(self.cfg, self.model_i, k)
                # 计数已完成的SGD步数和训练轮次

                # count_s：用于统计已完成的SGD（随机梯度下降）步数。每处理一个数据批次，count_s增加1，表示完成了一次SGD更新。
                # count_e：用于统计已完成的训练轮次（epochs）。在每轮次结束时，count_e增加1，这样可以记录客户端本地数据已完整遍历的次数。
                count_s, count_e = 0, 0  # finished SGD steps or epochs
                # Start local training 开始本地训练过程
                train_flag = True  # flag for keep local training
                while train_flag:  # update client i's model parameters u_i
                    # check the criterion after each epoch
                    if self._stop_check(count_e):
                        break
                    # 遍历每个批次的数据，执行SGD
                    for data, labels in data_loader:  # iterate over batches
                        # 将数据和标签传入设备（如GPU）
                        self.data, self.labels = data.to(
                            self.device), labels.to(self.device)
                        # 执行一次本地梯度下降（FGD或SGD）
                        # perform single FGD/SGD step
                        self._local_step(self.cfg)
                        # 更新本地模型参数
                        optimizer.step()  # update client's model parameters u
                        count_s += 1
                    # 增加已完成的训练轮次计数
                    count_e += 1
                # Personalized actions of different algorithms.
                self._local_step_personalize(self.cfg)
                # 将客户端i的模型参数保存到模型字典中
                self.models[self.i] = self.model_i.state_dict()
                self.steps[self.i][self.k] = count_e
                # 记录客户端i的训练轮次和步数
                # record local FGD/SGD steps
                self.res_dict["steps"] = self.steps
                # self.send_back 通常指的是客户端更新完成后，将模型更新的结果返回给服务器端进行聚合。在联邦学习或分布式优化框架中，send_back包含客户端的模型参数或对偶变量更新，以便服务器进行全局更新。
                #
                # 具体来说，send_back的数据可能包括：
                #
                # 本地模型参数：用于服务器的全局模型聚合。
                # 训练步数或状态：帮助服务器跟踪每个客户端的训练进展。
        return self.send_back

    def _stop_check(self, count):
        """Determine whether to stop client's local training."""
        stop_training = False
        # Check the maximum number. 检查是否达到最大训练轮次 self.cfg.E
        if count >= self.cfg.E:
            stop_training = True  # 达到最大轮次，设置停止标记为 True
        # Check the admm inexactness criterion.
        elif self.alg in ["admm_insa", "admm_in"]:  # 检查 ADMM 的不精确准则，仅在特定算法下启用
            if self._inexact_check(count):  # 如果符合不精确条件
                stop_training = True  # stop training # 设置停止标记为 True
        return stop_training  # 返回最终的停止标记

    def _local_step(self, cfg):
        """Client i performs single FGD/SGD step.""""""客户端 i 执行一次 FGD/SGD 更新。"""
        # 如果选择了 FedAvg 算法，则使用标准的梯度计算
        if self.alg in ["fedavg"]:
            self._grad(self.model_i, self.data, self.labels)
        # 如果选择了 ADMM 类算法（包括 Vanilla ADMM 和自适应 ADMM），进行 ADMM 特定的更新
        elif self.alg in ["admm", "admm_insa", "admm_in"]:
            # 对于自适应 ADMM 算法（admm_insa 或 admm_in），且批量大小为 0（即非批量处理），跳过梯度计算
            # 在 admm_insa 和 admm_in 算法中，当批量大小 (bs) 为 0 且在不精确检查中已计算过梯度时跳过梯度计算，
            # 主要是为了避免重复计算梯度，从而减少计算负担和通信成本。这种情况下，梯度已在 _inexact_check 中预先计算并用于判断是否满足不精确准则，
            # 因此不需要在本地更新中重复进行。这一策略优化了资源使用，尤其在资源受限的分布式环境中提高了效率。
            if self.alg in ["admm_insa", "admm_in"] and self.cfg.bs == 0:
                pass  # grad calculated during the inexactness check, see self._inexact_e()# 如果已在不精确检查中计算梯度，则此处跳过
            else:  # 否则执行 ADMM 的 u 更新步骤
                self._admm_u(self.model_i, self.data, self.labels)
        else:  # 如果算法无效，抛出错误
            raise ValueError(f"Invalid algorithm.")

    def _local_step_personalize(self, cfg):
        """Personalized actions of different algorithms."""
        # ADMM related local update process.# 针对 ADMM 相关算法的个性化更新
        if self.alg in ["admm_insa", "admm_in", "admm"]:
            beta_i_k = self.beta[self.i][self.k]  # 获取当前客户端和轮次的惩罚参数 beta
            # update dual variables lambda  # 更新拉格朗日乘子 lambda
            self._update_lambda(beta_i_k)
            if self.alg == "admm_insa":  # udpate penlaty parameter beta 如果是 admm_insa（自适应 ADMM），更新惩罚参数 beta
                self._adp_beta(beta_i_k)  # 自适应调整 beta
            # 在 admm 模式下的惩罚参数更新通常不会进行自适应调整，因此 beta 参数保持恒定。
            # 具体来说，admm 直接使用固定的惩罚参数 beta_i_k，并在每轮更新中保持不变，以确保每个客户端遵循相同的惩罚标准。
            # record params to send back to the server  # 记录更新后的 beta 参数，以便传回服务器
            self.send_back["beta"] = self.beta[:, self.k]  # beta_i^k use new
            self.res_dict["beta"] = self.beta  # record beta to plot

    def _update_lambda(self, beta):
        """Update the dual variable lambda."""
        # 获取客户端和服务器模型的参数状态字典
        u_state = self.model_i.state_dict()
        z_state = self.model_server.state_dict()
        # 遍历每个模型参数（不包括 BN 统计量）
        for name, _ in self.model_i.named_parameters():  # without BN statistics
            # 更新拉格朗日乘子：lambda = lambda - beta * (u - z)
            # self.lamda[self.i][name] -= beta * (u_state[name] - z_state[name])
            # 根据文章1的拉格朗日乘子更新公式，更新逻辑应改为加法而非减法。
            self.lamda[self.i][name] += beta * (u_state[name] - z_state[name])

    def _adp_beta(self, beta):
        """Update beta based on the adaptive penalty scheme."""
        # 获取当前轮次和上一轮次的模型
        u_kplus1, z_k = self.model_i, self.model_server  # 当前模型（本轮）
        u_k = state2model(self.cfg, self.models[self.i])  # 上一轮次的客户端模型
        # 原始残差，用于度量本地模型和上次更新间的差异，调整约束强度。
        primal_residual = beta * l2_norm(u_kplus1, u_k)
        dual_residual = l2_norm(u_kplus1, z_k)  # 对偶残差，度量本地模型和全局模型间差异。
        # 自适应调整 beta
        # 增加 beta（增强约束）：当原始残差相对较小时，说明客户端模型与上一轮次的更新差异较小，但与全局模型的差异（对偶残差）较大。
        # 此时需要增强一致性约束，使本地模型更快速地逼近全局模型。
        #
        # 减少 beta（放宽约束）：当对偶残差相对较小时，说明本地模型与全局模型已经接近，但本地更新的步幅较大（原始残差较大）。
        # 此时降低 beta，允许本地模型拥有更大的更新灵活性，有利于适应本地数据的特性。
        if self.cfg.mu * primal_residual < dual_residual:
            beta *= self.cfg.tau  # 增加 beta
        elif self.cfg.mu * dual_residual < primal_residual:
            beta /= self.cfg.tau  # 减少 beta
        self.beta[self.i][self.k + 1:] = beta  # update beta

    def _inexact_check(self, count):
        """Check the inexactness criterion."""
        # 计算自适应惩罚参数 beta 的调整值
        beta_tilde = self.beta[self.i][self.k] / self.cfg.c_i
        # 计算不精确准则的因子 sigma，用于衡量当前模型变化
        sigma = 0.999 * np.sqrt(2) / (np.sqrt(2) + np.sqrt(beta_tilde))
        # first time or using SGD, when e_u remains unchanged, calculate once and reuse
        # 初始检查（count 为 0），或使用 SGD 时计算初始的 e_u 并缓存以重复使用
        if count == 0:
            # use z^k instead of u_i^k# 使用全局模型参数 z^k
            self.e_u = self._inexact_e(self.model_server)
        # 计算新的不精确度 e_u
        e_u_new = self._inexact_e(self.model_i)
        # 如果新的不精确度 e_u_new 小于 sigma * self.e_u，则满足不精确准则，返回 True，否则返回 False
        return True if e_u_new <= sigma * self.e_u else False

    def _inexact_e(self, model: nn.Module):
        """Calculate the l2 norm of the inexactness residual e(u)."""

        data_loader_tmp = DataLoader(self.dataset_i, batch_size=500)
        model.eval()  # if using .train(), normalization layers cause problems
        model.zero_grad()
        for batch_idx, (data, labels) in enumerate(data_loader_tmp):
            data, labels = data.to(self.device), labels.to(self.device)
            pred = model(data)
            loss = model.loss(pred, labels)  # default reduction == "mean"
            loss.backward()  # accumulate grads
        u_state = model.state_dict()  # u_i
        z_state = self.model_server.state_dict()  # z
        for name, param in model.named_parameters():  # without BN statistics
            param.grad /= batch_idx + 1  # fix accumulated grads
            param.grad -= self.lamda[self.i][name]
            param.grad += self.beta[self.i][self.k] * \
                (u_state[name] - z_state[name])
        res = 0
        for param in model.parameters():  # without BN statistics
            res += torch.linalg.norm(param.grad).square()
        return res.sqrt().item()

    def _grad(self, model, data, labels):
        """Calculate the gradients of the local update."""
        # 将模型设置为训练模式并清空梯度
        model.train()
        model.zero_grad()
        # 前向传播以获得预测
        pred = model(data)
        # 计算损失
        # default reduction == "mean"# 默认使用 "mean" 作为损失的归约方式
        loss = model.loss(pred, labels)
        loss.backward()  # compute the gradients of f_i(u_i)# 反向传播计算损失相对于模型参数的梯度

    def _admm_u(self, model, data, labels):
        """计算 ADMM 的 u 子问题的梯度。"""
        """Calculate the gradients of the ADMM u-subproblem."""
        # 计算本地模型的梯度，使用数据和标签
        self._grad(model, data, labels)  # get gradients
        # 获取当前客户端模型和服务器全局模型的参数状态
        u_state = model.state_dict()  # u_i# 本地客户端模型 u_i
        z_state = self.model_server.state_dict()  # z# 全局模型 z
        # 更新梯度以满足 ADMM 的 u 子问题
        for name, param in model.named_parameters():  # without BN statistics# 忽略 BN 统计量
            # param.grad -= self.lamda[self.i][name]
            param.grad += self.lamda[self.i][name] * self.alpha[self.i]
            # param.grad += self.beta[self.i][self.k] * (u_state[name] - z_state[name])
            param.grad += self.beta[self.i][self.k] * \
                (u_state[name] - z_state[name]) * self.alpha[self.i]

    def _init_params(self, cfg, model_server):
        """Initialize parameters and hyperparameters."""
        # 获取服务器模型的参数状态字典
        model_state = model_server.state_dict()
        # 复制服务器模型参数为每个客户端的初始模型
        self.models = [copy.deepcopy(model_state) for _ in range(cfg.m)]
        # 将要返回给服务器的结果self.send_back 是一个字典，用于存储客户端在训练过程中生成的结果，这些结果将被发送回服务器。它包括以下内容：
        #
        # 本地模型参数：每个客户端的当前模型参数。
        # 拉格朗日乘子：用于更新和维护约束条件的对偶变量。
        # 惩罚参数
        # 𝛽 用于调整模型一致性的参数。
        self.send_back = {}  # Results to send back to the server
        self.send_back["models"] = self.models  # local model parameters
        # to save local FGD/SGD steps# 保存本地FGD/SGD步数
        self.steps = np.full((cfg.m, cfg.K + 1), 0)
        # local model accuracy# 保存本地模型准确率
        self.accus = np.full((cfg.m, cfg.K + 1), 0, dtype=float)
        # 针对ADMM算法，初始化对偶变量和惩罚参数
        if self.alg in ["admm_insa", "admm_in", "admm"]:
            # dual variables lambda
            lamda = copy.deepcopy(model_state)  # 初始化拉格朗日乘子 lambda
            for key in model_state.keys():
                lamda[key].zero_()  # 将乘子初始化为零
            self.lamda = [copy.deepcopy(lamda)
                          for _ in range(cfg.m)]  # 为每个客户端复制拉格朗日乘子
            self.send_back["lambda"] = self.lamda
            # penalty parameter beta # 初始化惩罚参数 beta 在函数 _init_params 中，惩罚参数
            # 𝛽 被初始化为配置文件 cfg 中的指定值 cfg.beta。
            self.beta = np.full((cfg.m, cfg.K + 2), cfg.beta,
                                dtype=float)  # round k from 1 to K
            self.send_back["beta"] = [cfg.beta] * cfg.m


# auxiliary functions
def init_optimizer(cfg, model, k):
    """Initialize the optimizer."""
    # 计算学习率，考虑学习率衰减
    # 如果设置了学习率衰减因子 lr_decay，则根据当前训练轮次 k 计算学习率
    lr = cfg.lr * np.power(cfg.lr_decay, k - 1) if cfg.lr_decay else cfg.lr
    # 根据配置选择不同的优化器
    if cfg.optimizer == "sgd":
        # 初始化 SGD 优化器，使用学习率 lr、权重衰减系数 decay 和动量 momentum
        optimizer = torch.optim.SGD(
            model.parameters(), lr=lr, weight_decay=cfg.decay, momentum=cfg.momentum
        )
    # 初始化 Adam 优化器，使用学习率 lr
    elif cfg.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # 最后返回初始化后的优化器，用于后续的模型训练步骤。
    return optimizer


def init_data_loader(cfg, dataset_i, shuffle=True):
    """Initialize the dataloader."""
    # 如果配置中设置了批次大小 bs，则使用小批量随机梯度下降 (SGD)
    if cfg.bs:  # SGD
        data_loader = DataLoader(dataset_i, batch_size=cfg.bs, shuffle=shuffle)
    else:  # FGD
        # 使用 FGD (全数据梯度下降)
        # 如果 bs 为 0，则使用完整数据集，即全梯度下降（FGD）
        data_loader = DataLoader(
            dataset_i, batch_size=len(dataset_i), shuffle=shuffle)
    return data_loader


def l2_norm(x: nn.Module, y: nn.Module, square=False):
    """Calculate the l2 norm of x and y."""
    res = 0  # 初始化 L2 范数计算结果
    # 计算 L2 范数，在不进行梯度更新的情况下
    with torch.no_grad():
        # 遍历两个模型的参数
        # without BN statistics
        for x_item, y_item in zip(x.parameters(), y.parameters()):
            # 计算每个参数之间的差异，并累积平方值
            res += torch.linalg.norm(x_item - y_item).square()
        # 如果 square 为 False，则返回平方根（即 L2 范数）
        res = res if square else torch.sqrt(res)
        # 返回 L2 范数或其平方的值
    return res.item()


def state2model(cfg: dict, state: dict):
    """Create a model with given parameters."""
    # 初始化模型，根据给定配置 cfg
    model = init_model(cfg, True)
    # 加载参数字典到模型中，将 state 中的参数加载到新模型中
    model.load_state_dict(state)
    # 返回这个具有指定参数的模型
    return model
