import pylab as p
import torch

from fedlearn.federate import FL_train
from fedlearn.server import Server
from fedlearn.client import Clients
from rich.progress import Progress
from utils.dataset import init_dataset, cal_alpha
from utils.args import args_parser
from utils.utils import load_config, set_seed
from scipy.io import loadmat
from utils.model import init_model

import scipy
import numpy as np

from matplotlib import pyplot as plt

import os
from gcn import gcniot


def main(cfg, alg, times, alg_index):
    # Setup the random seed.
    set_seed(cfg.seed)

    for i in range(times):
        # Initialize dataset, server and clients.
        dataset = init_dataset(cfg)  # Initialize datasets.
        server = Server(cfg, alg)  # Initialize the server.
        server.select_clients(cfg.frac)  # Randomly select clients.
        cfg.alpha = cal_alpha(cfg, dataset)  # Calculate the percentage of each client's dataset.
        clients = Clients(cfg, server.model, dataset, alg)  # Initialize clients.

        # Start FL training.
        res_dict = FL_train(cfg, server, clients, dataset, alg_index, add_noise=alg_index != 0)

        # Save the results.
        res_dir = "results/" + alg + str(alg_index)
        if not os.path.exists(res_dir):
            os.makedirs(res_dir)
        res_path = os.path.join(res_dir, "results_" + str(i) + ".mat")
        scipy.io.savemat(res_path, res_dict)

    # Average the results.
    loss = np.zeros((times, cfg.K + 1))
    accu = np.zeros((times, cfg.K + 1))

    for i in range(times):
        res_path = "results/" + alg + str(alg_index) + "/results_" + str(i) + ".mat"
        res_dict = scipy.io.loadmat(res_path)
        loss[i, :] = res_dict["loss"]
        accu[i, :] = res_dict["accu_test"]

    loss_mean = np.mean(loss, axis=0)
    accu_mean = np.mean(accu, axis=0)

    # Save the averaged results.
    res_path = "results/" + alg + str(alg_index) + "/results_avg.mat"
    scipy.io.savemat(res_path, {"loss": loss_mean, "accu_test": accu_mean})


if __name__ == "__main__":
    # Load the configuration.
    cfg_file = "config.yaml"  # "config.yaml"
    # Load arguments from the command line.
    arg = args_parser(cfg_file=cfg_file)
    # Load the configuration from "./utils/config.yaml".
    cfg = load_config(arg)
    print(cfg)
    # communication model
    # Some users are located in position I: -5 < x < 5, 45 < y < 50, z = 0
    ref1 = 1
    N0 = 1e-12
    m = int(cfg.m * cfg.frac)
    p2 = np.ones((m, 1)) * cfg.Pmax_up  # 0.1W
    dx = -5 + 10 * np.random.rand(1, m)
    dy = 50 - 5 * np.random.rand(1, m)
    dz = np.zeros((1, m))
    User_loc = np.vstack((dx, dy, dz)).T
    print('User_loc ', User_loc)
    cfg.L = cfg.M_v * cfg.M_h

    # # Generate uplink channel
    # cfg.G, cfg.h_r, cfg.h_d = gcniot(m, cfg.Na, cfg.M_h, cfg.M_v, User_loc, ref1)
    # # Generate downlink channel
    # cfg.G_down, cfg.h_r_down, cfg.h_d_down = gcniot(m, cfg.Na, cfg.M_h, cfg.M_v, User_loc, ref1)

    # 加载 channel1121.mat
    channel_data = loadmat('channel1203.mat')
    cfg.G = channel_data['G']
    cfg.h_r = channel_data['h_r']
    cfg.h_d = channel_data['h_d']

    cfg.G_down = channel_data['G_down']
    cfg.h_r_down = channel_data['h_r_down']
    cfg.h_d_down = channel_data['h_d_down']

    # # Generate uplink channel with matlab
    # import matlab.engine
    #
    # eng = matlab.engine.start_matlab()
    # User_loc_matlab = matlab.double(User_loc.tolist())
    # G, h_r, h_d = eng.gcniot(float(m), float(cfg.Na), float(cfg.M_h), float(cfg.M_v), User_loc_matlab, float(ref1), nargout=3)
    # G = np.array(G)
    # h_r = np.array(h_r)
    # h_d = np.array(h_d)
    #
    # # # 打印结果以确认
    # # print("G:", G)
    # # print("h_r:", h_r)
    # # print("h_d:", h_d)
    #
    # # Generate downlink channel
    #
    # # 调用 MATLAB 函数 gcniot 生成下行信道
    # G_down, h_r_down, h_d_down = eng.gcniot(float(m), float(cfg.Na), float(cfg.M_h), float(cfg.M_v), User_loc_matlab, float(ref1),
    #                                         nargout=3)
    #
    # # 将 MATLAB 返回的数据转换为 NumPy 数组
    # G_down = np.array(G_down)
    # h_r_down = np.array(h_r_down)
    # h_d_down = np.array(h_d_down)

    # # 打印下行信道结果以确认
    # print("Downlink G_down:", G_down)
    # print("Downlink h_r_down:", h_r_down)
    # print("Downlink h_d_down:", h_d_down)

    ini_para_data = loadmat('ini_para2.mat')
    theta = ini_para_data['thetaini'].flatten()
    # theta = np.exp(1j * np.random.rand(cfg.L) * 2 * np.pi)
    cfg.thetaini = theta

    h = np.zeros((cfg.Na, m), dtype=complex)
    snr = np.zeros(m)
    snr_DB = np.zeros(m)
    for i in range(m):
        h[:, i] = cfg.G @ np.diag(cfg.h_r[:, i]) @ theta + cfg.h_d[:, i]  # Cascaded link
        snr[i] = (p2[i] * (np.linalg.norm(h[:, i]) ** 2)) / N0
        snr_DB[i] = 10 * np.log10(snr[i])

    # Direct link SNR
    hd = np.zeros((cfg.Na, m), dtype=complex)
    snrd = np.zeros(m)
    snr_DBd = np.zeros(m)

    for i in range(m):
        hd[:, i] = cfg.h_d[:, i]  # Direct link
        snrd[i] = (p2[i] * (np.linalg.norm(hd[:, i]) ** 2)) / N0
        snr_DBd[i] = 10 * np.log10(snrd[i])

    # Mean SNR in dB
    mean_snr_DB = np.mean(snr_DB)
    mean_snr_DBd = np.mean(snr_DBd)

    print("Mean SNR (dB) with RIS:", mean_snr_DB)
    print("Mean SNR (dB) without RIS:", mean_snr_DBd)

    p = ini_para_data['pini']
    cfg.pini = p
    w_b = ini_para_data['wini']
    cfg.wini = w_b
    # import matlab.engine
    # eng = matlab.engine.start_matlab()
    # eng.cvx_setup(nargout=0)  # 运行 cvx_setup
    # f = np.exp(1j * np.random.rand(cfg.Na, 1) * 2 * np.pi)
    # f_matlab = matlab.double(f.tolist(), is_complex=True)
    # h_d_matlab = matlab.double(hd.tolist(), is_complex=True)
    # p_matlab = eng.find_inip(h_d_matlab, f_matlab, float(m), float(cfg.Pmax_up), float(N0))
    # cfg.pini = np.array(p_matlab).flatten()
    # w_b = np.exp(1j * np.random.rand(cfg.Na, 1) * 2 * np.pi)  # Initialize w_b
    # cfg.wini = np.sqrt(cfg.Pmax_down) * w_b
    # 打印结果以确认
    # print("Initial p:", p)

    # alg_list = ["fedavg", "admm_insa", "admm_in", "admm"]
    alg_list = ["admm", "admm", "admm", "admm"]

    # frac_list = [0.1, 0.2, 0.3, 0.4]

    model = init_model(cfg)
    cfg.param_size = sum(p.numel() for p in model.parameters())
    print(f"Model size: {cfg.param_size}")  # 878538

    # real_part = torch.randn(cfg.Na, cfg.param_size, dtype=torch.float32)  # 实部噪声
    # imag_part = torch.randn(cfg.Na, cfg.param_size, dtype=torch.float32)  # 虚部噪声
    #
    # # 组合成复数矩阵
    # cfg.n_up = torch.sqrt(torch.tensor(N0 / 2)) * (real_part + 1j * imag_part)

    # 生成实部和虚部的随机噪声，维度为 (cfg.Na,)
    real_part = torch.randn(cfg.Na, dtype=torch.float32)  # 实部噪声
    imag_part = torch.randn(cfg.Na, dtype=torch.float32)  # 虚部噪声

    # 组合成复数向量 (cfg.Na,)
    first_column = torch.sqrt(torch.tensor(N0 / 2)) * (real_part + 1j * imag_part)

    # 将第一列扩展到 (cfg.Na, cfg.param_size)，使每一列都相同
    cfg.n_up = first_column.unsqueeze(1).expand(-1, cfg.param_size)

    # 打印 cfg.n_up 确认
    # print("cfg.n_up:", cfg.n_up)

    # fh_hmul_pm_list = loadmat('param.mat')['fh_hmul_pm']
    # fh_nul_list = loadmat('param.mat')['fh_nul']

    with Progress() as progress:  # progress bar
        task = progress.add_task("[green]Main loop:", total=1)  # main loop bar
        cfg.progress = progress  # for the sub inner loop
        for i in range(1, len(alg_list)):
            # cfg.frac = frac_list[i]
            main(cfg, alg_list[i], 5, i)
        progress.update(task, advance=1)

    loss = np.zeros((len(alg_list), cfg.K + 1))
    accu = np.zeros((len(alg_list), cfg.K + 1))

    for i in range(len(alg_list)):
        res_path = "results/" + alg_list[i] + str(i) + "/results_avg.mat"
        res_dict = scipy.io.loadmat(res_path)
        loss[i, :] = res_dict["loss"]
        accu[i, :] = res_dict["accu_test"]

    fig1, ax1 = plt.subplots()  # loss图表
    fig2, ax2 = plt.subplots()  # accu图表

    linestyles = ['-', '--', '-.', ':']

    min1 = loss.min() - 0.1
    max1 = loss.max() + 0.1

    for i in range(len(alg_list)):
        ax1.plot(loss[i, :], label=alg_list[i], linestyle=linestyles[i])
        ax1.set_ylim(min1, max1)
        ax1.legend(loc='upper right', prop={'size': 10})

    ax1.set_title('Loss')
    ax1.set_xlabel('Communication rounds')
    ax1.set_ylabel('Loss')
    ax1.figure.savefig("results/loss.png")

    min2 = accu.min() - 0.05
    max2 = accu.max() + 0.05

    for i in range(len(alg_list)):
        ax2.plot(accu[i, :], label=alg_list[i], linestyle=linestyles[i])
        ax2.set_ylim(min2, max2)
        ax2.legend(loc='lower right', prop={'size': 10})

    ax2.set_title('Accuracy')
    ax2.set_xlabel('Communication rounds')
    ax2.set_ylabel('Accuracy')
    ax2.figure.savefig("results/accuracy.png")
