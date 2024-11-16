import pylab as p
import torch

from fedlearn.federate import FL_train
from fedlearn.server import Server
from fedlearn.client import Clients
from rich.progress import Progress
from utils.dataset import init_dataset
from utils.args import args_parser
from utils.utils import load_config, set_seed

from utils.model import init_model

import scipy
import numpy as np

from matplotlib import pyplot as plt

import os


def main(cfg, alg, fh_hmul_pm, fh_nul, times, alg_index):
    # Setup the random seed.
    set_seed(cfg.seed)

    for i in range(times):
        # Initialize dataset, server and clients.
        dataset = init_dataset(cfg)  # Initialize datasets.
        server = Server(cfg, alg)  # Initialize the server.
        clients = Clients(cfg, server.model, dataset, alg)  # Initialize clients.

        # reshape fh_nul
        fh_nul_reshape, offset = [], 0
        for param in server.model.parameters():
            fh_nul_reshape.append(fh_nul[offset:offset + param.numel()].reshape(param.shape))
            offset += param.numel()

        server.load_noise_args(fh_hmul_pm, fh_nul_reshape)

        # Start FL training.
        res_dict = FL_train(cfg, server, clients, dataset, add_noise=False)

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

    # alg_list = ["fedavg", "admm_insa", "admm_in", "admm"]
    alg_list = ["admm", "admm", "admm", "admm"]

    frac_list = [0.1, 0.2, 0.3, 0.4]

    model = init_model(cfg)
    param_size = sum(p.numel() for p in model.parameters())
    print(f"Model size: {param_size}")  # 878538

    fh_hmul_pm_list = [0.001 * torch.ones(cfg.m, 1, dtype=torch.complex64).to(cfg.device) for _ in range(len(alg_list))]
    fh_nul_list = [torch.zeros(param_size, dtype=torch.complex64).to(cfg.device) for _ in range(len(alg_list))]

    # fh_hmul_pm_list = loadmat('param.mat')['fh_hmul_pm']
    # fh_nul_list = loadmat('param.mat')['fh_nul']

    with Progress() as progress:  # progress bar
        task = progress.add_task("[green]Main loop:", total=1)  # main loop bar
        cfg.progress = progress  # for the sub inner loop
        for i in range(len(alg_list)):
            cfg.frac = frac_list[i]
            main(cfg, alg_list[i], fh_hmul_pm_list[i], fh_nul_list[i], 5, i)
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
