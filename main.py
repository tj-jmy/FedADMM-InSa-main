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

from scipy.io import loadmat


def main(cfg, alg, fh_hmul_pm, fh_nul):
    # Setup the random seed.
    set_seed(cfg.seed)

    # Initialize dataset, server and clients.
    dataset = init_dataset(cfg)  # Initialize datasets.
    server = Server(cfg, alg)  # Initialize the server.
    clients = Clients(cfg, server.model, dataset, alg)  # Initialize clients.

    server.load_noise_args(fh_hmul_pm, fh_nul)

    # Start FL training.
    FL_train(cfg, server, clients, dataset)


if __name__ == "__main__":
    # Load the configuration.
    cfg_file = "config.yaml"  # "config.yaml"
    # Load arguments from the command line.
    arg = args_parser(cfg_file=cfg_file)
    # Load the configuration from "./utils/config.yaml".
    cfg = load_config(arg)

    alg_list = ["fedavg", "admm_insa", "admm_in", "admm"]

    model = init_model(cfg)
    param_size = sum(p.numel() for p in model.parameters())
    print(f"Model size: {param_size}")  # 878538

    fh_hmul_pm_list = [torch.rand(cfg.m, 1, dtype=torch.complex64).to(cfg.device) for _ in range(len(alg_list))]
    fh_nul_list = [torch.rand(param_size, dtype=torch.complex64).to(cfg.device) for _ in range(len(alg_list))]

    fh_hmul_pm_list = loadmat('param.mat')['fh_hmul_pm']
    fh_nul_list = loadmat('param.mat')['fh_nul']

    with Progress() as progress:  # progress bar
        task = progress.add_task("[green]Main loop:", total=1)  # main loop bar
        cfg.progress = progress  # for the sub inner loop
        for i in range(len(alg_list)):
            main(cfg, alg_list[i], fh_hmul_pm_list[i], fh_nul_list[i])
        progress.update(task, advance=1)
