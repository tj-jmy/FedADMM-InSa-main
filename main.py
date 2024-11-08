import torch

from fedlearn.federate import FL_train
from fedlearn.server import Server
from fedlearn.client import Clients
from rich.progress import Progress
from utils.dataset import init_dataset
from utils.args import args_parser
from utils.utils import load_config, set_seed


def main(cfg):
    # Setup the random seed.
    set_seed(cfg.seed)

    # Initialize dataset, server and clients.
    dataset = init_dataset(cfg)  # Initialize datasets.
    server = Server(cfg)  # Initialize the server.
    clients = Clients(cfg, server.model, dataset)  # Initialize clients.

    fh_hmul_pm = torch.zeros(cfg.m, 1, dtype=torch.complex64).to(cfg.device)

    fh_nul=[torch.zeros(p.shape, dtype=torch.complex64).to(cfg.device) for p in server.model.parameters()]

    print(fh_hmul_pm.shape)
    print(fh_nul[0].shape)

    server.load_noise_args(fh_hmul_pm, fh_nul)

    # Start FL training.
    FL_train(cfg, server, clients, dataset)


if __name__ == "__main__":
    # Load the configuration.
    cfg_file = "config.yaml"  #  "config.yaml"
    arg = args_parser(cfg_file=cfg_file)  # Load arguments from the command line.
    cfg = load_config(arg)  # Load the configuration from "./utils/config.yaml".

    with Progress() as progress:  # progress bar
        task = progress.add_task("[green]Main loop:", total=1)  # main loop bar
        cfg.progress = progress  # for the sub inner loop
        main(cfg)
        progress.update(task, advance=1)
