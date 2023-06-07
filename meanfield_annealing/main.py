import os
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from train import TrainMeanField


def start_run(device):
    os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = device
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.95"

    config = {
        "dataset_name": "RB_iid_100",
        "problem_name": "MaxCut",
        "jit": True,
        "wandb": True,

        "seed": 123,
        "epochs": 1000,
        "lr": 1e-4,
        "batch_size": 32, # H
        "N_basis_states": 30, # n_s

        "random_node_features": True,
        "n_random_node_features": 6,

        "annealing": True,
        "T_max": 0.05,
        "N_warmup": 0,
        "N_anneal": 2000,
        "N_equil": 0,

        "n_features_list_prob": [64, 64, 2],
        "n_features_list_nodes": [64, 64],
        "n_features_list_edges": [64, 64],
        "n_features_list_messages": [64, 64],
        "n_features_list_encode": [64],
        "n_features_list_decode": [64],
        "n_message_passes": 8,
        "message_passing_weight_tied": False,
        "linear_message_passing": True,
    }

    wandb_project = f"MeanField__{config['dataset_name']}_{config['problem_name']}"
    if config['T_max'] > 0.:
        wandb_group = f"{config['seed']}_LMP_T_{config['T_max']}_warmup_{config['N_warmup']}_anneal_{config['N_anneal']}_MPasses_{config['n_message_passes']}_n_rnf_{config['n_random_node_features']}"
    else:
        wandb_group = f"{config['seed']}_LMP_T_{config['T_max']}_anneal_{config['N_anneal']}_MPasses_{config['n_message_passes']}"

    wandb_run = f"lr_{config['lr']}_batchsize_{config['batch_size']}_basisstates_{config['N_basis_states']}_rnf_{config['random_node_features']}"

    train = TrainMeanField(config, wandb_project=wandb_project, wandb_group=wandb_group, wandb_run=wandb_run)
    print(f"\nGPU: {device}\nwandb PROJECT: {wandb_project}\nwandb GROUP: {wandb_group} \nwandb RUN: {wandb_run}\n")
    train.train()


if __name__ == '__main__':
    start_run(device='0')
