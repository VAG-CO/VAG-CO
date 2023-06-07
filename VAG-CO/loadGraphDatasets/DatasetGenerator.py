import jax.random
import torch
from torch.utils.data import DataLoader
import os
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader as pygDataLoader
import numpy as np
from matplotlib import pyplot as plt
import jax.numpy as jnp
import jraph
from unipath import Path
from omegaconf import OmegaConf
from jraph_utils import utils as jutils
from loadGraphDatasets.jraph_Dataloader import JraphSolutionDataset, RRGDataset
from utils import string_utils
from loadGraphDatasets.loadTwitterGraph import TWITTER


gpu_np = jnp
cpu_np = np

def get_num_nodes_v1(pyg_graph):
    num_nodes = pyg_graph.x.shape[0]
    return num_nodes

def get_num_nodes_v2(pyg_graph):

    num_nodes = pyg_graph.num_nodes
    return num_nodes

class Generator:

    def __init__(self, cfg, num_workers = 0, shuffle_seed = None):
        self.num_workers = num_workers
        ### TODO try to set num workers
        self.cfg = cfg
        self.batch_size = cfg["Train_params"]["H_batch_size"]
        self.Nb = cfg["Train_params"]["n_basis_states"]
        self.time_horizon = cfg["Train_params"]["PPO"]["time_horizon"]
        self.random_node_features = cfg["Ising_params"]["n_rand_nodes"]

        self.test_batch_size = cfg["Test_params"]["n_test_graphs"]

        self.dataset_name = cfg["Ising_params"]["IsingMode"]

        self.shuffle_seed = cfg["Ising_params"]["shuffle_seed"]
        self.graph_padding_factor = cfg["Ising_params"]["graph_padding_factor"]

        if(self.dataset_name == "COLLAB" or self.dataset_name == "IMDB-BINARY"):
            self.get_num_nodes_fuc = get_num_nodes_v2
        else:
            self.get_num_nodes_fuc = get_num_nodes_v1

        self.dataset_names = ["ENZYMES", "PROTEINS", "MUTAG", "COLLAB", "IMDB-BINARY"]

        ### TODO implement some exceptions for COLLAB ind IMDB dataset because they have empty node attributes
        if(self.dataset_name in self.dataset_names):
            pass
        else:
            ValueError("This dataset does not exist")

        self.collate_from_torch_to_jraph_fn = lambda data: self.collate_from_torch_to_jraph(data, add_padded_node=True,
                                                                                  time_horizon=self.time_horizon)

        self.path = cfg["Paths"]["work_path"]

        if("RRG" not in self.dataset_name):
            if(self.dataset_name != "TWITTER"):
                self.init_TUDataset()
            else:
                self.init_TWITTERDataset()
        else:
            self.init_RRGDataset()

        #print("train dataset length ", "train", len(self.pyg_train_dataset), "val", len(self.pyg_val_dataset), "test",len(self.pyg_test_dataset))
        #### TODO sort test and val by size so that the code will run faster
        # self.jraph_test_loader = DataLoader(self.test_dataset, batch_size=self.test_batch_size, shuffle=False, collate_fn=self.collate_from_torch_to_jraph_val_and_test, num_workers=0)
        # self.jraph_val_loader = DataLoader(self.val_dataset, batch_size=self.test_batch_size, shuffle=False, collate_fn=self.collate_from_torch_to_jraph_val_and_test, num_workers=0)
        try:
            collate_data = lambda data: jutils.collate_jraphs_to_max_size(data, self.random_node_features)
            self.val_dataset = JraphSolutionDataset(cfg, mode = "val", seed=self.shuffle_seed)
            self.jraph_val_loader = DataLoader(self.val_dataset, batch_size=self.test_batch_size,
                                               collate_fn=collate_data, num_workers=self.num_workers)


            self.test_dataset = JraphSolutionDataset(cfg, mode="test", seed=self.shuffle_seed)
            self.jraph_test_loader = DataLoader(self.test_dataset, batch_size=self.test_batch_size,
                                                collate_fn=collate_data, num_workers=self.num_workers)
        except:
            UserWarning("Dataset has not been solved yet")

        self.Nb_spins_list = [cpu_np.zeros((self.Nb, graph.nodes.shape[0])) for graph in self.graph_list]
        self.Nb_spins = cpu_np.concatenate(self.Nb_spins_list, axis = 1)

        self.max_env_steps = np.array(self.batched_graphs.n_node)
        self.curr_env_steps = -1*np.ones((self.batch_size,), dtype = cpu_np.int32)

        self.finished_Energies = []
        self.tracked_Entropies = []

    def init_TUDataset(self):
        self.dataset = TUDataset(root=f'{os.getcwd()}/loadGraphDatasets/tmp/{self.dataset_name}', name=self.dataset_name)

        if(self.dataset_name == "COLLAB" and self.cfg["Ising_params"]["EnergyFunction"] == "MIS"):
            self.dataset = self.dataset[0:1000]
            full_dataset_len = 1000
        else:
            full_dataset_len = len(self.dataset)

        print("dataset name", self.dataset_name, full_dataset_len)

        full_dataset_arganged = np.arange(0, full_dataset_len)
        ### TODO shuffle with jax here to make it deterministic by seed

        if(self.shuffle_seed != -1):
            np.random.seed(self.shuffle_seed)
            np.random.shuffle(full_dataset_arganged)

        if(self.cfg["Ising_params"]["EnergyFunction"] == "MIS"):
            ts = 0.6
            vs = 0.1
        else:
            ts = 0.7
            vs = 0.1

        train_dataset_len = int(ts*full_dataset_len)
        val_dataset_len = int(vs*full_dataset_len)
        test_dataset_len = full_dataset_len - train_dataset_len - val_dataset_len
        self.train_dataset_idxs = full_dataset_arganged[0:train_dataset_len]
        self.val_dataset_idxs = full_dataset_arganged[train_dataset_len:train_dataset_len+val_dataset_len]
        self.test_dataset_idxs = full_dataset_arganged[train_dataset_len+val_dataset_len:]

        self.pyg_train_dataset = self.dataset[self.train_dataset_idxs]
        self.pyg_val_dataset = self.dataset[self.val_dataset_idxs]
        self.pyg_test_dataset = self.dataset[self.test_dataset_idxs]

        node_list = []
        for data in iter(pygDataLoader(self.pyg_train_dataset, batch_size=1, shuffle=True)):
            num_nodes = self.get_num_nodes_fuc(data)
            node_list.append(num_nodes)

        node_arr = np.array(node_list)
        self.normalisation_factor = float(np.mean(node_arr))
        #print("average number of nodes is", self.normalisation_factor)
        self.reset_collate_func = self.collate_from_torch_to_jraph_fn
        self.pyg_loader = iter(DataLoader(self.pyg_train_dataset, batch_size=1, shuffle=True, collate_fn=self.collate_from_torch_to_jraph_fn))

        self.jraph_dataloader = DataLoader(self.pyg_train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=self.collate_from_torch_to_jraph_fn, num_workers=self.num_workers)
        self.jraph_loader = iter(self.jraph_dataloader )

        self.batched_graphs, self.graph_list = next(self.jraph_loader)

    def init_RRGDataset(self):

        ### TODO add normalisation factor
        ### TODO add random node embeddings
        integers = string_utils.separate_integers(self.dataset_name)
        num_nodes = integers[0]
        k = integers[1]
        self.normalisation_factor = num_nodes

        self.pyg_train_dataset = RRGDataset(self.cfg)

        self.reset_collate_func = lambda data: jutils.collate_jraphs_to_horizon(data, self.time_horizon, self.random_node_features)
        self.pyg_loader = iter(DataLoader(self.pyg_train_dataset, batch_size=1, shuffle=True, collate_fn=self.reset_collate_func))


        self.jraph_dataloader = DataLoader(self.pyg_train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=self.reset_collate_func, num_workers=self.num_workers)
        self.jraph_loader = iter(self.jraph_dataloader )

        self.batched_graphs, self.graph_list = next(self.jraph_loader)

    def init_TWITTERDataset(self):
        self.reset_collate_func = lambda data: jutils.collate_jraphs_to_horizon(data, self.time_horizon, self.random_node_features)
        self.pyg_train_dataset = TWITTER(self.cfg, mode = self.mode)
        self.pyg_loader = iter(self.pyg_train_dataset)

        self.jraph_dataloader = DataLoader(self.pyg_train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=self.reset_collate_func, num_workers=self.num_workers)
        self.jraph_loader = iter(self.jraph_dataloader )

        self.batched_graphs, self.graph_list = next(self.jraph_loader)

    def update_metrics(self, Nb_Hb_Energy):
        self.curr_Energy = Nb_Hb_Energy

    def update_env(self, Nb_Hb_bins, Nb_curr_env_steps):
        self.Nb_spins = cpu_np.where(Nb_Hb_bins == 3, cpu_np.zeros_like(Nb_Hb_bins), 2 * Nb_Hb_bins - 1)
        self.curr_env_steps = np.array(Nb_curr_env_steps[0, :, -1])

    def reset_iterator(self):
        self.pyg_loader = iter(DataLoader(self.pyg_train_dataset, batch_size=1, shuffle=True, collate_fn=self.reset_collate_func))

    def _nearest_bigger_power_of_k(self, x: int, k: float) -> int:
        """Computes the nearest power of two greater than x for padding."""
        exponent = np.log(x) / np.log(k)

        return int(k**(int(exponent) + 1))

    ### TODO add paddint to nearest power of k
    def pad_graph_to_nearest_power_of_k(self,
                                        graphs_tuple: jraph.GraphsTuple, np_ = jnp) -> jraph.GraphsTuple:
        ### TODO track padding
        # Add 1 since we need at least one padding node for pad_with_graphs.
        pad_nodes_to = self._nearest_bigger_power_of_k(np_.sum(graphs_tuple.n_node), self.graph_padding_factor ) + 1
        pad_edges_to = self._nearest_bigger_power_of_k(np_.sum(graphs_tuple.n_edge), self.graph_padding_factor )
        # Add 1 since we need at least one padding graph for pad_with_graphs.
        # We do not pad to nearest power of two because the batch size is fixed.
        pad_graphs_to = graphs_tuple.n_node.shape[0] + 1
        return jraph.pad_with_graphs(graphs_tuple, pad_nodes_to, pad_edges_to,
                                     pad_graphs_to)



    def update_graphs(self):
        self.max_env_steps = np.array(self.batched_graphs.n_node) ### TODO update this in overwrite terminated graphs
        dones = self.curr_env_steps >= self.max_env_steps - 1

        self.batched_graphs, self.graph_list = self.overwrite_termianted_graphs(self.graph_list, dones,
                                                                      add_padded_node=True)
        self.curr_env_steps += 1
        self.curr_env_steps = np.where(dones == True, np.zeros_like(self.curr_env_steps), self.curr_env_steps)

        self.padded_batched_graphs = self.pad_graph_to_nearest_power_of_k(self.batched_graphs)

        #### TODO test if padding od spins is neccesary here
        num_spins_padded  = self.padded_batched_graphs.nodes.shape[0]
        self.padded_Nb_spins = cpu_np.zeros((self.Nb,num_spins_padded))
        #self.padded_Nb_spins = self.padded_Nb_spins.at[:,0:self.Nb_spins.shape[1]].set(self.Nb_spins)
        self.padded_Nb_spins[:,0:self.Nb_spins.shape[1]] = self.Nb_spins
        return self.batched_graphs, self.padded_batched_graphs, self.graph_list, self.curr_env_steps, self.padded_Nb_spins

    def overwrite_termianted_graphs(self, graph_list, dones, **kargs):
        ### this seems to be slow
        n_node = self.batched_graphs.n_node

        g_idx = np.arange(0, len(graph_list))[np.array(dones)]

        for idx in g_idx:
            self.finished_Energies.append(self.curr_Energy[:,idx])
            print("replace graph number", idx)
            try:
                (new_graph, _) = next(self.pyg_loader)
            except StopIteration:
                self.reset_iterator()
                (new_graph, _) = next(self.pyg_loader)

            graph_list[idx] = new_graph

            self.Nb_spins = overwrite_batched_spins(n_node, new_graph, self.Nb_spins, idx)
            #n_node = n_node.at[idx].set(new_graph.nodes.shape[0])
            n_node[idx] = new_graph.nodes.shape[0]

        if(len(g_idx) > 0):
            self.batched_graphs = jraph.batch_np(graph_list)


        return self.batched_graphs, graph_list

    def collate_from_torch_to_jraph(self, datas, **kargs):
        jdata_list = [self.from_pyg_to_jraph(data, **kargs) for data in datas]
        batched_jdata = jraph.batch_np(jdata_list)
        return (batched_jdata, jdata_list)

    def collate_from_torch_to_jraph_val_and_test(self, datas):

        num_nodes = max([self.get_num_nodes_fuc(data) for data in datas])

        jdata_list = [self.from_pyg_to_jraph(data, add_padded_node=True, time_horizon= num_nodes) for data in datas]
        batched_jdata = jraph.batch_np(jdata_list)
        return (batched_jdata, jdata_list)

    def from_pyg_to_jraph(self, pyg_graph, add_padded_node=False, time_horizon=0):
        if (not add_padded_node):
            num_nodes = self.get_num_nodes_fuc(pyg_graph)
        else:
            num_nodes = self.get_num_nodes_fuc(pyg_graph)
            rest = num_nodes % max([time_horizon, 1])
            if(rest != 0):
                additional_nodes = time_horizon - rest
                num_nodes = num_nodes + additional_nodes
            else:
                additional_nodes = 0

        num_edges = pyg_graph.edge_index.shape[1]

        nodes = cpu_np.zeros((num_nodes, 1), dtype=cpu_np.float32)

        if(self.random_node_features > 0):
            random_node_features = np.random.uniform(size = (nodes.shape[0], self.random_node_features))
            nodes = np.concatenate([nodes, random_node_features], axis = -1)
        #mask = nodes.at[:num_nodes-additional_nodes].set(jnp.ones((num_nodes-additional_nodes,1)))
        mask = np.zeros((num_nodes, 1))
        mask[:num_nodes-additional_nodes] = cpu_np.ones((num_nodes-additional_nodes,1))

        mask_and_nodes = cpu_np.concatenate([nodes, mask], axis = -1)

        unshuffled_senders = cpu_np.array(pyg_graph.edge_index[0, :])
        unshuffled_receivers = cpu_np.array(pyg_graph.edge_index[1, :])

        senders, receivers = self.permute_graph(num_nodes - additional_nodes, unshuffled_senders, unshuffled_receivers)

        edges = cpu_np.ones((num_edges, 1), dtype=cpu_np.float32) / self.normalisation_factor
        n_node = cpu_np.array([num_nodes])
        n_edge = cpu_np.array([num_edges])

        jgraph = jraph.GraphsTuple(nodes=mask_and_nodes, senders=senders, receivers=receivers,
                                   edges=edges, n_node=n_node, n_edge=n_edge, globals= cpu_np.zeros((1,)))
        return jgraph

    def permute_graph(self, num_nodes, senders, receivers):

        aranged_indeces = cpu_np.arange(0, num_nodes)

        # self.shuffle_key, subkey = jax.random.split(self.shuffle_key)
        # shuffled_idxs = jax.random.shuffle(subkey, aranged_indeces)

        cpu_np.random.shuffle(aranged_indeces)

        shuffled_senders = aranged_indeces[senders]
        shuffled_receivers = aranged_indeces[receivers]
        return shuffled_senders, shuffled_receivers


### TODO test if correct
def overwrite_batched_spins(n_node, new_graph, Nb_spins, idx):

    new_spins = cpu_np.zeros((Nb_spins.shape[0], new_graph.nodes.shape[0]))
    node_idx = cpu_np.concatenate([cpu_np.array([0]),n_node], axis = -1)

    global_node_idx = jax.lax.cumsum(node_idx)
    if(idx == 0):
        end_idx = global_node_idx[idx + 1]
        next_spins = Nb_spins[:, end_idx:]
        Nb_spins = cpu_np.concatenate([new_spins, next_spins], axis=1)
    elif(idx <= len(n_node)):
        start_idx = global_node_idx[idx]
        end_idx = global_node_idx[idx + 1]
        prev_spins = Nb_spins[:, 0:start_idx]
        next_spins = Nb_spins[:, end_idx:]
        Nb_spins = cpu_np.concatenate([prev_spins, new_spins, next_spins], axis=1)
        if(idx == len(n_node)-1):
            print("here")
            pass

        # if(idx == len(n_node) - 1):
        #     print(idx)

    return Nb_spins


if(__name__ == "__main__"):

    pass
    # p = Path( os.getcwd())
    # path = p.parent
    # print(path)
    # cfg = OmegaConf.load(path + "/Experiment_configs/HydraBaseConfig.yaml")
    # cfg["Ising_params"]["IsingMode"] = "PROTEINS"
    # gen = Generator(cfg)
