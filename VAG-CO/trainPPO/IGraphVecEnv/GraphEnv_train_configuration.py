import gym
import numpy as np
from numpy import random
from trainPPO.OwnVecEnv import SubprocVecEnv
import time
import igraph as ig
import jax
import jraph
import os
import random
import copy
import numba
from loadGraphDatasets import GetDataLoaders
from numba import guvectorize, float64, int64
from trainPPO.IGraphVecEnv import DataContainer
from jraph_utils import utils as jutils


class IGraphEnv(gym.Env):
    ### use jax random keys?
    """Custom Environment that follows gym interface"""
    metadata = {"render.modes": ["human", "rgb_array"], "render_fps": 4}

    ### TODO since always the first node is selected actually one igraph is enough! it will make the code much faster!
    def __init__(self,cfg, H_seed, mode = "train"):
        super(IGraphEnv, self).__init__()
        random.seed(H_seed)
        self.cfg = cfg
        self.node_embedding_type = cfg["Ising_params"]["node_embedding_type"]
        self.edge_embedding_type = cfg["Ising_params"]["edge_embedding_type"]
        self.H_seed = H_seed
        self.mode = mode
        self.Nb = cfg["Train_params"]["n_basis_states"]
        self.EnergyFunction = cfg["Ising_params"]["EnergyFunction"]
        self.dataset_name = cfg["Ising_params"]["IsingMode"]
        self.ordering = cfg["Ising_params"]["ordering"]
        self.n_rand_nodes = cfg["Ising_params"]["n_rand_nodes"]

        self.gamma = 1.
        self.lam = cfg["Train_params"]["PPO"]["lam"]

        self.time_horizon = cfg["Train_params"]["PPO"]["time_horizon"]

        ### TODO align code with ReplayBuffer_vectorised
        print("init spin env", mode)
        start = time.time()
        Dataset_dict = GetDataLoaders.init_Dataset(cfg, H_idx = H_seed, mode = mode)
        end = time.time()
        print("init time needed", end-start)

        self.num_graphs = Dataset_dict["num_graphs"]
        print("num_Graphs =", self.num_graphs)

        self.next_graph_func = Dataset_dict["loader_func"]
        self.loader = Dataset_dict["loader"]
        self.finished = False
        self.global_reset = False

        self.order_func = self.init_graph_ordering_func()

        self.mean_energy = Dataset_dict["mean_energy"]
        self.std_energy = Dataset_dict["std_energy"]

        self.policy_global_features = cfg["Network_params"]["policy_MLP_features"]
        self.n_classes = self.policy_global_features[-1]
        self.n_sampled_sites = int(np.log2(self.n_classes))
        self.sampling_mode = "normal"


    def normalize_Energy(self, Energy_arr):
        return (Energy_arr-self.mean_energy)/self.std_energy

    def compute_unormalized_Energy(self, Energy_arr):
        return (Energy_arr * self.std_energy + self.mean_energy)

    def convert_to_jraph(self, igraph):

        couplings = np.array(igraph.es["couplings"])
        edge_list = igraph.get_edgelist()
        edge_arr = np.array(edge_list)

        #print("ecount", igraph.ecount())
        #print(edge_list)
        if (igraph.ecount() > 0):
            receivers = edge_arr[:, 0]
            senders = edge_arr[:, 1]
            edges = couplings
        else:
            receivers = np.zeros((0,), dtype=np.int64)
            senders = receivers
            edges = np.ones((0, 1))
            edges = edges

        N = igraph.vcount()

        node_features = np.array(igraph.vs["node_features"])

        jgraph = jraph.GraphsTuple(nodes=node_features, edges=edges, receivers=receivers,
                                              senders=senders,
                                              n_node= np.array([N]),
                                              n_edge=np.array([senders.shape[0]]), globals= np.array([N]))

        return jgraph

    def _add_empty_nodes(self, Igraph, orig_Igraph, gs):
        ### todo add node, external fields
        num_nodes = Igraph.vcount()
        mod = num_nodes%self.n_sampled_sites
        new_gs = gs
        if(mod != 0):
            additional_nodes = self.n_sampled_sites - mod
            self.additional_nodes = additional_nodes
            ### TODO add all atributes
            external_fields = Igraph.vs["ext_fields"]
            external_fields.extend([np.array([0.]) for i in range(additional_nodes)])
            Igraph.add_vertices(additional_nodes)
            Igraph.vs["ext_fields"] = external_fields

            if(orig_Igraph != None):
                orig_Igraph.add_vertices(additional_nodes)

            new_gs = np.ones((num_nodes+additional_nodes,1))
            new_gs[:num_nodes,0] = gs
        #print("ext field output", np.array(Igraph.vs["ext_fields"]).shape)

        return Igraph, orig_Igraph, new_gs

    def init_graph_func(self):
        i_graph, orig_igraph, self.gs, self.finished, self.loader = self.next_graph_func(self.global_reset, self.finished, self.loader)
        self.orig_n_nodes = i_graph.vcount()
        self.global_reset = False
        i_graph, orig_igraph, self.gs = self._add_empty_nodes(i_graph, orig_igraph, self.gs)
        self.N_spin = i_graph.vcount()
        return i_graph, orig_igraph

    def shuffle_graph(self, Igraph):
        perm = np.arange(0, Igraph.vcount())
        np.random.shuffle(perm)
        Igraph = Igraph.permute_vertices(list(perm))
        return Igraph, perm

    def bfs_order_graph(self, Igraph):
        Igraph.vs["index"] = np.arange(0,  Igraph.vcount())

        ### TODO padded nodes are apparently always first, maybe make them last
        ### TODO randomize the order of disjoint graphs
        disjoint_graphs = Igraph.decompose()
        ordered_subgraphs = []
        for subgraph in disjoint_graphs:
            vc = subgraph.vcount()
            idx = np.random.randint(0, high=vc)
            res = subgraph.bfs(idx)
            order = res[0]
            subgraph = subgraph.permute_vertices(list(order))
            ordered_subgraphs.append(subgraph)

        ordered_IGraph = ig.disjoint_union(ordered_subgraphs)
        return ordered_IGraph, ordered_IGraph.vs["index"]

    def dfs_order_graph(self, Igraph):
        Igraph.vs["index"] = np.arange(0,  Igraph.vcount())
        disjoint_graphs = Igraph.decompose()
        ordered_subgraphs = []
        for subgraph in disjoint_graphs:
            vc = subgraph.vcount()
            idx = np.random.randint(0, high=vc)
            res = subgraph.dfs(idx)
            order = res[0]
            subgraph = subgraph.permute_vertices(list(order))
            ordered_subgraphs.append(subgraph)

        ordered_IGraph = ig.disjoint_union(ordered_subgraphs)

        return ordered_IGraph, ordered_IGraph.vs["index"]

    def init_graph_ordering_func(self):
        if(self.ordering == "None" ):
            func = lambda x: self.shuffle_graph(x)
        elif(self.ordering == "BFS"):
            func = lambda x: self.bfs_order_graph(x)
        elif(self.ordering == "DFS"):
            func = lambda x: self.dfs_order_graph(x)
        else:
            ValueError("Non Valid graph ordering function")
        return func

    def from_Hb_Igraph_to_Nb_Hb_Jgraph(self, Igraph):
        Igraph, self.perm = self.order_func(Igraph)

        if(self.node_embedding_type == "random"):
            Igraph.vs["node_features"] = np.random.normal(0,1, size = (Igraph.vcount(), self.n_rand_nodes))
        else:
            Igraph.vs["node_features"] = np.zeros((Igraph.vcount(), 1))

        self.EnergyIgraph = Igraph
        ext_fields = np.array(Igraph.vs["ext_fields"])
        Nb_ext_fields = np.repeat(ext_fields[np.newaxis,:], self.Nb, axis = 0)

        self.N_spin_arr = self.N_spin*np.ones((self.Nb, ), dtype=np.int64)
        Jgraph = self.convert_to_jraph(self.EnergyIgraph)

        return Jgraph, Nb_ext_fields

    def reset(self):
        self.dEnergies = []
        self.env_step = 0
        self.time_horizon_step = -1

        Igraph, self.orig_Igraph = self.init_graph_func()
        self.num_edges = Igraph.ecount()
        self.gt_Energy = Igraph["gt_Energy"]
        self.EnergyJgraph, self.Nb_external_fields = self.from_Hb_Igraph_to_Nb_Hb_Jgraph(Igraph)

        self.Jgraph, self.Igraph = self._get_compl_graph(self.EnergyJgraph, self.EnergyIgraph)
        self.return_Jgraph = self.Jgraph

        self.init_EnergyJgraph = copy.deepcopy(self.EnergyJgraph)
        self.init_Nb_external_fields = copy.copy(self.Nb_external_fields)

        self.Nb_spins = np.zeros((self.Nb, self.N_spin, 1))

        self.DataContainer = DataContainer.DataContainer(self.cfg, self.time_horizon, self.Nb, self.N_spin, lam = self.lam, n_sampled_spins=self.n_sampled_sites)

        return {"H_graph": {"Nb_ext_fields": self.Nb_external_fields, "jgraph": self.return_Jgraph}}


    def compute_laplace_embeding_on_edges(self, eigenvectors, H_graph):
        ev_dist = np.abs((eigenvectors[H_graph.receivers, :]- eigenvectors[H_graph.senders, :]))
        ev_mul = eigenvectors[H_graph.receivers, :]*eigenvectors[H_graph.senders, :]
        edge_features = np.concatenate([ev_dist, ev_mul], axis = -1)
        return edge_features

    def compute_Energy_cropped_graph_np(self, Nb_external_fields, Nb_spins, self_senders, self_receivers, self_loop_weight):
        Nb_ext_fields_edges = 0.5*Nb_spins[:,self_senders]*Nb_spins[:,self_receivers] * self_loop_weight[np.newaxis,:]
        Nb_energy_from_self_loops = np.sum(Nb_ext_fields_edges, axis = -2)

        Nb_Energy_per_node = Nb_external_fields * Nb_spins
        Nb_Energy = np.sum(Nb_Energy_per_node,axis = -2) + Nb_energy_from_self_loops

        return Nb_Energy


    def delete_node_and_edges(self, spin_configuration):
        #start_overall = time.time()
        #start_node_idxs = time.time()
        site_indices = np.arange(0, self.n_sampled_sites)
        Nb_spins = np.zeros((spin_configuration.shape[0], self.EnergyJgraph.nodes.shape[0]))
        Nb_spins[:, 0:self.n_sampled_sites] = spin_configuration
        Nb_spins = np.expand_dims(Nb_spins, axis = -1)

        ### TODO
        new_senders = self.EnergyJgraph.senders
        new_receivers = self.EnergyJgraph.receivers
        couplings = self.EnergyJgraph.edges

        #start_energy_compute = time.time()
        dEnergy = self.compute_Energy_cropped_graph_np(self.Nb_external_fields, Nb_spins, new_senders, new_receivers, couplings)
        dEnergy = np.squeeze(dEnergy, axis=-1)
        #end_energy_compute = time.time()

        ### TODO add flag where no nodes are deleted and spins are added to ext fields
        self.EnergyIgraph.delete_vertices(site_indices)
        self.Igraph.delete_vertices(site_indices)
        #end_spin_delete = time.time()

        #start_ext_delete = time.time()
        self.Nb_external_fields = self.update_external_field(new_senders, new_receivers, couplings, self.Nb_external_fields, Nb_spins, site_indices)
        #end_ext_delete = time.time()

        self.N_spin_arr -= 1

        self.EnergyJgraph = self.convert_to_jraph(self.EnergyIgraph)
        self.Jgraph = self.convert_to_jraph(self.Igraph)
        #end_overall_time = time.time()

        # print("overall_time", end_overall_time - start_overall, self.num_edges, self.N_spin)
        # print("spin_delete", end_spin_delete-start_spin_delete, self.num_edges, self.N_spin)
        # print("energy", end_energy_compute-start_energy_compute, self.num_edges, self.N_spin)
        #print("ext delete", end_ext_delete-start_ext_delete, self.num_edges, self.N_spin)
        # print("idx stuff", end_node_idxs - start_node_idxs)

        return  dEnergy

    def compute_new_external_fields(self, senders, receivers, edges, new_spins, ext_fields):
        ext_fields_edges = new_spins[senders] * edges
        ext_fields = 0.5 * jax.ops.segment_sum(ext_fields_edges, receivers, ext_fields.shape[0]) + ext_fields
        return ext_fields

    def vmapped_np_external_field_update(self, Nb_ext_fields, receivers, Nb_ext_fields_edges):
        Nb_ext_fields = np.squeeze(Nb_ext_fields, axis= -1)
        ext_fields_Nb = np.swapaxes(Nb_ext_fields , 0,1)
        Nb_ext_fields_edges = np.squeeze(Nb_ext_fields_edges, axis= -1)
        ext_fields_edges_Nb = np.swapaxes(Nb_ext_fields_edges, 0, 1)
        #Nb_ext_fields = np.squeeze(Nb_ext_fields, axis= -1)
        np.add.at(ext_fields_Nb, receivers, ext_fields_edges_Nb)

        return np.expand_dims(np.swapaxes(ext_fields_Nb , 0,1),axis = -1)

    def update_external_field(self, senders, receivers, edges, Nb_external_fields, Nb_new_spin_values, site_indices):

        if (senders.shape[0] > 0):
            ext_fields_edges = Nb_new_spin_values[:,senders] * edges[np.newaxis, :]
            Nb_ext_fields = copy.copy(Nb_external_fields)
            #Nb_receivers = np.repeat(receivers[np.newaxis,:], self.Nb, axis = 0)
            #np.add.at(Nb_ext_fields, Nb_receivers, ext_fields_edges)
            Nb_ext_fields = self.vmapped_np_external_field_update(Nb_ext_fields, receivers, ext_fields_edges)
            #Nb_ext_fields = self.vmapped_compute_new_external_fields(senders, receivers, edges, Nb_new_spin_values, Nb_external_fields)
        else:
            Nb_ext_fields = Nb_external_fields

        Nb_ext_fields = self.delete_nodes(Nb_ext_fields, site_indices)
        return Nb_ext_fields

    def delete_nodes(self, ext_fields, sampled_node_indeces):
        # new_spins = np.delete(new_spin_values, sampled_node_indeces)
        # new_spins = np.expand_dims(new_spins, axis=-1)
        ext_fields = np.delete(ext_fields, sampled_node_indeces, axis = -2)
        return ext_fields#, new_spins

    def make_env_step(self, data):
        sampled_class = data[:,0]
        log_probs = data[:,2]
        Temperatures = data[:,-1]

        spin_configurations = self._from_class_to_spins(sampled_class)
        dEnergy = self.delete_node_and_edges(spin_configurations)


        self.Nb_spins[:, self.env_step: self.env_step + self.n_sampled_sites,0] = spin_configurations
        self.env_step += self.n_sampled_sites
        self.time_horizon_step += 1
        self.dones = self.env_step >= self.N_spin

        # dEnergy = self.normalize_dEnergy(dEnergy, self.N_spin)
        self.dEnergies.append(dEnergy)

        reward = -(dEnergy + Temperatures * log_probs)
        self.update_Datacontainer(data, reward)

        # if(self.dones):
        #     self.check_energy()
        if (self.dones):
            self.finished_energy = self.compute_unormalized_Energy(np.sum(np.array(self.dEnergies), axis=0))
            if (self.mode != "train"):
                self.old_gt_Energy = self.compute_unormalized_Energy(self.gt_Energy)
            else:
                self.old_gt_Energy = self.gt_Energy

            old_finished = self.finished
            old_num_edges = self.num_edges
            old_num_nodes = self.N_spin
            self.orig_graph_dict = {"num_edges": old_num_edges, "num_nodes": old_num_nodes,
                                    "gt_Energy": self.old_gt_Energy, "pred_Energy": self.finished_energy}

            self.DataContainer.update_traces()
            self.filled_DataContainer = copy.deepcopy(self.DataContainer)
            self.reset()
            self.dEnergies = []
        else:
            old_finished = self.finished

        self.return_Jgraph = self.Jgraph

        return self.dones, dEnergy, 0, {"H_graph": {"jgraph": self.return_Jgraph, "Nb_ext_fields": self.Nb_external_fields}, "finished": old_finished}

    def _get_compl_graph(self, Jgraph, Igraph):
        ### TODO implement EnergyIgraph, normal Igraph
        if(self.orig_Igraph != None):
            self.orig_Igraph.vs["node_features"] = Igraph.vs["node_features"]
            self.orig_Igraph.vs["indices"] = np.arange(0, self.orig_Igraph.vcount())
            self.orig_Igraph = self.orig_Igraph.permute_vertices(self.perm)
            complIgraph = self.orig_Igraph
            complJgraph = jutils.from_igraph_to_jgraph(self.orig_Igraph)
        else:
            del self.orig_Igraph
            complJgraph = copy.deepcopy(Jgraph)
            complIgraph = copy.deepcopy(Igraph)
        return complJgraph, complIgraph

    def _from_class_to_spins(self, sampled_class):
        bin_arr = np.unpackbits(sampled_class.reshape((-1, 1)).astype(np.uint8), axis=1, count=self.n_sampled_sites, bitorder="little")
        spin_configuration = 2*np.array(bin_arr, dtype = np.int32) -1
        # print(f"num = {sampled_class}")
        # print(f"spins = {spin_configuration}")
        return spin_configuration

    def update_Datacontainer(self, data, rewards):
        actions = data[:,0]
        values = data[:,1]
        log_probs = data[:,2]
        self.DataContainer.append(self.return_Jgraph, self.old_Nb_external_fields, values, rewards, log_probs,actions)

    def step(self, data):
        # print("env step, time horizon")
        # print(self.env_step, self.time_horizon)
        self.old_Nb_external_fields = copy.copy(self.Nb_external_fields)
        if(self.time_horizon_step == self.time_horizon - 1):
            self.time_horizon_step = -1
            #print("last step in time horizon")
            reward = np.zeros_like(data[:,0])
            self.update_Datacontainer(data, reward)
            self.DataContainer.update_traces()
            self.filled_DataContainer = copy.deepcopy(self.DataContainer)
            ### TODO make own Datacontainer becuase env steps are different now
            #print("test", self.N_spin - self.env_step , self.return_Jgraph.nodes.shape[0])
            self.DataContainer = DataContainer.DataContainer(self.cfg, self.time_horizon, self.Nb, self.return_Jgraph.nodes.shape[0], lam=self.lam, n_sampled_spins=self.n_sampled_sites)
            return True, np.zeros((self.Nb)), 0, {"H_graph": {"jgraph": self.return_Jgraph, "Nb_ext_fields": self.Nb_external_fields}, "finished": False}
        else:
            return self.make_env_step(data)


    def check_energy(self):
        ### TODO check why this is wrong
        gs = self.gs[self.perm]
        Nb_gs = np.repeat(gs[np.newaxis,:], self.Nb, axis = 0)
        Nb_gs_spins = 2*Nb_gs-1
        self.init_Nb_graph = jraph.batch_np([self.init_EnergyJgraph for i in range(self.Nb)])
        flattened_init_Nb_external_field = np.expand_dims(np.ravel(self.init_Nb_external_fields), axis = -1)
        Energy_check_gs = compute_Energy_full_graph(flattened_init_Nb_external_field,self.init_Nb_graph,np.expand_dims(np.ravel(Nb_gs_spins), axis = -1), A= 1., B = 1.)
        Energy_check = compute_Energy_full_graph(flattened_init_Nb_external_field, self.init_Nb_graph,np.expand_dims(np.ravel(self.Nb_spins), axis = -1), A= 1., B = 1.)

        # ground_state = self.ground_state[self.perm]
        # ground_state = np.repeat(ground_state[np.newaxis,:], self.Nb, axis = 0)
        #
        # gs_Energy = compute_Energy_full_graph(self.init_Nb_graph,np.expand_dims(np.ravel(ground_state), axis = -1), A= 1., B = 1.)
        # gs_Energy = self.normalize_Energy(gs_Energy)

        T_Nb_Energy = np.array(self.dEnergies)
        #print(self.Nb_spins)
        print("Energy_check", self.compute_unormalized_Energy(np.ravel(Energy_check)))
        print("denergy check", self.compute_unormalized_Energy(np.sum(T_Nb_Energy, axis = 0)))
        print("Energy_check_gs", self.compute_unormalized_Energy(np.ravel(Energy_check_gs)))
        print("gs energy", self.compute_unormalized_Energy( self.gt_Energy), np.sum(gs))
        #print("gs energy", self.compute_unormalized_Energy(gs_Energy))
        print("end")


def MVC_Energy(H_graph, Nb_bins, A=1, B=1.1):
    ### remove self loops that were not there in original graph
    iH_graph = jutils.from_jgraph_to_igraph(H_graph)
    H_graph = jutils.from_igraph_to_jgraph(iH_graph)
    # print("nodes", nodes.shape, H_graph.edges.shape, spins.shape)
    Energy_messages = B * (1 - Nb_bins[:, H_graph.senders]) * (1 - Nb_bins[:, H_graph.receivers])
    ### TODO add HB_per_node
    Nb_Energy_messages = np.squeeze(Energy_messages, axis=-1)
    Energy_messages_Nb = np.swapaxes(Nb_Energy_messages, 0, 1)
    HB_per_node_Nb = np.zeros((H_graph.nodes.shape[0], Nb_bins.shape[0]))
    np.add.at(HB_per_node_Nb, H_graph.receivers, Energy_messages_Nb)

    HB = 0.5 * np.sum(Energy_messages, axis=-2)
    HA = A * np.sum(Nb_bins, axis=-2)
    # HB = jax.ops.segment_sum(HB_per_node, node_gr_idx, n_graph)
    # HA = jax.ops.segment_sum(HA_per_node, node_gr_idx, n_graph)

    return float(HB + HA), float(HA), float(HB), np.swapaxes(HB_per_node_Nb, 0, 1)

def MIS_Energy(H_graph, Nb_bins, A=1, B=1.1):
    ### remove self loops that were not there in original graph
    iH_graph = jutils.from_jgraph_to_igraph(H_graph)
    H_graph = jutils.from_igraph_to_jgraph(iH_graph)

    # print("nodes", nodes.shape, H_graph.edges.shape, spins.shape)
    Energy_messages = B * Nb_bins[:, H_graph.senders] * Nb_bins[:, H_graph.receivers]

    Nb_Energy_messages = np.squeeze(Energy_messages, axis=-1)
    Energy_messages_Nb = np.swapaxes(Nb_Energy_messages, 0, 1)
    HB_per_node_Nb = np.zeros((H_graph.nodes.shape[0], Nb_bins.shape[0]))
    np.add.at(HB_per_node_Nb, H_graph.receivers, Energy_messages_Nb)

    HB = 0.5 * np.sum(Energy_messages, axis=-2)
    HA = A * np.sum(Nb_bins, axis=-2)
    # HB = jax.ops.segment_sum(HB_per_node, node_gr_idx, n_graph)
    # HA = jax.ops.segment_sum(HA_per_node, node_gr_idx, n_graph)

    return float(HB + HA), float(HA), float(HB), np.swapaxes(HB_per_node_Nb, 0, 1)

def compute_Energy_full_graph(Nb_external_fields,H_graph, spins, A = 1.0, B = 1.1):
    nodes = H_graph.nodes
    n_node = H_graph.n_node
    n_graph = n_node.shape[0]
    graph_idx = np.arange(n_graph)
    sum_n_node = H_graph.nodes.shape[0]
    node_gr_idx = np.repeat(graph_idx, n_node, axis=0)

    #print("nodes", nodes.shape, H_graph.edges.shape, spins.shape)
    Energy_messages = H_graph.edges * spins[H_graph.senders] * spins[H_graph.receivers]
    Energy_per_node =  0.5 * jax.ops.segment_sum(Energy_messages, H_graph.receivers, sum_n_node) + spins*Nb_external_fields
    Energy = jax.ops.segment_sum(Energy_per_node, node_gr_idx, n_graph)

    return Energy


### TODO inherit test environments
def myfunc(cfg,H_seed, mode = "train"):
    return lambda: IGraphEnv(cfg, H_seed, mode = mode)

def envstep():
    from unipath import Path
    from omegaconf import OmegaConf
    from utils import split_dataset

    from loadGraphDatasets.jraph_Dataloader import JraphSolutionDataset
    # import warnings
    # warnings.filterwarnings('error', category=UnicodeWarning)
    # warnings.filterwarnings('error', message='*equal comparison failed*')
    os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    ### TODO debug Energy for MIS
    p = Path( os.getcwd())
    path = p.parent.parent
    cfg = OmegaConf.load(path + "/Experiment_configs/HydraBaseConfig.yaml")
    OmegaConf.update(cfg, "Paths.path", str(path) + "/model_checkpoints", merge=True)
    OmegaConf.update(cfg, "Paths.work_path", str(path), merge=True)
    cfg["Ising_params"]["IsingMode"] = "TWITTER"
    cfg["Ising_params"]["EnergyFunction"] = "MVC"
    cfg["Ising_params"]["ordering"] = "BFS"
    time_horizon = 10
    cfg["Train_params"]["PPO"]["time_horizon"] = time_horizon
    Hb = 1
    cfg["Test_params"]["n_test_graphs"] = Hb
    cfg["Train_params"]["H_batch_size"] = Hb
    cfg["Train_params"]["PPO"]["mini_Hb"] = 10
    cfg["Train_params"]["PPO"]["mini_Nb"] = 10
    cfg["Train_params"]["PPO"]["mini_Sb"] = 5
    n_sampled_sites = 5
    cfg["Network_params"]["policy_MLP_features"] = [120, 64, 2**n_sampled_sites]
    Nb = 30
    cfg["Train_params"]["n_basis_states"] = Nb
    mode = "train"
    shuffle_seed = cfg["Ising_params"]["shuffle_seed"]
    # split_idxs = split_dataset.get_split_datasets(Dataset, Hb)
    #
    # Dataset_index_func = lambda indices: list(map(lambda x: Dataset[x], indices))
    #
    # start_fast = time.time()
    # SpinEnv = SubprocVecEnv([myfunc(Dataset_index_func(split_idxs[H_seed]),cfg, H_seed, mode = mode) for H_seed in range(Hb)])
    # end_fast = time.time()

    start_slow = time.time()
    SpinEnv = SubprocVecEnv([myfunc(cfg, H_seed, mode = mode) for H_seed in range(Hb)])
    end_slow = time.time()

    print("slow", end_slow - start_slow)


    run_train(cfg, SpinEnv, Hb, Nb, time_horizon, n_sampled_sites)
    #run(SpinEnv, Hb, Nb)


def run_train(cfg,SpinEnv, Hb, Nb, time_horizon, n_sampled_sites):
    from trainPPO.DataContainer_Dataloader import ContainerDataset
    from trainPPO.DataContainer_Dataloader import collate_data_dict
    from torch.utils.data import DataLoader
    from jraph_utils import utils as jutils
    import jax.numpy as jnp
    ### TODO check eglibilty traces
    SpinEnv.set_attr("global_reset", True)
    orig_H_graphs_dict = SpinEnv.reset()
    orig_H_graphs = [H_graph["H_graph"]["jgraph"] for H_graph in orig_H_graphs_dict]

    Hb_Nb_H_graphs = jraph.batch_np(orig_H_graphs)

    for i in range(1000):
        terminated = False
        DataContainerList = []
        #while(True):
        T = 1.1
        for i in range(time_horizon + 1):
            spin_value = np.random.randint(0, 2**n_sampled_sites, (Hb, Nb, 1))

            spin_value[:,0,:] = spin_value[:,1,:]

            action = spin_value
            Temperatures = T*np.ones_like(action)
            log_probs = np.log(0.5*np.ones_like(action))
            Hb_idxs = 100 * np.arange(0, Hb)
            Nb_idxs = np.arange(0, Nb)
            values = np.expand_dims(Hb_idxs[:, np.newaxis] + Nb_idxs[np.newaxis, :], axis=-1)
            data = np.concatenate([action, values, values, Temperatures], axis = -1)

            start = time.time()
            done, reward, terminated, H_graph_dict = SpinEnv.step(data)
            #print("reward", reward)

            start_for_loop = time.time()
            if(np.any(done)):
                g_idx = np.arange(0, Hb)[np.array(done)]
                # print(done)
                # print(g_idx)
                dc = SpinEnv.get_attr("filled_DataContainer", g_idx)
                DataContainerList.extend(dc)

            end_for_loop = time.time()

            start_repeat = time.time()
            done = np.repeat(done[:, np.newaxis], Nb, axis=-1)
            end_repeat = time.time()
            end = time.time()
            print("repeat", end_repeat- start_repeat)
            print("for_loop", end_for_loop-start_for_loop)
            print("duration", end - start)
           # print([H_graph["batched_H_graph"].nodes.shape for H_graph in H_graph_dict])

            start_batching = time.time()
            print([H_graph["H_graph"]["jgraph"].nodes.shape for H_graph in H_graph_dict])
            print([H_graph["H_graph"]["Nb_ext_fields"].shape for H_graph in H_graph_dict])
            val_Hb_graphs = jraph.batch_np([H_graph["H_graph"]["jgraph"] for H_graph in H_graph_dict])
            val_Hb_graphs_jnp = jutils.cast_Tuple_to(val_Hb_graphs, np_ = jnp)

            end_batching = time.time()

            print("batching time v2", end_batching - start_batching)


        start_dataset = time.time()
        Dataset = ContainerDataset(cfg)
        Dataset.overwrite_data(DataContainerList)
        for i in range(10):
            Dataset.reshuffle()
        end_dataset = time.time()
        print("Dataset time", end_dataset - start_dataset)
        DataContainerLoader = DataLoader(Dataset, shuffle=True, num_workers=1, batch_size=10, persistent_workers=True,
                                         collate_fn=collate_data_dict)

        start_time = time.time()
        for minib_H_graphs, compl_H_graph, minib_actions, minib_A_k, minib_log_probs, minib_value_target, Hb_ext_field_list in DataContainerLoader:
            end_time = time.time()

            print("data", end_time - start_time)
            start_time = end_time

        Dataset.overwrite_data(DataContainerList)
        Dataset.reshuffle()


    ### TODO return DataContainer where env_step = time_Horizon + 1
    env_steps = SpinEnv.get_attr("env_step")

    print("DatacontainerList contains", len(DataContainerList))

    start_dataset = time.time()
    Dataset = ContainerDataset(cfg)
    Dataset.overwrite_data(DataContainerList)
    for i in range(10):
        Dataset.reshuffle()
    end_dataset = time.time()
    print("Dataset time",end-start_dataset)
    DataContainerLoader = DataLoader(Dataset, shuffle = True, num_workers = 2, batch_size=3, collate_fn=collate_data_dict)

    start_time = time.time()
    for data in DataContainerLoader:
        end_time = time.time()
        print("data", end_time-start_time)
        start_time = end_time


    print("finished")



def sample_from_ReplayBuffer(DataContainer_list):
    pass


def calc_traces( rewards, values, not_dones, time_horizon =5, gamma = 1., lam = 1.):
    advantage = np.zeros_like(values)
    for t in reversed(range(time_horizon)):
        delta = rewards[t] + gamma * not_dones[t+1]*values[t+1] - values[t]
        advantage[t] = delta + gamma*lam *not_dones[t+1]*advantage[t+1]

    value_target = (advantage + values)[0:time_horizon]
    return value_target, advantage[0:time_horizon]



if(__name__ == "__main__"):

    envstep()
    pass