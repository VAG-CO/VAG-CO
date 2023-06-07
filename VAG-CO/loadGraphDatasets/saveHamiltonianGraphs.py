from utils import sympy_utils, SympyHamiltonians
from EnergyFunctions import jraphEnergy
from omegaconf import OmegaConf
from unipath import Path
import os
import pickle
import numpy as np
from copy import deepcopy
import time
from jraph_utils import utils as jutils
import scipy
import scipy.sparse as sp
from GlobalProjectVariables import MVC_A, MVC_B, MaxCl_B
from GreedyAlgorithms import GreedyGeneral
from tqdm import tqdm

def load_solution_dict(dataset_name, EnergyFunction,seed = 0, mode = "val", parent = False):

    if(EnergyFunction == "MaxCl_compl" or EnergyFunction == "MaxCl_EGN"):
        loadEnergyFunction = "MaxCl"
    else:
        loadEnergyFunction = EnergyFunction

    p = Path(os.getcwd())
    if(parent):
        path = p.parent
    else:
        path = p
    cfg = OmegaConf.load(path + "/Experiment_configs/HydraBaseConfig.yaml")

    OmegaConf.update(cfg, "Paths.path", str(path) + "/model_checkpoints", merge=True)
    OmegaConf.update(cfg, "Paths.work_path", str(path), merge=True)

    cfg["Ising_params"]["IsingMode"] = dataset_name
    cfg["Ising_params"]["EnergyFunction"] = EnergyFunction
    cfg["Ising_params"]["shuffle_seed"] = -1
    path = os.path.join(cfg["Paths"]["work_path"], "loadGraphDatasets", "DatasetSolutions", "no_norm",
                        dataset_name, f"{mode}_{loadEnergyFunction}_seed_{seed}_solutions.pickle")

    file = open(path, "rb")
    solution_dict = pickle.load(file)
    return solution_dict, cfg

def norm_graph(H_graph, mean_Energy, std_Energy):
    num_nodes = H_graph.nodes.shape[0]
    nodes = H_graph.nodes/std_Energy
    couplings = H_graph.edges[:-2*num_nodes]/std_Energy
    self_connections = (H_graph.edges[-2*num_nodes:]-mean_Energy/num_nodes)/std_Energy
    edges = np.concatenate([couplings,self_connections], axis = 0)

    normed_H_graph = H_graph._replace(nodes = nodes, edges = edges)
    return normed_H_graph, self_connections, couplings

def norm_orig_graph(H_graph, mean_Energy, std_Energy, self_connections, couplings):
    #### TODO add self loops

    senders = H_graph.senders
    receivers = H_graph.receivers

    try:
        edges = np.ones_like(H_graph.edges)*couplings[0,0]
    except:
        edges = np.zeros_like(H_graph.edges)

    self_senders = np.arange(0, H_graph.nodes.shape[0])
    self_receivers = np.arange(0, H_graph.nodes.shape[0])
    self_edges = np.zeros((H_graph.nodes.shape[0], 1))

    all_senders = np.concatenate([senders, self_senders, self_receivers], axis = -1)
    all_receivers = np.concatenate([receivers, self_receivers, self_senders], axis = -1)
    all_edges = np.concatenate([edges, self_edges, self_edges], axis = 0)

    num_nodes = H_graph.nodes.shape[0]
    nodes = H_graph.nodes/std_Energy
    couplings = all_edges[:-2*num_nodes]/std_Energy
    edges = np.concatenate([couplings,self_connections], axis = 0)

    normed_H_graph = H_graph._replace(senders = all_senders, receivers = all_receivers, nodes = nodes, edges = edges)
    return normed_H_graph

def get_graph_laplacian(igraph, k = 10):  ### TODO k has to be as large as larges number of nodes in dataset
    num_nodes = igraph.vcount()
    k = min([k, num_nodes])
    adj = np.array(igraph.get_adjacency()._get_data())
    adj[np.arange(0,num_nodes), np.arange(0,num_nodes)] = 0

    if(False):
        edge_list = igraph.get_edgelist()
        A = np.zeros((num_nodes, num_nodes))
        for (s,r) in edge_list:
            if(s != r):
                A[s,r] = +1

        print("test",A)
        print(adj)

    L = sp.csgraph.laplacian(adj)
    w, v = scipy.linalg.eigh(L)

    igraph.vs["eigenvalue_embedding"] = w[0:k]
    igraph.vs["eigenvector_embedding"] = v[:,0:k]

    return igraph

def load(dataset_name, train_dataset_name, EnergyFunction, mode = "val", seed = 0, parent = False):
    from unipath import Path
    ### TODO load one graph after another

    solution_dict, cfg = load_solution_dict(dataset_name, EnergyFunction, mode = mode, seed = seed, parent = parent)
    train_solution_dict, _ = load_solution_dict(train_dataset_name, EnergyFunction, mode = "train", seed = seed, parent = parent)

    print(EnergyFunction)
    print("train",calculate_mean_and_std(train_solution_dict["Energies"]))
    run_path = os.getcwd()

    path_list = [cfg["Paths"]["work_path"], "loadGraphDatasets", "DatasetSolutions", "no_norm_H_graph",dataset_name,f"{mode}_{EnergyFunction}_seed_{seed}_solutions"]
    for path_el in path_list:
        run_path = os.path.join(run_path, path_el)
        if not os.path.exists(run_path):
            os.mkdir(run_path)

    solution_dict["spin_graph"] = []

    print("graphs are going to be translated to spin formulation")
    zip_data = list(zip(solution_dict["Energies"], solution_dict["H_graphs"], solution_dict["gs_bins"]))
    for idx, (Energy, j_graph, bins) in enumerate(tqdm(zip_data, total = len(zip_data))):
        # if(idx > 20):
        #     break

        if(EnergyFunction == "MaxCl" or EnergyFunction == "MaxCl_compl"):
            i_graph = jutils.from_jgraph_to_igraph(j_graph)
            Compl_igraph = i_graph.complementer(loops = False)
            Compl_jgraph = jutils.from_igraph_to_jgraph(Compl_igraph)
            H_graph = SympyHamiltonians.MIS_sparse(Compl_jgraph, B = MaxCl_B)
        elif(EnergyFunction == "MVC"):
            s2 = time.time()
            H_graph = SympyHamiltonians.MVC_sparse(j_graph)
            e2 = time.time()
            #print("MVC", e2-s2)
        elif(EnergyFunction == "MIS"):
            s2 = time.time()
            H_graph = SympyHamiltonians.MIS_sparse(j_graph)
            e2 = time.time()
            #print("MIS", e2-s2)
        elif(EnergyFunction == "WMIS"):
            s2 = time.time()
            H_graph = SympyHamiltonians.WMIS_sparse(j_graph)
            e2 = time.time()
            #print("WMIS", e2-s2)
        elif(EnergyFunction == "MaxCut"):
            s2 = time.time()
            H_graph = SympyHamiltonians.MaxCut(j_graph)
            e2 = time.time()
            #print("MaxCut", e2-s2)

        else:
            ValueError("No such EnergyFunction is implemented")

        spins = 2*bins -1
        spins = np.expand_dims(spins, axis=-1)
        #Energy_from_graph = jraphEnergy.compute_Energy_full_graph(H_graph, spins)
        #print("Energy from Gurobi", Energy, "Energy from Graph", np.squeeze(Energy_from_graph))
        solution_dict["spin_graph"].append(H_graph)

    new_solution_dict = {}
    new_solution_dict["gs_bins"] = []
    new_solution_dict["normed_igraph"] = []
    new_solution_dict["normed_Energies"] = []
    new_solution_dict["orig_igraph"] = []

    ### TODO always calc this on train set
    if(mode == "train"):

        if(EnergyFunction == "MaxCl" or EnergyFunction == "MaxCl_compl" or EnergyFunction == "WMIS"):
            iter_fraction = 3
        else:
            iter_fraction = 1

        mean_greedy_Energy, std_greedy_Energy = calculate_greedy_mean_and_std(solution_dict["spin_graph"], solution_dict["Energies"], iter_fraction=iter_fraction)
    else:
        path = os.path.join(cfg["Paths"]["work_path"], "loadGraphDatasets", "DatasetSolutions", "no_norm_H_graph",
                            train_dataset_name, f"train_{EnergyFunction}_seed_{seed}_solutions.pickle")

        file = open(path, "rb")
        train_sol_dict = pickle.load(file)
        mean_greedy_Energy = train_sol_dict["val_mean_Energy"]
        std_greedy_Energy = train_sol_dict["val_std_Energy"]

    new_solution_dict["val_mean_Energy"] = mean_greedy_Energy
    new_solution_dict["val_std_Energy"] = std_greedy_Energy

    print("Energy scale is going to be standartized")
    zip_data = list(tqdm(zip(solution_dict["Energies"], solution_dict["spin_graph"], solution_dict["H_graphs"], solution_dict["gs_bins"])))
    for idx, (Energy, H_graph, orig_graph, bins) in enumerate(tqdm(zip_data, total=len(zip_data))):

        spins = 2*bins -1
        spins = np.expand_dims(spins, axis = -1)

        normed_H_graph, self_connections, couplings = norm_graph(H_graph, mean_greedy_Energy, std_greedy_Energy)
        normed_Energy = (Energy - mean_greedy_Energy)/std_greedy_Energy
        #print("normed Energy", normed_Energy, jraphEnergy.compute_Energy_full_graph(normed_H_graph, spins))

        normed_ig = jutils.from_jgraph_to_igraph_normed(normed_H_graph)
        #get_graph_laplacian(normed_ig)
        if(EnergyFunction == "MaxCl_compl" or EnergyFunction == "MaxCl_EGN"):
            orig_graph = orig_graph._replace(nodes = normed_H_graph.nodes)
            normed_orig_graph = norm_orig_graph(orig_graph, mean_greedy_Energy, std_greedy_Energy, self_connections, couplings)
            orig_ig = jutils.from_jgraph_to_igraph_normed(normed_orig_graph)
        else:
            orig_ig = None

        new_solution_dict["normed_igraph"].append(normed_ig)
        new_solution_dict["normed_Energies"].append(normed_Energy)
        new_solution_dict["gs_bins"].append(bins)
        new_solution_dict["orig_igraph"].append(orig_ig)

        graph_dict = {}
        graph_dict["orig_igraph"] = orig_ig
        graph_dict["normed_igraph"] = normed_ig
        graph_dict["normed_Energies"] = normed_Energy
        graph_dict["gs_bins"] = bins
        graph_dict["val_mean_Energy"] = mean_greedy_Energy
        graph_dict["val_std_Energy"] = std_greedy_Energy
        graph_dict["n_graphs"] = len(solution_dict["Energies"])

        path = os.path.join(cfg["Paths"]["work_path"], "loadGraphDatasets", "DatasetSolutions", "no_norm_H_graph",
                            dataset_name, f"{mode}_{EnergyFunction}_seed_{seed}_solutions", f"_idx_{idx}.pickle")

        file = open(path, "wb")

        pickle.dump(graph_dict, file)

    path = os.path.join(cfg["Paths"]["work_path"], "loadGraphDatasets", "DatasetSolutions", "no_norm_H_graph",
                        dataset_name, f"{mode}_{EnergyFunction}_seed_{seed}_solutions.pickle")


    file = open(path, "wb")
    pickle.dump(new_solution_dict, file)

def calculate_mean_and_std(Energy_list):
    Energy_arr = np.array(Energy_list)
    #Energy_arr = np.round(Energy_arr)

    mean_energy = np.mean(Energy_arr)
    std_energy = np.std(Energy_arr)

    if(std_energy < 10**-10):
        std_energy = 1

    return mean_energy, std_energy

def calculate_greedy_mean_and_std(H_graphs, gs_Energies, norm_mode = "random_greedy", iter_fraction = 1):
    greedy_Energies = []
    print("graphs are solved with RGA")
    zip_data = list(zip(H_graphs, gs_Energies))
    for H_graph, gs_Energy in tqdm(zip_data, total=len(zip_data)):
        if(norm_mode == "autoregressive"):
            Energy, spins = GreedyGeneral.AutoregressiveGreedy(H_graph )
        elif("random_greedy"):
            # Energy, spins = GreedyGeneral.random_greedy(H_graph , iter_fraction= 1)
            # print("frac 1", Energy)
            Energy, spins = GreedyGeneral.random_greedy(H_graph , iter_fraction= iter_fraction)
            #print("frac ",iter_fraction,  Energy)
        elif("random"):
            Energy, spins = GreedyGeneral.random(H_graph)
        greedy_Energies.append(Energy)
        #print("greedy Energy", Energy, "vs gs Energy", gs_Energy, H_graph.nodes.shape[0], H_graph.edges.shape[0])
        if(Energy < gs_Energy):
            ValueError("gs Energy is larger than greedy Energy")

    greedy_Energies = np.array(greedy_Energies)
    mean_greedy_Energy = np.mean(greedy_Energies)
    std_greedy_Energy = np.std(greedy_Energies)
    print(f"{norm_mode} Energy", mean_greedy_Energy, std_greedy_Energy)
    print((gs_Energies - mean_greedy_Energy)/std_greedy_Energy )
    return mean_greedy_Energy, std_greedy_Energy

def compute_loading_time(dataset_name, EnergyFunction, mode = "val", seed = 123 ):
    solution_dict, cfg = load_solution_dict(EnergyFunction, mode = mode, seed = seed)
    for idx in range(100):
        path = os.path.join(cfg["Paths"]["work_path"], "loadGraphDatasets", "DatasetSolutions", "no_norm_H_graph",
                            dataset_name +  f"_{mode}_{EnergyFunction}_seed_{seed}_solutions", f"_idx_{idx+1}.pickle")
        start = time.time()

        file = open(path, "rb")

        graph_dict = pickle.load(file)
        end = time.time()
        print("time",end -start)

        start = time.time()

        graph = graph_dict["normed_igraph"]
        end = time.time()
        print("time2",end -start)

def make_RB_test_graphs():

    modes = ["test"]
    EnergyFunctions = ["MVC"]

    seeds = [123]
    p_list = np.linspace(0.25, 1, num = 10)
    for p in p_list:
        dataset_name = f"RB_iid_200_p_{p}"
        train_dataset_name = f"RB_iid_200"
        for seed in seeds:
                for mode in modes:
                    for EnergyFunction in EnergyFunctions:
                        print("finished", dataset_name, seed)
                        #compute_loading_time(dataset_name, EnergyFunction, mode = mode, seed = seed )
                        load(dataset_name, train_dataset_name, EnergyFunction, mode = mode, seed = seed )
    pass


def solve(dataset, problem, seeds = [123]):
    modes = ["train", "val","test"]
    dataset_names = [dataset]
    # modes = ["test"]
    # EnergyFunctions = ["MVC"]
    EnergyFunctions = [problem]

    for seed in seeds:
        for dataset_name in dataset_names:
            for mode in modes:
                for EnergyFunction in EnergyFunctions:
                    print("The following data is translated to spin formulation:", dataset_name, seed, mode, EnergyFunction)
                    # compute_loading_time(dataset_name, EnergyFunction, mode = mode, seed = seed )
                    load(dataset_name, dataset_name, EnergyFunction, mode=mode, seed=seed)

if(__name__ == "__main__"):
    ### TODO greedy normalise RRG_100 and RRG_1000

    if(True):
        #dataset_names = ["ExtToyMIS_[50, 100]_[2, 10]_0.7_0.8", "ToyMIS_13_3_10_harder"]
        modes = ["train", "val", "test"]
        dataset_names = ["TWITTER"]
        #modes = ["test"]
        #EnergyFunctions = ["MVC"]
        EnergyFunctions = ["MaxCl_compl"]

        seeds = [123]
        for seed in seeds:
            for dataset_name in dataset_names:
                for mode in modes:
                    for EnergyFunction in EnergyFunctions:
                        print("finished", dataset_name, seed, mode, EnergyFunction)
                        #compute_loading_time(dataset_name, EnergyFunction, mode = mode, seed = seed )
                        load(dataset_name, dataset_name, EnergyFunction, mode = mode, seed = seed , parent = True)


    # dataset_names = ["COLLAB","TWITTER", "IMDB-BINARY"]
    # modes = ["train","val", "test"]
    # #dataset_names = ["ToyMIS_13_3_5", "ToyMIS_13_3_3", "ToyMIS_13_3_10"]
    # #modes = ["val", "test"]
    # EnergyFunctions = ["MVC"]
    #
    # seeds = [124, 125]
    # for seed in seeds:
    #     for dataset_name in dataset_names:
    #         for mode in modes:
    #             for EnergyFunction in EnergyFunctions:
    #                 print("finished", dataset_name, seed)
    #                 #compute_loading_time(dataset_name, EnergyFunction, mode = mode, seed = seed )
    #                 load(dataset_name, EnergyFunction, mode = mode, seed = seed )
    #
    # dataset_names = ["RB_iid_100", "RB_iid_200"]
    # modes = ["train","val", "test"]
    # #dataset_names = ["ToyMIS_13_3_5", "ToyMIS_13_3_3", "ToyMIS_13_3_10"]
    # #modes = ["val", "test"]
    # EnergyFunctions = ["MVC"]
    #
    # seeds = [123]
    # for seed in seeds:
    #     for dataset_name in dataset_names:
    #         for mode in modes:
    #             for EnergyFunction in EnergyFunctions:
    #                 print("finished", dataset_name, seed)
    #                 #compute_loading_time(dataset_name, EnergyFunction, mode = mode, seed = seed )
    #                 load(dataset_name, EnergyFunction, mode = mode, seed = seed )