import numpy as np
from jraph_utils import utils as jutils
import igraph as ig
from Gurobi import GurobiSolver
from unipath import Path
import os
import pickle

class ToyMIS():
    def __init__(self, max_num_nodes_per_minimum, min_node_per_minima, num_minima = 2, A_max = 0.5, B = 1.1, fractional = 0.8):

        self.fractional = fractional
        self.A_max = A_max
        self.B = B
        self.num_minima = num_minima
        self.min_node_per_minima = min_node_per_minima #(local minimum)

        self.max_num_nodes_per_minimum = max_num_nodes_per_minimum

        self.gs_Energy = -self.A_max

    def generate(self):

        a = np.random.uniform(0.01, self.B)
        self.A_max_curr = a

        nodes_per_minimum = []
        for i in range(self.num_minima):
            num_nodes = np.random.randint(self.min_node_per_minima, self.max_num_nodes_per_minimum)
            nodes_per_minimum.append(num_nodes)

        minima_dict = {}

        node_idx = 0
        for i in range(self.num_minima+1):
            minima_dict[f"minimum_{i}"] = {}
            if(i == 0):
                minima_dict[f"minimum_{i}"]["external_field"] = np.array([self.A_max_curr])
                minima_dict[f"minimum_{i}"]["node_idx"] = np.array([node_idx])
                node_idx += 1
            else:
                num_nodes = nodes_per_minimum[i-1]
                #ext_fields = np.random.uniform(0,self.fractional*self.A_max/num_nodes, size=(num_nodes))
                ext_fields = self.fractional * self.A_max_curr / num_nodes*np.ones((num_nodes))
                minima_dict[f"minimum_{i}"]["external_field"] = ext_fields
                minima_dict[f"minimum_{i}"]["node_idx"] = np.arange(node_idx, node_idx+num_nodes)
                node_idx += num_nodes

        edge_list = []
        ext_fields = []
        for idx_1, key_1 in enumerate(minima_dict):
            ext_fields.extend(list(minima_dict[key_1]["external_field"]))
            for idx_2, key_2 in enumerate(minima_dict):
                if(idx_1 < idx_2):
                        node_idxs_1 = minima_dict[key_1]["node_idx"]
                        node_idxs_2 = minima_dict[key_2]["node_idx"]
                        for n_idx_1 in node_idxs_1:
                            for n_idx_2 in node_idxs_2:
                                edge_list.append((n_idx_1, n_idx_2))

        ext_field_arr = np.array(ext_fields)
        igraph = ig.Graph(edge_list)
        igraph.vs["ext_fields"] = ext_field_arr
        return igraph

class ExtendedToyMIS(ToyMIS):
    def __init__(self, node_range = [100, 200], num_minima_range = [2, 20], max_nodes_in_minimum_fraction = 0.5, A_max = 1, B = 1.1, fractional = 0.8, wide_minimum_f = 0.7, seed = 123):

        self.max_nodes_in_minimum_fraction = max_nodes_in_minimum_fraction
        self.fractional = fractional
        self.wide_minimum_f = wide_minimum_f
        self.A_max = A_max
        self.B = B
        self.num_minima_range = num_minima_range
        self.node_range = node_range

        self.gs_Energy = -self.A_max
        np.random.seed(seed)

    def generate(self):
        return self.generate_minima()

    def generate_minima(self):
        num_minima = np.random.randint(self.num_minima_range[0], self.num_minima_range[1])
        num_nodes = np.random.randint(self.node_range[0], self.node_range[1])

        nodes_per_minimum = [ 1 for i in range(num_minima)]
        sample_from_nodes = num_nodes - num_minima
        for min_idx in range(num_minima):
            if(sample_from_nodes > 1):
                add_node_in_minimum = np.random.randint(1, np.min([sample_from_nodes, self.max_nodes_in_minimum_fraction*num_nodes ]))
                sample_from_nodes -= add_node_in_minimum

                nodes_per_minimum[min_idx] += add_node_in_minimum
            else:
                nodes_per_minimum[min_idx] += 1
                break

        sorted_nodes_per_minimum = sorted(nodes_per_minimum)
        sorted_nodes_per_minimum[-1] += 1
        print(sorted_nodes_per_minimum)
        return self.add_ext_fields(sorted_nodes_per_minimum)

    def add_ext_fields(self, nodes_per_minimum):
        minima_dict = {}

        overall_ext_field_per_minimum = [0 for i in range(len(nodes_per_minimum))]

        node_idx = 0
        for i in range(len(nodes_per_minimum)):
            minima_dict[f"minimum_{i}"] = {}

            num_nodes = nodes_per_minimum[i]
            # ext_fields = np.random.uniform(0,self.fractional*self.A_max/num_nodes, size=(num_nodes))
            ext_fields = self.fractional * self.A_max / num_nodes * np.ones((num_nodes))
            minima_dict[f"minimum_{i}"]["external_field"] = ext_fields
            minima_dict[f"minimum_{i}"]["node_idx"] = np.arange(node_idx, node_idx + num_nodes)
            node_idx += num_nodes
            overall_ext_field_per_minimum[i] = self.fractional * self.A_max


        ### set global minimum
        global_minimum_index_range = len(nodes_per_minimum[:-1])
        global_minimum_index = np.random.randint(0, global_minimum_index_range)

        num_nodes = nodes_per_minimum[global_minimum_index]
        minima_dict[f"minimum_{global_minimum_index}"]["external_field"] = self.A_max / num_nodes * np.ones((num_nodes))
        overall_ext_field_per_minimum[global_minimum_index] = self.A_max


        wide_minimum_index = len(nodes_per_minimum[:-1])
        num_nodes = nodes_per_minimum[wide_minimum_index]
        minima_dict[f"minimum_{wide_minimum_index}"]["external_field"] = self.wide_minimum_f *self.A_max / num_nodes * np.ones((num_nodes))
        overall_ext_field_per_minimum[wide_minimum_index] = self.wide_minimum_f *self.A_max

        igraph = self.generate_graph(minima_dict)

        return igraph

    def generate_graph(self, minima_dict):

        edge_list = []
        ext_fields = []
        for idx_1, key_1 in enumerate(minima_dict):
            ext_fields.extend(list(minima_dict[key_1]["external_field"]))
            for idx_2, key_2 in enumerate(minima_dict):
                if(idx_1 < idx_2):
                        node_idxs_1 = minima_dict[key_1]["node_idx"]
                        node_idxs_2 = minima_dict[key_2]["node_idx"]
                        for n_idx_1 in node_idxs_1:
                            for n_idx_2 in node_idxs_2:
                                edge_list.append((n_idx_1, n_idx_2))

        ext_field_arr = np.array(ext_fields)
        igraph = ig.Graph(edge_list)
        igraph.vs["ext_fields"] = ext_field_arr
        return igraph

def solve_instances(Generator, dataset_name, EnergyFunction, path, mode = "train", seed = 123):
    if (mode == "train"):
        n_graphs = 1000
        time_limit = 1
    elif (mode == "val"):
        n_graphs = 100
        time_limit = float("inf")
    else:
        n_graphs = 300
        time_limit = float("inf")

    solution_dict = {}
    solution_dict["Energies"] = []
    solution_dict["H_graphs"] = []
    solution_dict["gs_bins"] = []
    solution_dict["runtimes"] = []

    for i in range(n_graphs):
        Generator.A_max = 1.
        igraph = Generator.generate()
        ext_fields = np.array(igraph.vs["ext_fields"])
        ext_fields = np.expand_dims(ext_fields, axis=-1)
        jgraph = jutils.from_igraph_to_jgraph(igraph)
        weighted_jgraph = jgraph._replace(nodes=ext_fields)

        if (mode == "train"):
            model, energy, solution, runtime = GurobiSolver.solveWMIS_as_MIP(weighted_jgraph, time_limit=time_limit)

            if (False):
                ds = np.arange(0, igraph.vcount())
                Energy_at_d = []
                d_list = []
                for d in ds:
                    model, energy, _, runtime = GurobiSolver.solveWMIS_as_MIP_at_d(weighted_jgraph, solution, d)
                    Energy_at_d.append(energy)
                    d_list.append(d)

                plt.figure()
                plt.plot(d_list, Energy_at_d)
                plt.show()
                print("here")

        else:
            model, energy, solution, runtime = GurobiSolver.solveWMIS_as_MIP(weighted_jgraph, time_limit=time_limit)

        print("mode", mode, energy, igraph.vcount(), -np.max(weighted_jgraph.nodes))
        solution_dict["Energies"].append(energy)
        solution_dict["H_graphs"].append(weighted_jgraph)
        solution_dict["gs_bins"].append(solution)
        solution_dict["runtimes"].append(runtime)

    newpath = path + f"/loadGraphDatasets/DatasetSolutions/no_norm/{dataset_name}"
    if not os.path.exists(newpath):
        os.makedirs(newpath)

    save_path = path + f"/loadGraphDatasets/DatasetSolutions/no_norm/{dataset_name}/{mode}_{EnergyFunction}_seed_{seed}_solutions.pickle"
    pickle.dump(solution_dict, open(save_path, "wb"))

def generate_and_save_ToyMIS_instances(max_local_minimum_size = 13, min_local_minimum_size = 3, num_minima = 10, parent = True, mode = "val", seed = 123, B = 1.1):
    from matplotlib import pyplot as plt

    np.random.seed(seed)
    p = Path(os.getcwd())
    if(parent):
        path = p.parent
    else:
        path = p

    EnergyFunction = "WMIS"
    dataset_name = f"ToyMIS_{max_local_minimum_size}_{min_local_minimum_size}_{num_minima}_harder"

    Generator = ToyMIS(max_local_minimum_size, min_local_minimum_size, num_minima = num_minima, B = B)

    solve_instances(Generator, dataset_name, EnergyFunction, path, mode = mode)

def generate_and_save_ExtToyMIS_instances(node_range = [100, 200], num_minima_range = [2, 20], fractional = 0.7, wide_minimum_f = 0.8 ,parent = True, mode = "val", seed = 123, B = 1.1):
    from matplotlib import pyplot as plt

    np.random.seed(seed)
    p = Path(os.getcwd())
    if(parent):
        path = p.parent
    else:
        path = p

    EnergyFunction = "WMIS"
    dataset_name = f"ExtToyMIS_{node_range}_{num_minima_range}_{fractional}_{wide_minimum_f}"

    Generator = ExtendedToyMIS( node_range = node_range, num_minima_range = num_minima_range, fractional=fractional, wide_minimum_f = wide_minimum_f, B = B, seed = seed)

    solve_instances(Generator, dataset_name, EnergyFunction, path, mode = mode, seed = seed)




if(__name__ == "__main__"):
    # modes = ["train", "val", "test"]
    #
    # for mode in modes:
    #     print("start solving ", mode)
    #     generate_and_save_ToyMIS_instances(mode = mode)

    modes = ["train", "val", "test"]

    for mode in modes:
        print("start solving ", mode)
        generate_and_save_ExtToyMIS_instances(mode = mode, node_range = [50, 100], num_minima_range = [2, 10])
    # modes = ["train", "val", "test"]
    # num_minima_list = [1]
    # k = 6
    # for num_minima in num_minima_list:
    #     for mode in modes:
    #         print("start solving ", mode)
    #         generate_and_save_instances(max_local_minimum_size=35, min_local_minimum_size=30, num_minima = num_minima, mode = mode)
    pass



