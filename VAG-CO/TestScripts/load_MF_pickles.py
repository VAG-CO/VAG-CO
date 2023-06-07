import numpy as np
import pickle
import matplotlib.pyplot as plt


def load_RRG():
    path = ""

    ids = ["100_ka5gqgl1_RRG_100_k_=all.pickle", "100_lfgf2hq1_RRG_100_k_=all.pickle"]
    res_dict = test_load(path, ids)
    return res_dict

def load_RB_200(N = 8):
    path = ""

    list_over_p_anneal = []
    list_over_p_no_anneal = []

    rel_error_over_p = {}
    rel_error_over_p["anneal"] = []
    rel_error_over_p["no_anneal"] = []
    std_rel_error_over_p = {}
    std_rel_error_over_p["anneal"] = []
    std_rel_error_over_p["no_anneal"] = []
    ps = np.linspace(0.25, 1, num=10)
    for p in ps:
        print("curr p is", p)
        #### TODO calc mean over this stuff here
        ### TODO find aut what is annealing and what not
        ids = [f"100_k8rdh96k_RB_iid_200_p_{p}.pickle", f"100_4pr5dly3_RB_iid_200_p_{p}.pickle"]
        res_dict =  test_load(path, ids)

        anneal_keys = ["anneal", "no_anneal"]
        for anneal_key in anneal_keys:
            rel_error_over_p[anneal_key].append(res_dict[anneal_key]["rel_error_over_N"][N-1])
            std_rel_error_over_p[anneal_key].append(res_dict[anneal_key]["std_rel_error_over_N"][N-1])
        list_over_p_anneal.append(res_dict["anneal"]["std_rel_error_per_graph"])
        list_over_p_no_anneal.append(res_dict["no_anneal"]["std_rel_error_per_graph"])

    print("calculate mean Energy anneal")
    anneal_Energy_at_N = np.array([el[N - 1] for el in list_over_p_anneal])
    print(anneal_Energy_at_N.shape)
    N_samples = anneal_Energy_at_N.shape[0]*anneal_Energy_at_N.shape[1]
    print("anneal mean Energy", np.mean(anneal_Energy_at_N), np.std(anneal_Energy_at_N)/np.sqrt(N_samples))
    no_anneal_Energy_at_N = np.array([el[N - 1] for el in list_over_p_no_anneal])
    print("no anneal mean Energy", np.mean(no_anneal_Energy_at_N), np.std(no_anneal_Energy_at_N)/np.sqrt(N_samples))

    N_list = res_dict["N_list"]
    arr_over_p_anneal = np.array(list_over_p_anneal)
    Nb_Np_Hb_data_anneal = np.swapaxes(arr_over_p_anneal, 0, 1)
    Nb_data_anneal = np.reshape(Nb_Np_Hb_data_anneal, (Nb_Np_Hb_data_anneal.shape[0], Nb_Np_Hb_data_anneal.shape[1]*Nb_Np_Hb_data_anneal.shape[2]))
    arr_over_p_no_anneal = np.array(list_over_p_no_anneal)
    Nb_Np_Hb_data_no_anneal = np.swapaxes(arr_over_p_no_anneal, 0, 1)
    Nb_data_no_anneal = np.reshape(Nb_Np_Hb_data_no_anneal, (Nb_Np_Hb_data_no_anneal.shape[0], Nb_Np_Hb_data_no_anneal.shape[1]*Nb_Np_Hb_data_no_anneal.shape[2]))

    mean_Nb_data_anneal = np.mean(Nb_data_anneal, axis = -1)
    std_Nb_data_anneal = np.std(Nb_data_anneal, axis = -1)/np.sqrt(Nb_data_anneal.shape[-1])
    mean_Nb_data_no_anneal = np.mean(Nb_data_no_anneal, axis = -1)
    std_Nb_data_no_anneal = np.std(Nb_data_no_anneal, axis = -1)/np.sqrt(Nb_data_no_anneal.shape[-1])


    plt.figure()
    plt.errorbar(N_list, mean_Nb_data_anneal, yerr=std_Nb_data_anneal, fmt = "-x", label = "anneal")
    plt.errorbar(N_list, mean_Nb_data_no_anneal, yerr=std_Nb_data_no_anneal, fmt = "-x", label = "no_anneal")
    plt.show()

    data_dict = {}

    data_dict["N_list"] = N_list
    data_dict["anneal"] = {}
    data_dict["anneal"]["mean_over_N"] = mean_Nb_data_anneal
    data_dict["anneal"]["std_over_N"] = std_Nb_data_anneal
    data_dict["no_anneal"] = {}
    data_dict["no_anneal"]["mean_over_N"] = mean_Nb_data_no_anneal
    data_dict["no_anneal"]["std_over_N"] = std_Nb_data_no_anneal

    for anneal_key in anneal_keys:
        data_dict[anneal_key]["ps"] = ps
        data_dict[anneal_key]["rel_error_over_p"] = np.array(rel_error_over_p[anneal_key])
        data_dict[anneal_key]["std_rel_error_over_p"] = np.array(std_rel_error_over_p[anneal_key])


    return data_dict


def test_load(path, ids):
    n_different_random_node_features = 100
    file_path_annealing = path + ids[1]
    file_path_no_annealing = path + ids[0]

    with open(file_path_annealing, 'rb') as file:
        data_annealing = pickle.load(file)

    with open(file_path_no_annealing, 'rb') as file:
        data_no_annealing = pickle.load(file)

    n_edges = data_annealing['results']["n_edges"]
    n_nodes = data_annealing['results']["n_nodes"]
    rel_error_matrix = data_annealing['results']['rel_error_matrix']
    rel_error_matrix_no_annealing = data_no_annealing['results']['rel_error_matrix']

    def calc_rel_errors(data, n_node_inits=8):

        idxs_ = np.arange(0, data.shape[0])
        np.random.shuffle(idxs_)
        selected_idxs = idxs_[0:n_node_inits]
        H_idxs = np.arange(0, data.shape[1])

        data = data[selected_idxs[ :, np.newaxis], H_idxs[np.newaxis, :]]

        min_rel_error_per_graph = np.min(data, axis = 0)
        mean_rel_err = np.mean(min_rel_error_per_graph)
        std_rel_err = np.std(min_rel_error_per_graph)/ np.sqrt(min_rel_error_per_graph.shape[0])
        return mean_rel_err, std_rel_err, min_rel_error_per_graph

    def plot_list(data):
        rel_errors_list = []
        std_rel_errors_list = []
        rel_errors_per_graph_over_N = []

        for i in range(n_different_random_node_features):
            n_node_inits = i + 1
            mean_rel_error, std_rel_error, rel_error_per_graph = calc_rel_errors(data, n_node_inits)
            rel_errors_list.append(mean_rel_error)
            rel_errors_per_graph_over_N.append(rel_error_per_graph)
            std_rel_errors_list.append(std_rel_error)
        return np.array(rel_errors_list), np.array(std_rel_errors_list), np.array(rel_errors_per_graph_over_N)

    scale = 1.5

    rel_errors_over_N_anneal, std_rel_error_anneal, rel_error_per_graph_over_N_anneal =  plot_list(rel_error_matrix)
    rel_errors_over_N_no_anneal, std_rel_error_no_anneal, rel_error_per_graph_over_N_no_anneal =  plot_list(rel_error_matrix_no_annealing)

    Nb_list = [i + 1 for i in range(n_different_random_node_features)]
    if(False):
        plt.figure(figsize=(10 * scale, 6 * scale))
        plt.grid()
        plt.errorbar(Nb_list, rel_errors_over_N_anneal, yerr=std_rel_error_anneal,
                 label= f'RRG_100_k_=all_MIS - CE | $T_0={data_annealing["T"]}$')
        plt.errorbar(Nb_list, rel_errors_over_N_no_anneal, yerr=  std_rel_error_no_anneal,
                 label= f'RRG_100_k_=all_MIS - CE | $T_0={data_no_annealing["T"]}$')

        plt.legend()
        plt.yscale('log')
        plt.ylabel('rel_error')
        plt.xlabel('number of different node inits')
        plt.axvline(x=8, c='red', linestyle='-.', linewidth=.75)
        plt.show()

    res_dict = {}
    res_dict["N_list"] = Nb_list
    res_dict["n_edges"] = Nb_list
    res_dict["n_nodes"] = Nb_list
    res_dict["anneal"] = {}
    res_dict["anneal"]["rel_error_over_N"] = rel_errors_over_N_anneal
    res_dict["anneal"]["std_rel_error_over_N"] = std_rel_error_anneal
    res_dict["anneal"]["std_rel_error_per_graph"] = rel_error_per_graph_over_N_anneal
    res_dict["no_anneal"] = {}
    res_dict["no_anneal"]["rel_error_over_N"] = rel_errors_over_N_no_anneal
    res_dict["no_anneal"]["std_rel_error_over_N"] = std_rel_error_no_anneal
    res_dict["no_anneal"]["std_rel_error_per_graph"] = rel_error_per_graph_over_N_no_anneal
    return res_dict


if(__name__ == "__main__"):
    load_RRG()
    load_RB_200()