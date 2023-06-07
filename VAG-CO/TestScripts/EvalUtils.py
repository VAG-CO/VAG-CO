import pickle

import matplotlib.pyplot as plt
import numpy as np
import argparse


def load_model(path):
    base_path = path#"/publicdata/sanokows/CombOpt/PPO/SK_deeper_PPOGNN/N_anneal_=_20000/23cjk58i"

    param_path = base_path + "/best_val_rel_error_weights.pickle"
    best_param_path = base_path + "/best_val_best_rel_error_weights.pickle"
    config_path = base_path + "/config.pickle"

    file = open(param_path, 'rb')
    params = pickle.load(file)

    file = open(best_param_path, 'rb')
    best_param_path = pickle.load(file)

    file = open(config_path, 'rb')
    config = pickle.load(file)
    return config, params, best_param_path

def load_model_weights(path):
    base_path = path#"/publicdata/sanokows/CombOpt/PPO/SK_deeper_PPOGNN/N_anneal_=_20000/23cjk58i"

    param_path = base_path + "/best_val_rel_error_weights.pickle"
    config_path = base_path + "/config.pickle"

    file = open(param_path, 'rb')
    params = pickle.load(file)

    file = open(config_path, 'rb')
    config = pickle.load(file)
    return config, params

def evaluate_on_data(path, eval_mode_dict):
    cfg, params = load_model_weights(path)

    from trainPPO import trainPPO_configuration

    Model = trainPPO_configuration.HiVNAPPo()

    n_test_graphs = eval_mode_dict["n_test_graphs"]
    Ns = eval_mode_dict["Ns"]

    mode = eval_mode_dict["mode"]

    if (mode == "OG"):
        n_perm = eval_mode_dict["n_perm"]
    else:
        n_perm = 1

    log_dict = Model.eval_on_testdata(params, cfg, n_test_graphs, padding_factor = 1.1, n_perm = n_perm, Nb = Ns, mode = mode)

    save_path = path + f"/log_dict_params_{mode}_N={Ns}_p_{n_perm}.pickle"
    with open(path + f"/log_dict_params_{mode}_N={Ns}_p_{n_perm}.pickle", "wb") as f:
        pickle.dump(log_dict, f)

    calc_AR(save_path, n_perms = n_perm)

        # log_dict_best = Model.eval_on_testdata(best_params, cfg, n_test_graphs, padding_factor=1.1, n_perm=n_perm,
        #                                        Nb=Nb, mode=mode)
        #
        # with open(path + f"/log_dict_best_params_{mode}_N={Nb}_p_{n_perm}.pickle", "wb") as f:
        #     pickle.dump(log_dict_best, f)

    return log_dict

def calc_AR(path, n_perms = 8, params = "params"):
    AR_over_seeds = []

    save_path = path
    with open(save_path, 'rb') as file:
        res_dict = pickle.load( file)

    pred_energy = res_dict['pred_Energy_per_graph']

    gt_Energy = res_dict['gt_Energy_per_graph']

    min_pred_energy = np.min(pred_energy[:,0:n_perms, 0], axis = -1)
    min_pred_energy = np.expand_dims(min_pred_energy, axis = -1)
    AR_per_graph = min_pred_energy/gt_Energy
    mean_AR = np.mean(AR_per_graph)
    AR_over_seeds.append(mean_AR)

    AR_over_seeds = np.array(AR_over_seeds)
    mean_AR = np.mean(AR_over_seeds)
    std_AR = np.std(AR_over_seeds)/np.sqrt(AR_over_seeds.shape[0])
    print("permutation Results", params)
    print("mean AR", mean_AR)
    print("std_AR", std_AR)
    pass

def load_different_seeds(path_list, n_perms = 8, params = "params"):
    from unipath import Path
    overall_logs = []
    mode = "perm"
    N = 30
    n_perm = 30
    AR_over_seeds = []
    paths = path_list
    for path in paths:
        save_path = path + f"/log_dict_{params}_{mode}_N={N}_p_{n_perm}.pickle"
        with open(save_path, 'rb') as file:
            res_dict = pickle.load( file)

        try:
            pred_energy = res_dict['pred_Engery_per_graph']
        except:
            pred_energy = res_dict['pred_Energy_per_graph']

        gt_Energy = res_dict['gt_Energy_per_graph']

        min_pred_energy = np.min(pred_energy[:,0:n_perms, 0], axis = -1)
        min_pred_energy = np.expand_dims(min_pred_energy, axis = -1)
        AR_per_graph = min_pred_energy/gt_Energy
        mean_AR = np.mean(AR_per_graph)
        AR_over_seeds.append(mean_AR)

    AR_over_seeds = np.array(AR_over_seeds)
    mean_AR = np.mean(AR_over_seeds)
    std_AR = np.std(AR_over_seeds)/np.sqrt(AR_over_seeds.shape[0])
    print("permutation Results", params)
    print("mean AR", mean_AR)
    print("std_AR", std_AR)

    mode = "perm+Nb"
    n_perm = 1
    AR_over_seeds = []
    for path in paths:
        save_path = path + f"/log_dict_{params}_{mode}_N={N}_p_{n_perm}.pickle"
        with open(save_path, 'rb') as file:
            res_dict = pickle.load( file)

        try:
            pred_energy = res_dict['pred_Engery_per_graph']
        except:
            pred_energy = res_dict['pred_Energy_per_graph']

        gt_Energy = res_dict['gt_Energy_per_graph']


        AR_per_graph = pred_energy[:,0, :]/gt_Energy
        mean_AR = np.mean(AR_per_graph)
        AR_over_seeds.append(mean_AR)

    AR_over_seeds = np.array(AR_over_seeds)
    mean_AR = np.mean(AR_over_seeds)
    std_AR = np.std(AR_over_seeds)/np.sqrt(AR_over_seeds.shape[0])
    print("mean results", params)
    print("mean AR", mean_AR)
    print("std_AR", std_AR)
    pass



def iterate_over_seeds(path_list, eval_mode_dict):
    from unipath import Path
    overall_logs = []
    paths = path_list
    for path in paths:
        log_dict = evaluate_on_data(path, eval_mode_dict)
        overall_logs.append(log_dict)

    p = Path(path)
    path = p.parent

    save_path = path + "/overall_log_dict.pickle"
    file = open(save_path, 'wb')
    pickle.dump(overall_logs, file)


def load_Nb_p_data(N = 100, p = 8, path = "", params = "best_params", add = ""):
    ### TODO add p + greedy
    N_list = np.arange(1, N+1)
    path = path + f"/log_dict_{params}_perm+Nb{add}_N={N}_p_{p}.pickle"
    with open(path, "rb") as f:
        AR_dict = pickle.load(f)

    ps = [1, 2, 8]

    p_dict = {}
    for p in ps:
        p_dict[p] = {}
        AR_list = []
        std_AR_list = []
        rel_error_per_N_list = []
        AR_per_graph_list = []
        for Nb in N_list:
            print("processing ", Nb)

            ARs_per_graph = []
            AR_nodes = AR_dict["n_nodes"]
            AR_edges = AR_dict["n_edges"]
            AR_pred_Energies = AR_dict["pred_Engery_per_graph"][:, 0:p, :]
            AR_gt_Energies = AR_dict["gt_Energy_per_graph"]

            Hb_Nb_pred_Energy_arr = AR_pred_Energies  #
            Hb_Nb_gt_Energy_arr = AR_gt_Energies

            idxs_ = np.arange(0, N)
            np.random.shuffle(idxs_)
            selected_idxs = idxs_[0:Nb]
            H_idxs = np.arange(0, Hb_Nb_pred_Energy_arr.shape[0])

            Hb_Nb_pred_Energy_arr = Hb_Nb_pred_Energy_arr[H_idxs[:, np.newaxis], :, selected_idxs[np.newaxis, :]]
            Hb_Nb_pred_Energy_arr = np.reshape(Hb_Nb_pred_Energy_arr, (
            Hb_Nb_pred_Energy_arr.shape[0], Hb_Nb_pred_Energy_arr.shape[1] * Hb_Nb_pred_Energy_arr.shape[2]))
            best_pred_Energy = np.expand_dims(np.min(Hb_Nb_pred_Energy_arr, axis=-1), axis=-1)

            AR_per_graph = np.abs(best_pred_Energy / Hb_Nb_gt_Energy_arr)
            AR_per_graph_list.append(AR_per_graph)
            best_AR = np.mean(AR_per_graph)
            best_std_err_AR = np.std(AR_per_graph) / np.sqrt(Hb_Nb_gt_Energy_arr.shape[0])
            AR_list.append(best_AR)
            std_AR_list.append(best_std_err_AR)
            rel_error_per_N_list.append(np.abs((Hb_Nb_gt_Energy_arr - best_pred_Energy) / Hb_Nb_gt_Energy_arr))

        p_dict[p]["AR_list"] = AR_list
        p_dict[p]["std_AR_list"] = std_AR_list
        p_dict[p]["rel_error_per_N_list"] = rel_error_per_N_list
        p_dict[p]["AR_per_graph_list"] = AR_per_graph_list

    return p_dict, N_list


def load_greedy_p(N=100, p=8, path="", params="best_params", add = ""):
    ### TODO add p + greedy

    path = path + f"/log_dict_{params}_perm{add}_N={N}_p_{p}.pickle"
    with open(path, "rb") as f:
        AR_dict = pickle.load(f)

    ps = np.arange(1,p + 1)

    AR_list = []
    std_AR_list = []
    rel_error_per_N_list = []
    AR_per_graph_list = []

    for p in ps:
        AR_nodes = AR_dict["n_nodes"]
        AR_edges = AR_dict["n_edges"]
        try:
            AR_pred_Energies = AR_dict["pred_Engery_per_graph"]
        except:
            AR_pred_Energies = AR_dict["pred_Energy_per_graph"]
        AR_gt_Energies = AR_dict["gt_Energy_per_graph"]

        Hb_Nb_pred_Energy_arr = AR_pred_Energies  #
        Hb_Nb_gt_Energy_arr = AR_gt_Energies

        Hb_Nb_pred_Energy_arr = Hb_Nb_pred_Energy_arr[:, 0:p, 0]
        best_pred_Energy = np.expand_dims(np.min(Hb_Nb_pred_Energy_arr, axis=-1), axis=-1)

        AR_per_graph = np.abs(best_pred_Energy / Hb_Nb_gt_Energy_arr)
        best_AR = np.mean(AR_per_graph)
        best_std_err_AR = np.std(AR_per_graph) / np.sqrt(Hb_Nb_gt_Energy_arr.shape[0])

        AR_per_graph_list.append(AR_per_graph)
        AR_list.append(best_AR)
        std_AR_list.append(best_std_err_AR)
        rel_error_per_N_list.append(np.abs((Hb_Nb_gt_Energy_arr - best_pred_Energy) / Hb_Nb_gt_Energy_arr))

    return ps, AR_list, std_AR_list, AR_per_graph_list, AR_nodes, AR_edges