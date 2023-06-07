
import pickle

import matplotlib.pyplot as plt
import numpy as np
from TestScripts import EvalUtils


def evaluate_dataset(paths, mode = "OG", n_perm = 8, Ns = 1, n_test_graphs = 2):
    eval_mode_dict = {}
    eval_mode_dict["mode"] = mode
    eval_mode_dict["n_perm"] = n_perm
    eval_mode_dict["Ns"] = Ns
    eval_mode_dict["n_test_graphs"] = n_test_graphs

    EvalUtils.iterate_over_seeds(paths, eval_mode_dict)

def COLLAB_MVC():
    ### T == 0.05
    Nb = 30
    path1 = ""
    path2 = ""
    path3 = ""
    EvalUtils.iterate_over_seeds([path3])
    pass

def load_COLLAB_MVC():
    ### T == 0.05
    Nb = 30
    path1 = ""
    path2 = ""
    path3 = ""
    ps = [8,16,30]
    for p in ps:
        print("COLLAB MVC p = ", p)
        EvalUtils.load_different_seeds([path1, path2, path3], n_perms=p)

def TWITTER_MVC():
    Nb = 30
    path1 = ""
    path2 = ""
    path3 = ""
    EvalUtils.iterate_over_seeds([path1, path2, path3])
    pass

def load_TWITTER_MVC():
    print("TWITTER RESULTS")
    Nb = 30
    ### TODO replace paths
    path1 = ""
    path2 = ""
    path3 = ""
    ps = [8,16,30]
    for p in ps:
        print("TWITTER MIS p = ", p)
        EvalUtils.load_different_seeds([path1, path2, path3], n_perms=p)

def IMDB_MVC():
    Nb = 30
    path1 = ""
    path2 = ""
    path3 = ""
    EvalUtils.iterate_over_seeds([path1, path2, path3])
    pass

def load_IMDB_MVC():
    path1 = ""
    path2 = ""
    path3 = ""
    ps = [8,16,30]
    for p in ps:
        print("IMDB MVC p = ", p)
        EvalUtils.load_different_seeds([path1, path2, path3], n_perms=p)


def RRG_MIS():
    ### T == 0.05
    Nb = 30
    path1 = ""
    EvalUtils.iterate_over_seeds([path1], n_perm_init = 100, modes = ["perm"])

def MUTAG_MIS():
    Nb = 30
    path1 = ""
    path2 = ""
    path3 = ""
    EvalUtils.iterate_over_seeds([path1, path2, path3])

def load_MUTAG():
    path1 = ""
    path2 = ""
    path3 = ""
    ps = [8, 16, 30]
    for p in ps:
        print("MUTAG MIS p = ", p)
        EvalUtils.load_different_seeds([path1, path2, path3], n_perms=p)

def ENZYMES_MIS():
    Nb = 30
    ### TODO replace paths
    path1 = ""
    path2 = ""
    path3 = ""
    EvalUtils.iterate_over_seeds([path1, path2, path3])
    pass

def load_ENZYMES_MIS():
    print("ENZYMES MIS")
    path1 = ""
    path2 = ""
    path3 = ""
    ps = [8,16,30]
    for p in ps:
        print("ENZMES MIS p = ", p)
        EvalUtils.load_different_seeds([path1, path2, path3], n_perms=p)

def PROTEINS_MIS():
    Nb = 30
    path1 = ""
    path2 = ""
    path3 = ""
    EvalUtils.iterate_over_seeds([ path1, path2, path3])
    pass

def load_PROTEINS_MIS():
    path1 = ""
    path2 = ""
    path3 = ""
    ps = [8, 16, 30]
    for p in ps:
        print("PROTEINS MIS p = ", p)
        EvalUtils.load_different_seeds([path1, path2, path3], n_perms=p)

def COLLAB_MIS():
    Nb = 30
    path1 = ""
    path2 = ""
    path3 = ""
    EvalUtils.iterate_over_seeds([ path1, path2, path3])

def load_COLLAB_MIS():
    print("COLLAB MIS")
    path1 = ""
    path2 = ""
    path3 = ""
    ps = [8, 16, 30]
    for p in ps:
        print("COLLAB MIS p = ", p)
        EvalUtils.load_different_seeds([path1, path2, path3], n_perms=p)

def IMDB_MIS():
    Nb = 30
    path1 = ""
    path2 = ""
    path3 = ""
    EvalUtils.iterate_over_seeds([path1, path2, path3])
    pass

def load_IMDB_MIS():
    print("IMDB MIS")
    path1 = ""
    path2 = ""
    path3 = ""
    ps = [8,16,30]
    for p in ps:
        print("IMDB MIS p = ", p)
        EvalUtils.load_different_seeds([path1, path2, path3], n_perms=p)

def load_general(path, N_list = [1,4,8,16, 32, 64, 100]):
    N = 100
    with open(path + "/log_dict_best_params_perm+Nb_N=100_p_8.pickle", "rb") as f:
        AR_dict = pickle.load(f)


    AR_list = []
    std_AR_list = []
    rel_error_per_N_list = []
    for Nb in N_list:
        print("processing ", Nb)

        ARs_per_graph = []
        AR_nodes = AR_dict["n_nodes"]
        AR_edges = AR_dict["n_edges"]
        AR_pred_Energies = AR_dict["pred_Engery_per_graph"]
        AR_gt_Energies = AR_dict["gt_Energy_per_graph"]

        Hb_Nb_pred_Energy_arr = np.min(AR_pred_Energies, axis=1)  #
        np.min(np.min(AR_pred_Energies, axis=1), axis = -1)
        Hb_Nb_gt_Energy_arr = AR_gt_Energies

        idxs_ = np.arange(0, N)
        np.random.shuffle(idxs_)
        selected_idxs = idxs_[0:Nb]
        H_idxs = np.arange(0, Hb_Nb_pred_Energy_arr.shape[0])

        Hb_Nb_pred_Energy_arr = Hb_Nb_pred_Energy_arr[H_idxs[:, np.newaxis], selected_idxs[np.newaxis, :]]
        best_pred_Energy = np.expand_dims(np.min(Hb_Nb_pred_Energy_arr, axis=-1), axis=-1)

        AR_per_graph = np.abs(best_pred_Energy / Hb_Nb_gt_Energy_arr)
        best_AR = np.mean(AR_per_graph)
        best_std_err_AR = np.std(AR_per_graph) / np.sqrt(Hb_Nb_gt_Energy_arr.shape[0])
        AR_list.append(best_AR)
        std_AR_list.append(best_std_err_AR)
        rel_error_per_N_list.append(np.abs((Hb_Nb_gt_Energy_arr - best_pred_Energy) / Hb_Nb_gt_Energy_arr))

def load_RRG_AR():
    ### TODO add error bars for Mean Field
    path = ""
    p_dict, N_list = EvalUtils.load_Nb_p_data(path = path, params = "best_params")

    greedy_k_list, greedy_rel_error, greedy_std_error, overall_results = load_and_greedy_solve(num_nodes = 100, k = "all")

    AR_greedy_k_list, AR_greedy_rel_error, AR_greedy_std_error, _, AR_nodes, AR_edges = EvalUtils.load_greedy_p(params ="params", path = path, N = 30, p = 100)
    #AR_best_greedy_k_list, AR_best_greedy_rel_error, AR_best_greedy_std_error = get_greedy_sampling_data(params = "_best")

    greedy_AR = np.mean(1 - overall_results)
    std_err_greedy_AR = np.std(1 - overall_results)/np.sqrt(overall_results.shape[0])

    from TestScripts import load_MF_pickles

    res_dict = load_MF_pickles.load_RRG()

    Nbs = res_dict["N_list"]
    mean_anneal = res_dict["anneal"]["rel_error_over_N"]
    std_anneal = res_dict["anneal"]["std_rel_error_over_N"]
    mean_no_anneal = res_dict["no_anneal"]["rel_error_over_N"]
    std_no_anneal = res_dict["no_anneal"]["std_rel_error_over_N"]

    plt.figure()
    plt.title("RRG-100")
    for p in p_dict:
        AR_list = np.array(p_dict[p]["AR_list"])
        n_points = int(AR_list.shape[0]/p)+1
        AR_arr = AR_list[0:n_points]
        std_AR_list = np.array(p_dict[p]["std_AR_list"])[0:n_points]
        p_N_list = p*np.array(N_list)[0:n_points]
        print(AR_list.shape, std_AR_list.shape, p_N_list.shape, n_points)
        #rel_error_per_N_list = p_dict[p]["rel_error_per_N_list"]
        plt.errorbar(p_N_list, AR_arr, yerr=std_AR_list, fmt = "-", label = f"AR + Anneal: n_perm = {p}")
    plt.errorbar(AR_greedy_k_list, AR_greedy_rel_error, yerr=AR_greedy_std_error, fmt="-", label=f"AR + Anneal greedy")
    #plt.errorbar(AR_best_greedy_k_list, AR_best_greedy_rel_error, yerr=AR_best_greedy_std_error, fmt="-", label=f"AR + Anneal best greedy")
    plt.fill_between(N_list, (greedy_AR-std_err_greedy_AR)*np.ones((len(N_list))) , (greedy_AR+std_err_greedy_AR)*np.ones((len(N_list))), alpha = 0.5, color = "r")
    plt.plot(N_list, (greedy_AR)*np.ones((len(N_list))) , "-", label = "DB-Greedy", color = "r")
    plt.plot(Nbs, 1-mean_anneal, "-", label="MFA + Anneal: CE")
    plt.plot(Nbs, 1-mean_no_anneal, "-", label="MFA: CE")
    plt.xlabel("number of states per graph")
    plt.ylabel("Approximation rate")
    plt.xlim(right = 100, left = 0)
    plt.axvline(x=8, c='red', linestyle='-.', linewidth=.75)
    plt.legend()
    plt.show()

    import itertools
    marker = itertools.cycle(('-x', '-+', '-v', '-o', '-*', "-^", "-s"))
    plt.figure()
    plt.title("RRG-100 MIS Dataset")
    for p in p_dict:
        n_points = int(AR_list.shape[0]/p)+1
        AR_list = np.array(p_dict[p]["AR_list"])
        AR_arr = AR_list[0:n_points]
        std_AR_list = np.array(p_dict[p]["std_AR_list"])[0:n_points]
        p_N_list = p*np.array(N_list)[0:n_points]
        #rel_error_per_N_list = p_dict[p]["rel_error_per_N_list"]
        if(p == 1):
            label = fr"VAG-CO: S"
        else:
            label = fr"VAG-CO: OS; $n_O$ = {p}"

        plt.errorbar(p_N_list, 1 - AR_arr, yerr=std_AR_list, fmt = next(marker), label = label)
    plt.errorbar(AR_greedy_k_list, 1- np.array(AR_greedy_rel_error), yerr=AR_greedy_std_error, fmt=next(marker), label=f"VAG-CO: OG")
    #plt.errorbar(AR_best_greedy_k_list, 1- np.array(AR_greedy_rel_error), yerr=AR_best_greedy_std_error, fmt="-", label=f"AR + Anneal perm best greedy")
    plt.errorbar(Nbs, mean_anneal, yerr=std_anneal, fmt = next(marker), label="MFA-Anneal: CE")
    plt.errorbar(Nbs, mean_no_anneal, yerr = std_anneal ,fmt = next(marker), label="MFA: CE")
    plt.fill_between(N_list, 1 - (greedy_AR-std_err_greedy_AR)*np.ones((len(N_list))) , 1 - (greedy_AR+std_err_greedy_AR)*np.ones((len(N_list))), alpha = 0.5, color = 'magenta')
    plt.plot(N_list, 1 - (greedy_AR)*np.ones((len(N_list))) , "-", label = "DB-Greedy", color = 'magenta')
    plt.xlabel(r"$n_S$", fontsize = 22)
    plt.ylabel(r"$\epsilon^*_{rel}$", fontsize = 22)
    plt.legend(loc = 'upper right', fontsize = 11, ncol = 2)
    plt.xlim(right = 100, left = 0)
    plt.axvline(x=8, c='red', linestyle='-.', linewidth=.75)
    plt.yscale("log")
    plt.tight_layout()
    plt.show()

    N = 8-1
    print("mean MF Anneal Energy", 1 - mean_anneal[N],  std_anneal[N])

    print("mean MF no Anneal Energy", 1 -mean_no_anneal[N],  std_anneal[N])

    print("mean DB_Greedy Energy", greedy_AR,  std_err_greedy_AR)

    print("mean AR PG greedy Energy", AR_greedy_rel_error[N],  AR_greedy_std_error[N])

    # plt.figure()
    # plt.title("RRG-100 Dataset")
    # for N, rel_errors in zip(N_list, rel_error_per_N_list):
    #     AR_anneal_k_list, AR_anneal_rel_error, AR_anneal_std_err = degree_plot(AR_nodes, AR_edges, rel_errors)
    #     plt.errorbar(AR_anneal_k_list, AR_anneal_rel_error, yerr = AR_anneal_std_err ,fmt = "x", label = f"AR-Anneal: N = {N}")
    # plt.errorbar(greedy_k_list, greedy_rel_error, yerr = greedy_std_error,  fmt = "x", label = "DB-Greedy")
    # plt.xlabel("degree")
    # plt.ylabel(r"$\epsilon_{\mathrm{rel}}$", fontsize=22)
    # plt.legend()
    # plt.tight_layout()
    # plt.show()


def plot_RRG_figure():
    from matplotlib import pyplot as plt
    from TestScripts.Figures import FigureFonts
    import numpy as np

    np.random.seed(0)
    FigureFonts.init_figure_font()

    AR_path = ""
    with open(AR_path, "rb") as f:
        AR_dict = pickle.load(f)

    AR_nodes = AR_dict["n_nodes"][0,:]
    AR_edges = AR_dict["n_edges"][0,:]
    AR_pred_Energies = np.min(AR_dict["pred_Energy_per_graph"], axis = -2)
    AR_pred_Energies = np.expand_dims(AR_pred_Energies, axis = -1)
    AR_gt_Energies = AR_dict["gt_Energy_per_graph"]
    AR_rel_errors = [ np.abs(gt_e-pred_e)/np.abs(gt_e) for gt_e, pred_e in zip(AR_gt_Energies, AR_pred_Energies)]


    n_nodes, n_edges, rel_errors = load_RRG_MIS_MF()
    MF_k_list, MF_rel_error, MF_std_err = degree_plot(n_nodes, n_edges, rel_errors)

    n_nodes, n_edges, rel_errors = load_RRG_MIS_MF_annealing()
    MF_anneal_k_list, MF_anneal_rel_error, MF_anneal_std_err = degree_plot(n_nodes, n_edges, rel_errors)

    AR_k_list, AR_rel_error, AR_std_error = degree_plot(AR_nodes, AR_edges, AR_rel_errors)

    greedy_k_list, greedy_rel_error, greedy_std_error, overall_results = load_and_greedy_solve(num_nodes = 100, k = "all")

    # params = {'text.usetex': True,
    #           'text.latex.preamble': [r'\usepackage{cmbright}', r'\usepackage{amsmath}']}
    # plt.rcParams.update(params)
    import itertools
    marker = itertools.cycle(('-x', '-+', '-v', '-o', '-*'))
    plt.figure()
    #plt.title("RRG-100 Dataset")
    plt.errorbar(AR_k_list, AR_rel_error, yerr = AR_std_error ,fmt = next(marker), label = "VAG-CO (ours)")
    plt.errorbar(MF_anneal_k_list, MF_anneal_rel_error, yerr = MF_anneal_std_err, fmt = next(marker), label = "MFA-Anneal: CE")
    plt.errorbar(MF_k_list, MF_rel_error, yerr = MF_std_err, fmt = next(marker), label = "MFA: CE")
    plt.errorbar(greedy_k_list, greedy_rel_error, yerr = greedy_std_error,  fmt = next(marker), label = "DB-Greedy")
    plt.xlabel("$d$", fontsize = 30)
    plt.ylabel(r"$\epsilon^*_{\mathrm{rel}}$", fontsize=35)
    plt.legend(fontsize = 16)
    plt.xticks(fontsize = 16)
    plt.yticks(fontsize = 16)
    plt.ylim(top = 0.12)
    plt.tight_layout()
    plt.show()


def degree_plot(AR_nodes, AR_edges, rel_errors):
    res_dict = {}

    for n_node, n_edge, rel_err in zip(AR_nodes, AR_edges, rel_errors):

        k = int(np.round(((n_edge-2*n_node)/n_node)))
        if(k in res_dict.keys()):
            res_dict[k].append(rel_err)
        else:
            res_dict[k] = [rel_err]

    print("keys",res_dict.keys())
    res_dict = dict(sorted(res_dict.items()))
    k_list = [int(k) for k in res_dict.keys()]
    rel_error = [np.mean(np.array(res_dict[k])) for k in res_dict]
    std_error = [np.std(np.array(res_dict[k]))/np.sqrt(len(res_dict[k])) for k in res_dict]

    overall_rel_error = np.mean(np.array(rel_error))
    print("overall_rel_error", overall_rel_error)

    return k_list, rel_error, std_error


def load_RRG_MIS_MF_annealing():
    path = ""
    with open(path, "rb") as f:
        res_dict = pickle.load(f)
        print(res_dict.keys())

    print(res_dict["results"])
    n_nodes = res_dict["results"]["n_nodes"]
    n_edges = res_dict["results"]["n_edges"]
    rel_error = res_dict["results"]["rel_error_CE"]
    return n_nodes, n_edges, rel_error

def load_RRG_MIS_MF():
    path = ""
    with open(path, "rb") as f:
        res_dict = pickle.load(f)
        print(res_dict.keys())

    print(res_dict["results"])
    n_nodes = res_dict["results"]["n_nodes"]
    n_edges = res_dict["results"]["n_edges"]
    rel_error = res_dict["results"]["rel_error_CE"]
    return n_nodes, n_edges, rel_error


def load_and_greedy_solve(num_nodes = 100, k = "all", EnergyFunction = "MIS", mode = "test", seed = 123, parent = True):
    from matplotlib import pyplot as plt
    from GreedyAlgorithms import GreedyMIS
    from unipath import Path
    p = Path(os.getcwd())
    if(parent):
        path = p.parent
    else:
        path = p

    dataset_name = f"RRG_{num_nodes}_k_={k}"
    load_path = path + f"/loadGraphDatasets/DatasetSolutions/no_norm/{dataset_name}/{mode}_{EnergyFunction}_seed_{seed}_solutions.pickle"
    with open(load_path, "rb") as f:
        solution_dict = pickle.load(f)


    res_dict = {}
    print("num_graphs", len(solution_dict["H_graphs"]))
    overall_results = []

    for idx, (H_graph, Energy) in enumerate(zip(solution_dict["H_graphs"], solution_dict["Energies"])):
        n_edge = H_graph.edges.shape[0]
        n_node = H_graph.nodes.shape[0]
        pred_Energy = GreedyMIS.solveMIS(H_graph)
        gt_Energy = Energy
        rel_err = np.abs(gt_Energy-pred_Energy)/np.abs(gt_Energy)
        overall_results.append(rel_err)
        print(idx,"rel_error", rel_err)

        k = int(np.round(((n_edge)/n_node)))
        if(k in res_dict.keys()):
            res_dict[k].append(rel_err)
        else:
            res_dict[k] = [rel_err]

    print("keys",res_dict.keys())
    res_dict = dict(sorted(res_dict.items()))
    k_list = [int(k) for k in res_dict.keys()]
    rel_error = [np.mean(np.array(res_dict[k])) for k in res_dict]
    std_error = [np.std(np.array(res_dict[k]))/np.sqrt(len(res_dict[k])) for k in res_dict]

    plt.figure()
    plt.errorbar(k_list, rel_error, yerr = std_error, fmt = "x")
    plt.show()

    overall_results = np.array(overall_results)
    return k_list, rel_error, std_error, overall_results

def load_and_plot_over_basis_states():
    from matplotlib import pyplot as plt
    path1 = ""
    file_path = path1 + "/log_dict_best_params_normal_N=60.pickle"

    with open(file_path,"rb") as f:
        log_dict = pickle.load(f)

    AR_list = []
    std_AR_list = []
    N = 60
    Nbs = [4,8,16,32,50, 60]
    for Nb in Nbs:
        Hb_Nb_pred_Energy_arr = np.squeeze(log_dict["pred_Energy_arr"], axis=0)  #
        Hb_Nb_gt_Energy_arr = np.squeeze(log_dict["gt_Energy_arr"], axis=0)

        idxs_ = np.arange(0, N)
        np.random.shuffle(idxs_)
        selected_idxs = idxs_[0:Nb]
        H_idxs = np.arange(0, Hb_Nb_pred_Energy_arr.shape[0])

        Hb_Nb_pred_Energy_arr = Hb_Nb_pred_Energy_arr[H_idxs[:, np.newaxis], selected_idxs[np.newaxis, :]]
        best_pred_Energy = np.expand_dims(np.min(Hb_Nb_pred_Energy_arr, axis=-1), axis=-1)

        AR_per_graph = np.abs(best_pred_Energy / Hb_Nb_gt_Energy_arr)
        best_AR = np.mean(AR_per_graph)
        best_std_err_AR = np.std(AR_per_graph) / np.sqrt(Hb_Nb_gt_Energy_arr.shape[0])
        AR_list.append(best_AR)
        std_AR_list.append(best_std_err_AR)

    plt.figure()
    plt.title("COLLAB MVC")
    plt.errorbar(Nbs, np.array(AR_list)-1, yerr=std_AR_list, fmt = "x")
    plt.ylabel("AR")
    plt.xlabel("n basis states")
    plt.yscale("log")
    plt.tight_layout()
    plt.show()

if(__name__ == "__main__"):
    ### TWITTER runs
    import os

    #os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
    os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"#str(args.GPUs[0])

    #MUTAG_MIS()
    # load_IMDB_MVC()
    # load_ENZYMES_MIS()
    # load_IMDB_MIS()
    # load_TWITTER_MVC()
    # load_MUTAG()
    # load_COLLAB_MIS()
    # load_PROTEINS_MIS()
    # load_COLLAB_MVC()
    #plot_RRG_figure()
    #
    load_RRG_AR()
    #load_and_plot_over_basis_states()
    #RRG_MIS()
    #COLLAB_MaxCl()
    #plot_RRG_figure()
    #Collab_MVC()
