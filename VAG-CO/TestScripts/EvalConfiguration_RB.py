
import pickle
import numpy as np
import argparse
from TestScripts import EvalUtils


def load_model(path):
    base_path = path#"/publicdata/sanokows/CombOpt/PPO/SK_deeper_PPOGNN/N_anneal_=_20000/23cjk58i"
    ### todo check what happens if best best weigts are loaded
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


def evaluate_on_data(path, Nb, dataset_name = ""):
    cfg, params, best_params = load_model(path )

    cfg["Ising_params"]["IsingMode"] = dataset_name
    ### todo increase nb to 100
    from trainPPO import trainPPO_configuration

    Model = trainPPO_configuration.HiVNAPPo()
    n_test_graphs = 1
    Nb = 100

    # log_dict = Model.eval_CE(params, cfg, n_test_graphs, Nb = 100)
    #
    # with open(path + f"/log_dict_CE.pickle", "wb") as f:
    #     pickle.dump(log_dict, f)

    #modes = ["normal","perm", "perm+Nb", "perm+beam_search", "beam_search"]
    modes = ["perm"]
    ### TODO fix beam search it does not seem to work corectly
    ### TODO look at probs of normal sampling vs probs in beam search
    for mode in modes:
        # log_dict = Model.eval_on_testdata(params, cfg, n_test_graphs, padding_factor = 1.1, n_perm = n_perm, Nb = Nb, mode = mode)
        #
        # with open(path + f"/log_dict_params_{mode}_{dataset_name}_N={Nb}_p_{n_perm}.pickle", "wb") as f:
        #     pickle.dump(log_dict, f)
        if(mode == "perm"):
            n_perm = 100
            Nb = None
        else:
            n_perm = 8


        log_dict_best = Model.eval_on_testdata(best_params, cfg, n_test_graphs, padding_factor = 1.1, n_perm = n_perm, Nb = Nb, mode = mode)

        with open(path + f"/log_dict_best_params_{mode}_{dataset_name}_N={Nb}_p_{n_perm}.pickle", "wb") as f:
            pickle.dump(log_dict_best, f)



    return log_dict_best

def RB_100_MVC():
    base_path = ""

    Nb = 30
    ### T = 0.05
    ### TODO continue 16k run
    id_paths_large_T = ["N_anneal_=_1000/rf1mk8x1", "N_anneal_=_2000/nyhrfm46", "N_anneal_=_4000/3x36hh4x", "N_anneal_=_8000/89yg2fss"]
    for id_path in id_paths_large_T:
        path1 = base_path + id_path
        calc_and_save([path1], Nb)

    ### T = 0.01
    if(False):
        id_paths_small_T = ["N_anneal_=_1000/r5zitn7s", "N_anneal_=_2000/4lqoroe4"]
        for id_path in id_paths_small_T:
            path1 = base_path + id_path
            calc_and_save([path1], Nb)

def RB_MVC():
    ### T == 0.05
    Nb = 30
    path1 = ""
    calc_and_save([path1], Nb)


def calc_and_save(path_list, Nb):
    from unipath import Path
    APR_dict = {}
    APR_dict["APR"] = []
    APR_dict["best_APR"] = []
    paths = path_list
    ps = np.linspace(0.25, 1, num=10)
    for path in paths:
        for p in ps:
            dataset_name = f"RB_iid_200_p_{p}"
            log_dict = evaluate_on_data(path, Nb = Nb, dataset_name = dataset_name)
            APR_dict["APR"].append(log_dict["APR"])
            APR_dict["best_APR"].append(log_dict["best_APR"])

    p = Path(path)
    path = p.parent

    save_path = path + "/APR_dict.pickle"
    file = open(save_path, 'wb')
    pickle.dump(APR_dict, file)

    best_APR_arr = np.array(APR_dict["best_APR"])
    APR_arr = np.array(APR_dict["APR"])

    print("best_APR", np.mean(best_APR_arr), np.std(best_APR_arr) / np.sqrt(len(path_list)))
    print("APR", np.mean(APR_arr), np.std(APR_arr) / np.sqrt(len(path_list)))

def load_and_greedy_solve(num_nodes = 200, EnergyFunction = "MVC", mode = "test", seed = 123, parent = True):
    from matplotlib import pyplot as plt
    import pickle
    from unipath import Path
    import os
    from GreedyAlgorithms import GreedyMIS
    p = Path(os.getcwd())
    if(parent):
        path = p.parent
    else:
        path = p

    res_dict = {}
    ps = np.linspace(0.25, 1, num = 10)
    overall_rel_error_list = []
    for p in ps:
        dataset_name = f"RB_iid_{num_nodes}_p_{p}"
        load_path = path + f"/loadGraphDatasets/DatasetSolutions/no_norm/{dataset_name}/{mode}_{EnergyFunction}_seed_{seed}_solutions.pickle"
        with open(load_path, "rb") as f:
            solution_dict = pickle.load(f)

        print("num_graphs", len(solution_dict["H_graphs"]))

        for idx, (H_graph, Energy) in enumerate(zip(solution_dict["H_graphs"], solution_dict["Energies"])):
            pred_Energy = H_graph.nodes.shape[0] + GreedyMIS.solveMIS(H_graph)
            gt_Energy = Energy
            rel_err = np.abs(pred_Energy)/np.abs(gt_Energy)
            print(idx,"rel_error", rel_err)
            overall_rel_error_list.append(rel_err)
            k = p
            if(k in res_dict.keys()):
                res_dict[k].append(rel_err)
            else:
                res_dict[k] = [rel_err]

    overall_rel_error_arr = np.array(overall_rel_error_list)
    overall_rel_error = np.mean(overall_rel_error_arr)
    overall_std_rel_error = np.std(overall_rel_error_arr)/np.sqrt(overall_rel_error_arr.shape[0])
    print("overall rel error", overall_rel_error, overall_std_rel_error)
    print("keys",res_dict.keys())
    k_list = [float(k) for k in res_dict.keys()]
    rel_error = [np.mean(np.array(res_dict[k])) for k in res_dict]
    std_rel_error = [np.std(np.array(res_dict[k]))/(np.sqrt(len(res_dict[k]))) for k in res_dict]

    return k_list, rel_error, std_rel_error, overall_rel_error, overall_std_rel_error

def load_model_results(mode = "normal", params = "params", N = 100, N_list = [4,8,16, 32, 50, 75, 100]):
    ps = np.linspace(0.25, 1, num=10)
    path = ""

    res_dict = {}
    for Nb in N_list:
        AR_list = []
        std_AR_list = []
        p_list = []

        ARs_per_graph = []

        for p in ps:
            try:
                dataset_name = f"RB_iid_200_p_{p}"

                with open(path + f"/log_dict_{params}_{mode}_{dataset_name}_N={N}.pickle","rb") as f:
                    log_dict = pickle.load(f)
                    print(log_dict.keys())

                Hb_Nb_pred_Energy_arr = np.squeeze(log_dict["pred_Energy_arr"],axis = 0)#
                Hb_Nb_gt_Energy_arr = np.squeeze(log_dict["gt_Energy_arr"], axis = 0)

                idxs_ = np.arange(0, N)
                np.random.shuffle(idxs_)
                selected_idxs = idxs_[0:Nb]
                H_idxs = np.arange(0, Hb_Nb_pred_Energy_arr.shape[0])

                Hb_Nb_pred_Energy_arr = Hb_Nb_pred_Energy_arr[H_idxs[:,np.newaxis], selected_idxs[np.newaxis,:]]
                best_pred_Energy = np.expand_dims(np.min(Hb_Nb_pred_Energy_arr, axis = -1), axis = -1)

                AR_per_graph = np.abs(best_pred_Energy/Hb_Nb_gt_Energy_arr)
                best_AR = np.mean(AR_per_graph)
                best_std_err_AR = np.std(AR_per_graph)/np.sqrt(Hb_Nb_gt_Energy_arr.shape[0])
                AR_list.append(best_AR)
                std_AR_list.append(best_std_err_AR)
                p_list.append(p)

                ARs_per_graph.append(AR_per_graph)
            except:
                print("loading failed", Nb, p)
        res_dict[Nb] = {}
        res_dict[Nb]["p_list"] = p_list
        res_dict[Nb]["AR_list"] = AR_list
        res_dict[Nb]["std_AR_list"] = std_AR_list
        res_dict[Nb]["AR_per_graph_list"] = ARs_per_graph

    return res_dict

def load_model_results_greedy():
    ps = np.linspace(0.25, 1, num=10)
    path = ""
    greedy_params = "best_params"
    AR_over_p = []
    std_AR_over_p = []
    for p in ps:

        dataset_name = f"RB_iid_200_p_{p}"
        # perms, AR_list, std_AR_list, AR_per_graph_list, _,_ = EvalUtils.load_greedy_p(path=path, p=8, N=None,
        #                                                                          add=f"_{dataset_name}",
        #                                                                          params=greedy_params)
        # print("load 8 perms now", p)
        # perms, AR_list, std_AR_list, AR_per_graph_list, _, _ = EvalUtils.load_greedy_p(path=path, p=8, N=100,
        #                                                                          add=f"_{dataset_name}",
        #                                                                          params=greedy_params)
        try:
            perms, AR_list, std_AR_list, AR_per_graph_list, _, _ = EvalUtils.load_greedy_p(path = path, p = 8, N = 100, add = f"_{dataset_name}", params = greedy_params)
            print("load 8 perms now", p)
        except:
            perms, AR_list, std_AR_list, AR_per_graph_list, _ ,_ = EvalUtils.load_greedy_p(path=path, p=50, N=100,
                                                                                     add=f"_{dataset_name}",
                                                                                     params=greedy_params)
            print("load 50 perms now", p)

        AR_over_p.append(AR_list[-1])
        std_AR_over_p.append(std_AR_list[-1])

    return ps, AR_over_p, std_AR_over_p


def plot_RB_200_MVC_over_Nb(path = "", params = "best_params"):
    ### TODO add MF
    ps = np.linspace(0.25, 1, num=10)
    path = ""


    AR_per_graph_for_all_ps = {}
    for p in ps:
        try:
            dataset_name = f"RB_iid_200_p_{p}"
            p_dict, N_list = EvalUtils.load_Nb_p_data(path = path, p = 8, N = 100, add = f"_{dataset_name}", params = params)

            for key in p_dict:
                if (key not in AR_per_graph_for_all_ps):
                    AR_per_graph_for_all_ps[key] = []

                AR_per_graph_over_N = np.array(p_dict[key]["AR_per_graph_list"])
                AR_per_graph_for_all_ps[key].append(AR_per_graph_over_N)
        except:
            print("loading failed", p)

    ARs_over_N_over_perm = {}
    for perm_key in AR_per_graph_for_all_ps:
        ARs_over_N_over_perm[perm_key] = {}
        ### (Ns, graphs)
        res = np.concatenate(AR_per_graph_for_all_ps[perm_key] ,axis = -2)
        ARs_over_N_over_perm[perm_key]["Ns"] = perm_key * np.array(N_list)
        ARs_over_N_over_perm[perm_key]["AR_mean"] = np.squeeze(np.mean(res, axis = -2), axis = -1)
        ARs_over_N_over_perm[perm_key]["AR_std"] = np.squeeze(np.std(res, axis=-2), axis = -1)/( np.sqrt(res.shape[-2]))

    greedy_params = "best_params"
    AR_per_graph_for_all_ps_greedy = {}
    for p in ps:

        dataset_name = f"RB_iid_200_p_{p}"
        # try:
        #     perms, AR_list, std_AR_list, AR_per_graph_list, _, _ = EvalUtils.load_greedy_p(path = path, p = 8, N = 100, add = f"_{dataset_name}", params = greedy_params)
        #     print("load 8 perms now", p)
        # except:
        #     perms, AR_list, std_AR_list, AR_per_graph_list, _, _ = EvalUtils.load_greedy_p(path=path, p=50, N=100,
        #                                                                              add=f"_{dataset_name}",
        #                                                                              params=greedy_params)
        #     print("load 50 perms now", p)
        perms, AR_list, std_AR_list, AR_per_graph_list, _, _ = EvalUtils.load_greedy_p(path=path, p=100, N=None,
                                                                                       add=f"_{dataset_name}",
                                                                                       params=greedy_params)
        #     print("load 8 perms now", p)

        for perm_idx, key in enumerate(perms):
            if (key not in AR_per_graph_for_all_ps_greedy):
                AR_per_graph_for_all_ps_greedy[key] = []

            AR_per_graph_over_N = np.array(AR_per_graph_list[perm_idx])
            AR_per_graph_for_all_ps_greedy[key].append(AR_per_graph_over_N)


    ARs_over_N_over_perm_greedy = {}
    for perm_key in AR_per_graph_for_all_ps_greedy:
        ARs_over_N_over_perm_greedy[perm_key] = {}
        ### (Ns, graphs)
        res = np.concatenate(AR_per_graph_for_all_ps_greedy[perm_key] ,axis = -2)
        ARs_over_N_over_perm_greedy[perm_key]["Ns"] = perm_key * np.array(N_list)
        ARs_over_N_over_perm_greedy[perm_key]["AR_mean"] = np.squeeze(np.mean(res, axis = -2), axis = -1)
        ARs_over_N_over_perm_greedy[perm_key]["AR_std"] = np.squeeze(np.std(res, axis=-2), axis = -1)/( np.sqrt(res.shape[-2]))

    from TestScripts import load_MF_pickles
    DB_ps, DB_AR, std_AR,overall_AR, overall_std_AR = load_and_greedy_solve()
    data_dict = load_MF_pickles.load_RB_200()

    from matplotlib import pyplot as plt
    import itertools
    marker = itertools.cycle(('-x', '-+', '-v', '-o', '-*', "-^", "-s"))
    plt.figure()
    plt.title(f"RB-200 MVC Dataset")
    for perm_key in ARs_over_N_over_perm:
        n_points = int(ARs_over_N_over_perm[perm_key]["Ns"].shape[0]/perm_key)+1
        AR_arr = ARs_over_N_over_perm[perm_key]["AR_mean"][0:n_points]
        std_AR_list = ARs_over_N_over_perm[perm_key]["AR_std"][0:n_points]
        p_N_list = ARs_over_N_over_perm[perm_key]["Ns"][0:n_points]

        if(perm_key == 1):
            label = fr"VAG-CO: S"
        else:
            label = fr"VAG-CO: OS; $n_O$ = {perm_key}"

        plt.errorbar(p_N_list, AR_arr - 1,
                     yerr = std_AR_list, label = label, fmt = next(marker))

    greedy_AR_arr = np.array([ ARs_over_N_over_perm_greedy[key]["AR_mean"] for key in ARs_over_N_over_perm_greedy])
    greedy_AR_arr_std = np.array([ARs_over_N_over_perm_greedy[key]["AR_std"] for key in ARs_over_N_over_perm_greedy])
    plt.errorbar(perms, greedy_AR_arr - 1,
                 yerr = greedy_AR_arr_std, label = f"VAG-CO: OG ", fmt = next(marker))
    plt.errorbar(data_dict["N_list"], data_dict["anneal"]["mean_over_N"],
                 yerr = data_dict["anneal"]["std_over_N"], label = f"MFA-Anneal: CE", fmt = next(marker))
    plt.errorbar(data_dict["N_list"], data_dict["no_anneal"]["mean_over_N"],
                 yerr = data_dict["no_anneal"]["std_over_N"], label = f"MFA: CE  ", fmt = next(marker))
    N_list = np.arange(0,100)
    plt.fill_between( N_list, ( overall_AR - 1 - overall_std_AR)*np.ones_like(N_list), (overall_AR -1  + overall_std_AR)*np.ones_like(N_list), alpha = 0.3, color = 'magenta')
    plt.plot( N_list, ( overall_AR - 1)*np.ones_like(N_list), "-", label = "DB-Greedy", alpha = 1., color = 'magenta')
    plt.yscale("log")
    plt.legend(loc = "upper right", fontsize = 11, ncol = 2)
    plt.xlabel(r"$n_S$", fontsize = 22)
    #plt.ylabel(r"$\epsilon_\mathrm{rel}$")
    plt.xlim(0, 100)
    plt.axvline(x=8, c='red', linestyle='-.', linewidth=.75)
    plt.tight_layout()
    plt.show()

    N = 8-1
    print("mean MF Anneal Energy", data_dict["anneal"]["mean_over_N"][N],  data_dict["anneal"]["std_over_N"][N])

    print("mean MF no Anneal Energy", data_dict["no_anneal"]["mean_over_N"][N],  data_dict["no_anneal"]["std_over_N"][N])

    print("mean DB_Greedy Energy", overall_AR,  overall_std_AR)

    print("mean AR PG greedy Energy", greedy_AR_arr[N],  greedy_AR_arr_std[N])



def load_MF(wandb_id = "k8rdh96k", Nb = ""):
    path = ""
    ps = np.linspace(0.25, 1, num=10)
    mean_rel_error_list = []
    std_rel_error_list = []
    for p in ps:
        if(Nb == ""):
            file = path + f"/{wandb_id}_RB_iid_200_p_{p}.pickle"
        else:
            file = path + f"/{Nb}_{wandb_id}_RB_iid_200_p_{p}.pickle"

        with open(file, "rb") as f:
            sol_dict = pickle.load(f)
            print(sol_dict.keys())
        mean_rel_error = np.mean(sol_dict["results"]["APR_CE"])
        std_err_rel_error = np.std(sol_dict["results"]["APR_CE"])/(np.sqrt(sol_dict["results"]["APR_CE"].shape[0]))
        mean_rel_error_list.append(mean_rel_error)
        std_rel_error_list.append(std_err_rel_error)
    return ps, mean_rel_error_list, std_rel_error_list

def plot_model_vs_greedy():
    from matplotlib import pyplot as plt
    from TestScripts import load_MF_pickles
    import numpy as np

    np.random.seed(0)
    AR_res_dict_best = load_model_results(params = "params")
    #AR_res_dict = load_model_results(params = "best_params")

    p_list, AR_over_p, std_AR_over_p = load_model_results_greedy()

    # MFAnneal_ps, MFAnneal_rel_error, MFAnneal_std_error = load_MF(wandb_id="4pr5dly3", Nb = 8)
    # MF_ps, MF_rel_error, MF_std_error = load_MF()

    data_dict = load_MF_pickles.load_RB_200()
    MFAnneal_ps, MFAnneal_rel_error, MFAnneal_std_error = data_dict["anneal"]["ps"], data_dict["anneal"]["rel_error_over_p"], data_dict["anneal"]["std_rel_error_over_p"]
    MF_ps, MF_rel_error, MF_std_error = data_dict["no_anneal"]["ps"], data_dict["no_anneal"]["rel_error_over_p"], data_dict["no_anneal"]["std_rel_error_over_p"]

    DB_ps, DB_AR, std_rel_error,overall_rel_error, overall_std_rel_error = load_and_greedy_solve()

    from TestScripts.Figures import FigureFonts

    FigureFonts.init_figure_font()

    import itertools
    marker = itertools.cycle(('-x', '-+', '-v', '-o', '-*'))

    plt.figure()
    plt.errorbar(p_list, AR_over_p , yerr=std_AR_over_p, fmt = next(marker), label = f"VAG-CO (ours)")
    plt.errorbar(MFAnneal_ps, 1 + MFAnneal_rel_error, yerr=MFAnneal_std_error, fmt = next(marker), label = "MFA-Anneal: CE")
    plt.errorbar(MF_ps, 1 + MF_rel_error, yerr=MF_std_error, fmt = next(marker), label = "MFA: CE")
    # for key in AR_res_dict:
    #     AR_ps = AR_res_dict[key]["p_list"]
    #     model_AR_list = AR_res_dict[key]["AR_list"]
    #     std_AR_list = AR_res_dict[key]["std_AR_list"]
    #     plt.errorbar(AR_ps, model_AR_list , yerr=std_AR_list, fmt = "x", label = f"AR + Anneal checkpoint best mean; Nb = {key}")

    plt.errorbar(DB_ps, np.array(DB_AR), yerr=std_rel_error, fmt=next(marker), label="DB-Greedy")
    plt.legend(fontsize = 16, loc = "upper right")
    plt.xticks(fontsize = 16)
    plt.yticks(fontsize = 16)
    plt.ylim(top = 1.022)
    plt.xlabel(r"$p$", fontsize=30)
    plt.ylabel(r"$AR^*$", fontsize=24)
    plt.tight_layout()
    plt.show()


    plt.figure()
    plt.errorbar(DB_ps, DB_AR, yerr=std_rel_error, fmt = "x", label = "DB-Greedy")
    plt.errorbar(MFAnneal_ps, 1 + MFAnneal_rel_error, yerr=MFAnneal_std_error, fmt = "x", label = "MF + Anneal Nb = 8")
    plt.errorbar(MF_ps, 1 + MF_rel_error, yerr=MF_std_error, fmt = "x", label = "MF Nb = 8")

    # for key in AR_res_dict:
    #     AR_ps = AR_res_dict[key]["p_list"]
    #     model_AR_list = AR_res_dict[key]["AR_list"]
    #     std_AR_list = AR_res_dict[key]["std_AR_list"]
    #     plt.errorbar(AR_ps, model_AR_list , yerr=std_AR_list, fmt = "x", label = f"AR + Anneal checkpoint best mean; Nb = {key}")

    for key in AR_res_dict_best:
        AR_ps = AR_res_dict_best[key]["p_list"]
        model_AR_list = AR_res_dict_best[key]["AR_list"]
        std_AR_list = AR_res_dict_best[key]["std_AR_list"]
        plt.errorbar(AR_ps, model_AR_list , yerr=std_AR_list, fmt = "x", label = f"AR + Anneal bs; Nb = {key}")
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2),
              ncol=3, fancybox=True, shadow=True)
    plt.xlabel("p")
    plt.ylabel("AR")
    plt.tight_layout()
    plt.show()

    plt.figure()
    plt.title("normal sampling")
    for key in AR_res_dict_best:
        if(key == 8):
            AR_ps = AR_res_dict_best[key]["p_list"]
            model_AR_list = AR_res_dict_best[key]["AR_list"]
            std_AR_list = AR_res_dict_best[key]["std_AR_list"]
            plt.errorbar(AR_ps, model_AR_list , yerr=std_AR_list, fmt = "x", label = f"AR-Anneal; Nb = {key}")
    plt.errorbar(MFAnneal_ps, 1 + MFAnneal_rel_error, yerr=MFAnneal_std_error, fmt = "x", label = "MF-Anneal Nb = 8")
    plt.errorbar(MF_ps, 1 + MF_rel_error, yerr=MF_std_error, fmt = "x", label = "MF Nb = 8")
    # for key in AR_res_dict:
    #     AR_ps = AR_res_dict[key]["p_list"]
    #     model_AR_list = AR_res_dict[key]["AR_list"]
    #     std_AR_list = AR_res_dict[key]["std_AR_list"]
    #     plt.errorbar(AR_ps, model_AR_list , yerr=std_AR_list, fmt = "x", label = f"AR + Anneal checkpoint best mean; Nb = {key}")

    plt.errorbar(DB_ps, np.array(DB_AR), yerr=std_rel_error, fmt="x", label="DB-Greedy")
    plt.legend()
    plt.xlabel("p", fontsize=15)
    plt.ylabel(r"Approximation Ratio", fontsize=15)
    plt.tight_layout()
    plt.show()




if(__name__ == "__main__"):
    ### TWITTER runs
    import os

    #os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
    os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"#str(args.GPUs[0])
    #TWITTER_MaxCl()
    #plot_RB_200_MVC_over_Nb(params="params")
    plot_RB_200_MVC_over_Nb(params="best_params")
    #plot_model_vs_greedy()
    #RB_MVC()
    #COLLAB_MaxCl()
    #plot_RRG_figure()
    #Collab_MVC()
