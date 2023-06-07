from loadGraphDatasets import solveTUDatasets, saveHamiltonianGraphs, GenerateRB_graphs, GenerateRRGs

import argparse

TU_datasets = ["MUTAG", "TWITTER", "ENZYMES", "PROTEINS", "IMDB-BINARY"]
RB_datasets = ["RB_iid_200", "RB_iid_100"]
RRG_datasets = ["RRG_100"]
dataset_choices = TU_datasets + RB_datasets + RRG_datasets
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='MUTAG', choices = dataset_choices, help='Define the dataset')
parser.add_argument('--problem', default='MIS', choices = ["MIS", "MVC"], help='Define the CO problem')
args = parser.parse_args()

### To be updated so that datasets can be prepared in a simpler way
### TODO add seeds as an argument

if(__name__ == "__main__"):
    ### TODO first set up your gurobi licence before running this code. Otherwise large CO Problem Instances cannot be solved by gurobi!

    if(args.dataset in TU_datasets):
        print("Solving dataset with gurobi")
        solveTUDatasets.solve_datasets(args.dataset, args.problem, parent = False)
        print("Translating dataet into spin formulation")
        saveHamiltonianGraphs.solve(args.dataset, args.problem)
    elif(args.dataset in RB_datasets):
        ### TODO add size as an argument
        if(args.dataset == "RB_iid_100"):
            sizes = ["100"]
        else:
            sizes = ["200"]
        GenerateRB_graphs.create_and_solve_graphs_MVC(parent = False, sizes = sizes)
        saveHamiltonianGraphs.solve(args.dataset, args.problem)
    elif(args.dataset in RRG_datasets):
        GenerateRRGs.make_dataset(parent = False)
        saveHamiltonianGraphs.solve(args.dataset, args.problem)
    else:
        ValueError("Dataset is not defined")

    pass