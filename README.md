# Variational Annealing on Graphs for Combinatorial Optimization

This repository is the official implementation of "Variational Annealing". 

The folder "/VAG-CO" contains code with the VAG-CO implementation and also contains code that solves CO problem instances on various datasets.
The folder "/meanfield_annealing" contains code with the Mean Field Approximation implementation.

Within the folder /VAG-CO do the following:
## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

Get an academic gurobi licence at https://www.gurobi.com/.
This is a neccesary step to run prepare_datasets.py, which uses gurobi to obtain optimal results for CO problem instances.

## Create Datasets

run prepare_datasets.py

e.g.
```setup
python prepare_datasets.py --dataset RB_iid_100 --problem MVC
```

All possible datasets and CO problems are listed within prepare_datasets.py.

## Training

After you created the datasets you can train your model with:

e.g.
```train
python PPO_configuration.py --IsingMode RB_iid_100 --temps 0.05 --N_anneal 4000 --GPUs 0 --time_horizon 20 --batch_epochs 2 --AnnealSchedule cosine_frac --num_hidden_neurons 40 --mini_Sb 10 --mini_Nb 10 --mini_Hb 10 --EnergyFunction MVC --n_val_workers 30 --encode_node_features 40 --encode_edge_features 30 --lr 0.0005 --message_nh 30 --seed 123 --n_GNN_layers 3 --n_sample_spins 5 --project_name "myfirstrun"
```

You can find other parser_args that were used in the paper either in parser_MVC.txt or parser_MIS.txt.
The list of all parser_args will be added soon.

## Evaluation

You can evaluate on a dataset by running evaluate.py

e.g.

```train
python evaluate.py --GPU 0 --batchsize 2 --Ns 8 --sample_mode OG
```

This code will run evaluation on the RRG-100 MIS dataset and calculate an average APR.
If you want to evaluate on another dataset you will have to change the "overall_path" in evaluate.py to a path that contains the config and model weights of a model that is trained on another dataset.

## Pre-trained Models

A pretrained model that is trained on RRG-100 dataset is available.
See Evaluation. 

## Results

see Paper.


