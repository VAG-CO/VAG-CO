import numpy as np
import jax
import jax.numpy as jnp
import time as time
import random
import jraph
import os

from Networks.policy import Policy
from train import TrainMeanField
from Networks.load_network import LoadNetwork
from Data.RandomGraphs import ErdosRenyiGraphs
from ConditionalExpectation import ConditionalExpectation


import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


if __name__ == '__main__':
	os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
	os.environ['CUDA_VISIBLE_DEVICES'] = "2"
	os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.95"
	#os.environ['WANDB_SILENT'] = "true"






