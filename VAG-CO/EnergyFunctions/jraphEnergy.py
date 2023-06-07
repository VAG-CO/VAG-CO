import numpy as np
import jax
from GlobalProjectVariables import MVC_B, MVC_A

def compute_Energy_full_graph(H_graph, spins, A = MVC_A, B = MVC_B):
    n_node = H_graph.n_node
    n_graph = n_node.shape[0]
    graph_idx = np.arange(n_graph)
    sum_n_node = H_graph.nodes.shape[0]
    node_gr_idx = np.repeat(graph_idx, n_node, axis=0)

    #print("nodes", nodes.shape, H_graph.edges.shape, spins.shape)
    Energy_messages = H_graph.edges * spins[H_graph.senders] * spins[H_graph.receivers]
    Energy_per_node =  0.5 * jax.ops.segment_sum(Energy_messages, H_graph.receivers, sum_n_node) + spins*H_graph.nodes
    Energy = jax.ops.segment_sum(Energy_per_node, node_gr_idx, n_graph)

    return Energy