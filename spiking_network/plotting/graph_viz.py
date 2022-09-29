import torch
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
from networks import FilterParams, SpikingNetwork


def plot_graph_from_file(file_path):
    data = torch.load(file_path)
    print(data)
    plot_graph(data)

def plot_network(network):
    graph = to_networkx(network.data, remove_self_loops=True)
    pos = nx.nx_agraph.graphviz_layout(graph, prog='neato')
    nx.draw(graph, pos, with_labels=False, node_size=20, node_color='red', arrowsize=5)
    plt.show()

if __name__ == '__main__':
    filter_params = FilterParams(n_neurons=20, n_clusters=1, n_hubneurons=0)
    network = SpikingNetwork(filter_params)
    plot_network(network)
