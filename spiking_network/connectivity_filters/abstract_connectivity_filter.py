from abc import ABC, abstractmethod
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

class AbstractConnectivityFilter(ABC):
    def __init__(self, W0, time_scale, filter_params):
        self.W0 = W0
        self.time_scale = time_scale
        self._filter_parameters = filter_params
        self._build_W(W0, time_scale, filter_params)

    @abstractmethod
    def time_dependence(self, t, filter_params):
        pass

    @property
    def filter_parameters(self):
        return self._filter_parameters

    @property
    def W(self):
        return self._W

    @property
    def edge_index(self):
        return self._edge_index

    @property
    def n_neurons(self):
        return self.W0.shape[0]

    @property
    def n_edges(self):
        return self._W.shape[0]

    def _build_W(self, W0: torch.Tensor, time_scale: int, filter_params: dict) -> torch.Tensor:
        """Constructs a connectivity filter W from the weight matrix W0"""
        edge_index = W0.nonzero()
        W0 = W0[edge_index[:, 0], edge_index[:, 1]]
        time = torch.arange(self.time_scale).repeat(W0.shape[0], 1)
        i, j = edge_index[:, 0], edge_index[:, 1]
        W = self.time_dependence(W0, time, filter_params, i, j)

        self._edge_index = edge_index.T
        self._W = W

    def _update(self, filter_params):
        """Updates the connectivity filter W with new parameters"""
        self._W = self._build_W(self.W0, filter_params)
        self._edge_index = self._W.nonzero().T

    def to_device(self, device):
        self._W = self._W.to(device)
        self._edge_index = self._edge_index.to(device)

    def tensor(self):
        sparse_W = torch.sparse_coo_tensor(self.edge_index, self.W, self.W0.shape)
        return sparse_W.to_dense()

    def plot(self) -> None:
        """Plots the graph of the connectivity filter"""
        data = Data(num_nodes=self.W0.shape[0], edge_index=self._edge_index)
        graph = to_networkx(data, remove_self_loops=True)
        pos = nx.nx_agraph.graphviz_layout(graph, prog='neato')
        nx.draw(graph, pos, with_labels=False, node_size=20, node_color='red', arrowsize=5)
        plt.show()
