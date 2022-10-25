from torch_geometric.nn import MessagePassing
import torch
import numpy as np
from torch_scatter import scatter_add

class AbstractModel(MessagePassing):
    def __init__(self, W, edge_index, device="cpu"):
        super(AbstractModel, self).__init__(aggr='add')
        self.W0 = W.to(device)
        self.edge_index = edge_index.to(device)
        self.to(device)
        self.device = device

    def time_dependence(self, W, edge_index):
        return W

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor):
        r"""Calculates the new state of the network

        Parameters:
        ----------
        x: torch.Tensor
            The state of the network from time t - time_scale to time t [n_neurons, time_scale]
        edge_index: torch.Tensor
            The connectivity of the network [2, n_edges]
        edge_attr: torch.Tensor
            The edge weights of the connectivity filter [n_edges, time_scale]

        Returns:
        -------
        new_state: torch.Tensor
            The new state of the network from time t+1 - time_scale to time t+1 [n_neurons]
        """
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j: torch.Tensor, edge_attr: torch.Tensor):
        """Calculates the activation of the neurons

        Parameters:
        ----------
        x_j: torch.Tensor
            The state of the source neurons from time t - time_scale to time t [n_edges, time_scale]
        edge_attr: torch.Tensor
            The edge weights of the connectivity filter [n_edges, time_scale]

        Returns:
        -------
        activation: torch.Tensor
            The activation of the neurons at time t[n_edges]
        """
        return torch.sum(x_j*edge_attr, dim=1, keepdim=True)

    def save(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
