from torch_geometric.nn import MessagePassing
import torch
import torch.sparse

class SparseNetwork(MessagePassing):
    def __init__(self, W, edge_index, n_neurons, n_networks=1, threshold=-5, seed=0, aggr='add'):
        super().__init__(aggr=aggr)
        self.threshold = threshold
        self.W = W
        self.edge_index = edge_index
        self.filter_length = W.shape[1]
        self.n_edges = W.shape[0] // n_networks
        self.n_neurons = n_neurons
        self.n_networks = n_networks
        self.rng = torch.Generator().manual_seed(seed)

    def forward(self, x):
        return self.propagate(self.edge_index, x=x)

    def message(self, x_j):
        return torch.sparse.sum(self.W * x_j, dim=1).to_dense().unsqueeze(1)

    def update(self, activation):
        return torch.sigmoid(activation - self.threshold)

    def step(self, x):
        probs = self.forward(x)
        return torch.bernoulli(probs, generator=self.rng)

