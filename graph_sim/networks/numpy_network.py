import numpy as np
from scipy.special import expit
from scipy.sparse import csr_array
from pathlib import Path

class NumpyNetwork:
    def __init__(self, W, edge_index, n_neurons, n_networks, seed=0, threshold=-5):
        self.threshold = threshold
        self.W = W
        self.edge_index = edge_index
        self.filter_length = W.shape[1]
        self.n_edges = W.shape[0] // n_networks
        self.n_neurons = n_neurons
        self.n_networks = n_networks
        self.rng = np.random.default_rng(seed)
        self.backend = "numpy"

    def forward(self, x):
        x_j = x[self.edge_index[0]]
        messages = self.message(x_j)
        activation = self.aggregate(messages)
        return self.update(activation)

    def message(self, x_j):
        return (x_j * self.W).sum(axis=1)

    def update(self, activation):
        return expit(activation - self.threshold)

    def aggregate(self, x):
        a = np.zeros(self.n_neurons*self.n_networks)
        np.add.at(a, self.edge_index[1], x)
        return a

    def step(self, x):
        probs = self.forward(x)
        return self.rng.binomial(1, probs, size=(self.n_neurons*self.n_networks,))
    
    def next(self, t):
        rel_x = self.x[:, t:t+self.filter_length]
        probs = self.forward(rel_x)
        self.x[:, self.filter_length + t] = self.rng.binomial(1, probs, size=(self.n_neurons*self.n_networks,))

    def run(self, n_steps):
        self.prepare(n_steps)
        for t in range(n_steps):
            self.next(t)

    def prepare(self, n_steps, equilibration_steps=100):
        self.initialize_x(n_steps + equilibration_steps)
        self.to_device()

        for t in range(equilibration_steps):
            self.next(t)

        self.x = self.x[:, equilibration_steps:]
    
    def save_spikes(self, data_path):
        """Save the spikes to files in data_path"""
        for i, spikes_i in enumerate(self.spikes):
            W_i = self.W[i*self.n_edges:(i+1)*self.n_edges, :]
            edge_index_i = self.edge_index[:, i*self.n_edges:(i+1)*self.n_edges] - i*self.n_neurons
            np.savez(data_path / Path(f"{i}.npz"), spikes = spikes_i, W=W_i, edge_index=edge_index_i)

    def _to_sparse(self):
        """Convert the spikes to sparse matrix"""
        spikes = np.split(self.x[:, self.filter_length:], self.n_neurons, axis=0)
        return [csr_array(s) for s in spikes]

    def initialize_x(self, n_steps):
        self.x = np.zeros((self.n_neurons * self.n_networks, n_steps + self.filter_length))
        self.x[:, self.filter_length - 1] = self.rng.integers(0, 2, (self.n_neurons * self.n_networks,))

    @property
    def spikes(self):
        return self._to_sparse()

    def to_device(self):
        pass
