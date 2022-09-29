from torch_geometric.nn import MessagePassing
import numpy as np
import torch
from pathlib import Path
from scipy.sparse import csr_array

class SpikingNetwork(MessagePassing):
    def __init__(self, filter_params, seed=0):
        super().__init__()
        self.rng = torch.Generator().manual_seed(seed)

        # W0
        self.n_neurons = filter_params.n_neurons
        self.dist_params = filter_params.dist_params
        self.n_clusters = filter_params.n_clusters
        self.n_hub_neurons = filter_params.n_hub_neurons

        # Filter
        self.ref_scale = filter_params.ref_scale
        self.abs_ref_scale = filter_params.abs_ref_scale
        self.spike_scale = filter_params.spike_scale
        self.abs_ref_strength = filter_params.abs_ref_strength
        self.rel_ref_strength = filter_params.rel_ref_strength
        self.decay_offdiag = filter_params.decay_offdiag
        self.decay_diag = filter_params.decay_diag
        self.threshold = filter_params.threshold

        self.W, self.edge_index = self.build_clusters(self.n_clusters, self.n_hub_neurons, seed)
        self.n_edges = self.edge_index.shape[1]

        self.filter_length = self.W.shape[1]
        self.n_edges = self.W.shape[0] //self.n_clusters
        self.seed = seed

    def forward(self, x):
        return self.propagate(self.edge_index, x=x)

    def message(self, x_j):
        return torch.sum(x_j * self.W, dim=1, keepdim=True)

    def update(self, activation):
        return torch.sigmoid(activation - self.threshold)

    def next(self, t):
        rel_x = self.x[:, t:t+self.filter_length]
        probs = self.forward(rel_x)
        self.x[:, self.filter_length + t] = torch.bernoulli(probs, generator=self.rng).squeeze()

    def prepare(self, n_steps, equilibration_steps=100):
        self.initialize_x(n_steps + equilibration_steps)
        self.to_device()

        for t in range(equilibration_steps):
            self.next(t)

        self.x = self.x[:, equilibration_steps:]

    def simulate(self, n_steps, save_spikes=False, data_path=None, equilibration_steps=100):
        self.prepare(n_steps, equilibration_steps)
        for t in range(n_steps):
            self.next(t)
        if save_spikes == True:
            self.save(data_path)

    def to_device(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.W = self.W.to(device)
        self.edge_index = self.edge_index.to(device)
        self.x = self.x.to(device)

    def save_network(self, filename):
        torch.save(self.state_dict(), filename)

    def initialize_x(self, n_steps):
        self.x = torch.zeros((self.n_neurons * self.n_clusters, n_steps + self.filter_length), dtype=torch.float32)
        self.x[:, self.filter_length - 1] = torch.randint(0, 2, (self.n_neurons * self.n_clusters,), dtype=torch.float32, generator=self.rng)

    def save(self, data_path):
        sparse_x = csr_array(self.x[:, self.filter_length:])
        np.savez(data_path / Path(f"{self.seed}.npz"), spikes = sparse_x, W=self.W, edge_index=self.edge_index)

    def build_clusters(self, n_clusters, hub_nodes, seed, backend="torch"):
        Ws = []
        edge_indices = []
        w_seed = seed
        for i in range(n_clusters):
            W, edge_index = self.build_filter(w_seed)
            Ws.append(W)
            edge_indices.append(edge_index)
            w_seed += 1

        W = torch.concat(Ws, dim=0)
        edge_index = torch.concat(edge_indices, dim=1)
        shift = (torch.arange(n_clusters)*(self.n_neurons)).repeat_interleave(W.shape[0] // n_clusters, 0).repeat(2, 1)
        edge_index += shift

        for i in range(n_clusters):
            for j in range(hub_nodes):
                hub_node = (torch.randint(0, self.n_neurons, (1,), generator=self.rng)[0] + i*self.n_neurons)
                end_cluster = torch.randperm(n_clusters, generator=self.rng)[0]
                while end_cluster == i:
                    end_cluster = torch.randperm(n_clusters, generator=self.rng)[0]
                end_node = torch.randint(0, self.n_neurons, (1,))[0] + end_cluster * self.n_neurons
                new_edge = torch.tensor([hub_node, end_node]).unsqueeze(1)
                w = self._new_single_weight().unsqueeze(0).flip(1)
                edge_index = torch.cat((edge_index, new_edge), dim=1)
                W = torch.cat((W, w), dim=0)

        return W, edge_index


    def build_filter(self, seed):
        """Generates a new connectivity filter W"""
        W0 = self._generate_w0(self.dist_params, seed)
        W, edge_index = self.build_W(W0)
        return W, edge_index

    def _generate_w0(self, dist_params, seed):
        """Generates a normally-drawn connectivity matrix W0 that follows Dale's law and has zeros on the diagonal"""
        if dist_params.name == 'glorot':
            half_W0 = self._generate_glorot_w0(self.n_neurons, dist_params.mean, dist_params.std, seed)
        if dist_params.name == 'normal':
            half_W0 = self._generate_normal_w0(self.n_neurons, dist_params.mean, dist_params.std, seed)
        elif dist_params.name == 'uniform':
            half_W0 = self._generate_uniform_w0((self.n_neurons, self.n_neurons), seed)
        elif dist_params.name == 'mexican_hat':
            half_W0 = self._generate_mexican_hat_w0((self.n_neurons, self.n_neurons), seed)
        W0 = self._dales_law(half_W0)
        return W0

    def _dales_law(self, W0):
        """Applies Dale's law to the connectivity matrix W0"""
        W0 = torch.concat((W0 * (W0 > 0), W0 * (W0 < 0)), 0)
        W0 = torch.concat((W0, W0), 1)
        return W0

    def _generate_normal_w0(self, n_neurons, mean, std, seed):
        """Generates a normal n_neurons/2*n_neurons/2 matrux from a normal distribution"""
        rng = torch.Generator().manual_seed(seed)
        half_n_neurons = n_neurons // 2
        W0 = torch.normal(mean, std, (half_n_neurons, half_n_neurons), generator=rng)
        return W0

    def _generate_glorot_w0(self, n_neurons, mean, std, seed):
        """Generates a normal n_neurons/2*n_neurons/2 matrux from a normal distribution"""
        normal_W0 = self._generate_normal_w0(n_neurons, mean, std, seed)
        return normal_W0 / torch.sqrt(torch.tensor(n_neurons // 2))
    
    def build_W(self, W0):
        """Constructs a connectivity filter W from the weight matrix W0 and the filter parameters"""
        # Sets the diagonal elements of the connectivity filter
        diagonal_identity = torch.eye(W0.shape[0]).unsqueeze(2)

        # Sets self-influence during the absolute refractory period to self.abs_ref_strength
        diag = torch.zeros(self.ref_scale)
        diag[:self.abs_ref_scale] = self.abs_ref_strength 
        rel_ref_idx = torch.arange(self.abs_ref_scale, self.ref_scale) # [3, 4, 5, 6, 7, 8, 9]

        # Sets self-influence during the relative refractory period to -self.rel_ref_strength * exp(-self.beta * (t+3))
        diag[rel_ref_idx] = self.rel_ref_strength * torch.exp(-self.decay_diag * (rel_ref_idx - self.abs_ref_scale))

        # Sets the diagonal part of the connectivity filter (in decreasing time order)
        W = diagonal_identity @ diag.flip(dims=(0,)).unsqueeze(0)

        # Sets the off-diagonal elements of the connectivity filter based on W0 with decay self.alpha (in decreasing time order)
        offdiag_idx = torch.arange(self.spike_scale)
        offdiag = torch.exp(-self.decay_offdiag * offdiag_idx)
        W[:, :, -self.spike_scale:] += W0.unsqueeze(2) @ offdiag.flip(dims=(0,)).unsqueeze(0)
        edge_index = torch.nonzero(W[:, :, -1]).T
        flattened_W = W[edge_index[0], edge_index[1]]
        return flattened_W, edge_index

    def _new_single_weight(self):
        """Generates a new weight for the connectivity filter W"""
        w = torch.zeros(self.ref_scale)
        idx = torch.arange(self.spike_scale)
        if self.dist_params.name == 'glorot':
            w0 = torch.normal(self.dist_params.mean, self.dist_params.std, (1,)) / torch.sqrt(torch.tensor(self.n_neurons))
        elif self.dist_params.name == 'normal':
            w0 = torch.normal(self.dist_params.mean, self.dist_params.std, (1,))
        else:
            raise NotImplementedError
        w[:self.spike_scale] = w0 * torch.exp(-self.decay_offdiag * idx)
        return w
    
    def save_spikes(self, data_path):
        """Save the spikes to files in data_path"""
        for i, spikes_i in enumerate(self.spikes):
            W_i = self.W[i*self.n_edges:(i+1)*self.n_edges, :]
            edge_index_i = self.edge_index[:, i*self.n_edges:(i+1)*self.n_edges] - i*self.n_neurons
            np.savez(data_path / Path(f"{i}.npz"), spikes = spikes_i, W=W_i, edge_index=edge_index_i)

    def _to_sparse(self):
        """Convert the spikes to sparse matrix"""
        spikes = torch.split(self.x[:, self.filter_length:], self.n_neurons, dim=0)
        return [csr_array(s) for s in spikes]
