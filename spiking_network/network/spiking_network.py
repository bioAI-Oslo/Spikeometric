from torch_geometric.nn import MessagePassing
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
import numpy as np
import torch
from pathlib import Path
from scipy.sparse import csr_array
from network.filter_params import FilterParams, DistributionParams

class SpikingNetwork(MessagePassing):
    def __init__(self, filter_params: FilterParams, seed=0) -> None:
        super().__init__()
        self.rng = torch.Generator().manual_seed(seed)

        if filter_params.n_neurons % filter_params.n_clusters != 0:
            raise ValueError("n_neurons must be divisible by n_clusters")
        cluster_size = filter_params.n_neurons // filter_params.n_clusters
        if cluster_size % 2 != 0:
            raise ValueError("cluster_size must be even to generate W0")
        if filter_params.n_hubneurons > cluster_size:
            raise ValueError("n_hubneurons must be smaller than cluster_size")
        if filter_params.n_clusters == 0:
            raise ValueError("Must have at least one cluster")
        if filter_params.n_clusters < 2 and filter_params.n_hubneurons > 0:
            raise ValueError("n_clusters must be at least 2 to use hubneurons")

        # W0 parameters
        self.cluster_size = filter_params.n_neurons // filter_params.n_clusters
        self.n_neurons = filter_params.n_neurons
        self.dist_params = filter_params.dist_params
        self.n_clusters = filter_params.n_clusters
        self.n_hubneurons = filter_params.n_hubneurons

        # Filter parameters
        self.ref_scale = filter_params.ref_scale
        self.abs_ref_scale = filter_params.abs_ref_scale
        self.spike_scale = filter_params.spike_scale
        self.abs_ref_strength = filter_params.abs_ref_strength
        self.rel_ref_strength = filter_params.rel_ref_strength
        self.decay_offdiag = filter_params.decay_offdiag
        self.decay_diag = filter_params.decay_diag
        self.threshold = filter_params.threshold

        # Graph
        self.W, self.edge_index = self._build_clusters(self.n_clusters, self.cluster_size, self.n_hubneurons, start_seed=seed)
        self.data = Data(num_nodes=self.n_neurons, edge_index=self.edge_index, edge_attr=self.W)
        self.n_edges = self.data.num_edges
        self.filter_length = self.W.shape[1]

        self.seed = seed

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the network, starts the message passing, finishes with update"""
        return self.propagate(self.edge_index, x=x)

    def message(self, x_j: torch.Tensor) -> torch.Tensor:
        """Defines the message function"""
        return torch.sum(x_j * self.W, dim=1, keepdim=True)

    def update(self, activation: torch.Tensor) -> torch.Tensor:
        """Calculates new spikes based on the activation of the neurons"""
        probs = torch.sigmoid(activation - self.threshold) # Calculates the probability of a neuron firing
        return torch.bernoulli(probs, generator=self.rng).squeeze()

    def _next(self, t: int) -> None:
        """Calculates the next step of the network"""
        rel_x = self.x[:, t:t+self.filter_length] # Gets the spikes from the last filter_length steps
        self.x[:, self.filter_length + t] = self.forward(rel_x)

    def prepare(self, n_steps: int, equilibration_steps=100) -> None:
        """Prepares the network for simulation"""
        self._initialize_x(n_steps + equilibration_steps) # Sets up matrix X to store spikes
        self.to_device() # Moves network to GPU if available

        for t in range(equilibration_steps): # Simulates the network for equilibration_steps
            self._next(t)

        self.x = self.x[:, equilibration_steps:] # Removes equilibration steps from X

    def simulate(self, n_steps: int, save_spikes=False, data_path=None, equilibration_steps=100) -> None:
        """Simulates the network for n_steps"""
        self.prepare(n_steps, equilibration_steps) # Prepares the network for simulation
        for t in range(n_steps): # Simulates the network for n_steps
            self._next(t) 
        if save_spikes == True: # Saves the spikes if save_spikes is True
            self.save(data_path)

    def to_device(self) -> None:
        """Moves the network to GPU if available"""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.W = self.W.to(device)
        self.edge_index = self.edge_index.to(device)
        self.x = self.x.to(device)

    def save_network(self, filename: str) -> None:
        """Saves the network structrue to a file"""
        torch.save(self.state_dict(), filename)

    def _initialize_x(self, n_steps: int) -> None:
        """Initializes the matrix X to store spikes"""
        self.x = torch.zeros((self.n_neurons, n_steps + self.filter_length), dtype=torch.float32)
        self.x[:, self.filter_length - 1] = torch.randint(0, 2, (self.n_neurons,), dtype=torch.float32, generator=self.rng)

    def save(self, data_path:str) -> None:
        """Saves the spikes to a file"""
        data_path = Path(data_path)
        data_path.mkdir(parents=True, exist_ok=True)
        sparse_x = csr_array(self.x[:, self.filter_length:])
        np.savez(data_path / Path(f"{self.seed}.npz"), spikes = sparse_x, W=self.W, edge_index=self.edge_index)

    def _build_clusters(self, n_clusters: int, cluster_size: int, n_hubneurons: int, start_seed) -> Tuple[torch.Tensor, torch.Tensor]:
        """Builds the clusters for the network"""
        Ws = []
        edge_indices = []
        w_seed = start_seed
        for i in range(n_clusters): # Builds the internal structure of each cluster
            W, edge_index = self._build_filter(w_seed)
            Ws.append(W)
            edge_indices.append(edge_index)
            w_seed += 1

        W = torch.concat(Ws, dim=0)
        edge_index = torch.concat(edge_indices, dim=1)

        # Shifts the edge_index to account for the clusters
        edge_index = self._shift_edge_index(edge_index, Ws)

        # Adds connections between clusters
        for i in range(n_clusters):
            for j in range(n_hubneurons):
                hub_node = (torch.randint(0, cluster_size, (1,), generator=self.rng)[0] + i*cluster_size) # Selects a random node in the cluster
                end_cluster = torch.randperm(n_clusters, generator=self.rng)[0] # Selects a random cluster to receive a connection
                while end_cluster == i: # Ensures that the cluster does not connect to itself
                    end_cluster = torch.randperm(n_clusters, generator=self.rng)[0]
                end_node = torch.randint(0, cluster_size, (1,))[0] + end_cluster * cluster_size # Selects a random node in the end_cluster
                new_edge = torch.tensor([hub_node, end_node]).unsqueeze(1) # Creates the new edge
                w = self._new_single_weight().unsqueeze(0).flip(1) # Creates the new weight

                # Adds the new edge and weight to the network
                edge_index = torch.cat((edge_index, new_edge), dim=1)
                W = torch.cat((W, w), dim=0)

        return W, edge_index

    def _shift_edge_index(self, edge_index: torch.Tensor, Ws: List[torch.Tensor]) -> torch.Tensor:
        """Shifts the edge_index to account for the clusters"""
        shifts = []
        for i in range(len(Ws)):
            shift_i = torch.full((Ws[i].shape[0],), i * self.cluster_size)
            shifts.append(shift_i)
        shift = torch.cat(shifts)

        return edge_index + shift


    def _build_filter(self, seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Builds the internal structure of a cluster"""
        W0 = self._generate_w0(self.dist_params, seed)
        W, edge_index = self._build_W(W0)
        return W, edge_index

    def _generate_w0(self, dist_params: DistributionParams, seed: int) -> torch.Tensor:
        """Generates a normally-drawn connectivity matrix W0 that follows Dale's law and has zeros on the diagonal"""
        if dist_params.name == 'glorot':
            W0 = self._generate_glorot_w0(self.cluster_size, dist_params.mean, dist_params.std, seed)
        if dist_params.name == 'normal':
            W0 = self._generate_normal_w0(self.cluster_size, dist_params.mean, dist_params.std, seed)
        elif dist_params.name == 'uniform':
            W0 = self._generate_uniform_w0((self.cluster_size, self.cluster_size), seed)
        elif dist_params.name == 'mexican_hat':
            W0 = self._generate_mexican_hat_w0((self.cluster_size, self.cluster_size), seed)
        W0 = self._dales_law(W0)
        W0 = W0.fill_diagonal_(0)
        return W0

    def _dales_law(self, W0):
        """Applies Dale's law to the connectivity matrix W0"""
        W0 = torch.concat((W0 * (W0 > 0), W0 * (W0 < 0)), 0)
        return W0

    def _generate_normal_w0(self, n_neurons, mean, std, seed):
        """Generates a normal n_neurons/2*n_neurons/2 matrux from a normal distribution"""
        rng = torch.Generator().manual_seed(seed)
        half_n_neurons = n_neurons // 2
        W0 = torch.normal(mean, std, (half_n_neurons, n_neurons), generator=rng)
        return W0

    def _generate_glorot_w0(self, n_neurons, mean, std, seed):
        """Generates a normal n_neurons/2*n_neurons/2 matrux from a normal distribution"""
        normal_W0 = self._generate_normal_w0(n_neurons, mean, std, seed)
        glorot_W0 = normal_W0 / torch.sqrt(torch.tensor(n_neurons, dtype=torch.float32))

        return glorot_W0
    
    def _build_W(self, W0):
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
        """Generates a new weight for one of the hub neurons"""
        w = torch.zeros(self.ref_scale)
        idx = torch.arange(self.spike_scale)
        if self.dist_params.name == 'glorot':
            w0 = torch.normal(self.dist_params.mean, self.dist_params.std, (1,)) / torch.sqrt(torch.tensor(self.cluster_size))
        elif self.dist_params.name == 'normal':
            w0 = torch.normal(self.dist_params.mean, self.dist_params.std, (1,))
        else:
            raise NotImplementedError
        w[:self.spike_scale] = w0 * torch.exp(-self.decay_offdiag * idx)
        return w

    def show_graph(self):
        """Plots the graph of the connectivity filter"""
        graph = to_networkx(self.data, remove_self_loops=True)
        pos = nx.nx_agraph.graphviz_layout(graph, prog='neato')
        nx.draw(graph, pos, with_labels=False, node_size=20, node_color='red', arrowsize=5)
        plt.show()

