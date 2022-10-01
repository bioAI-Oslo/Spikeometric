from tkinter import W
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

class RecursiveNetwork(MessagePassing):
    def __init__(self, W, edge_index, n_neurons, filter_params: FilterParams, n_clusters=1, hub_neurons=[], rng=torch.Generator()):
        super().__init__(aggr='add')
        self.rng = rng
        self.initial_seed = self.rng.initial_seed()

        self.n_neurons = n_neurons
        self.cluster_size = n_neurons // n_clusters
        self.n_clusters = n_clusters
        self.hub_neurons = hub_neurons

        self._filter_params = filter_params
        self.threshold = filter_params.threshold

        self.W = W
        self.edge_index = edge_index
        
        self.n_edges = self.edge_index.shape[1]
        self.time_scale = self.W.shape[1]

        self.x = None
    
    @classmethod
    def build_recursively(cls, n_clusters: int, cluster_size: int, filter_params: FilterParams, rng: torch.Generator):
        """Builds the network recursively"""
        if n_clusters < 1:
            raise ValueError("n_clusters must be at least 1")

        if n_clusters == 1:
            if cluster_size % 2 != 0:
                raise ValueError("cluster_size must be even to generate W0")
            hub_neuron = torch.randint(cluster_size, (1,), dtype=torch.long)
            W, edge_index = cls._build_cluster(cluster_size=cluster_size, filter_params=filter_params, rng=rng)
            return cls(W, edge_index, cluster_size, filter_params, n_clusters, hub_neuron, rng)
        
        clusters = []
        for i in range(n_clusters): # Builds the internal structure of each cluster
            rng.manual_seed(rng.initial_seed() + i)
            cluster = cls.build_recursively(cluster_size=cluster_size, n_clusters=1, filter_params=filter_params, rng=rng)
            clusters.append(cluster)

        network = clusters[0]
        for i in range(1, n_clusters): # Builds the connections between the clusters
            network += clusters[i]

        # Connects the hub neurons and adds them to the graph
        W_hub, edge_index_hub = cls._connect_hub_neurons(network.hub_neurons, cluster_size, filter_params)
        W = torch.cat((network.W, W_hub), dim=0)
        edge_index = torch.cat((network.edge_index, edge_index_hub), dim=1)

        return cls(W, edge_index, network.n_neurons, network._filter_params, network.n_clusters, network.hub_neurons, rng)

    # Methods for calculating the spikes at the next time step
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
        x_over_last_time_steps = self.x[:, t:t+self.time_scale] # Gets the spikes from the last time_scale steps
        self.x[:, self.time_scale + t] = self.forward(x_over_last_time_steps) # Calculates the spikes for the next step

    def _initialize_x(self, n_steps: int) -> None:
        """Initializes the matrix X to store spikes, and randomly sets the initial spikes"""
        self.x = torch.zeros((self.n_neurons, n_steps + self.time_scale), dtype=torch.float32)
        self.x[:, self.time_scale - 1] = torch.randint(0, 2, (self.n_neurons,), dtype=torch.float32, generator=self.rng)

    def prepare(self, n_steps: int, equilibration_steps=100) -> None:
        """Prepares the network for simulation by initializing the spikes, sending the tensors to device and equilibrating the network"""
        self._initialize_x(n_steps + equilibration_steps) # Sets up matrix X to store spikes
        self.to_device() # Moves network to GPU if available

        for t in range(equilibration_steps): # Simulates the network for equilibration_steps
            self._next(t)

        self.x = self.x[:, equilibration_steps:] # Removes equilibration steps from X

    def simulate(self, n_steps: int, data_path: str, equilibration_steps=100, is_parallel=False) -> None:
        """Simulates the network for n_steps"""
        self.prepare(n_steps, equilibration_steps) # Prepares the network for simulation
        for t in range(n_steps): # Simulates the network for n_steps
            self._next(t) 
        self.save(data_path, is_parallel) # Saves the spikes to data_path
    
    @staticmethod
    def _connect_hub_neurons(hub_neurons: list[int], cluster_size, filter_params) -> tuple[torch.Tensor, torch.Tensor]:
        """For each hubneuron, connects it to a randomly selected hubneuron in another cluster"""
        edge_index = torch.tensor([], dtype=torch.long)
        W = torch.tensor([])
        for i in hub_neurons:
            available_neurons = [hub_neuron for hub_neuron in hub_neurons if hub_neuron != i]
            j = available_neurons[torch.randint(len(available_neurons), (1,))[0]]
            new_edge = torch.tensor([i, j], dtype=torch.long).unsqueeze(1)

            chm_action = "excitatory" if (i % cluster_size) < (cluster_size // 2) else "inhibitory"  # Determines the type of connection based on which half of the cluster the hubneuron is in
            w = RecursiveNetwork._new_single_weight(chemical_action=chm_action, cluster_size=cluster_size, filter_params=filter_params).unsqueeze(0).flip(1)
            edge_index = torch.cat((edge_index, new_edge), dim=1)
            W = torch.cat((W, w), dim=0)
        return W, edge_index

    @staticmethod
    def _build_cluster(cluster_size: int, filter_params: FilterParams, rng: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
        """Builds the internal structure of a cluster"""
        W0 = RecursiveNetwork._generate_w0(cluster_size, filter_params.dist_params, rng)
        W, edge_index = RecursiveNetwork._build_cluster_W_and_edge_index(W0, filter_params)
        return W, edge_index

    @staticmethod
    def _generate_w0(cluster_size, dist_params: DistributionParams, rng: torch.Generator) -> torch.Tensor:
        """Generates a normally-drawn connectivity matrix W0 that follows Dale's law and has zeros on the diagonal"""
        if dist_params.name == 'glorot':
            W0 = RecursiveNetwork._generate_glorot_w0(cluster_size, dist_params.mean, dist_params.std, rng)
        if dist_params.name == 'normal':
            W0 = RecursiveNetwork._generate_normal_w0(cluster_size, dist_params.mean, dist_params.std, rng)
        elif dist_params.name == 'uniform':
            W0 = RecursiveNetwork._generate_uniform_w0((cluster_size, cluster_size), rng)
        elif dist_params.name == 'mexican_hat':
            W0 = RecursiveNetwork._generate_mexican_hat_w0((cluster_size, cluster_size), rng)
        W0 = RecursiveNetwork._dales_law(W0)
        W0 = W0.fill_diagonal_(0)
        return W0
    
    @staticmethod
    def _dales_law(W0: torch.Tensor) -> torch.Tensor:
        """Applies Dale's law to the connectivity matrix W0"""
        W0 = torch.concat((W0 * (W0 > 0), W0 * (W0 < 0)), 0)
        return W0

    @staticmethod
    def _generate_normal_w0(cluster_size: int, mean: float, std: float, rng: torch.Generator) -> torch.Tensor:
        """Generates a normal n_neurons/2*n_neurons/2 matrux from a normal distribution"""
        half_cluster_size = cluster_size // 2
        W0 = torch.normal(mean, std, (half_cluster_size, cluster_size), generator=rng)
        return W0
    
    @staticmethod
    def _generate_glorot_w0(cluster_size: int, mean: float, std: float, rng: torch.Generator) -> torch.Tensor:
        """Generates a normal n_neurons/2*n_neurons/2 matrux from a normal distribution"""
        normal_W0 = RecursiveNetwork._generate_normal_w0(cluster_size, mean, std, rng)
        glorot_W0 = normal_W0 / torch.sqrt(torch.tensor(cluster_size, dtype=torch.float32))
        return glorot_W0

    @staticmethod 
    def _build_cluster_W_and_edge_index(W0: torch.Tensor, filter_params: FilterParams) -> tuple[torch.Tensor, torch.Tensor]:
        """Constructs a connectivity filter W from the weight matrix W0 and the filter parameters"""
        # Sets the diagonal elements of the connectivity filter
        diagonal_identity = torch.eye(W0.shape[0]).unsqueeze(2)

        # Sets self-influence during the absolute refractory period to abs_ref_strength
        diag = torch.zeros(filter_params.ref_scale)
        diag[:filter_params.abs_ref_scale] = filter_params.abs_ref_strength 
        rel_ref_idx = torch.arange(filter_params.abs_ref_scale, filter_params.ref_scale) # [3, 4, 5, 6, 7, 8, 9]

        # Sets self-influence during the relative refractory period to -rel_ref_strength * exp(decay_diag * (t+3))
        diag[rel_ref_idx] = filter_params.rel_ref_strength * torch.exp(-filter_params.decay_diag * (rel_ref_idx - filter_params.abs_ref_scale))

        # Sets the diagonal part of the connectivity filter (in decreasing time order)
        W = diagonal_identity @ diag.flip(dims=(0,)).unsqueeze(0)

        # Sets the off-diagonal elements of the connectivity filter based on W0 with decay decay_offdiag (in decreasing time order)
        offdiag_idx = torch.arange(filter_params.spike_scale)
        offdiag = torch.exp(-filter_params.decay_offdiag * offdiag_idx)
        W[:, :, -filter_params.spike_scale:] += W0.unsqueeze(2) @ offdiag.flip(dims=(0,)).unsqueeze(0)
        edge_index = torch.nonzero(W[:, :, -1]).T
        flattened_W = W[edge_index[0], edge_index[1]]

        return flattened_W, edge_index

    @staticmethod
    def _new_single_weight(chemical_action: str, cluster_size, filter_params: FilterParams) -> torch.Tensor:
        """Generates a new weight for one of the hub neurons"""
        w = torch.zeros(filter_params.ref_scale)
        idx = torch.arange(filter_params.spike_scale)
        if filter_params.dist_params.name == 'glorot':
            w0 = torch.normal(filter_params.dist_params.mean, filter_params.dist_params.std, (1,)) / torch.sqrt(torch.tensor(cluster_size))
        elif filter_params.dist_params.name == 'normal':
            w0 = torch.normal(filter_params.dist_params.mean, filter_params.dist_params.std, (1,))
        else:
            raise NotImplementedError
        w[:filter_params.spike_scale] = w0 * torch.exp(-filter_params.decay_offdiag * idx)
        if chemical_action == 'excitatory':
            w  = w * torch.sign(w0)
        elif chemical_action == 'inhibitory':
            w  = w * -torch.sign(w0)
        return w

    # Helper methods
    def to_device(self) -> None:
        """Moves the network to GPU if available"""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.W = self.W.to(device)
        self.edge_index = self.edge_index.to(device)
        self.x = self.x.to(device)

    def save(self, data_path:str, is_parallel: bool) -> None:
        """Saves the spikes to a file"""
        if is_parallel:
            self._save_parallel(data_path)
        data_path = Path(data_path)
        data_path.mkdir(parents=True, exist_ok=True)
        sparse_x = csr_array(self.x[:, self.time_scale:])
        np.savez(data_path / Path(f"{self.initial_seed}.npz"), spikes = sparse_x, W=self.W, edge_index=self.edge_index)
    
    def _save_parallel(self, data_path: str) -> None:
        """Saves the spikes to a file"""
        data_path = Path(data_path)
        data_path.mkdir(parents=True, exist_ok=True)
        x = self.x[:, self.time_scale:]
        x_sims = torch.split(x, self.cluster_size, dim=0)
        for i, x_sim in enumerate(x_sims):
            sparse_x = csr_array(x_sim)
            np.savez(data_path / Path(f"{self.initial_seed}_{i}.npz"), spikes = sparse_x, W=self.W, edge_index=self.edge_index)

    def show_graph(self) -> None:
        """Plots the graph of the connectivity filter"""
        data = Data(num_nodes=self.n_neurons, edge_index=self.edge_index, edge_attr=self.W)
        graph = to_networkx(data, remove_self_loops=True)
        pos = nx.nx_agraph.graphviz_layout(graph, prog='neato')
        nx.draw(graph, pos, with_labels=False, node_size=20, node_color='red', arrowsize=5)
        plt.show()

    def __add__(self, other):
        """Adds two RecursiveNetworks together"""
        if self._filter_params != other._filter_params:
            raise ValueError("The two networks have different filter parameters")
        new_W = torch.cat((self.W, other.W), dim=0)
        new_edge_index = torch.cat((self.edge_index, other.edge_index + self.n_neurons), dim=1)
        new_hub_neurons = torch.cat((self.hub_neurons, other.hub_neurons + self.n_neurons))
        n_neurons = self.n_neurons + other.n_neurons
        n_clusters = self.n_clusters + other.n_clusters
        new_seed = self.initial_seed + other.initial_seed
        new_rng = torch.Generator().manual_seed(new_seed)
        return RecursiveNetwork(new_W, new_edge_index, n_neurons, self._filter_params, n_clusters, new_hub_neurons, new_rng)