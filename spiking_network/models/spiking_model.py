from spiking_network.models.abstract_model import AbstractModel
import torch
import torch.nn as nn
from tqdm import tqdm

class SpikingModel(AbstractModel):
    def __init__(self, W0, edge_index, n_neurons, stimulation=[], seed=0, device="cpu"):
        super(SpikingModel, self).__init__(W0, edge_index, device)
        self._seed = seed
        self._rng = torch.Generator(device=device).manual_seed(seed)

        # Learnable parameters
        self.threshold = nn.Parameter(torch.tensor(5.0, device=device))
        self.alpha = nn.Parameter(torch.tensor(0.2, device=device))
        self.beta = nn.Parameter(torch.tensor(0.5, device=device))

        # Network parameters
        self.n_neurons = torch.tensor(n_neurons, device=device)
        self.abs_ref_strength = torch.tensor(-100., device=device)
        self.rel_ref_strength = torch.tensor(-30., device=device)
        self.time_scale = torch.tensor(10, device=device, dtype=torch.long)
        self.abs_ref_scale = torch.tensor(3, device=device, dtype=torch.long)
        self.rel_ref_scale = torch.tensor(7, device=device, dtype=torch.long)
        self.influence_scale = torch.tensor(5, device=device, dtype=torch.long)
        
    def time_dependence(self, W0, edge_index):
        r"""Determines the time-dependendence of the connection between neuorns i, j"""
        i, j = edge_index
        t = torch.arange(self.time_scale).repeat(W0.shape[0], 1).to(self.device) # Time steps
        is_self_edge = (i==j).unsqueeze(1).repeat(1, self.time_scale) # Is the edge a self-edge?
        self_edges = (
                self.abs_ref_strength * (t < self.abs_ref_scale) + 
                self.rel_ref_strength * torch.exp(-torch.abs(self.beta) * (t - self.abs_ref_scale)) * (self.abs_ref_scale <= t) * (t <= self.abs_ref_scale + self.rel_ref_scale)
            ) # values for self-edges
        other_edges = torch.einsum("i, ij -> ij", W0, torch.exp(-torch.abs(self.alpha) * t) * (t < self.influence_scale)) # values for other edges

        return torch.where(is_self_edge, self_edges, other_edges).flip(1) # Flip to get latest time step last

    def simulate(self, n_steps, stimulation=[]) -> torch.Tensor:
        """Simulates the network for n_steps"""
        W = self.time_dependence(self.W0, self.edge_index)
        spikes = torch.zeros(self.n_neurons, n_steps + self.time_scale, device=self.device)
        with torch.no_grad():
            self.eval()
            for t in (pbar := tqdm(range(n_steps), colour="#3E5641")):
                pbar.set_description(f"Simulating... t={t}")
                activation = self(spikes[:, t:t+self.time_scale], self.edge_index, W)
                for stim in stimulation:
                    activation += stim(t)
                spikes[:, t + self.time_scale] = self._update(activation)
        return spikes[:, self.time_scale:]

    def tune(self, p, lr = 0.01, N=1, max_iter=1000, epsilon=1e-4):
        """Tunes the parameters of the network to match the desired firing rate"""
        self.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        p = torch.tensor(p, device=self.device)
        for epoch in (pbar := tqdm(range(max_iter), colour="#3E5641")):
            optimizer.zero_grad()
            sum_probabilities = 0
            W = self.time_dependence(self.W0, self.edge_index)
            x = torch.zeros(self.n_neurons, self.time_scale + N, device=self.device)
            x[:, -1] = torch.randint(0, 2, (self.n_neurons,), device=self.device)
            for t in range(N):
                activation = self(x[:, t:t+self.time_scale], self.edge_index, W)
                x[:, t+self.time_scale] = self._update(activation)
                sum_probabilities += torch.sigmoid(activation - self.threshold).mean()
            avg_probabilities = sum_probabilities / N
            loss = loss_fn(avg_probabilities, p)
            if loss < epsilon:
                break
            loss.backward()
            optimizer.step()
            pbar.set_description(f"Tuning... p={avg_probabilities.item():.5f}")

    def _update(self, activation):
        """Samples the spikes of the neurons"""
        probabilities = torch.sigmoid(activation - self.threshold).squeeze()
        return torch.bernoulli(probabilities)

    def save_parameters(self):
        return {
            "alpha": self.alpha,
            "beta": self.beta,
            "threshold": self.threshold,
            "time_scale": self.time_scale,
            "abs_ref_scale": self.abs_ref_scale,
            "rel_ref_scale": self.rel_ref_scale,
            "influence_scale": self.influence_scale,
            "abs_ref_strength": self.abs_ref_strength,
            "rel_ref_strength": self.rel_ref_strength,
            }
