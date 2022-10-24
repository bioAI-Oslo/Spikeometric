from spiking_network.models.abstract_model import AbstractModel
import torch
import torch.nn as nn
from tqdm import tqdm


class ConnectivityModel(AbstractModel):
    def __init__(self, connectivity_filter, seed=0, device="cpu"):
        super().__init__(W=connectivity_filter.W, edge_index=connectivity_filter.edge_index, device=device)
        self._seed = seed
        self._rng = torch.Generator(device=device).manual_seed(seed)
        self.connectivity_filter = connectivity_filter
        self.device = device

        # Learnable parameters
        self.threshold = nn.Parameter(torch.tensor(5.0, device=device))

    def simulate(self, n_steps, stimulation=None) -> torch.Tensor:
        """Simulates the network for n_steps"""
        if not stimulation:
            stimulation = lambda t: torch.zeros((self.connectivity_filter.n_neurons, 1), device=self.device)

        spikes = torch.zeros(self.connectivity_filter.n_neurons, n_steps + self.connectivity_filter.time_scale, device=self.device)
        with torch.no_grad():
            self.eval()
            for t in (pbar := tqdm(range(n_steps), colour="#3E5641")):
                pbar.set_description(f"Simulating... t={t}")
                activation = self.forward(spikes[:, t:t+self.connectivity_filter.time_scale], self.connectivity_filter.edge_index, self.connectivity_filter.W)
                activation += stimulation(t)
                spikes[:, t + self.connectivity_filter.time_scale] = self._update(activation)

        return spikes[:, self.connectivity_filter.time_scale:]

    def tune(self, p, lr = 0.01, N=100, max_iter=1000, epsilon=1e-6):
        """Tunes the parameters of the network to match the desired firing rate"""
        self.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        p = torch.tensor(p, device=self.device)
        convergence_count = 0
        for epoch in (pbar := tqdm(range(max_iter), colour="#3E5641")):
            optimizer.zero_grad()

            self.connectivity_filter.update()
            activation = torch.zeros((self.connectivity_filter.n_neurons, N), device=self.device)
            x = torch.zeros(self.connectivity_filter.n_neurons, self.connectivity_filter.time_scale + N, device=self.device)
            x[:, self.connectivity_filter.time_scale] = torch.randint(0, 2, (self.connectivity_filter.n_neurons,), device=self.device)

            for t in range(N):
                act = self.forward(x[:, t:t+self.connectivity_filter.time_scale], self.connectivity_filter.edge_index, self.connectivity_filter.W)
                activation[:, t] = act.squeeze()
                x[:, t+self.connectivity_filter.time_scale] = self._update(act)
            
            probabilities = self._probability_of_spiking(activation).mean()
            loss = loss_fn(probabilities, p)
            pbar.set_description(f"Tuning... p={probabilities.item():.5f}")

            loss.backward()
            optimizer.step()
            if loss < epsilon:
                convergence_count += 1
            if convergence_count > 100:
                break

    def _probability_of_spiking(self, activation):
        return torch.sigmoid(activation - self.threshold).squeeze()

    def _update(self, activation):
        """Samples the spikes of the neurons"""
        probabilities = self._probability_of_spiking(activation)
        return torch.bernoulli(probabilities, generator=self._rng)

    def save_parameters(self):
        return {
            "threshold": self.threshold,
            }
