from spiking_network.models.spiking_model import AbstractModel
from torch_geometric.nn import MessagePassing
import torch
from torch import nn
from tqdm import tqdm


class HermanModel(AbstractModel):
    def __init__(self, W0, edge_index, n_neurons, stimulation=[], seed=0, device="cpu"):
        super(HermanModel, self).__init__(W0, edge_index, stimulation, device)
        self._seed = seed
        self._rng = torch.Generator(device=device).manual_seed(seed)

        # Learnable parameters
        #  self.r = torch.nn.Parameter(torch.tensor(0.025, device=device))
        #  self.threshold = torch.nn.Parameter(torch.tensor(1.378e-3, device=device))
        #  self.b = torch.nn.Parameter(torch.tensor(0.001, device=device))
        #  self.noise_sparsity = torch.nn.Parameter(torch.tensor(1.5, device=device))
        #  self.tau = torch.nn.Parameter(torch.tensor(0.01, device=device))
            
        self.r = torch.tensor(0.025, device=device)
        self.threshold = torch.tensor(1.378e-3, device=device)
        self.noise_std = torch.tensor(0.3, device=device) # noise amplitude
        self.b = torch.tensor(0.001, device=device) #uniform feedforward input
        self.noise_sparsity = torch.tensor(1.0, device=device) #noise is injected with the prob that a standard normal exceeds this

        self.tau = torch.tensor(0.01, device=device)
        self.dt = torch.tensor(0.0001, device=device)
        self.n_neurons = n_neurons


    def simulate(self, n_steps) -> torch.Tensor:
        """Simulates the network for n_steps"""
        W = self.time_dependence(self.W0, self.edge_index)

        spikes = torch.zeros((self.n_neurons, n_steps), device=self.device)
        activation = torch.zeros((self.n_neurons, 1), device=self.device)
        with torch.no_grad():
            self.eval()
            for t in (pbar := tqdm(range(n_steps), colour="#3E5641")):
                pbar.set_description(f"Simulating... t={t}")
                activation = self(activation, self.edge_index, W)
                for stim in self._stimulation:
                    stim(t, activation)
                spikes[:, t] = self._update(activation) > self.threshold
                activation += spikes[:, t].unsqueeze(-1) - (activation / self.tau) * self.dt
        return spikes

    def tune(self, p, lr = 0.01, N=100, max_iter=1000):
        """Tunes the parameters of the network to match the desired firing rate"""
        self.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        p = torch.tensor(p, device=self.device)
        for epoch in (t := tqdm(range(max_iter), colour="#3E5641")):
            optimizer.zero_grad()
            probabilities = 0
            W = self.time_dependence(self.W0, self.edge_index)
            spikes = torch.zeros((self.n_neurons, N), device=self.device)
            activation = torch.zeros((self.n_neurons, 1), device=self.device)
            for i in range(N):
                activation = self(activation, self.edge_index, W)
                spikes[:, i] = self._update(activation)
                activation = activation + spikes[:, i].unsqueeze(-1) - (activation / self.tau) * self.dt
                probabilities += spikes[:, i].sum()
            avg_probabilities = probabilities / (N * self.n_neurons)
            loss = loss_fn(avg_probabilities, p)
            loss.backward()
            optimizer.step()
            t.set_description(f"Tuning... p={avg_probabilities.item():.3f}, s={avg_spikes.item():.3f}")

    def _update(self, activation: torch.Tensor):
        noise = torch.normal(0., self.noise_std, size=activation.shape, device=activation.device)
        filtered_noise = torch.normal(0., 1., size=activation.shape, device=activation.device) > self.noise_sparsity
        b_term = self.b * (1 + noise * filtered_noise)
        l = self.r * activation + b_term
        spikes = l > self.threshold
        return spikes.squeeze()

    def save_parameters(self):
        return {
            "r": self.r,
            "threshold": self.threshold,
            "noise_std": self.noise_std,
            "b": self.b,
            "noise_sparsity": self.noise_sparsity,
            "tau": self.tau,
            "dt": self.dt,
            "n_neurons": self.n_neurons,
        }
