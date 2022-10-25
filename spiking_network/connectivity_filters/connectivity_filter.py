import torch
import torch.nn as nn

class ConnectivityFilter(nn.Module):
    def __init__(self, W0, edge_index, n_neurons, device="cpu"):
        super().__init__()
        # Device
        self.device = device

        # Learnable parameters
        self.alpha = nn.Parameter(torch.tensor(0.2, device=device))
        self.beta = nn.Parameter(torch.tensor(0.5, device=device))

        # Constants
        self.n_neurons = torch.tensor(n_neurons, device=device)
        self.abs_ref_strength = torch.tensor(-100., device=device)
        self.rel_ref_strength = torch.tensor(-30., device=device)
        self.time_scale = torch.tensor(10, device=device, dtype=torch.long)
        self.abs_ref_scale = torch.tensor(3, device=device, dtype=torch.long)
        self.rel_ref_scale = torch.tensor(7, device=device, dtype=torch.long)
        self.influence_scale = torch.tensor(5, device=device, dtype=torch.long)

        # Construct W
        self.edge_index = edge_index.to(device)
        self.W0 = W0.to(device)
        self.W = self._construct_W(self.W0, self.edge_index)

    def _construct_W(self, W0, edge_index):
        i, j = edge_index
        t = torch.arange(self.time_scale).repeat(W0.shape[0], 1).to(self.device) # Time steps
        is_self_edge = (i==j).unsqueeze(1).repeat(1, self.time_scale) # Is the edge a self-edge?
        self_edges = (
                self.abs_ref_strength * (t < self.abs_ref_scale) + 
                self.rel_ref_strength * torch.exp(-torch.abs(self.beta) * (t - self.abs_ref_scale)) * (self.abs_ref_scale <= t) * (t <= self.abs_ref_scale + self.rel_ref_scale)
            ) # values for self-edges
        other_edges = torch.einsum("i, ij -> ij", W0, torch.exp(-torch.abs(self.alpha) * t) * (t < self.influence_scale)) # values for other edges

        return torch.where(is_self_edge, self_edges, other_edges).flip(1) # Flip to get latest time step last

    def update(self):
        self.W = self._construct_W(self.W0, self.edge_index)

    def save_parameters():
        return {
            "alpha": self.alpha.item(),
            "beta": self.beta.item(),
            "abs_ref_strength": self.abs_ref_strength.item(),
            "rel_ref_strength": self.rel_ref_strength.item(),
            "time_scale": self.time_scale.item(),
            "abs_ref_scale": self.abs_ref_scale.item(),
            "rel_ref_scale": self.rel_ref_scale.item(),
            "influence_scale": self.influence_scale.item()
        }
