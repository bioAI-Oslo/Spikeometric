from spiking_network.models.abstract_model import AbstractModel
from torch_geometric.nn import MessagePassing, HeteroConv
import torch

class StimulationModel(AbstractModel):
    def __init__(self, n_steps, seed=0, device="cpu", equilibration_steps = 100):
        super(StimulationModel, self).__init__(W=None, edge_index = None, n_steps=n_steps, seed=seed, device=device, equilibration_steps=equilibration_steps)
        self._layer = SpikeLayer()
        self._stim_layer = StimulationLayer()
        self.conv = HeteroConv({("neuron", "drives", "neuron"): self._layer, ("stimulus", "stimulates", "neuron"): self._stim_layer}, aggr="sum")

    def forward(self, x_dict, edge_index_dict, edge_attr_dict) -> torch.Tensor:
        """Simulates the network for n_steps"""
        initial_conditions = x_dict["neuron"]
        time_scale = initial_conditions.size(1)
        n_neurons = initial_conditions.size(0)

        x = self._equilibrate(x_dict["neuron"], edge_index_dict[("neuron", "drives", "neuron")], edge_attr_dict[("neuron", "drives", "neuron")])
        self._spikes = torch.zeros(n_neurons, self._n_steps + time_scale, device=x.device)
        self._spikes[:, :time_scale] = x
        self._stim_times = x_dict["stimulus"]
        for t in range(self._n_steps):
            x_dict["neuron"] = self._spikes[:, t:t+time_scale]
            x_dict["stimulus"] = self._stim_times[:, t].unsqueeze(-1)
            total_act = self.conv(x_dict, edge_index_dict, edge_attr_dict)
            self._spikes[:, t+time_scale] = self._sample(total_act["neuron"])
        self.to_device("cpu") 
        return self._spikes[:, time_scale:]

    def _equilibrate(self, x, edge_index, edge_attr):
        """Equilibrate the network"""
        x = torch.roll(x, 1, -1)
        for _ in range(self._equilibration_steps):
            act = self._layer(x, edge_index, edge_attr)
            x[:, -1] = self._sample(act)
            x = torch.roll(x, 1, -1)
        return x

    def _sample(self, x):
        """Samples the spikes of the neurons"""
        probabilities = torch.sigmoid(x).squeeze()
        return torch.bernoulli(probabilities, generator=self._rng)

    def to_device(self, device):
        self._spikes = self._spikes.to(device)
        self._rng = torch.Generator(device = device).manual_seed(self._seed)

class StimulationLayer(MessagePassing):
    def __init__(self):
        super(StimulationLayer, self).__init__(aggr='add')

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j, edge_attr):
        return (x_j.squeeze(1) * edge_attr).unsqueeze(1)


class SpikeLayer(MessagePassing):
    def __init__(self):
        super(SpikeLayer, self).__init__(aggr='add')
        self.threshold = 5

    def forward(self, state: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor):
        r"""Calculates the new state of the network

        Parameters:
        ----------
        state: torch.Tensor
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
        return self.propagate(edge_index, x=state, edge_attr=edge_attr)

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
