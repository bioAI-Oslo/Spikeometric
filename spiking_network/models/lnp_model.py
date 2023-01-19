from spiking_network.models.base_model import BaseModel
import torch

class LNPModel(BaseModel):
    def __init__(self, parameters={}, rng=None, stimulation=None):
        super().__init__(stimulation)
        params = self._default_parameters
        params.update(parameters)

        for key, value in params.items():
            if key not in self._valid_parameters:
                raise ValueError(f"Invalid parameter {key}")
            self.register_buffer(key, torch.tensor(value))

        self._rng = rng if rng is not None else torch.Generator()
    
    def message(self, x_j, W):
        """
        Compute the message from x_j to x_i
        
        Parameters
        ----------
        x_j : torch.Tensor [n_edges, 1]
            The activation of the neurons at the previous time step.
        W : torch.Tensor [n_edges, 1]
            The edge weights of the connectivity filter.

        Returns
        -------
        message : torch.Tensor [n_edges, 1]
        """
        return W * x_j

    def propagate(self, edge_index, x, W, current_activation):
        current_activation = current_activation.unsqueeze(dim=1)
        current_activation += x - (current_activation / self.tau) * self.dt
        return super().propagate(edge_index, x=current_activation, W=W)

    def update_state(self, activation):
        """
        Update the state of the neurons.
        The network will spike if the activation is above the threshold.
        Explanation of the model <-- here

        Parameters
        ----------
        activation : torch.Tensor [n_neurons, time_scale]
            The activation of the neurons at the current time step.

        Returns
        -------
        spikes : torch.Tensor [n_neurons, time_scale]
            The spikes of the network at the current time step.
        """
        noise = torch.normal(0., self.noise_std, size=activation.shape, device=activation.device)
        filtered_noise = torch.normal(0., 1., size=activation.shape, device=activation.device) > self.noise_sparsity
        b_term = self.b * (1 + noise * filtered_noise)
        l = self.r * activation + b_term
        spikes = l > self.threshold
        return spikes.squeeze()

    def connectivity_filter(self, W0, edge_index):
        return W0.unsqueeze(1)

    def initialize_state(self, n_neurons):
        return torch.zeros(n_neurons, self.time_scale, device=self.time_scale.device, dtype=torch.uint8)

    @property
    def _default_parameters(self):
        return {
            "r": 0.025,
            "b": 0.001,
            "tau": 0.01,
            "dt": 0.0001,
            "noise_std": 0.3,
            "noise_sparsity": 1.0,
            "threshold": 1.378e-3,
            "time_scale": 1,
        }

    @property
    def _valid_parameters(self) -> list:
        return list(self._default_parameters.keys())

