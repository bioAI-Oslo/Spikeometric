from torch_geometric.nn import MessagePassing
import torch
import torch.nn as nn 
from pathlib import Path
from spiking_network.stimulation import BaseStimulation

class BaseModel(MessagePassing):
    """
    Base class for all models. Extends the MessagePassing class from torch_geometric by adding stimulation support and
    a forward method that calculates the activation of the network and then updates the state of the network using
    the update_state method that must be implemented by the child class. There are also methods for saving and loading
    the model.
    """
    def __init__(self, stimulation):
        """Sets the parameters of the model and adds the stimulation if it is not None"""
        super(BaseModel, self).__init__(aggr='add') 
        if stimulation is not None:
            self.add_stimulation(stimulation)
        else:
            self.stimulation = lambda t: 0
    
    @classmethod
    def load(cls, path):
        """Loads the model"""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"File {path} not found, please tune the model first")
        model = cls()
        model.load_state_dict(torch.load(path))
        return model

    def update_state(self, activations: torch.Tensor) -> torch.Tensor:
        """Calculates the spikes of the network at time t from the probabilites"""
        raise NotImplementedError
    
    def message(self, x_j: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
        """Calculates the activation of the neurons"""
        raise NotImplementedError
    
    def initialize_state(self, n_neurons):
        """Initializes the state of the network"""
        raise NotImplementedError
    
    def activation(self, x: torch.Tensor, edge_index: torch.Tensor, W: torch.Tensor, current_activation=None, t=-1, stimulation_targets=None) -> torch.Tensor:
        """Calculates the activation of the network"""
        return self.propagate(edge_index, x=x, W=W, current_activation=current_activation).squeeze() + self.stimulate(t, stimulation_targets, x.shape[0])
    
    def probability_of_spike(self, activation):
        """Calculates the probability of a neuron to spike"""
        return activation

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, W: torch.Tensor, current_activation=None, t=-1, stimulation_targets=None) -> torch.Tensor:
        r"""
        Calculates the new state of the network at time t+1 from the state at time t by first calculating the activation of the network
        and then calculating the new state of the network using the activation.

        Parameters:
        ----------
        x: torch.Tensor
            The state of the network from time t - time_scale to time t [n_neurons, time_scale]
        edge_index: torch.Tensor
            The connectivity of the network [2, n_edges]
        W: torch.Tensor
            The edge weights of the connectivity filter [n_edges, time_scale]
        current_activation: torch.Tensor
            The activation of the network from time t - time_scale to time t [n_neurons, time_scale]
        t: int
            The current time step
        stimulation_targets: torch.Tensor
            The indices of the neurons to be stimulated [n_stimulations, n_stimulated_neurons]
        

        Returns:
        -------
        spikes: torch.Tensor
            The new state of the network from time t+1 - time_scale to time t+1 [n_neurons]
        """
        activation = self.activation(
            edge_index=edge_index,
            x=x,
            W=W,
            current_activation=current_activation,
            t=t,
            stimulation_targets=stimulation_targets
        )
        return self.update_state(activation)

    def add_stimulation(self, stimulation):
        """Adds stimulation to the network"""
        if isinstance(stimulation, list):
            self.stimulations = torch.nn.ModuleDict({f"{stim.__class__.__name__}_{i}": stim for i, stim in enumerate(stimulation)})
            self.stimulation = lambda t, targets, n_neurons: sum([self.stimulations[key](t, targets[i], n_neurons) for i, key in enumerate(self.stimulations.keys())])
        else:
            self.stimulation = stimulation
    
    def stimulate(self, t, targets, n_neurons):
        """Stimulates the network"""
        if targets is None:
            return self.stimulation(t)
        return self.stimulation(t, targets, n_neurons)
    
    @property
    def _default_parameters(self):
        """Returns the default parameters of the model"""
        return {
            "time_scale": 1,
        }

    @property
    def _valid_parameters(self) -> list:
        """Returns the parameters of the model"""
        return self._default_parameters.keys()

    def tune(self, parameters):
        """Sets requires_grad to True for the parameters to tune"""
        for param in parameters:
            parameter_dict = dict(self.named_parameters())
            if param not in parameter_dict.keys():
                raise ValueError(f"Parameter {param} not found in the model")
            parameter_dict[param].requires_grad = True

    def save(self, path):
        """Saves the model"""
        torch.save(self.state_dict(), path)

    def to(self, device):
        """Moves the model to the device"""
        self = super().to(device)
        if hasattr(self, "_rng"):
            seed = self._rng.seed()
            self._rng = torch.Generator(device=device).manual_seed(seed)
        return self