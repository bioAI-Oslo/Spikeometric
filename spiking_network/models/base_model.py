from torch_geometric.nn import MessagePassing
import torch
import torch.nn as nn 
from pathlib import Path
from spiking_network.stimulation import BaseStimulation

class BaseModel(MessagePassing):
    def __init__(self, parameters, device="cpu", stimulation=None):
        super(BaseModel, self).__init__() 
        self.device = device

        # Parameters
        params = self._default_parameters
        self._check_parameters(parameters)
        params.update(parameters)

        self._params = self._init_parameters(params, device)
        self.time_scale = self._params["time_scale"].item()
        
        if stimulation is not None:
            self.add_stimulation(stimulation)
        else:
            self.stimulation = lambda t: 0

    def update_state(self, probabilites):
        """Calculates the spikes of the network at time t from the probabilites"""
        raise NotImplementedError
    
    def message(self, x_j: torch.Tensor, W: torch.Tensor):
        """Calculates the activation of the neurons"""
        raise NotImplementedError
    
    def initialize_state(self, n_neurons):
        """Initializes the state of the network"""
        raise NotImplementedError
    
    def activation(self, x: torch.Tensor, edge_index: torch.Tensor, W: torch.Tensor, current_activation=None, t=-1) -> torch.Tensor:
        """Calculates the activation of the network"""
        return self.propagate(edge_index, x=x, W=W, current_activation=current_activation).squeeze() + self.stimulate(t)
    
    def probability_of_spike(self, activation):
        """Calculates the probability of a neuron to spike"""
        return activation

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, W: torch.Tensor, current_activation=None, t=-1) -> torch.Tensor:
        """Forward pass of the network"""
        r"""Calculates the new state of the network

        Parameters:
        ----------
        x: torch.Tensor
            The state of the network from time t - time_scale to time t [n_neurons, time_scale]
        edge_index: torch.Tensor
            The connectivity of the network [2, n_edges]
        W: torch.Tensor
            The edge weights of the connectivity filter [n_edges, time_scale]
        t: int
            The current time step
        activation: torch.Tensor
            The activation of the network from time t - time_scale to time t [n_neurons, time_scale]

        Returns:
        -------
        spikes: torch.Tensor
            The new state of the network from time t+1 - time_scale to time t+1 [n_neurons]
        """
        activation = self.activation(edge_index=edge_index, x=x, W=W, current_activation=current_activation)
        probabilites = self.probability_of_spike(activation)
        return self.update_state(probabilites)

    def add_stimulation(self, stimulation):
        """Adds stimulation to the network"""
        if isinstance(stimulation, list):
            self.stimulation = lambda t: sum([stimulation_i(t) for stimulation_i in stimulation])
            for stimulation_i in stimulation:
                self._add_parameters_from_stimulation(stimulation_i)
        else:
            self.stimulation = stimulation
            if isinstance(stimulation, BaseStimulation):
                self._add_parameters_from_stimulation(stimulation)

    def stimulate(self, t):
        """Stimulates the network"""
        return self.stimulation(t)

    def save(self, path):
        """Saves the model"""
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path):
        """Loads the model"""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"File {path} not found, please tune the model first")
        model = cls()
        model.load_state_dict(torch.load(path))
        return model

    def _init_parameters(self, params, device):
        """Initializes the parameters of the model"""
        return nn.ParameterDict(
                {
                key: nn.Parameter(torch.tensor(value, device=device), requires_grad=False)
                for key, value in params.items()
            }
        )

    @property
    def _default_parameters(self):
        """Returns the default parameters of the model"""
        return {
            "time_scale": 1,
        }

    def _add_parameters_from_stimulation(self, stimulation):
        """Adds the parameters to the model"""
        self._params.update(stimulation._params)
    
    def set_tunable_parameters(self, params_to_tune):
        """Sets the parameters of the model to tune"""
        self._check_tunable_parameters(params_to_tune)
        for param in params_to_tune:
            self._params[param].requires_grad = True
    
    def _check_parameters(self, parameters):
        if any([p not in self._default_parameters.keys() for p in parameters]):
            raise ValueError("Parameters {} not recognised".format([p for p in parameters if p not in self._default_parameters.keys()]))
    
    def _check_tunable_parameters(self, tunable_parameters):
        if any([p in self._untunable_parameters or p not in self._params.keys() for p in tunable_parameters]):
            raise ValueError("Parameters {} cannot be tuned".format(self._untunable_parameters))
    
    @property 
    def parameter_dict(self) -> dict:
        """Returns the parameters of the model"""
        d = {}
        for key, value in self._params.items():
            d[key] = value.data
        return d
