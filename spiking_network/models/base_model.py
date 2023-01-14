from torch_geometric.nn import MessagePassing
import torch
import torch.nn as nn 
from pathlib import Path
from spiking_network.stimulation import BaseStimulation

class BaseModel(MessagePassing):
    def __init__(self, parameters, stimulation, device="cpu"):
        super(BaseModel, self).__init__() 
        self.device = device

        params = self._default_parameters.copy()
        self._check_parameters(parameters)
        params.update(parameters)
        for key, value in params.items():
            setattr(self, key, value)

        self._tunable_params = self._init_parameters(
            {k: v for k, v in self.__dict__.items() if k in self._tunable_parameter_keys},
            device
        )
        
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

    def update_state(self, probabilites):
        """Calculates the spikes of the network at time t from the probabilites"""
        raise NotImplementedError
    
    def message(self, x_j: torch.Tensor, W: torch.Tensor):
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
        activation = self.activation(edge_index=edge_index, x=x, W=W, current_activation=current_activation, t=t, stimulation_targets=stimulation_targets)
        probabilites = self.probability_of_spike(activation)
        return self.update_state(probabilites)

    def add_stimulation(self, stimulation):
        """Adds stimulation to the network"""
        if isinstance(stimulation, list):
            self.stimulation = lambda t, targets, n_neurons: sum([stimulation[i](t, targets[i], n_neurons) for i in range(len(stimulation))])
            for i, stimulation_i in enumerate(stimulation):
                if isinstance(stimulation_i, BaseStimulation):
                    self._add_parameters_from_stimulation(stimulation_i, prefix=f"stimulation_{i}")
        else:
            self.stimulation = stimulation
            if isinstance(stimulation, BaseStimulation):
                self._add_parameters_from_stimulation(stimulation)
    
    def _add_parameters_from_stimulation(self, stimulation, prefix="stimulation"):
        """Adds the parameters to the model"""
        if prefix:
            params = {f"{prefix}_{key}": value for key, value in stimulation._tunable_params.items()}
        else:
            params = stimulation._params
        self._tunable_params.update(params)

    def stimulate(self, t, targets, n_neurons):
        """Stimulates the network"""
        if targets is None:
            return self.stimulation(t)
        return self.stimulation(t, targets, n_neurons)

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

    @property
    def _tunable_parameter_keys(self):
        """Returns the tunable parameters of the model"""
        return []

    @property
    def tunable_parameters(self):
        """Returns the tunable parameters of the model"""
        return {key: value.data.item() for key, value in self._tunable_params.items()}
    
    @property
    def _parameter_keys(self) -> list:
        """Returns the parameters of the model"""
        return self._default_parameters.keys()
    
    @property
    def parameter_dict(self):
        """Returns the parameters of the model"""
        tunable_parameters = {key: value.data.item() for key, value in self._tunable_params.items()}
        untunable_parameters = {k: v for k, v in self.__dict__.items() if k in self._parameter_keys and k not in tunable_parameters.keys()}
        return {**tunable_parameters, **untunable_parameters}
    
    def set_tunable_parameters(self, params_to_tune):
        """Sets the parameters of the model to tune"""
        self._check_tunable_parameters(params_to_tune)
        for param in params_to_tune:
            self._tunable_params[param].requires_grad = True
    
    def _check_parameters(self, parameters):
        if any([p not in self._parameter_keys for p in parameters]):
            raise ValueError("Parameters {} not recognised".format([p for p in parameters if p not in self._parameter_keys]))
    
    def _check_tunable_parameters(self, tunable_parameters):
        if any([p not in self._tunable_params.keys() for p in tunable_parameters]):
            raise ValueError("Parameters {} cannot be tuned".format(tunable_parameters))
    
    def save(self, path):
        """Saves the model"""
        torch.save(self.state_dict(), path)