from torch_geometric.nn import MessagePassing
from torch_sparse import SparseTensor
import torch
import torch.nn as nn 
import numpy as np
from torch_scatter import scatter_add
from tqdm import tqdm
from pathlib import Path

class BaseModel(MessagePassing):
    def __init__(self, device="cpu"):
        super(BaseModel, self).__init__(aggr='add')
        self.device = device
    
    @classmethod
    def load(cls, path):
        """Loads the model"""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"File {path} not found, please tune the model first")
        model = cls()
        model.load_state_dict(torch.load(path))
        return model

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward pass of the network"""
        return self.propagate(edge_index, x=x, **kwargs).squeeze()

    def message(self, x_j: torch.Tensor, W: torch.Tensor):
        """Message function"""
        raise NotImplementedError

    def _init_state(self, n_neurons, time_scale):
        """Initializes the state of the network"""
        raise NotImplementedError

    def _update_state(self, activation):
        """Updates the state of the network"""
        raise NotImplementedError

    def _init_parameters(self, params, tuneable, device):
        """Initializes the parameters of the model"""
        return nn.ParameterDict(
                {
                key: nn.Parameter(torch.tensor(value, device=device), requires_grad=key in tuneable)
                for key, value in params.items()
            }
        )

    def connectivity_filter(self, W0, edge_index):
        """Computes the connectivity filter"""
        raise NotImplementedError
    
    def save(self, path):
        """Saves the model"""
        torch.save(self.state_dict(), path)
