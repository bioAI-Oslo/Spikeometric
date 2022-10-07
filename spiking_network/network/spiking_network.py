from torch_geometric.nn import MessagePassing
import dataclasses
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
import numpy as np
import torch
from pathlib import Path
from spiking_network.network.abstract_network import AbstractNetwork
from spiking_network.network.filter_params import FilterParams, DistributionParams

class SpikingNetwork(AbstractNetwork):
    def __init__(self, connectivity_filter, seed, trainable=False):
        super().__init__(connectivity_filter, seed, trainable)

    def message(self, x_j: torch.Tensor) -> torch.Tensor:
        """Defines the message function"""
        if self.trainable:
            self.connectivity_filter.update(self.params)
        return torch.sum(x_j * self.connectivity_filter.W, dim=1, keepdim=True)

    def update(self, activation: torch.Tensor) -> torch.Tensor:
        """Calculates new spikes based on the activation of the neurons"""
        probs = torch.sigmoid(activation + self.connectivity_filter.filter_parameters["threshold"]) # Calculates the probability of a neuron firing
        return torch.bernoulli(probs, generator=self.rng).squeeze()

