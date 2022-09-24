import torch
from simulator.torch_simulator import TorchSimulator
from models import SparseGraphGLM
from simulator.simulator import Simulator

class SparseSimulator(TorchSimulator):
    def __init__(self, n_steps, p_sims, n_neurons, threshold):
        self.model = SparseGraphGLM(threshold=threshold)
        super(TorchSimulator, self).__init__(n_steps, p_sims, n_neurons)

    def _forward(self, x):
        y = x.to_sparse()
        return super()._forward(y)

    def _forward_equi(self, x):
        y = x.to_sparse()
        probs = self.model.forward(y)
        x[:, -1] = torch.bernoulli(probs, generator=self.rng).squeeze()
        return torch.roll(x, -1, dims=1)
