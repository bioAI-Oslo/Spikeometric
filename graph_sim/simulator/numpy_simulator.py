import numpy as np
from simulator import Simulator
from numpy.random import default_rng
from models import NumpyGraphGLM
from scipy.sparse import csr_array, lil_array

class NumpySimulator(Simulator):
    def __init__(self, n_steps, p_sims, n_neurons, threshold):
        model = NumpyGraphGLM(threshold=threshold)
        super().__init__(model, n_steps, p_sims, n_neurons)

    def run(self, W, edge_index, seed):
        self.rng = default_rng(seed)
        x = self._initialize_x(filter_length=W.shape[1])
        self.model.set_weights(W, edge_index)

        # Equilibrate the system
        equilibration_steps = 100
        x = self._equilibrate(x, equilibration_steps)

        # Run the system
        spikes = [[] for _ in range(self.p_sims)]
        for t in range(self.n_steps):
            new_spikes = np.where(x[:, -1])[0]
            for i in new_spikes:
                spikes[np.floor_divide(i, self.n_neurons)].append((i % self.n_neurons, t))

            x = self._forward(x)

        return spikes

    def _initialize_x(self, filter_length):
        """Initialize the initial state of the system"""
        x = np.zeros((self.p_sims*self.n_neurons, filter_length))
        rand_init = self.rng.integers(0, 2, self.p_sims*self.n_neurons)
        x[:, -1] = rand_init
        return x

    def _forward(self, x):
        """Forward pass of the simulator"""
        prob = self.model.forward(x)
        tmp = np.zeros_like(x)
        tmp[: , :-1] = x[:, 1:]
        tmp[:, -1] = self.rng.binomial(1, prob, size=prob.shape)
        return tmp

