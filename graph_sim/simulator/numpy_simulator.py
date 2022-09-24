import numpy as np
from simulator import Simulator
from numpy.random import default_rng
from models import NumpyGraphGLM
from scipy.sparse import csr_array, lil_array

class NumpySimulator(Simulator):
    def __init__(self, n_steps, p_sims, n_neurons, threshold):
        self.model = NumpyGraphGLM(threshold=threshold)
        super().__init__(n_steps, p_sims, n_neurons)

    def run(self, W, edge_index, seed):
        self.rng = default_rng(seed)
        filter_length = W.shape[1]
        x = self._initialize_x(filter_length=filter_length)
        self.model.set_weights(W, edge_index)

        # Equilibrate the system
        equilibration_steps = 100
        x[:, :filter_length] = self._equilibrate(x[:, :filter_length], equilibration_steps)

        # Run the system
        spikes = [[] for _ in range(self.p_sims)]
        for t in range(self.n_steps):
            # new_spikes = np.where(x[:, -1])[0]
            # for i in new_spikes:
                # spikes[np.floor_divide(i, self.n_neurons)].append((i % self.n_neurons, t))

            x[:, filter_length + t] = self._forward(x[:, t:t+filter_length]) # Step forward to next time step

        return x[:, filter_length:]

    def _initialize_x(self, filter_length):
        """Initialize the initial state of the system"""
        x = np.zeros((self.p_sims*self.n_neurons, self.n_steps + filter_length))
        rand_init = self.rng.integers(0, 2, self.p_sims*self.n_neurons)
        x[:, filter_length - 1] = rand_init
        return x

    def _forward_equi(self, x):
        """Forward pass of the simulator"""
        prob = self.model.forward(x)
        x[:, -1] = self.rng.binomial(1, prob, size=prob.shape)
        return np.roll(x, -1, axis=1)

    def _forward(self, x):
        """Forward pass of the simulator"""
        prob = self.model.forward(x)
        return self.rng.binomial(1, prob, size=prob.shape)

    def _to_sparse(self, x):
        """Converts a dense array to a sparse array for each of the p_sims simulations"""
        return [csr_array(x[i*self.n_neurons:(i+1)*self.n_neurons, :]) for i in range(self.p_sims)]

