import numpy as np

class Simulator:
    def __init__(self, n_steps, p_sims, n_neurons):
        self.n_steps = n_steps
        self.p_sims = p_sims
        self.n_neurons = n_neurons

    def _equilibrate(self, x_equi, n_steps):
        """Equilibrate the system in order to get a steady state"""
        for i in range(n_steps):
            x_equi = self._forward_equi(x_equi)
        return x_equi
    
    def _to_numpy(self, spikes):
        """Convert the spikes to numpy"""
        return [np.array(s) for s in spikes]
