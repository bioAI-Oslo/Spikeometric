import numpy as np

class Simulator:
    def __init__(self, model, n_steps, p_sims, n_neurons):
        self.model = model
        self.n_steps = n_steps
        self.p_sims = p_sims
        self.n_neurons = n_neurons

    def _equilibrate(self, x, n_steps):
        """Equilibrate the system in order to get a steady state"""
        for i in range(n_steps):
            x = self._forward(x)
        return x
    
    def _to_numpy(self, spikes):
        """Convert the spikes to numpy"""
        return [np.array(s) for s in spikes]
