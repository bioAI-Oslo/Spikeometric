import torch
from simulator import Simulator
from models import TorchGraphGLM

class TorchSimulator(Simulator):
    def __init__(self, n_steps, p_sims, n_neurons, threshold):
        model = TorchGraphGLM(threshold=threshold)
        super().__init__(model, n_steps, p_sims, n_neurons)

    def run(self, W, edge_index, seed):
        """Run the simulator"""
        self.rng = torch.Generator().manual_seed(seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x = self._initialize_x(filter_length=W.shape[1])
        x = self._send_to_device(x, device)
        
        self.model.set_weights(W, edge_index)

        # Equilibrate the system
        equilibration_steps = 100
        x = self._equilibrate(x, equilibration_steps) # Makes sure the system is in a steady state

        # Run the system
        spikes = [[] for _ in range(self.p_sims)]
        for t in range(self.n_steps):
            new_spikes = torch.where(x[:, -1])[0] # Check which neurons spiked
            for i in new_spikes:
                spikes[torch.div(i, self.n_neurons, rounding_mode="floor")].append((i % self.n_neurons, t)) # Save the spikes

            x = self._forward(x) # Step forward to next time step

        return self._to_numpy(spikes)

    def _initialize_x(self, filter_length):
        """Initialize the initial state of the system"""
        x = torch.zeros((self.p_sims*self.n_neurons, filter_length))
        rand_init = torch.randint(0, 2, (self.p_sims*self.n_neurons, ), generator=self.rng)
        x[:, -1] = rand_init
        return x

    def _send_to_device(self, x, device):
        """Send the data to the device"""
        x = x.to(device)
        self.model.to(device)
        return x

    def _forward(self, x):
        """Forward pass of the simulator"""
        prob = self.model(x)
        x[:, -1] = torch.bernoulli(prob, generator=self.rng).squeeze()
        return torch.roll(x, -1, dims=1)


