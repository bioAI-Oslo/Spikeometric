import torch
from simulator import Simulator
from models import TorchGraphGLM
from scipy.sparse import csr_array

class TorchSimulator(Simulator):
    def __init__(self, n_steps, p_sims, n_neurons, threshold):
        self.model = TorchGraphGLM(threshold=threshold)
        super().__init__(n_steps, p_sims, n_neurons)

    def run(self, W, edge_index, seed):
        """Run the simulator"""
        filter_length = W.shape[1]
        self.rng = torch.Generator().manual_seed(seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x = self._initialize_x(filter_length=filter_length)
        x = self._send_to_device(x, device)


        self.model.set_weights(W, edge_index)

        # Equilibrate the system
        equilibration_steps = 100
        x[:, :filter_length] = self._equilibrate(x[:, :filter_length], equilibration_steps) # Makes sure the system is in a steady state

        # Run the system
        for t in range(self.n_steps):
            x[:, filter_length + t] = self._forward(x[:, t:t+filter_length]) # Step forward to next time step

        return self._to_sparse(x[:, filter_length:])

    def _initialize_x(self, filter_length):
        """Initialize the initial state of the system"""
        x = torch.zeros((self.p_sims*self.n_neurons, self.n_steps+filter_length), dtype=torch.float32)
        rand_init = torch.randint(0, 2, (self.p_sims*self.n_neurons, ), generator=self.rng)
        x[:, filter_length-1] = rand_init
        return x

    def _send_to_device(self, x, device):
        """Send the data to the device"""
        x = x.to(device)
        self.model.to(device)
        return x

    def _forward_equi(self, x):
        """Forward pass of the simulator"""
        prob = self.model(x)
        x[:, -1] = torch.bernoulli(prob, generator=self.rng).squeeze()
        return torch.roll(x, -1, dims=1)

    def _forward(self, x):
        """Forward pass of the simulator"""
        prob = self.model(x)
        return torch.bernoulli(prob).squeeze()

    def _to_sparse(self, spikes):
        """Convert the spikes to sparse matrix"""
        spikes = torch.split(spikes, self.n_neurons, dim=0)
        return [csr_array(s) for s in spikes]


