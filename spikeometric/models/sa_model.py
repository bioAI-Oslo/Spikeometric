from spikeometric.models.base_model import BaseModel
import torch
from tqdm import tqdm
from spikeometric.stimulus import BaseStimulus
from torch_geometric.data import Data

class SAModel(BaseModel):
    r"""
    The Synaptic Activation model (SAModel) is a base model
    for models that use the synaptic activation as the state of the network and has an update rule
    for based on previous synaptic activation and spikes.

    In addition to the input, non_linearity and emit_spikes methods, SAModels must implement the update_activation method.
    """
    def __init__(self):
        super().__init__()
    
    def update_activation(self, spikes, activation):
        r"""The update rule for the synaptic activation."""
        raise NotImplementedError
    
    def simulate(self, data: Data, n_steps: int, verbose: bool =True, equilibration_steps: int =100, store_as_dtype: torch.dtype = torch.int):
        """
        Simulates the network for n_steps time steps given the connectivity.
        Returns the state of the network at each time step.

        Parameters
        -----------
        data: torch_geometric.data.Data
            The data containing the connectivity. 
        n_steps: int
            The number of time steps to simulate
        verbose: bool
            If True, a progress bar is shown
        equilibration: int
            The number of time steps to simulate before starting to record the state of the network.
        store_as_dtype: torch.dtype
            The dtype to store the state of the network

        Returns
        --------
        x: torch.Tensor[n_neurons, n_steps]
            The state of the network at each time step. The state is a binary tensor where 1 means that the neuron is active.
        """
        # Get the parameters of the network
        n_neurons = data.num_nodes
        edge_index = data.edge_index
        W0 = data.W0
        W = self.connectivity_filter(W0, edge_index)
        T = W.shape[1]
        device = edge_index.device

        # If verbose is True, a progress bar is shown
        pbar = tqdm(range(n_steps), colour="#3E5641") if verbose else range(n_steps)
        
        # Initialize the state of the network
        x = torch.zeros(n_neurons, n_steps, device=device, dtype=store_as_dtype)
        initial_activation = torch.rand((n_neurons, T), device=device)
        activation = self.equilibrate(edge_index, W, initial_activation, equilibration_steps, store_as_dtype=store_as_dtype)

        # Simulate the network
        for t in pbar:
            x[:, t] = self(edge_index=edge_index, W=W, state=activation, t=t)
            activation = self.update_activation(spikes=x[:, t:t+T], activation=activation)

        # If the stimulus is batched, we increment the batch in preparation for the next batch
        if isinstance(self.stimulus, BaseStimulus) and self.stimulus.n_batches > 1:
            self.stimulus.next_batch()

        # Return the state of the network at each time step
        return x

    def tune(
        self,
        data,
        firing_rate,
        tunable_parameters="all",
        lr = 0.1,
        n_steps=100,
        n_epochs=100,
        verbose=True
    ):
        """
        Tunes the model parameters to match a desired firing rate.

        Parameters
        -----------
        data: torch_geometric.data.Data
            The training data containing the connectivity.
        firing_rate: torch.Tensor
            The target firing rate of the network
        tunable_parameters: list or str
            The list of parameters to tune, can be "all", "stimulus", "model" or a list of parameter names
        lr: float
            The learning rate
        n_steps: int
            The number of time steps to simulate for each epoch
        n_epochs: int
            The number of epochs
        verbose: bool
            If True, a progress bar is shown
        """
        # If verbose is True, a progress bar is shown
        pbar = tqdm(range(n_epochs), colour="#3E5641") if verbose else range(n_epochs)
        
        # Get the device to use
        device = data.edge_index.device

        # Check parameters
        if not tunable_parameters:
            raise ValueError("No parameters to tune")
        elif not self.tunable_parameters:
            raise ValueError("The model has no tunable parameters")
        elif tunable_parameters == "all":
            tunable_parameters = self.tunable_parameters
        elif tunable_parameters == "stimulus":
            tunable_parameters = [param for param in self.tunable_parameters if param.startswith("stimulus")]
        elif tunable_parameters == "model":
            tunable_parameters = [param for param in self.tunable_parameters if not param.startswith("stimulus")]
        elif any([param not in self.tunable_parameters for param in tunable_parameters]):
            raise ValueError("Invalid parameter name. Valid parameter names are: {}".format(self.tunable_parameters))
            
        self.set_tunable(tunable_parameters)
        
        # Get the parameters of the network
        edge_index = data.edge_index
        W0 = data.W0
        W = self.connectivity_filter(W0, edge_index)
        T = W.shape[1]
        n_neurons = data.num_nodes
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        loss_fn = torch.nn.MSELoss()
        firing_rate = torch.tensor(firing_rate, device=device, dtype=torch.float)
        self.train()
        average_firing_rate = 0
        for epoch in pbar:
            optimizer.zero_grad()

            # Initialize the state of the network
            x = torch.zeros(n_neurons, n_steps, device=device)
            activation = torch.rand((n_neurons, 1), device=device)
            input = torch.zeros((n_neurons, n_steps), device=device)
            x[:, 0] = torch.randint(0, 2, (n_neurons,), device=device)

            # Simulate the network
            for t in range(1, n_steps):
                input[:, t] = self.input(
                    edge_index,
                    W=W,
                    state=activation,
                    t=t,
                )
                x[:, t] = self.emit_spikes(
                    self.non_linearity(input[:, t]),
                )
                activation = self.update_activation(
                    activation=activation,
                    spikes=x[:, t:t+1]
                )

            # Compute the loss
            firing_rate_hat =  self.dt / self.non_linearity(input[:, T:]).mean()
            loss = loss_fn(firing_rate_hat, firing_rate)
            average_firing_rate += (firing_rate_hat.item() - average_firing_rate) / (epoch + 1)

            if verbose:
                pbar.set_description(f"Tuning... fr={average_firing_rate:.5f}")

            # Backpropagate
            loss.backward()
            optimizer.step()

        # If the stimulus is batched, we increment the batch in preparation for the next batch
        if isinstance(self.stimulus, BaseStimulus) and self.stimulus.n_batches > 1:
            self.stimulus.next_batch()

        self.requires_grad_(False) # Freeze the parameters 

    def equilibrate(self, edge_index: torch.Tensor, W: torch.Tensor, initial_state: torch.Tensor, n_steps=100, store_as_dtype: torch.dtype = torch.int) -> torch.Tensor:
        """
        Equilibrate the network to a given connectivity matrix.

        Parameters
        -----------
        edge_index: torch.Tensor
            The connectivity of the network
        W: torch.Tensor
            The connectivity filter
        initial_state: torch.Tensor
            The initial state of the network
        n_steps: int
            The number of time steps to equilibrate for
        store_as_dtype: torch.dtype
            The dtype to store the state of the network as

        Returns
        --------
        x: torch.Tensor
            The state of the network at each time step
        """
        n_neurons = initial_state.shape[0]
        device = initial_state.device
        x_equi = torch.zeros((n_neurons, self.T + n_steps), device=device, dtype=store_as_dtype)
        activation_equi = initial_state

        # Equilibrate the network
        for t in range(self.T, self.T + n_steps):
            x_equi[:, t] = self(edge_index=edge_index, W=W, state=activation_equi)
            activation_equi = self.update_activation(spikes=x_equi[:, t:t+self.T], activation=activation_equi)
        
        return activation_equi