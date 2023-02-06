from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data
import torch
from tqdm import tqdm
from typing import Union

class BaseModel(MessagePassing):
    """
    Base class for all spiking neural networks.
    
    Extends the MessagePassing class from torch_geometric by adding stimulation support and
    a forward method that calculates the spikes of the network at time t using the following steps:
    - inpit: calculates the input to each neuron
    - non_linearity: applies a non-linearity to the input to get the neuron's response
    - emit_spikes: Determines the spikes of the network at time t from the response
    These methods are overriden by the child classes to implement different models.

    To simulate the network, a default simulate method is provided, but can be overriden by the child classes
    to implement different simulation methods if needed.

    If the models has any tunable parameters, they can be tuned to match a desired firing rate using the tune method.
    For other target functions, the tune method can be overriden by the child classes.

    There are also methods for saving and loading the model.
    """
    def __init__(self):
        super().__init__()
        self.stimulation = lambda t: 0 # The default stimulation function

    @property
    def tunable_parameters(self) -> dict:
        """Returns a list of the tunable parameters"""
        return dict(self.named_parameters())
    
    def connectivity_filter(self, W0: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        r"""
        Returns the connectivity filter of the network.
        The connectivity filter determines the time dependency of the weights of the network.
        The default connectivity filter is just the initial synaptic weights, which means that the spikes only
        affect the neurons for one time step.

        Parameters
        ----------
        W0: torch.Tensor
            The initial synaptic weights of the network [n_edges, T]
        edge_index: torch.Tensor
            The connectivity of the network [2, n_edges]
        
        Returns
        -------
        W: torch.Tensor
            The connectivity filter of the network [n_edges, T]
        """
        return W0.unsqueeze(1)

    def input(self, edge_index: torch.Tensor, W: torch.Tensor, state: torch.Tensor, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    def emit_spikes(self, inputs: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
    
    def non_linearity(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs
    
    def synaptic_input(self, edge_index: torch.Tensor, W: torch.Tensor, state: torch.Tensor, **kwargs) -> torch.Tensor:
        r"""
        Calculates the synaptic input to each neuron torch_geometric's message passing framework.
        The propagate method fist calls the message method to compute the message along each edge and 
        then aggregates the messages using the aggregation method (sum in this case). The result is then
        passed to the update method to compute the new state of the neurons. We only override the
        message method, and use the default aggregation and update methods.
        
        .. math::
            I_i(t) = \sum_{j \in \mathcal{N}_i} \mathbf{W}_{ij} \cdot \mathbf{x}_j(t)

        Parameters
        ----------
        edge_index: torch.Tensor
            The connectivity of the network [2, n_edges]
        W: torch.Tensor
            The weights of the edges [n_edges, T]
        state: torch.Tensor
            The state of the neurons [n_neurons, T]
        **kwargs:
            Additional arguments
        
        Returns
        -------
        synaptic_input: torch.Tensor
            The synaptic input to each neuron [n_neurons, 1]
        """
        return self.propagate(edge_index, W=W, state=state, **kwargs).squeeze()
    
    def message(self, state_j: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
        r"""
        Calculates the message from the j-th neuron to the i-th neuron. This method is called by the propagate method
        of torch_geometric's MessagePassing class.

        .. math::
            m_{ij} = \mathbf{W}_{ij} \cdot \mathbf{x}_j(t)
        
        Parameters
        ----------
        state_j: torch.Tensor
            The state of the j-th neuron [n_edges, T]
        W: torch.Tensor
            The weights of the edges [n_edges, T]
        
        Returns
        -------
        message: torch.Tensor
            The message from the j-th neuron to the i-th neuron [n_edges, 1]
        """
        return torch.sum(state_j*W, dim=1, keepdim=True)

    def stimulation_input(self, t: int, stimulus_mask: torch.Tensor) -> torch.Tensor:
        r"""
        Calculates the stimulation input to the network at time t.

        Parameters
        ----------
        t: int
            The current time step
        stimulus_mask: torch.Tensor[bool]
            A boolean tensor indicating which neurons are targeted by the stimulus [n_neurons]
        
        Returns
        -------
        stimulation_input: torch.Tensor
            The stimulation input to the network [n_neurons]
        """
        return self.stimulation(t*self.dt) * stimulus_mask

    def add_stimulation(self, stimulation: callable):
        """Adds a stimulus to the network"""
        if not callable(stimulation):
            raise TypeError("The stimulation must be a callable function")
        self.stimulation = stimulation

    def forward(self, edge_index: torch.Tensor, W: torch.Tensor, state: torch.Tensor, **kwargs) -> torch.Tensor:
        r"""
        Calculates the new state of the network at time t+1 from the state at time t.

        Parameters
        -----------
        edge_index: torch.Tensor
            The connectivity of the network [2, n_edges]
        W: torch.Tensor
            The edge weights of the connectivity filter [n_edges, T]
        state: torch.Tensor
            The state of the network from time t - T to time t [n_neurons, T]

        Returns
        --------
        spikes: torch.Tensor
            The spikes of the network at time t+1 [n_neurons]
        """
        input = self.input(
            edge_index=edge_index,
            W=W,
            state=state,
            **kwargs
        )
        rate = self.non_linearity(input)
        return self.emit_spikes(rate)
    
    def simulate(self, data, n_steps, stimulation=None, verbose=True, equilibration_steps=0):
        r"""
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
            The number of time steps to simulate before the we start recording the state of the network.

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
        stimulus_mask = data.stimulus_mask if hasattr(data, "stimulus_mask") else False
        device = edge_index.device

        # If verbose is True, a progress bar is shown
        pbar = tqdm(range(T, n_steps + T), colour="#3E5641") if verbose else range(T, n_steps + T)
        
        # Simulate the network
        x = torch.zeros((n_neurons, n_steps + T), device=device, dtype=torch.int)
        inital_state = torch.randint(0, 2, device=device, size=(n_neurons,), generator=self._rng)
        x[:, :T] = self.equilibrate(edge_index, W, inital_state, n_steps=equilibration_steps)
        for t in pbar:
            x[:, t] = self(edge_index=edge_index, W=W, state=x[:, t-T:t], t=t-T, stimulus_mask=stimulus_mask)
        
        # Return the state of the network at each time step
        return x[:, T:]
    
    def tune(
        self,
        data: Data,
        firing_rate: float,
        tunable_parameters: Union[str, list[str]] = "all",
        lr: float = 0.1,
        n_steps: int = 100,
        n_epochs: int = 100,
        verbose: bool = True
    ):
        """
        Tunes the model parameters to match a firing rate.

        Parameters
        -----------
        data: torch_geometric.data.Data
            The training data containing the connectivity.
        firing_rate: torch.Tensor
            The target firing rate of the network
        stimulation: torch.nn.Module
            The stimulation module. If None, no stimulation is applied.
        tunable_parameters: list or str
            The list of parameters to tune, can be "all", "stimulation", "model" or a list of parameter names
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
        elif tunable_parameters == "stimulation":
            tunable_parameters = [param for param in self.tunable_parameters if param.startswith("stimulation")]
        elif tunable_parameters == "model":
            tunable_parameters = [param for param in self.tunable_parameters if not param.startswith("stimulation")]
        elif any([param not in self.tunable_parameters for param in tunable_parameters]):
            raise ValueError("Invalid parameter name. Valid parameter names are: {}".format(self.tunable_parameters))
            
        self.set_tunable(tunable_parameters)
        
        # Get the parameters of the network
        edge_index = data.edge_index
        W0 = data.W0
        n_neurons = data.num_nodes
        stimulus_mask = data.stimulus_mask if hasattr(data, "stimulus_mask") else False
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        loss_fn = torch.nn.MSELoss()
        firing_rate = torch.tensor(firing_rate, device=device, dtype=torch.float)
        self.train()
        for _ in pbar:
            optimizer.zero_grad()
            
            # Compute the connectivity matrix using the current parameters
            W = self.connectivity_filter(W0, edge_index)
            T = W.shape[1]

            # Initialize the state of the network
            x = torch.zeros(n_neurons, n_steps + T, device=device, dtype=torch.int)
            activation = torch.zeros((n_neurons, n_steps + T), device=device)
            initial_state = torch.randint(0, 2, (n_neurons,), device=device, dtype=torch.int, generator=self._rng)
            with torch.no_grad():
                x[:, :T] = self.equilibrate(edge_index, W, initial_state, n_steps=10)

            # Simulate the network
            for t in range(T, n_steps + T):
                activation[:, t] = self.input(
                    edge_index,
                    W=W,
                    state=x[:, t-T:t],
                    t=t,
                    stimulus_mask=stimulus_mask,
                )
                x[:, t] = self.emit_spikes(
                    self.non_linearity(activation[:, t]),
                )

            # Compute the loss
            firing_rate_hat = self.non_linearity(activation[:, T:]).mean() * 1000 / self.dt

            loss = loss_fn(firing_rate, firing_rate_hat)
            if verbose:
                pbar.set_description(f"Tuning... fr={firing_rate_hat.item():.5f}")

            # Backpropagate
            loss.backward()
            optimizer.step()
        
        self.requires_grad_(False)
        
    def set_tunable(self, parameters: list):
        """Sets requires_grad to True for the parameters to be tuned"""
        for param in parameters:
            parameter_dict = dict(self.named_parameters())
            if param not in parameter_dict:
                raise ValueError(f"Parameter {param} not found in the model")
            parameter_dict[param].requires_grad = True

    def save(self, path: str):
        """Saves the model to the path"""
        torch.save(self.state_dict(), path)

    def load(self, path: str):
        """Loads the model from the path"""
        self.load_state_dict(torch.load(path))

    def to(self, device: str):
        """Moves the model to the device, including the random number generator"""
        self = super().to(device)
        if hasattr(self, "_rng"):
            if device == "cpu":
                state = self._rng.get_state()
                self._rng = torch.Generator().set_state(state)
            else:
                seed = self._rng.seed()
                self._rng = torch.Generator(device=device).manual_seed(seed)

        return self

    def equilibrate(self, edge_index: torch.Tensor, W: torch.Tensor, inital_state: torch.Tensor, n_steps=100) -> torch.Tensor:
        """
        Equilibrate the network to a given connectivity matrix.

        Parameters
        -----------
        data: torch_geometric.data.Data
            The data containing the connectivity matrix
        n_steps: int
            The number of time steps to simulate for

        Returns
        --------
        x: torch.Tensor
            The state of the network at each time step
        """
        n_neurons = inital_state.shape[0]
        device = inital_state.device
        T = W.shape[1]
        x_equi = torch.zeros((n_neurons, T + n_steps), device=device, dtype=torch.int)
        x_equi[:, T-1] = inital_state

        # Equilibrate the network
        for t in range(T, T + n_steps):
            x_equi[:, t] = self(edge_index=edge_index, W=W, state=x_equi[:, t-T:t])
        
        return x_equi[:, -T:]


