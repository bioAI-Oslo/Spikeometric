import torch
import torch.nn as nn
from tqdm import tqdm
from spiking_network.models import BaseModel
from spiking_network.stimulation.base_stimulation import BaseStimulation
import numpy as np
from torch_geometric.loader import DataLoader
from scipy.sparse import coo_matrix
from pathlib import Path

class StimulationLoader(DataLoader):
    def __init__(self, data, stimulation_targets, batch_size=1, shuffle=False):
        super().__init__(data, batch_size, shuffle)
        if all([isinstance(stimulation_target, int) for stimulation_target in stimulation_targets]):
            stimulation_targets = [torch.tensor(stimulation_targets)]
        if isinstance(stimulation_targets, torch.Tensor) and stimulation_targets.dim() == 1:
            stimulation_targets = [stimulation_targets]

        if not len(stimulation_targets) == len(data):
            raise ValueError(f"Must have stimulation targets for each graph in the dataset ({len(data)})")
        
        self.n_neurons_list = [data[i].num_nodes for i in range(len(data))]
        self.stimulation_targets = stimulation_targets

    def __iter__(self):
        for i, batch in enumerate(super().__iter__()):
            batch = self._add_stimulation_targets(batch, self.stimulation_targets, i)
            yield batch

    def _add_stimulation_targets(self, batch, stimulation_targets, batch_idx):
        """Adds stimulation targets to the data"""
        batch_stimulation_targets = stimulation_targets[self.batch_size * batch_idx : self.batch_size * (batch_idx + 1)]
        batch_adjusted_stimulation_target = torch.cat([batch_stimulation_targets[i] + i*self.n_neurons_list[i] for i in range(batch.num_graphs)], dim=-1)
        batch.stimulation_targets = batch_adjusted_stimulation_target
        return batch

def load_data(file):
    """
    Loads the data from the given file.

    Parameters:
    ----------
    file: str

    Returns:
    -------
    X: torch.Tensor
    W0: torch.Tensor
    """
    data = np.load(file, allow_pickle=True)

    X_sparse = data["X_sparse"].item()
    X = X_sparse.toarray()

    W0_sparse = data["w_0"].item()
    W0 = W0_sparse.toarray()

    return torch.from_numpy(X), torch.from_numpy(W0)

def save_data(x, model, w0_data, seed, data_path, stimulation=None):
    """Saves the spikes and the connectivity filter to a file"""
    if not isinstance(x, torch.Tensor):
        x = torch.cat(x, dim=0)
    x = x.cpu()
    xs = torch.split(x, w0_data[0].num_nodes, dim=0)
    for i, (x, network) in enumerate(zip(xs, w0_data)):
        sparse_x = coo_matrix(x)
        sparse_W0 = coo_matrix((network.W0, network.edge_index), shape=(network.num_nodes, network.num_nodes))
        np.savez_compressed(
            data_path / Path(f"{i}.npz"),
            X_sparse=sparse_x,
            w_0=sparse_W0,
            parameters=dict(model.state_dict()),
            stimulation=stimulation.parameter_dict if stimulation else None,
            seed=seed
        )

def calculate_isi(spikes, N, n_steps, dt=0.001) -> float:
    """
    Calculates the interspike interval of the network.

    Parameters:
    ----------
    spikes: torch.Tensor
        The spikes of the network
    N: int
        The number of neurons in the network
    n_steps: int
        The number of time steps the network was simulated for
    dt: float
        The time step size
    """
    return N * n_steps * dt / spikes.sum()

def calculate_firing_rate(spikes) -> float:
    return spikes.float().mean()

def simulate(model, data, n_steps, verbose=True) -> torch.Tensor:
    """
    Simulates the network for n_steps time steps given the connectivity.
    It is also possible to stimulate the network by passing a stimulation function.
    Returns the state of the network at each time step.

    Parameters:
    ----------
    model: BaseModel
        The model to use for the simulation
    data: torch_geometric.data.Data
        The data containing the connectivity. 
    n_steps: int
        The number of time steps to simulate
    stimulation: callable
        A function that takes the current time step and returns the stimulation at that time step
    verbose: bool
        If True, a progress bar is shown

    Returns:
    -------
    x: torch.Tensor[n_neurons, n_steps]
        The state of the network at each time step. The state is a binary tensor where 1 means that the neuron is active.
    """
    # Get the parameters of the network
    n_neurons = data.num_nodes
    edge_index = data.edge_index
    W0 = data.W0
    W = model.connectivity_filter(W0, edge_index)
    stimulation_targets = data.stimulation_targets if hasattr(data, "stimulation_targets") else None
    device = edge_index.device

    # If verbose is True, a progress bar is shown
    if verbose:
        pbar = tqdm(range(model.time_scale, n_steps + model.time_scale), colour="#3E5641")
    else:
        pbar = range(model.time_scale, n_steps + model.time_scale)
    
    # Initialize the state of the network
    x = torch.zeros(n_neurons, n_steps + model.time_scale, device=device, dtype=torch.uint8)
    activation = torch.zeros(n_neurons, device=device)
    x[:, :model.time_scale] = model.initialize_state(n_neurons)

    # Simulate the network
    model.eval()
    with torch.no_grad():
        for t in pbar:
            x[:, t] = model(x[:, t-model.time_scale:t], edge_index, W=W, t=t, current_activation=activation, stimulation_targets=stimulation_targets)

    # Return the state of the network at each time step
    return x[:, model.time_scale:]

def tune(model,
        data,
        firing_rate,
        tunable_parameters="all",
        lr = 0.1,
        n_steps=100,
        n_epochs=100,
        verbose=True
    ):
    """
    Tunes the model parameters to match the firing rate of the network.

    Parameters:
    ----------
    model: BaseModel
        The model to tune
    data: torch_geometric.data.Data
        The training data containing the connectivity.
    firing_rate: torch.Tensor
        The target firing rate of the network
    tunable_parameters: list
        The list of parameters to tune
    lr: float
        The learning rate
    n_steps: int
        The number of time steps to simulate for each epoch
    n_epochs: int
        The number of epochs
    verbose: bool
        If True, a progress bar is shown

    Returns:
    -------
    model: BaseModel
        The tuned model
    """
    # If verbose is True, a progress bar is shown
    if verbose:
        pbar = tqdm(range(n_epochs), colour="#3E5641")
    else:
        pbar = range(n_epochs)
    
    # Get the device to use
    device = data.edge_index.device

    # Check parameters
    if not tunable_parameters:
        raise ValueError("No parameters to tune")
    elif tunable_parameters == "all":
        tunable_parameters = [param for param, val in model.named_parameters()]
    elif tunable_parameters == "stimulation_parameters":
        tunable_parameters = [param for param, val in model.named_parameters() if param.startswith("stimulation")]
    elif tunable_parameters == "model_parameters":
        tunable_parameters = [param for param, val in model.named_parameters() if not param.startswith("stimulation")]
    elif any([param not in [param for param, val in model.named_parameters()] for param in tunable_parameters]):
        raise ValueError("Invalid parameter name. Valid parameter names are: {}".format([param for param, val in model.named_parameters()]))
        
    model.tune(tunable_parameters)
    
    # Get the parameters of the network
    edge_index = data.edge_index
    W0 = data.W0
    n_neurons = data.num_nodes
    time_scale = model.time_scale
    stimulation_targets = data.stimulation_targets if hasattr(data, "stimulation_targets") else None
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    firing_rate = torch.tensor(firing_rate, device=device)
    model.train()
    for epoch in pbar:
        optimizer.zero_grad()

        # Initialize the state of the network
        x = torch.zeros(n_neurons, n_steps + time_scale, device=device)
        activation = torch.zeros((n_neurons, n_steps + time_scale), device=device)
        x[:, :time_scale] = model.initialize_state(n_neurons)

        # Compute the connectivity matrix using the current parameters
        W = model.connectivity_filter(W0, edge_index)

        # Simulate the network
        for t in range(time_scale, n_steps + time_scale):
            activation[:, t] = model.activation(
                x[:, t-time_scale:t],
                edge_index,
                W=W,
                t=t,
                current_activation=activation[:, t-time_scale:t],
                stimulation_targets=stimulation_targets
            )
            x[:, t] = model.update_state(activation[:, t])

        # Compute the loss
        avg_probability_of_spike = model._probability_of_spike(activation[:, time_scale:]).mean()
        loss = loss_fn(avg_probability_of_spike, firing_rate)
        if verbose:
            pbar.set_description(f"Tuning... fr={avg_probability_of_spike.item():.5f}")

        # Backpropagate
        loss.backward(retain_graph=True)
        optimizer.step()

    return model
    
