import torch
import torch.nn as nn
from tqdm import tqdm
from spiking_network.models import BaseModel

def simulate(model, data, n_steps, stimulation=None, verbose=True) -> torch.Tensor:
    """
    Simulates the network for n_steps time steps given the connectivity.
    It is also possible to stimulate the network by passing a stimulation function.
    Returns the state of the network at each time step.

    Parameters:
    ----------
    model: BaseModel
    data: torch_geometric.data.Data
    n_steps: int
    stimulation: callable
    verbose: bool

    Returns:
    -------
    x: torch.Tensor
    """
    n_neurons = data.num_nodes
    edge_index = data.edge_index
    W0 = data.W0
    W = model.connectivity_filter(W0, edge_index)
    time_scale = W.shape[1]
    if stimulation is None:
        stimulation = lambda t: torch.zeros(n_neurons, device=model.device)

    if verbose:
        pbar = tqdm(range(time_scale, n_steps + time_scale), colour="#3E5641")
    else:
        pbar = range(time_scale, n_steps + time_scale)

    x = torch.zeros(n_neurons, n_steps + time_scale, device=model.device, dtype=torch.uint8)
    activation = torch.zeros((n_neurons,), device=model.device)
    x[:, :time_scale] = model._init_state(n_neurons, time_scale)
    with torch.no_grad():
        model.eval()
        for t in pbar:
            #  print()
            #  print("\n\x1b[31mForward out:", model.forward(x[:, t-time_scale:t], edge_index, W=W, t=t, activation=activation).shape, "\x1b[0m") # ]]
            #  exit()
            activation = model.forward(x[:, t-time_scale:t], edge_index, W=W, t=t, activation=activation)
            x[:, t] = model._update_state(activation + stimulation(t-time_scale))

    return x[:, time_scale:]

def tune(model, data, firing_rate, lr = 0.01, n_steps=1000, n_epochs=100, verbose=True) -> BaseModel:
    """
    Tunes the model parameters to match the firing rate of the network.

    Parameters:
    ----------
    model: BaseModel
    data: torch_geometric.data.Data
    firing_rate: torch.Tensor
    lr: float
    n_steps: int
    n_epochs: int
    verbose: bool

    Returns:
    -------
    model: BaseModel
    """
    if verbose:
        pbar = tqdm(range(n_epochs), colour="#3E5641")
    else:
        pbar = range(n_epochs)

    edge_index = data.edge_index
    W0 = data.W0
    n_neurons = data.num_nodes
    time_scale = model.connectivity_filter(W0, edge_index).shape[1]

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    firing_rate = torch.tensor(firing_rate, device=model.device)
    for epoch in pbar:
        optimizer.zero_grad()

        # Initialize the state of the network
        x = torch.zeros(n_neurons, n_steps + time_scale, device=model.device)
        activation = torch.zeros((n_neurons, n_steps + time_scale), device=model.device)
        x[:, :time_scale] = model._init_state(n_neurons, time_scale)

        for t in range(time_scale, n_steps + time_scale):
            activation[:, t] = model.forward(x[:, t-time_scale:t], edge_index, W=model.connectivity_filter(W0, edge_index), t=t, activation=activation[:, t-time_scale:t])
            x[:, t] = model._update_state(activation[:, t])

        # Compute the loss
        avg_probability_of_spike = model._spike_probability(activation[:, time_scale:]).mean()
        loss = loss_fn(avg_probability_of_spike, firing_rate)
        if verbose:
            pbar.set_description(f"Tuning... fr={avg_probability_of_spike.item():.5f}")

        # Backpropagate
        loss.backward()
        optimizer.step()
    
    return model
