
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import torch
import seaborn as sns
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx

sns.set_theme()

def visualize_spikes(X):
    """
    Plots the number of firings per neuron and per timestep.

    Parameters:
    ----------
    X: torch.Tensor
    """
    n_neurons = X.shape[0]
    n_timesteps = X.shape[1]
    n_bins = 100
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(5, 3))

    fig.set_figheight(4.5)
    fig.set_figwidth(12)
    axes[0].set_title("Firings per neuron")
    axes[0].set_ylabel("Firings")
    axes[0].set_xlabel("Neuron")
    axes[0].bar(range(1, n_neurons + 1), torch.sum(X, axis=1), lw=0)

    axes[1].set_title("Firings per timestep")
    axes[1].set_ylabel("Firings")
    axes[1].set_xlabel(f"Timebin ({n_timesteps // n_bins} steps per bin)")

    firings_per_bin = torch.sum(X, axis=0).reshape(n_timesteps // n_bins, -1).sum(axis=0)
    axes[1].plot(
        range(1, n_bins + 1),
        firings_per_bin,
    )

    plt.show()

def visualize_weights(W0):
    """
    Plots a heatmap of the weights of the network.
    
    Parameters:
    ----------
    W0: torch.Tensor
    """
    W0 = W0.clone()
    W0[torch.eye(W0.shape[0], dtype=torch.bool)] = 0
    fig, axs = plt.subplots(figsize=(10, 10))
    fig.tight_layout(pad=3.0)
    fig.set_figheight(10)
    fig.set_figwidth(10)
    axs.set_title(r"$W_0$")
    axs.tick_params(axis="both", which="both", labelbottom=False, labelleft=False, bottom=False, left=False)
    sns.heatmap(W0, ax=axs, square=True, vmin=W0.max() * -1, vmax=W0.max())

    plt.show()

def visualize_time_dependence(W):
    """
    Plots the time dependence of the weights.

    Parameters:
    ----------
    W: torch.Tensor
    """
    fig, axs = plt.subplots(figsize=(5, 10), ncols=2)
    fig.set_figheight(7)
    fig.set_figwidth(10)
    axs[0].plot(torch.arange(10), W[0, 0, :])
    axs[0].set_title("Self-influence")
    axs[0].set_ylabel("Weight")
    axs[0].set_xlabel("Time")

    axs[1].set_title("Influence on other neurons")
    axs[1].set_xlabel("Time")
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            if i != j and W[i, j, 0] > 0.0:
                axs[1].plot(torch.arange(10), W[i, j, :])
                break
        break

    plt.show()
