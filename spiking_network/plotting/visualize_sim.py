import numpy as np
import re
import random
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import torch
import seaborn as sns

sns.set_theme()

def load_data(file):
    params_match = re.search(r".*_(\d+)_neurons_(\d+)_steps.*", str(file))

    n_neurons = int(params_match.group(1))
    n_timesteps = int(params_match.group(2))

    data = np.load(file, allow_pickle=True)

    X_sparse = data["spikes"].item()
    X = X_sparse.todense()

    W = data["W"]
    edge_index = data["edge_index"]

    return torch.tensor(X), torch.tensor(W), torch.tensor(edge_index)

def visualize_spikes(X):
    n_neurons = X.shape[0]
    n_timesteps = X.shape[1]
    n_bins = 100
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(5, 3))

    fig.set_figheight(4.5)
    fig.set_figwidth(12)
    axes[0].set_title("Firings per neuron")
    axes[0].set_ylabel("Firings")
    axes[0].set_xlabel("Neuron")
    axes[0].bar(range(1, n_neurons + 1), torch.sum(X, axis=1))

    axes[1].set_title("Firings per timestep")
    axes[1].set_ylabel("Firings")
    axes[1].set_xlabel("Timebin (100 steps per bin)")

    firings_per_bin = torch.sum(X, axis=0).reshape(n_timesteps // n_bins, -1).sum(axis=0)
    axes[1].plot(
        range(1, n_bins + 1),
        firings_per_bin,
    )

    plt.show()

def visualize_weights(W, edge_index):
    W = reconstruct_full_W(W, edge_index)
    W0 = W[:, :, 0].fill_diagonal_(0)
    fig, ax = plt.subplots(figsize=(10, 10))
    plt.title(r"$W_0$")
    sns.heatmap(W0, ax=ax, square=True, vmin=W0.max() * -1, vmax=W0.max())
    plt.xlabel("Target neuron")
    plt.ylabel("Source neuron")
    plt.show()

def reconstruct_full_W(W, edge_index):
    n_neurons = edge_index.max() + 1
    W = W.flip(dims=(1,))
    sparse_W = torch.sparse_coo_tensor(edge_index, W, size=(n_neurons, n_neurons, W.shape[1]))
    return sparse_W.to_dense()

dataset_path = (
    Path("data")
)

directories = list(dataset_path.iterdir())
for i, path in enumerate(directories):
    print(f"{i+1}) {path}")

option = input("\nSelect option: ")

directory = directories[int(option) - 1]

while True:
    file = random.choice(list(directory.iterdir()))

    X, W, edge_index = load_data(file)

    visualize_spikes(X)
    visualize_weights(W, edge_index)


