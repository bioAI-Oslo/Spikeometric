import numpy as np
import re
import random
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import torch
import seaborn as sns
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx

sns.set_theme()

def load_data(file):

    data = np.load(file, allow_pickle=True)

    X_sparse = data["X_sparse"].item()
    X = X_sparse.toarray()

    W0_sparse = data["w_0"].item()
    W0 = W0_sparse.toarray()
    np.fill_diagonal(W0, 0)

    #  W = data["W"]
    #  edge_index = data["edge_index"]

    return X, W0

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
    axes[0].bar(range(1, n_neurons + 1), np.sum(X, axis=1), lw=0)

    axes[1].set_title("Firings per timestep")
    axes[1].set_ylabel("Firings")
    axes[1].set_xlabel(f"Timebin ({n_timesteps // n_bins} steps per bin)")

    firings_per_bin = np.sum(X, axis=0).reshape(n_timesteps // n_bins, -1).sum(axis=0)
    axes[1].plot(
        range(1, n_bins + 1),
        firings_per_bin,
    )

    plt.show()

def visualize_weights(W0):
    W0 = torch.from_numpy(W0)
    edge_index = W0.nonzero().t()
    fig, axs = plt.subplots(figsize=(10, 10), ncols=2)
    fig.tight_layout(pad=3.0)
    fig.set_figheight(10)
    fig.set_figwidth(10)
    axs[1].set_title(r"$W_0$")
    axs[1].tick_params(axis="both", which="both", labelbottom=False, labelleft=False, bottom=False, left=False)

    sns.heatmap(W0, ax=axs[1], square=True, vmin=W0.max() * -1, vmax=W0.max())

    colors = [float(w) for w in W0[edge_index[0], edge_index[1]]]
    data = Data(num_nodes = W0.shape[0], edge_index=edge_index, edge_attr=colors)
    graph = to_networkx(data, remove_self_loops=True)
    pos = nx.nx_agraph.graphviz_layout(graph, prog="neato")
    nx.draw(graph, pos, with_labels=False, node_size=10, edge_color=data.edge_attr, edge_vmin=W0.max()*-1, edge_vmax=W0.max(), arrowsize=5, ax=axs[0])
    axs[0].set_title("Network graph")

    plt.show()

def reconstruct_full_W(W, edge_index):
    n_neurons = edge_index.max() + 1
    W = W.flip(dims=(1,))
    sparse_W = torch.sparse_coo_tensor(edge_index, W, size=(n_neurons, n_neurons, W.shape[1]))
    return sparse_W.to_dense()


def main():
    dataset_path = (
        Path("data")
    )

    directories = list(dataset_path.iterdir())
    for i, path in enumerate(directories):
        print(f"{i+1}) {path}")

    option = input("\nSelect option: ")

    directory = directories[int(option) - 1]

    while True:
        files = list(directory.iterdir())
        for i, path in enumerate(files):
            print(f"{i+1}) {path}")

        option = input("\nSelect option: ")
        file = files[int(option) - 1]

        X, W0 = load_data(file)

        visualize_spikes(X)
        #  visualize_weights(W0)

if __name__ == "__main__":
    main()


