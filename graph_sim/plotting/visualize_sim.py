import numpy as np
import re
import random
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import torch

def load_data(file):
    params_match = re.search(r".*_(\d+)_neurons_(\d+)_steps.*", str(file))

    n_neurons = int(params_match.group(1))
    n_timesteps = int(params_match.group(2))

    data = np.load(file, allow_pickle=True)

    X_sparse = data["X_sparse"].item()
    # sparse_x = torch.sparse_coo_tensor(X_sparse, torch.ones(X_sparse.shape[1]), size = (n_neurons, n_timesteps))

    X = X_sparse.todense()

    return torch.tensor(X)

def visualize(X):
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


dataset_path = (
    Path("data")
)

directories = list(dataset_path.iterdir())
for i, path in enumerate(directories):
    print(f"{i+1}) {path}")

option = input("\nSelect option: ")

directory = directories[int(option) - 1]
# directory = directories[4]

while True:
    file = random.choice(list(directory.iterdir()))

    X = load_data(file)

    visualize(X)



    # y, y_hat = classify(file)
    # y, y_hat = regress(file)
    # plot_heat_maps(y, y_hat)

    # plot_timestep_dependence(file)

    # plot_classifications(y, y_hat)
    # plot_confusion_matrix(y, y_hat)

    # y, y_hats = prepare_heat_anim(file)
    # animate_heat_map(y, y_hats)
