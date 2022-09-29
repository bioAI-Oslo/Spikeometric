import argparse
from network import FilterParams, SpikingNetwork
from pathlib import Path

def make_dataset(n_neurons, n_steps, n_clusters, n_hubneurons, seed):
    filter_params = FilterParams(n_neurons=n_neurons, n_clusters=n_clusters, n_hubneurons=n_hubneurons)

    # Path to save results
    data_path = "data" / Path(f"jakob_{n_neurons}_neurons_{n_steps}_steps")
    data_path.mkdir(parents=True, exist_ok=True)

    network = SpikingNetwork(filter_params, seed=seed)
    network.simulate(n_steps, save_spikes=True, data_path=data_path)

if __name__ == "__main__":
    make_dataset(100, 1000, 5, 5, 0)
