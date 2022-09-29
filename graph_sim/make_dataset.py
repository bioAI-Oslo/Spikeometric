import argparse
from networks import NetworkBuilder, FilterParams, SpikingNetwork
from pathlib import Path

def make_dataset(n_neurons, n_steps, n_clusters, n_hub_neurons, seed):
    filter_params = FilterParams(n_neurons=n_neurons, n_clusters=n_clusters, n_hub_neurons=n_hub_neurons)
    network_builder = NetworkBuilder(filter_params)

    # Path to save results
    data_path = "data" / Path(f"jakob_{n_neurons}_neurons_{n_steps}_steps")
    data_path.mkdir(parents=True, exist_ok=True)

    network = SpikingNetwork(filter_params, seed=seed)
    network.simulate(n_steps, save_spikes=True, data_path=data_path)

if __name__ == "__main__":
    make_dataset(30, 1000, 3, 1, 0)
