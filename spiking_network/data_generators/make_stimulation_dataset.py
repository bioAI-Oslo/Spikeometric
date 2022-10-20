from spiking_network.models.stimulation_model import StimulationModel
from spiking_network.connectivity_filters.connectivity_filter import ConnectivityFilter
from spiking_network.stimulation.regular_stimulation import RegularStimulation
from spiking_network.spiking_data.spiking_data import SpikingData
from spiking_network.w0_generators.w0_generator import W0Generator, GlorotParams, NormalParams
from pathlib import Path
from tqdm import tqdm
import torch
from scipy.sparse import coo_matrix
import numpy as np

def save(x, connectivity_filter, n_steps, seed, data_path):
    """Saves the spikes and the connectivity filter to a file"""
    sparse_x = coo_matrix(x)
    sparse_W0 = coo_matrix(connectivity_filter.W0)
    np.savez_compressed(
        data_path,
        X_sparse=sparse_x,
        W=connectivity_filter.W,
        edge_index=connectivity_filter.edge_index,
        w_0=sparse_W0,
        parameters=connectivity_filter.parameters,
        seed=seed,
    )


def make_stimulation_dataset(n_clusters, cluster_size, n_cluster_connections, n_steps, n_datasets, data_path, is_parallel=False):
    """Generates a dataset"""
    # Set data path
    data_path = (
        Path(data_path)
        / f"n_clusters_{n_clusters}_cluster_size_{cluster_size}_n_cluster_connections_{n_cluster_connections}_n_steps_{n_steps}"
    )
    data_path.mkdir(parents=True, exist_ok=True)

    # Set parameters for W0
    dist_params = GlorotParams(0, 5)
    w0_generator = W0Generator(
        n_clusters, cluster_size, n_cluster_connections, dist_params
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    #  device = "cpu"
    with torch.no_grad():  # Disables gradient computation (the models are built on top of torch)
        for i in tqdm(range(n_datasets)):
            w0, n_neurons_list, n_edges_list, hub_neurons = w0_generator.generate(
                i
            )  # Generates a random W0

            connectivity_filter = ConnectivityFilter(
                w0
            )  # Creates a connectivity filter from W0

            stim = RegularStimulation([0, 1, 2], 10, 5, n_steps)  # Creates a regular stimulation

            spiking_data = SpikingData(connectivity_filter, stim).to(
                device
            )

            model = StimulationModel(
                n_steps, seed=i, device=device
            )

            spikes = model(spiking_data.x_dict, spiking_data.edge_index_dict, spiking_data.edge_attr_dict)

            save(spikes, connectivity_filter, n_steps, i, data_path / f"dataset_{i}.npz")



if __name__ == "__main__":
    make_dataset(10, 20, 1, 1000, 1)
