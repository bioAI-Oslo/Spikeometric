from spiking_network.models.spiking_model import SpikingModel
from spiking_network.w0_generators.w0_dataset import ConnectivityDataset
from spiking_network.w0_generators.w0_generator import W0Generator, GlorotParams
from spiking_network.data_generators.save_functions import save
from pathlib import Path
from tqdm import tqdm
import torch
from torch_geometric.loader import DataLoader


def make_sara_dataset(n_neurons, n_sims, n_steps, data_path, max_parallel):
    """Generates a dataset"""
    # Set data path
    data_path = (
        "spiking_network" / Path(data_path) / f"{n_neurons}_neurons_{n_sims}_datasets_{n_steps}_steps"
    )
    data_path.mkdir(parents=True, exist_ok=True)

    # Parameters for the simulation
    batch_size = min(n_sims, max_parallel)
    n_clusters = 3
    total_neurons = batch_size*n_neurons*n_clusters

    # You can generate a list of W0s here in your own way
    w0_generator = W0Generator(n_clusters=n_clusters, cluster_size=n_neurons, n_cluster_connections=1, dist_params=GlorotParams(0, 5))
    w0_list = w0_generator.generate_list(n_sims, seed=0) 

    # Now you can load the W0s into a dataset in this way. This makes it easy to parallelize.
    # Note that the square w0s will be split into a sparse representation with w0.shape = [n_edges] and edge_index.shape = [2, n_edges]
    w0_data = ConnectivityDataset.from_list(w0_list)
    data_loader = DataLoader(w0_data, batch_size=batch_size, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    for i, batch in enumerate(data_loader):
        batch = batch.to(device)

        # Initalize model
        model = SpikingModel(
            batch.W0, batch.edge_index, total_neurons, seed=i, device=device
        )

        # Simulate the model for n_steps
        spikes = model.simulate(n_steps)

        # Save the data and the model
        save(spikes, model, w0_data, i, data_path) # Insert your own way of saving the data here (see save_functions.py)

if __name__ == "__main__":
    make_dataset(10, 20, 1, 1000, 1)
