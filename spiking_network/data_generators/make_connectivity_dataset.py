from spiking_network.models.spiking_model import SpikingModel
from spiking_network.w0_generators.w0_dataset import W0Dataset
from spiking_network.stimulation.regular_stimulation import RegularStimulation
from spiking_network.stimulation.poisson_stimulation import PoissonStimulation
from spiking_network.stimulation.mixed_stimulation import MixedStimulation
from spiking_network.stimulation.sin_stimulation import SinStimulation
from spiking_network.w0_generators.w0_generator import GlorotParams, NormalParams
from spiking_network.data_generators.save_functions import save
from spiking_network.models.connectivity_model import ConnectivityModel
from spiking_network.connectivity_filters.connectivity_filter import ConnectivityFilter
from pathlib import Path
from tqdm import tqdm
import torch
from torch_geometric.loader import DataLoader


def make_connectivity_dataset(n_neurons, n_sims, n_steps, data_path, max_parallel, p=0.1):
    """Generates a dataset"""
    # Set data path
    data_path = (
        "spiking_network" / Path(data_path) / f"{n_neurons}_neurons_{n_sims}_datasets_{n_steps}_steps"
    )
    data_path.mkdir(parents=True, exist_ok=True)

    # Parameters for the simulation
    batch_size = min(n_sims, max_parallel)
    w0_params = GlorotParams(0, 5)
    #  w0_params = NormalParams(0, 5)

    # Generate W0s to simulate and put them in a data loader
    w0_data = W0Dataset(n_neurons, n_sims, w0_params, seed=0)
    data_loader = DataLoader(w0_data, batch_size=batch_size, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    for i, batch in enumerate(data_loader):
        batch = batch.to(device)

        connectivity_filter = ConnectivityFilter(batch.W0, batch.edge_index, batch.num_nodes, device=device)

        # Initalize model
        model = ConnectivityModel(
            connectivity_filter, seed=i, device=device
        )


        model_path = (
                Path("spiking_network/models/saved_models") / f"{w0_params.name}_{n_neurons}_neurons_{p}_probability.pt"
        )
        if model_path.exists():
            model.load(model_path)
        else:
            model.tune(p=p, epsilon=1e-6)

        # Simulate the model for n_steps
        spikes = model.simulate(n_steps)
        print(f"Average firing rate: {spikes.mean()}")

        #  Save the data and the model
        save(spikes, model, w0_data, i, data_path)
        model.save(model_path)
