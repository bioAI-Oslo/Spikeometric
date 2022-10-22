from spiking_network.models.spiking_model import SpikingModel
from spiking_network.w0_generators.w0_dataset import W0Dataset
from spiking_network.stimulation.regular_stimulation import RegularStimulation
from spiking_network.w0_generators.w0_generator import GlorotParams, NormalParams
from spiking_network.data_generators.save_functions import save
from pathlib import Path
from tqdm import tqdm
import torch
from torch_geometric.loader import DataLoader


def make_dataset(n_neurons, n_sims, n_steps, data_path, max_parallel, p=0.1):
    """Generates a dataset"""
    # Set data path
    data_path = (
        Path(data_path) / f"{n_neurons}_neurons_{n_sims}_datasets_{n_steps}_steps"
    )
    model_path = (
            Path("spiking_network/models/saved_models") / f"{n_neurons}_{p}_probability.pt"
    )
    data_path.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    w0_data = W0Dataset(n_neurons, n_sims, NormalParams(0, 5), seed=0)
    batch_size = min(n_sims, max_parallel)
    data_loader = DataLoader(w0_data, batch_size=batch_size, shuffle=False)

    for i, batch in enumerate(data_loader):
        batch = batch.to(device)

        regular_stimulation = RegularStimulation([0, 1, 2], rate=0.1, strength=5).to(device)

        model = SpikingModel(
            batch.W0, batch.edge_index, batch_size*n_neurons, seed=i, stimulation = [regular_stimulation], device=device
        )

        if model_path.exists():
            model.load(model_path)
        else:
            model.tune(p=p, epsilon=1e-6)

        spikes = model.simulate(n_steps)

        save(spikes, model, w0_data, i, data_path)
        model.save(model_path)

if __name__ == "__main__":
    make_dataset(10, 20, 1, 1000, 1)
