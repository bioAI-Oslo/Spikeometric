from spiking_network.models.spiking_model import SpikingModel
from spiking_network.w0_generators.w0_dataset import W0Dataset
from spiking_network.stimulation.regular_stimulation import RegularStimulation
from spiking_network.stimulation.poisson_stimulation import PoissonStimulation
from spiking_network.stimulation.sin_stimulation import SinStimulation
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
        "spiking_network" / Path(data_path) / f"{n_neurons}_neurons_{n_sims}_datasets_{n_steps}_steps"
    )
    data_path.mkdir(parents=True, exist_ok=True)

    # Parameters for the simulation
    batch_size = min(n_sims, max_parallel)
    w0_params = GlorotParams(0, 5)
    total_neurons = batch_size*n_neurons

    # Generate W0s to simulate and put them in a data loader
    w0_data = W0Dataset(n_neurons, n_sims, w0_params, seed=0)
    data_loader = DataLoader(w0_data, batch_size=batch_size, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    for i, batch in enumerate(data_loader):
        batch = batch.to(device)

        # Initalize model
        model = SpikingModel(
            batch.W0, batch.edge_index, total_neurons, seed=i, device=device
        )

        # If we already have a tuned model for this initial distribution, 
        # number of neurons and probability of firing, we can load it. Otherwise we need to tune it.
        # Note that p is the probability that each neuron will fire per timestep.
        model_path = (
                Path("spiking_network/models/saved_models") / f"{w0_params.name}_{n_neurons}_neurons_{p}_probability.pt"
        )
        if model_path.exists():
            model.load(model_path)
        else:
            model.tune(p=p, epsilon=1e-6)

        # Generate different types of stimulation
        regular_stimulation = RegularStimulation([0, 1, 2], rates=0.1, strengths=3, temporal_scales=2, duration=n_steps, n_neurons=total_neurons, device=device)
        poisson_stimulation = PoissonStimulation([5, 3, 9], periods=5, strengths=6, temporal_scales=4, duration=100, n_neurons=total_neurons, device=device)
        sin_stimulation = SinStimulation([4, 6, 8], amplitudes=10, frequencies=0.001, duration=n_steps, n_neurons=total_neurons, device=device)
        stimulation = [regular_stimulation, poisson_stimulation, sin_stimulation]

        # Simulate the model for n_steps
        spikes = model.simulate(n_steps, stimulation)

        # Save the data and the model
        save(spikes, model, w0_data, i, data_path, stimulation=stimulation)
        model.save(model_path)

if __name__ == "__main__":
    make_dataset(10, 20, 1, 1000, 1)
