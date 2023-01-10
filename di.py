from spiking_network.datasets import ConnectivityDataset, NormalConnectivityDataset, GlorotParams
import torch

from spiking_network.utils import tune
from spiking_network.models import GLMModel
from spiking_network.stimulation import RegularStimulation
from spiking_network.utils import simulate, calculate_firing_rate
from benchmarking.timing import time_model

dataset = NormalConnectivityDataset.generate_examples(20, 1, GlorotParams(0, 5), seed=14071789)
example_data = dataset[0]
glm_model = GLMModel(seed=14071789)

t = time_model(glm_model, example_data, 100, 10)
print(t)

