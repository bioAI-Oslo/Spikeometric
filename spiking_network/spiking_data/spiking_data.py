from torch_geometric.data import HeteroData
import torch

class SpikingData(HeteroData):
    def __init__(self, connectivity_filter, stimulation=[], **kwargs):
        super(SpikingData, self).__init__(**kwargs)
        self["neuron"].x = torch.zeros(connectivity_filter.n_neurons, connectivity_filter.time_scale, dtype=torch.bool)
        self["neuron", "drives", "neuron"].edge_attr = connectivity_filter.W
        self["neuron", "drives", "neuron"].edge_index = connectivity_filter.edge_index

        if isinstance(stimulation, list):
            for i, stim in enumerate(stimulation):
                self[f"stimulus_{i}", "stimulates", "neuron"].edge_attr = stim.stimulation_strength
                stim.edge_index[0] += connectivity_filter.n_neurons + i
                self[f"stimulus_{i}", "stimulates", "neuron"].edge_index = stim.edge_index
                self[f"stimulus_{i}"].x = stim.stimulation_times
        else:
            self["stimulus", "stimulates", "neuron"].edge_attr = stimulation.stimulation_strength
            stimulation.edge_index[0] += connectivity_filter.n_neurons
            self["stimulus", "stimulates", "neuron"].edge_index = stimulation.edge_index
            x = torch.zeros(connectivity_filter.n_neurons + 1, len(stimulation.stimulation_times), dtype=torch.bool)
            x[-1] = stimulation.stimulation_times
            self["stimulus"].x = x

