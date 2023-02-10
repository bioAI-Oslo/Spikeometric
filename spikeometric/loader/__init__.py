
from torch_geometric.loader import DataLoader
import torch

class ConnectivityLoader(DataLoader):
    def __init__(self, data, stimulus_targets, batch_size=1, shuffle=False):
        super().__init__(data, batch_size, shuffle)
        if all([isinstance(stimulus_target, int) for stimulus_target in stimulus_targets]):
            stimulus_targets = [torch.tensor(stimulus_targets)]
        if isinstance(stimulus_targets, torch.Tensor) and stimulus_targets.dim() == 1:
            stimulus_targets = [stimulus_targets]

        if not len(stimulus_targets) == len(data):
            raise ValueError(f"Must have stimulus targets for each graph in the dataset ({len(data)})")

        max_neurons = max([data[i].num_nodes for i in range(len(data))])
        max_stimulus_targets = max([stimulus.max() for stimulus in stimulus_targets]) 
        if max_stimulus_targets >= max_neurons:
            raise ValueError(f"Stimulus indices must be smaller than the number of neurons in the network ({max_neurons})")
     
        self.n_neurons_list = [data[i].num_nodes for i in range(len(data))]
        self.stimulus_targets = stimulus_targets

    def __iter__(self):
        for i, batch in enumerate(super().__iter__()):
            batch = self._add_stimulus_targets(batch, self.stimulus_targets, i)
            yield batch

    def _add_stimulus_targets(self, batch, stimulus_targets, batch_idx):
        """Adds stimulus targets to the data"""
        batch_stimulus_targets = stimulus_targets[self.batch_size * batch_idx : self.batch_size * (batch_idx + 1)]
        batch_adjusted_stimulus_target = torch.cat([batch_stimulus_targets[i] + i*self.n_neurons_list[i] for i in range(batch.num_graphs)], dim=-1)
        batch.stimulus_targets = batch_adjusted_stimulus_target
        return batch
