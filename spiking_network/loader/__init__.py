
from torch_geometric.loader import DataLoader
import torch

class ConnectivityLoader(DataLoader):
    def __init__(self, data, stimulation_targets, batch_size=1, shuffle=False):
        super().__init__(data, batch_size, shuffle)
        if all([isinstance(stimulation_target, int) for stimulation_target in stimulation_targets]):
            stimulation_targets = [torch.tensor(stimulation_targets)]
        if isinstance(stimulation_targets, torch.Tensor) and stimulation_targets.dim() == 1:
            stimulation_targets = [stimulation_targets]

        if not len(stimulation_targets) == len(data):
            raise ValueError(f"Must have stimulation targets for each graph in the dataset ({len(data)})")

        max_neurons = max([data[i].num_nodes for i in range(len(data))])
        max_stimulation_targets = max([stimulation.max() for stimulation in stimulation_targets]) 
        if max_stimulation_targets >= max_neurons:
            raise ValueError(f"Stimulation indices must be smaller than the number of neurons in the network ({max_neurons})")
     
        self.n_neurons_list = [data[i].num_nodes for i in range(len(data))]
        self.stimulation_targets = stimulation_targets

    def __iter__(self):
        for i, batch in enumerate(super().__iter__()):
            batch = self._add_stimulation_targets(batch, self.stimulation_targets, i)
            yield batch

    def _add_stimulation_targets(self, batch, stimulation_targets, batch_idx):
        """Adds stimulation targets to the data"""
        batch_stimulation_targets = stimulation_targets[self.batch_size * batch_idx : self.batch_size * (batch_idx + 1)]
        batch_adjusted_stimulation_target = torch.cat([batch_stimulation_targets[i] + i*self.n_neurons_list[i] for i in range(batch.num_graphs)], dim=-1)
        batch.stimulation_targets = batch_adjusted_stimulation_target
        return batch
