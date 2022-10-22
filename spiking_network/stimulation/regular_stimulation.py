import torch
from torch_scatter import scatter_add

class RegularStimulation(torch.nn.Module):
    def __init__(self, targets, rate, strength):
        self.targets = torch.tensor(targets)
        self.rate = rate
        self.stimulation_strength = strength * torch.ones(len(targets))

    def __call__(self, t, out=None):
        stimulate = t % (1 / self.rate) == 0
        stim = self.stimulation_strength * stimulate
        if out is None:
            return scatter_add(stim, self.targets, dim=0)
        else:
            scatter_add(stim, self.targets, dim=0, out=out.squeeze())
            return out

    def to(self, device):
        self.targets = self.targets.to(device)
        self.stimulation_strength = self.stimulation_strength.to(device)
        return self
        
