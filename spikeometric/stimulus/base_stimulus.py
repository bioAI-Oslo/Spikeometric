import torch
import torch.nn as nn
import math

class BaseStimulus(nn.Module):
    r"""
    Base class for stimuli. This class implements the logic for switching between batches of stimuli for simulations with batched networks.
    """
    @property
    def stimulus_masks(self):
         r"""
         Returns the batched stimulus masks.
         """
         return torch.split(self.conc_stimulus_masks, self.split_points.tolist(), dim=0)

    def batch_stimulus_masks(self, stimulus_masks: list, batch_size: int) -> list:
        r"""
        Batches the stimulus masks into batches of size :obj:`batch_size`, concatenates them, and returns the concatenated stimulus masks and the split points.
        """
        if batch_size > len(stimulus_masks):
            raise ValueError("Batch size must be smaller or equal to the number of networks.")
        n_neurons = torch.tensor([sm.shape[0] for sm in stimulus_masks])
        split_points = [sum(n) for n in torch.split(n_neurons, batch_size)]
        concatenated_stimulus_masks = torch.cat(stimulus_masks, dim=0)
        return concatenated_stimulus_masks, split_points
    
    @property
    def current_batch(self):
        r"""
        Returns the current batch of stimuli.
        """
        return int(self._idx)
    
    def reset(self):
        r"""
        Resets the stimulus to the first batch.
        """
        self._idx = 0

    def next_batch(self):
        r"""
        Switches to the next batch of stimuli.
        """
        if self.n_batches == 1:
            raise ValueError("There is only one batch.")
        self._idx = (self._idx + 1) % self.n_batches

    def set_batch(self, idx: int):
        r"""
        Switches to the batch of stimuli with the given index.

        Parameters
        ----------
        idx : int
            Index of the batch to switch to.
        """
        if idx < 0 or idx >= self.n_batches:
            raise ValueError("Index out of bounds.")
        self._idx = idx