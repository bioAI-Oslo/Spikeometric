import torch
import torch.nn as nn
import math
from spikeometric.stimulus.base_stimulus import BaseStimulus
from typing import Union

class LoadedStimulus(BaseStimulus):
    r"""
    Stimulus loaded from a file. The file should be a .pt file containing a torch.Tensor of shape (n_neurons, n_steps, n_networks),
    where n_neurons is the number of neurons in the network and n_steps is the number of time steps in the stimulus.
    At each time step, the stimulus is a :obj:`torch.Tensor` of length n_neurons, indicating the stimulus to each neuron.

    Example:
        >>> from spikeometric.stimulus import LoadedStimulus
        >>> stimulus = LoadedStimulus("example_directory/stimulus_file.pt", batch_size=3)
        >>> stimulus(0).shape
        torch.Size([60])
        >>> stimulus.set_batch(3)
        >>> stimulus(0).shape
        torch.Size([20])
        >>> stimulus.reset()
        >>> model.add_stimulus(stimulus)

    Parameters
    ----------
    path : str
        Path to the file containing the stimulus.
    batch_size : int, optional
        Number of networks in each batch. If the number of networks in the stimulus is not a multiple of the batch size,
        the last batch will contain fewer networks. Default: 1.
    """
    def __init__(self, path: str, batch_size: int = 1):
        super().__init__()
        stimuli = torch.load(path)
        self.register_buffer("n_neurons", torch.tensor(stimuli.shape[0], dtype=torch.int))
        self.register_buffer("n_steps", torch.tensor(stimuli.shape[1], dtype=torch.int))

        if len(stimuli.shape) < 4:
            n_networks = stimuli.shape[2] if len(stimuli.shape) > 2 else 1
            self.register_buffer("n_networks", torch.tensor(n_networks, dtype=torch.int))
        else:
            n_networks = stimuli.shape[3]
            self.register_buffer("n_networks", torch.tensor(n_networks, dtype=torch.int))

        self.register_buffer("batch_size", torch.tensor(batch_size, dtype=torch.int))
        if self.n_networks < batch_size:
            raise ValueError("The number of networks in the stimulus is smaller than the batch size.")
        
        self.register_buffer("n_batches", torch.tensor(math.ceil(n_networks / batch_size), dtype=torch.int))

        if len(stimuli.shape) == 3:
            stimuli = torch.concat(torch.split(stimuli, 1, dim=2), dim=0).squeeze(2)
        elif len(stimuli.shape) == 4:
            stimuli = torch.concat(torch.split(stimuli, 1, dim=3), dim=0).squeeze(3)

        neurons_per_batch = [self.n_neurons*self.batch_size] * (self.n_batches - 1)
        if n_networks % batch_size != 0:
            neurons_per_batch.append(self.n_neurons*(n_networks % batch_size))
        else:
            neurons_per_batch.append(self.n_neurons*self.batch_size)
        
        self.register_buffer("neurons_per_batch", torch.tensor(neurons_per_batch, dtype=torch.int))
        self.register_buffer("stimuli", stimuli)
        self._idx = 0

    @property
    def stimulus(self):
        stimulus = torch.split(self.stimuli, self.neurons_per_batch.tolist(), dim=0)
        return stimulus[self._idx]

    def __call__(self, t: Union[float, torch.Tensor]) -> torch.Tensor:
        r"""
        If :math:`t` is a float, returns the stimulus at time :math:`t`. If :math:`t` is a :obj:`torch.Tensor`, returns the stimulus at each time step
        in :math:`t`, and the returned tensor has shape (n_neurons, t.shape[0])

        Parameters
        ----------
        t : torch.Tensor or float
            Time :math:`t` at which to compute the stimulus.

        Returns
        -------
        torch.Tensor [n_neurons, t.shape[0]] or [n_neurons]
            Stimulus at time :math:`t`.
        """
        if torch.is_tensor(t) and not t.dim() == 0:
            return self.stimulus[..., :t.shape[0], ...]
        else:
            return self.stimulus[..., t, ...] if 0 <= t < self.n_steps else torch.zeros(self.neurons_per_batch[self._idx], dtype=torch.float, device=self.stimulus.device)