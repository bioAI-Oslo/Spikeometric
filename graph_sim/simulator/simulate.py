import torch
import numpy as np

def simulate(W, edge_index, model, n_steps, n_neurons, rng):
    x = torch.zeros((n_neurons, W.shape[-1]))
    rand_init = torch.randint(0, 2, (int(n_neurons/2), ), generator=rng)
    rand_init = torch.concat((rand_init, rand_init), dim=0)

    x[:, -1] = rand_init
    spikes = []
    for t in range(n_steps - 1):
        if x[:, -1].any():
            spikes.extend([(i, t) for i in np.where(x[:, -1])[0]])

        probabilities = model(x, edge_index, W_ij = W)

        x[:, -1] = torch.bernoulli(probabilities, generator=rng).squeeze()

        x = torch.roll(x, -1, 1)

    return np.array(spikes)

