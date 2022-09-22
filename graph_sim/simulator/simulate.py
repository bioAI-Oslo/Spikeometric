import torch
import numpy as np
from tqdm import tqdm

def simulate(W, edge_index, model, n_steps, n_neurons, rng):
    x = torch.zeros((n_neurons, 10))
    rand_init = torch.randint(0, 2, (n_neurons, ), generator=rng)
    x[:, -1] = rand_init

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device}")
    model = model.to(device)
    W = W.to(device)
    x = x.to(device)
    edge_index = edge_index.to(device)

    equi_steps = 10
    for e_t in tqdm(range(equi_steps), desc="Equilibrating...", leave=False, colour="#935518"):
        probabilities = model(x, edge_index, edge_attr=W)
        x[:, -1] = torch.bernoulli(probabilities, generator=rng).squeeze()
        x = torch.roll(x, -1, 1)

    spikes = []
    for t in tqdm(range(n_steps), desc=f"Simulating...", leave=False, colour="#435518"):
        if x[:, -1].any():
            spikes.extend([(i, t) for i in torch.where(x[:, -1])[0]])

        probabilities = model(x, edge_index, edge_attr = W)

        x[:, -1] = torch.bernoulli(probabilities, generator=rng).squeeze()
        x = torch.roll(x, -1, 1)

    return np.array(spikes)

