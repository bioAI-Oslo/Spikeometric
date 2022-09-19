from w0.normal import NormalWeights
from model.graph_glm import GraphGLM
from simulator.simulate import simulate
from pathlib import Path
from tqdm import tqdm
from numpy.random import default_rng
import numpy as np
import torch

# Priorities:
# 1. Flexible W, in terms of structure, distribution and time dependence
# 2. Flexible model, differnt link functions
# 3. Speed. Parallelize, use GPU, use sparse matrices

def main():
    mu = 0
    sigma = 5
    n_neurons = 1000
    n_steps = 1000
    n_sims = 1

    NW = NormalWeights(n_neurons, mu, sigma)
    model = GraphGLM()
    data_path = "data" / Path(f"jakob_{n_neurons}_neurons_{n_steps}_steps")

    data_path.mkdir(parents=True, exist_ok=True)

    print(
        f"Creating dataset with {n_neurons} neurons, {n_sims} sims, {n_steps} steps"
    )

    for seed in tqdm(range(n_sims), desc="Simulating", leave=False, colour="#435518"):
        rng = torch.Generator().manual_seed(seed)
        W0 = NW.build_W0(rng)
        W, edge_index = NW.build_W(W0)
        
        result = simulate(
            W=W,
            edge_index=edge_index,
            model=model,
            n_steps=n_steps,
            n_neurons=n_neurons,
            rng=rng,
        )

        fname = f"{seed}.npz"

        np.savez(
            data_path / fname,
            X_sparse=result,
            W0=W0
        )

if __name__ == "__main__":
    main()
