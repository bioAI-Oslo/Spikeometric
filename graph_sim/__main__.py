from connectivity import ConnectivityFilterGenerator, NormalParams, FilterParams
from simulator import TorchSimulator, NumpySimulator, SparseSimulator
from pathlib import Path
from tqdm import tqdm
from numpy.random import default_rng
import numpy as np
import torch

# Next steps
# 1. Deal with W and edge_index in numpy
# 2. Tuning with torch
# 3. Sparsify in numpy
# 4. Find better way of aggregating in numpy
# 4. Compare speed gains from parallelization
# 5. Compare speed gains from sparsification
# 6. Incorporate connectivity into the Simulator class?
# 7. Get torch_geometric on the cluster


def main():
    # Simulation parameters
    n_steps = 1000
    n_batches = 1
    p_sims = 1
    n_neurons = 1000
    threshold = -5

    # Connectivity parameters
    mu = 0
    sigma = 5
    ref_scale=10
    abs_ref_scale=3
    spike_scale=5
    abs_ref_strength=-100
    rel_ref_strength=-30
    decay_offdiag=0.2
    decay_diag=0.5

    normal_params = NormalParams(mu, sigma)
    filter_params = FilterParams(
            ref_scale=ref_scale,
            abs_ref_scale=abs_ref_scale,
            spike_scale=spike_scale,
            abs_ref_strength=abs_ref_strength,
            rel_ref_strength=rel_ref_strength,
            decay_offdiag=decay_offdiag,
            decay_diag=decay_diag
        )

    cf_generator = ConnectivityFilterGenerator(n_neurons, normal_params, filter_params)

    # Simulation
    simulator = TorchSimulator(n_steps, p_sims, n_neurons, threshold)

    # Path to save results
    data_path = "data" / Path(f"jakob_{n_neurons}_neurons_{n_steps}_steps")
    data_path.mkdir(parents=True, exist_ok=True)

    print(
        f"Creating dataset with {n_neurons} neurons, {n_batches*p_sims} sims, {n_steps} steps"
    )

    # Create dataset running n_batches of p_sims each
    for batch in tqdm(range(n_batches), desc="Simulation", leave=False, colour="#435518"):
        rng = torch.Generator().manual_seed(batch)
        W, edge_index = cf_generator.new_filter(p_sims, rng)

        result = simulator.run(
            W=W,
            edge_index=edge_index,
            seed=batch,
        )

        # Save results
        n_edges = edge_index.shape[1] // p_sims
        for i, spikes in enumerate(result):
            W_i = W[i*n_edges:(i+1)*n_edges, :]
            edge_index_i = edge_index[:, i*n_edges:(i+1)*n_edges] - i*n_neurons
            np.savez(data_path / f"batch_{batch}_sim_{i}.npz", X_sparse = spikes, W=W_i, edge_index=edge_index_i)


if __name__ == "__main__":
    main()
