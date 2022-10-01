from network import SpikingNetwork, FilterParams, RecursiveNetwork
import torch
from tqdm import tqdm

def main():
    filter_params = FilterParams()
    cluster_size = 80
    n_clusters = 8
    for i in tqdm(range(5)):
        rng = torch.Generator()
        network = RecursiveNetwork.build_recursively(n_clusters, cluster_size, filter_params, rng)
        network.simulate(n_steps=1000, data_path="data/example_data/")

if __name__ == '__main__':
    main()
