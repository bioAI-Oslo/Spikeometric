from network import SpikingNetwork, FilterParams, NormalParams
from tqdm import tqdm

def main():
    filter_params = FilterParams(n_neurons=500, n_clusters=1, n_hubneurons=0)

    for i in tqdm(range(5)):
        network = SpikingNetwork(filter_params, seed=i)
        network.simulate(n_steps=1000, save_spikes=True, data_path="data/example_data/")

if __name__ == '__main__':
    main()
