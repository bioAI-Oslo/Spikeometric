from network import SpikingNetwork, FilterParams
from tqdm import tqdm

def main():
    filter_params = FilterParams(n_neurons=20*50, n_clusters=50, n_hubneurons=0)

    for i in tqdm(range(10)):
        network = SpikingNetwork(filter_params, seed=i)
        network.simulate(n_steps=100000, save_spikes=True, data_path="data/example_data/")

if __name__ == '__main__':
    main()
