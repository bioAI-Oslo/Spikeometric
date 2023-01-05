from spiking_network.datasets import ConnectivityDataset, W0Dataset, GlorotParams

w0data = W0Dataset(20, 10000, GlorotParams(0, 5), seed=0, root="tests/test_data/big_dataset")
from IPython import embed; embed()