import numpy as np
from scipy.special import expit
from scipy.sparse import csr_array

class NumpyGraphGLM:
    def __init__(self, threshold):
        self.threshold = threshold

    def forward(self, x):
        x_j = x[self.edge_index[0]]
        messages = self.message(x_j)
        aggr_out = self.aggregate(messages)
        return self.update(aggr_out)

    def message(self, x_j):
        return (x_j * self.W).sum(axis=1)
        # return np.einsum("ij,ij->i", x_j, self.W) 

    def update(self, aggr_out):
        return expit(aggr_out - self.threshold)

    def aggregate(self, x):
        a = np.zeros(self.n_neurons)
        np.add.at(a, self.edge_index[1], x)
        return a

    def set_weights(self, W, edge_index):
        self.W = W
        self.edge_index = edge_index
        self.n_neurons = np.max(edge_index) + 1
