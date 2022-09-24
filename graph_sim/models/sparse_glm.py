from torch_geometric.nn import MessagePassing
import torch
import torch.sparse
import torch.nn.functional as F

class SparseGraphGLM(MessagePassing):
    def __init__(self, threshold, aggr='add'):
        super(SparseGraphGLM, self).__init__(aggr=aggr)
        self.threshold = threshold

    def forward(self, x):
        return self.propagate(self.edge_index, x=x)

    def message(self, x_j):
        return torch.sparse.sum(self.W * x_j, dim=1).to_dense().unsqueeze(1)

    def update(self, aggr_out):
        return torch.sigmoid(aggr_out - self.threshold)

    def set_weights(self, W, edge_index):
        self.W = W.to_sparse()
        self.edge_index = edge_index
