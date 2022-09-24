from torch_geometric.nn import MessagePassing
import torch
import torch.sparse
import torch.nn.functional as F

class TorchGraphGLM(MessagePassing):
    def __init__(self, threshold, aggr='add'):
        super(TorchGraphGLM, self).__init__(aggr=aggr)
        self.threshold = threshold

    def forward(self, x):
        return self.propagate(self.edge_index, x=x)

    def message(self, x_j):
        return torch.sum(x_j * self.W, dim=1).unsqueeze(1)

    def update(self, aggr_out):
        return torch.sigmoid(aggr_out - self.threshold)

    def set_weights(self, W, edge_index):
        self.W = W
        self.edge_index = edge_index
