from torch_geometric.nn import MessagePassing
import torch
import torch.nn.functional as F

class GraphGLM(MessagePassing):
    def __init__(self, aggr='add'):
        super(GraphGLM, self).__init__(aggr=aggr)

    def forward(self, x, edge_index, W_ij):
        return self.propagate(edge_index, x=x, W_ij=W_ij)

    def message(self, x_j, W_ij):
        return (W_ij * x_j).sum(dim=1).unsqueeze(1)

    def update(self, aggr_out):
        return torch.sigmoid(aggr_out - 5)
