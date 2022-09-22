from torch_geometric.nn import MessagePassing
import torch
import torch.sparse
import torch.nn.functional as F

class GraphGLM(MessagePassing):
    def __init__(self, aggr='add'):
        super(GraphGLM, self).__init__(aggr=aggr)

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j, edge_attr):
        return (x_j * edge_attr).sum(dim=1).unsqueeze(1)

    def update(self, aggr_out):
        return torch.sigmoid(aggr_out - 5)
