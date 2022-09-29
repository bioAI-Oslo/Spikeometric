import torch
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
from networks import NetworkBuilder, FilterParams

filter_params = FilterParams(n_neurons=10)
netbuilder = NetworkBuilder(filter_params)

# Create a random graph
n_clusters = 10
network = netbuilder.build_clusters(n_clusters=n_clusters, hub_nodes=0, seed=4, backend="torch")
data = Data(num_nodes=network.n_neurons*n_clusters, edge_index=network.edge_index)
G = to_networkx(data, remove_self_loops=True)

pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=False, node_size=10, arrowsize=4)
plt.show()



# n_nodes = int(w0.shape[0] / 2)

# #  s1 = w0.shape[1] / 2
# #  w0 = w0[:, : int(w0.shape[1] / 2)]
# w0_graph = w0.copy()
# # w0_graph[:, :] = 0

# G = nx.from_numpy_matrix(w0_graph, create_using=nx.DiGraph)

# print(len(G.edges))

# vertex_color = [
    # hsv_to_rgb(floor(idx / n_nodes) / 2, 1, 1) for idx in range(n_nodes * 2)
# ]

# pos = nx.circular_layout(G)

# plt.title(f"{len(G.edges)} edges")

# raw_weights = nx.get_edge_attributes(G, "weight")
# min_weight = min(raw_weights.values())
# max_weight = max(raw_weights.values())
# weights = {(u, v): f'{G[u][v]["weight"]:.2f}' for u, v in G.edges}
# # edge_colors = [
    # # hsv_to_rgb((float(weight) - max_weight) / (min_weight / max_weight) / 2, 1, 1)
    # # for weight in raw_weights.values()
# # ]

# edge_colors = [
    # float(weight) for weight in raw_weights.values()
# ]

# nx.draw(G, pos, node_color=vertex_color, edge_color = edge_colors, edge_cmap=plt.cm.seismic, with_labels=True)
