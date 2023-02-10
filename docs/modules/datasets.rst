Connectivity datasets
=====================

.. currentmodule:: spikeometric.datasets

The connectivity datasets are collections of graphs that represent the connectivity of a set of networks.

There are two ways to work with the connectivity datasets. The first is to use the :class:`ConnectivityDataset` class.
This class allows us to easily load weight matrices from a directory that contains a set of torch or numpy files,
and to convert them to a PyTorch Geometric dataset. This is useful if we have some predesigned connectivity matrices
that we want to use in our experiments.

The second way to work with the connectivity datasets is to use one of the :class:`ConnectivityGenerator` classes
that are provided in the package. These classes can be used to generate random connectivity matrices from a couple common
distributions, save them to disk and load them into a PyTorch Geometric dataset. It is also possible to generate them
on the fly, without saving them, by using the :func:`generate` method. This is useful if we want to generate
some example connectivity matrices to use in our experiments.

Common for both ways is that the connectivity matrices are represented as a list of PyTorch Geometric graphs. Each graph
represents the connectivity of a single network. This includes a :code:`edge_index` tensor of shape :code:`[2, num_edges]`
that contains the indices of the connected nodes, and a :code:`W0` tensor of shape :code:`[num_edges,]` that contains
the weights of the connections.

One of the main benefits of the list of :class:`torch_geometric.data.Data` format is that it is easy to convert them
to a :class:`torch_geometric.data.Batch` object by using the data loader :class:`torch_geometric.loader.DataLoader`.
This is useful since it is often much faster in total to simulate a batch of networks than to simulate them one by one.

Dataset classes
---------------
.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: autosummary/class.rst

   ConnectivityDataset

Dataset generators
------------------
.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: autosummary/class.rst

   NormalGenerator
   UniformGenerator
   MexicanHatGenerator

.. automodule:: spikeometric.datasets
   :members:
   :undoc-members: