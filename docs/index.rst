.. spikeometric documentation master file, created by
   sphinx-quickstart on Sun Jan 29 17:27:32 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Spikeometric
============

The `spikeometric` package is a framework for simulating spiking neural networks (SNNs)
using generalized linear models (GLMs) and Linear-Nonlinear-Poisson models (LNPs) in Python. 
It is built on top of the `PyTorch Geometric <https://pytorch-geometric.readthedocs.io/en/latest/>`_ package and
makes use of their powerful graph neural network (GNN) modules and efficient graph
representation.
It is designed to be fast, flexible and easy to use, and is intended for research purposes.

Installation
------------

To use this package you will need to first install the following packages:

    * `PyTorch <https://pytorch.org/get-started/locally/>`_
    * `PyTorch Geometric <https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html>`_

Make sure you get the cuda versions if you are planning to use a GPU.

Then you can install `spikeometric` using pip:

.. code-block:: bash

   pip install spikeometric


.. toctree::
   :maxdepth: 1
   :caption: Introduction

   introduction/introduction

.. toctree::
   :maxdepth: 1
   :caption: Tutorials
   
   tutorials/tune_model
   tutorials/implement_model
   tutorials/stimuli

.. toctree::
   :maxdepth: 1
   :caption: API Reference

   modules/datasets
   modules/models
   modules/stimulus