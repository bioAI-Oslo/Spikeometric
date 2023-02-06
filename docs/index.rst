.. snn-glm-simulator documentation master file, created by
   sphinx-quickstart on Sun Jan 29 17:27:32 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to snn-glm-simulator's documentation!
=============================================

The snn-glm-simulator package is a framework for simulating spiking neural networks (SNNs)
using generalized linear models (GLMs) and Linear-Nonlinear-Poisson models (LNPs) in Python. 
It is built on top of the `PyTorch Geometric <https://pytorch-geometric.readthedocs.io/en/latest/>`_ package and
makes use of their powerful graph neural network (GNN) modules and efficient graph
representation.
It is designed to be fast, flexible and easy to use, and is intended for research purposes.

.. toctree::
   :maxdepth: 1
   :caption: Installation

   install/installation

.. toctree::
   :maxdepth: 1
   :caption: Introduction

   introduction/introduction

.. toctree::
   :maxdepth: 1
   :caption: Tutorials
   
   tutorials/tune_model
   tutorials/advanced_example
   tutorials/create_model
   tutorials/create_dataset
   tutorials/create_stimulus

.. toctree::
   :maxdepth: 1
   :caption: API Reference

   modules/datasets
   modules/models
   modules/stimulation