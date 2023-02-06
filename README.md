# snn-glm-simulator
Spiking Neural Network Simulator based on Generalized Linear Models

# NeuraSIMPL - Neural Simulation, Inference and Modelling Package using LNP-models
This package provides a simple and scaleable way to simulate networks of neurons using either
Linear-Nonlinear-Poisson models (LNP) or its cousin the Generalized Linear Model (GLM). The framework
is built on top of torch.Module and lets you tune parameters in your model to match a certain firing rate
or spike-train pattern, provided the model is differentiable. One key application
is the problem of infering connectivity from spike data, where
LNP models are commonly used as generative models, and GLMs both as both generative and inference models.
The NeuraSIMPL framework provides a unified framework for generative and inference models in this setting.

## How are the models so scalable?
- The TuneLNP package uses a torch backend, which with its underlying C implementation is very fast. Torch
also makes running your models on GPU very easy, allowing us to scale the networks. These advantages
are taken a step further by adopting torch_geometric's framework for Graph Neural Networks (GNNs). This framework 
ignores non-existent edges in our connectivity and batches our simulations as subgraphs of a larger graph,
speeding up computations even further.

## What has been done to make the models simple to use?
- The models are run exactly the same way as you normally run torch models. This has become a familiar workflow for many,
and reduces the friction in picking up this new tool.

# Results
The maximal speed and size will depend on your hardware, but for reference the BernoulliGLM running on an NVIDIA A1000 with 80GB RAM can reach up to
23 000 neurons before running into memory issues.

# 