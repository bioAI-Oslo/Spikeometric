# Spikeometric - GLM-based Spiking Neural Networks with PyTorch Geometric

This package provides a simple and scaleable way to simulate networks of neurons using either
Linear-Nonlinear-Poisson models (LNP) or its cousin the Generalized Linear Model (GLM).

The framework is built on top of torch modules and let's you tune parameters in your model to match a certain firing rate, provided the model is differentiable. 

One key application is the problem of infering connectivity from spike data, where these models are often used both as generative and inference models.

# Install
Before installing `spikeometric` you will need to download versions of PyTorch and PyTorch Geometric that work with your hardware. When you have done that (for example in a conda environment), you are ready to download spikeometric with:

    pip install spikeometric

# Documentation

For more information about the package and a full API reference check out our [documentation](https://spikeometric.readthedocs.io/en/latest/).
