---
title: 'Spikeometric: Spiking Neural Networks using GLMs and LNPs in Pytorch Geometric'
tags:
  - Python
  - computational neuroscience
  - machine learning
  - spiking neural networks
  - generalized linear models
  - linear non-linear poisson models
authors:
  - name: Jakob L. Sønstebø
    affiliation: 1
  - name: Mikkel Elle Leppereød
    orcid: 0000-0002-4262-5549
    affiliation: "1, 2"
affiliations:
 - name: Department of Numerical Analysis and Scientific Computing, Simula Research Laboratory, Oslo, Norway
   index: 1
 - name: Institute of Basic Medical Sciences, University of Oslo, Oslo, Norway
   index: 2
date: 21 February 2023
bibliography: paper.bib
---

# Statement of need
In order to understand the dynamics of the brain, computational neuroscientists often study smaller scale networks using simple stochastic models such as the Linear-Nonlinear-Poisson (LNP) model and the Generalized Linear Model (GLM) [@gerstner_kistler_naud_paninski_2014; @10.3389/fnsys.2016.00109].

These come in many flavors, but all model the spike response of a network through the same sequence of stages. At each time step, a neuron receives input from its environment, converts it to a firing rate by a nonlinearity, and this firing rate parametrizes a probability distribution from which spikes are drawn. This is in contrast to threshold-based models, in which spikes are emitted whenever a variable representing the membrane potential exceeds a certain value.

While they don’t explicitly model the subcellular mechanisms of each neuron, and so lack some biological plausibility on the neuron level, stochastic models can give key insights into the behavior of a network on a systems level. They have been used to study how multi-neuron systems process incoming stimuli [@Pillow2008], and have even been used to study causality in the brain [@Lepperod463758]. As a bonus, we get models that have been shown to enjoy nice convexity properties and that can be fitted to observed spike data.

Another application of LNP/GLM models is as generative models for the task of training a machine learning model to infer connectivity from spike data [@Das2020]. As spike data is very sparse, this can be highly data intensive so having a fast and convenient data-generation tool is important.

# Summary
`Spikeometric` is a framework for simulating networks of neurons using LNP and GLM models.

It is designed to be a convenient and scalable modeling tool for neuroscientists, whether they want to run big simulations, many simulations or simply implement a quick idea. With its simple interface it can also act as an educational tool for students learning about neural encoding models for the first time.

The `torch` backend makes simulating large networks on a GPU easy, with the extra benefit of having a familiar use pattern, reducing the friction of picking up a new tool. The package relies heavily on `PyTorch Geometric` [@Fey/Lenssen/2019], with the networks being represented as `torch_geometric` `Data` objects and the models extending the `MessagePassing` base class. The `PyTorch Geometric` framework is a popular deep learning framework originally designed for Graph Neural Networks (GNNs), a class of neural networks for learning graph-related data [@DBLP:journals/corr/abs-2104-13478]. It is the perfect setting for simulating neural networks with tunable parameters, allowing us to us to formulate the model’s equations naturally in terms of vertices and edges, and the giving us access to easy automatic tuning of parameters to match a certain firing rate, provided that the nonlinearity in the model is differentiable.

In addition to the models, the package includes dataset classes that can generate random connectivity matrices from a distribution or load pre-constructed connectivity matrices into `torch_geometric`’s `Data` objects to be passed straight to the model. These objects hold a sparse representation of our connectivity matrices and can be batched together to form isolated subgraphs of a big graph, letting us simulate many networks simultaneously.

Finally, in order to facilitate the common use pattern of adding an external stimulus to the simulation and recording the resulting activity, we have also included various stimulation classes that can be easily added to the model, and even tuned to provoke a certain response.