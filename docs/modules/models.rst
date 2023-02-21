Models
=======

.. currentmodule:: spikeometric.models

The GLM and LNP models are implemented as classes that inherit from the :class:`MessagePassing` class from the `PyTorch Geometric <https://pytorch-geometric.readthedocs.io/en/latest/>`_ library.
We represent the network as a graph, where each node corresponds to a neuron and each edge corresponds to a synapse.
The nodes are connected according to a weight matrix :math:`\mathbf{W_0}`, where the element :math:`(W_0)_{i,j}` is the weight of the synapse between neuron :math:`i` and neuron :math:`j`.
The state of the network at time step :math:`t` is represented by a vector :math:`\mathbf{X}_t` where each element corresponds to the spike count of a neuron at time step :math:`t`.

While there are many different types of GLM and LNP models, they all share a common structure. 
For each neuron :math:`i`, the state at time step :math:`t+1` is computed in three steps:

#. The input stage
    The first step is to compute the synaptic input :math:`g_i(t+1)` to neuron :math:`i` at time step :math:`t+1`.
    Input can come from several sources.
    
    * If a neuron :math:`j` that is connected to :math:`i` has recently fired, :math:`i` will receive some input as a function of the synaptic weight :math:`W_{ji}` between the two neurons and how long it has been since :math:`j` fired.

    * If :math:`i` itself has recently fired, it will be in a refractory period and will receive some self-inhibiting input.

    * There might also be some background input that is independent of the network state.

    * Finally, :math:`i` may receive an external input, for example from an optogenetic stimulus.

#. The non-linear stage
    The second step is to compute the response of the neuron to the synaptic input by 
    applying a non-linear function to it.
    In the GLM, the non-linearity is the inverse link function, for example the inverse logit function (sigmoid) for the 
    Bernoulli model. In the LNP, the non-linearity might for example be the rectified linear unit (ReLU) function. 
    The output of this step is often interpreted as the expected spike count of the neuron.

#. The spike emission stage
    The final step is to draw a spike count from a probability distribution that depends on the output of the non-linear stage.
    The probability distribution is often a Poisson distribution, but it can also be a Bernoulli distribution for the GLM.

These three steps form the core of the GLM and LNP models, and each class must implement them in the
:func:`input`, :func:`non_linearity` and :func:`emit_spikes` functions, respectively.

The :class:`BaseModel` class implements the core functionality of the GLM and LNP models, and the :class:`SAModel` class implements the core functionality of the activation-based models.

Base models
------------
.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: autosummary/class.rst

   BaseModel
   SAModel

Spike-based models
-------------------------
.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: autosummary/class.rst

   RectifiedLNP
   PoissonGLM
   BernoulliGLM

Activation-based models
-------------------------------
.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: autosummary/class.rst

   RectifiedSAM
   ThresholdSAM