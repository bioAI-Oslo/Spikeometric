Stimulus
====================

.. currentmodule:: spikeometric.stimulus

The stimlation classes are used to define the external input to the network. They inherit from the :class:`Module` class 
from torch, which allows us to tune the parameters of the stimulus with the usual torch methods. A couple of stimulus models are provided in the package, but it is easy to implement
new ones by taking inspiration from the existing ones and extending the :class:`Module` class from torch.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: autosummary/class.rst

   PoissonStimulus
   RegularStimulus
   SinStimulus