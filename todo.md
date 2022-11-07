# Testing

-   We need to make tests for as many of the different methods as possible. This should include simulation of a small
    system where we can easily calculate the expected behaviour.

# Stimulation

-   We should make the stimulation parameters learnable. To do this we extend torch's module class.
-   Is the current way of using stimulation intuitive?
-   Make a base class for stimulation

# Visualization

-   These are relevant things to plot
    -   Distribution of firings across the different neurons
    -   Number of firings over time
    -   Time dependence of weights
    -   A way of visualizing W0, as graph or heatmap
-   Maybe these could be imported from spiking_network.utils? We could but the save functions in there as well.

# Other

-   Perhaps simulate and tune could be functions instead of methods in model? We'd then just pass the model as one of the parameters.
    This would make it more natural when we pass a learnable stimulation as well.
