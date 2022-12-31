
from spiking_network.stimulation import RegularStimulation
targets = [0, 4, 9]
intervals = [5, 7, 9]
strengths = 1
temporal_scales = 2
durations = 100
stimulation = RegularStimulation(
    targets=targets,
    intervals=intervals,
    strengths=strengths,
    temporal_scales=temporal_scales,
    durations=durations,
    total_neurons=20,
)
print(stimulation.parameter_dict)