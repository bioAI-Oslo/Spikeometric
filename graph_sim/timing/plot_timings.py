import numpy as np
import matplotlib.pyplot as plt

def load_data(filename):
    data = np.load(filename)
    timings = data["timings"]
    neuron_list = data["neuron_list"]
    return timings, neuron_list

def plot_timings(timings, neuron_list, filename):
    plt.figure()
    plt.plot(neuron_list, timings, label=["Mikkel", "Torch", "Numpy"])
    plt.xlabel("Number of neurons")
    plt.ylabel("Avg time (s)")
    plt.legend()
    plt.show()
    # plt.savefig(filename)

t, n = load_data("n_dependence.npz")
plot_timings(t, n, "n_dependence.png")

