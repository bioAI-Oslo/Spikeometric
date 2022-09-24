import numpy as np
import matplotlib.pyplot as plt

def load_data(filename):
    data = np.load("time_data/" + filename)
    timings = data["timings"]
    neuron_list = data["neuron_list"]
    return timings, neuron_list

def plot_timings(timings, neuron_list, filename):
    plt.figure()
    plt.plot(neuron_list, timings, label=["Mikkel", "Torch", "Numpy", "Sparse"])
    plt.xlabel("Number of neurons")
    plt.ylabel("Avg time (s)")
    plt.legend()
    plt.show()
    plt.savefig("plots/" + filename)

def compare_timings(timings_0, timings_1, neuron_list, filename):
    plt.figure()
    plt.plot(neuron_list, timings_0, label="Big X")
    plt.plot(neuron_list, timings_1, label="Rolling")
    plt.xlabel("Number of neurons")
    plt.ylabel("Avg time (s)")
    plt.legend()
    plt.show()
    plt.savefig("plots/" + filename)

t, n = load_data("comparison_sparse.npz")
# t_1, n_1 = load_data("torch_timings.npz")
plot_timings(t, n, "sims_dependence.pdf")
# compare_timings(t_0, t_1, n_0, "torch_compare.pdf")

