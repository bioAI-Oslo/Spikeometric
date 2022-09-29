import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Make plots with seaborn
sns.set_theme()
def load_data(filename):
    data = np.load("benchmarking/benchmark_data/" + filename)
    timings = data["timings"]
    neuron_list = data["neuron_list"]
    return timings, neuron_list

def plot_timings(timings, neuron_list, filename):
    plt.figure()
    plt.plot(neuron_list, timings, label=["Old simulation", "New simulation"])
    plt.xlabel("Number of neurons")
    plt.ylabel("Avg time (s)")
    plt.legend()
    plt.savefig("plots/" + filename)
    plt.show()

t, n = load_data("comparison_network_class.npz")
plot_timings(t, n, "old_vs_new_no_equi.png")

