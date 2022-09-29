from make_dataset import make_dataset
import argparse

# Next steps
# 2. Tuning with torch
# 4. Find better way of aggregating in numpy
# 4. Compare speed gains from parallelization
# 5. Compare speed gains from sparsification
# 7. Get torch_geometric on the cluster
# 8. Make network unaware of simulator

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--n_neurons", type=int, default=20, help="Number of neurons")
    parser.add_argument("-s", "--n_steps", type=int, default=10000, help="Number of steps in simulation")
    parser.add_argument("-p", "--n_paralell_networks", type=int, default=100, help="Number of networks to run in parallel")
    parser.add_argument("-t", "--n_total_networks", type=int, default=100, help="Number of total networks")
    args = parser.parse_args()

    make_dataset(args.n_neurons, args.n_steps, args.n_paralell_networks, args.n_total_networks)

if __name__ == "__main__":
    main()
