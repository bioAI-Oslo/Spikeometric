from spiking_network.data_generators.make_dataset import make_dataset
from spiking_network.data_generators.make_herman_dataset import make_herman_dataset
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
    parser.add_argument("-n", "--n_clusters", type=int, default=50, help="Number of clusters")
    parser.add_argument("-s", "--cluster_size", type=int, default=20, help="Size of each cluster")
    parser.add_argument("-t", "--n_steps", type=int, default=1000, help="Number of steps in simulation")
    parser.add_argument("-d", "--n_datasets", type=int, default=1, help="Number of datasets to generate")
    parser.add_argument("--herman", help="Run hermans simulation", action="store_true")
    parser.add_argument("-p", "--parallel", help="Run in parallel", action="store_true")
    parser.add_argument("--data_path", type=str, default="spiking_network/data", help="The path where the data should be saved")
    args = parser.parse_args()

    print("Generating datasets...")
    print(f"n_clusters: {args.n_clusters}")
    print(f"cluster_size: {args.cluster_size}")
    print(f"n_steps: {args.n_steps}")
    print(f"n_datasets: {args.n_datasets}")
    print(f"path: {args.data_path}")
    print(f"herman_sim: {args.herman}")
    print(f"is_parallel: {args.parallel}")
    
    if args.herman:
        make_herman_dataset(args.n_clusters, args.cluster_size, 0.025, args.n_steps, args.n_datasets, "spiking_network/data", is_parallel=(args.parallel == True))
    else:
        make_dataset(args.n_clusters, args.cluster_size, 0, args.n_steps, args.n_datasets, args.data_path, is_parallel = (args.parallel == True))


if __name__ == "__main__":
    main()
