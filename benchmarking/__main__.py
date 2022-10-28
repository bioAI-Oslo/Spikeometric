from benchmarking.timing import timing
import argparse
import torch

import sys
from IPython.core import ultratb

sys.excepthook = ultratb.FormattedTB(
    mode="Context", color_scheme="Linux", call_pdb=False
)

# Next steps
# 2. Tuning with torch
# 4. Find better way of aggregating in numpy
# 4. Compare speed gains from parallelization
# 5. Compare speed gains from sparsification
# 7. Get torch_geometric on the cluster
# 8. Make network unaware of simulator

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--n_neurons",    type=int,   default=20,                 help="Number of neurons in the network")
    parser.add_argument("-t", "--n_steps",      type=int,   default=100,                help="Number of steps in simulation")
    parser.add_argument("-N", "--n_samples",    type=int,   default=1,                  help="Number of simulations to run")
    parser.add_argument("--data_path",          type=str,   default="benchmarking/data",   help="The path where the data should be saved")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Timing model ...")
    print(f"Max number of neurons:                        {args.n_neurons}")
    print(f"Number of steps:                              {args.n_steps}")
    print(f"Number of samples:                            {args.n_samples}")
    print(f"Path to store data:                           {args.data_path}")

    timing(args.n_neurons, args.n_steps, args.n_samples, args.data_path)
if __name__ == "__main__":
    main()
