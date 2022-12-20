from benchmarking.timing import timing
from benchmarking.compare_to_old import compare_to_old
from benchmarking.parallelization import parallelization
import argparse
import torch
import sys
from IPython.core import ultratb

sys.excepthook = ultratb.FormattedTB(
    mode="Context", color_scheme="Linux", call_pdb=False
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--n_neurons",    type=int,   default=20,                 help="Number of neurons in the network")
    parser.add_argument("-t", "--n_steps",      type=int,   default=100,                help="Number of steps in simulation")
    parser.add_argument("-N", "--n_samples",    type=int,   default=5,                  help="Number of simulations to run")
    parser.add_argument("-c", "--compared",                                             help="Number of simulations to run", action="store_true")
    parser.add_argument("-p", "--parallel",                                             help="Number of simulations to run", action="store_true")
    parser.add_argument("--data_path",          type=str,   default="data/benchmark_data",   help="The path where the data should be saved")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Max number of neurons:                        {args.n_neurons}")
    print(f"Number of steps:                              {args.n_steps}")
    print(f"Number of samples:                            {args.n_samples}")
    print(f"Path to store data:                           {args.data_path}")
    print(f"Device:                                       {device}")

    if args.compared:
        print("Comparing models ...")
        compare_to_old(args.n_neurons, args.n_steps, args.n_samples, args.data_path, device)
    elif args.parallel:
        print("Timing model in parallel ...")
        parallelization(args.n_neurons, args.n_steps, args.n_samples, args.data_path, device)
    else:
        print("Timing model ...")
        timing(args.n_neurons, args.n_steps, args.n_samples, args.data_path, device)

if __name__ == "__main__":
    main()
