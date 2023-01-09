from simulation.simulate import run_simulation
from simulation.simulate_herman import run_herman
import argparse
import torch

import sys
from IPython.core import ultratb

sys.excepthook = ultratb.FormattedTB(
    mode="Context", color_scheme="Linux", call_pdb=False
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--n_neurons",    type=int,   default=20,                       help="Number of neurons in the network")
    parser.add_argument("-t", "--n_steps",      type=int,   default=20_000,                   help="Number of steps in simulation")
    parser.add_argument("-s", "--n_sims",       type=int,   default=1,                        help="Number of simulations to run")
    parser.add_argument("--data_path",          type=str,   default="data/simulation_data",   help="The path where the data should be saved")
    parser.add_argument("-p", "--max_parallel", type=int,   default=100,                      help="The max number of simulations to run in parallel")
    parser.add_argument("-f", "--firing_rate",  type=float, default=0.1,                      help="The average firing fate of the neurons")
    parser.add_argument("-r", "--r",            type=float, default=0.025,                    help="The r to use for the herman case")
    parser.add_argument("-th", "--threshold",   type=float, default=1.378e-3,                 help="The threshold to use for the herman case")
    parser.add_argument("-em", "--emptiness",   type=float, default=0.9,                      help="The sparsity of the w_0 matrices")
    parser.add_argument("-e", "--n_epochs",     type=int,   default=100,                      help="Number of epochs to train for")
    parser.add_argument("-fn", "--folder_name", type=str,   default="",                       help="The name for the folder for the saved data")
    parser.add_argument("--herman",                                                           help="Run hermans simulation", action="store_true")
    parser.add_argument("--seed",               type=int,   default=14071789,                 help="The seed to use for the simulation")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Generating datasets...")
    print(f"Number of neurons:                            {args.n_neurons}")
    print(f"Number of simulations:                        {args.n_sims}")
    print(f"Number of steps:                              {args.n_steps}")
    print(f"Path to store data:                           {args.data_path}")
    print(f"Max number of simulation to run in parallel:  {args.max_parallel}")
    print("Device:                                       ", device)


    if args.herman:
        run_herman(args.n_neurons, args.n_sims, args.n_steps, args.data_path, args.folder_name, args.max_parallel)
    else:
        run_simulation(args.n_neurons, args.n_sims, args.n_steps, args.data_path, args.folder_name, args.seed, args.max_parallel)

if __name__ == "__main__":
    main()
