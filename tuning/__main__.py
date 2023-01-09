from tuning.tune import run_tune
import argparse
import torch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--n_neurons",    type=int,   default=20,                     help="Number of neurons in the network")
    parser.add_argument("-t", "--n_steps",      type=int,   default=100,                    help="Number of steps per epoch")
    parser.add_argument("-s", "--dataset_size", type=int,   default=10,                     help="Number of datasets to generate")
    parser.add_argument("--model_path",         type=str,   default="data/saved_models",    help="The path where the model should be saved")
    parser.add_argument("-e", "--n_epochs",     type=int,   default=100,                    help="Number of epochs to train for")
    parser.add_argument("-f", "--firing_rate",  type=float, default=0.1,                    help="Target firing rate")
    parser.add_argument("-r", "--seed",         type=int,   default=0,                      help="Random seed")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Tuning model...")
    print(f"Number of neurons:                            {args.n_neurons}")
    print(f"Size of dataset:                              {args.dataset_size}")
    print(f"Number of steps per epoch:                    {args.n_steps}")
    print(f"Number of epochs:                             {args.n_epochs}")
    print(f"Path to store model:                          {args.model_path}")
    print(f"Target firing rate:                           {args.firing_rate}")
    print(f"Device:                                       {device}")

    run_tune(args.n_neurons, args.dataset_size, args.n_steps, args.n_epochs, args.model_path, args.firing_rate, args.seed)

if __name__ == "__main__":
    main()
