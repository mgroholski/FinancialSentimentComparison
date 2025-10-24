import argparse

from svm.svm import SVM
from quantum.quantum import Quantum
from randomforest.randomforest import RandomForest
from cnn.cnn import CNN
from lstm.lstm import LSTM
from datasets import load_dataset


def main(args):
    dataset = load_dataset("Zihan1004/FNSPID")

    model = None
    if args.pipeline == "svm":
        model = SVM()
    elif args.pipeline == "cnn":
        model = CNN()
    elif args.pipeline == "lstm":
        model = LSTM()
    elif args.pipeline == "quantum":
        model = Quantum()
    elif args.pipeline == "randomforest":
        model = RandomForest()
    else:
        raise ValueError(f"Unknown pipeline type: {args.pipeline}")


def parse_args():
    parser = argparse.ArgumentParser(description="Run a specific ML pipeline.")
    parser.add_argument(
        "--pipeline",
        required=True,
        choices=["svm", "cnn", "lstm", "quantum", "forest"],
        help="Select which pipeline to run.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
