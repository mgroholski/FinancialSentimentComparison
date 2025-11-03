import argparse

from svm.svm import SVM
from randomforest.randomforest import RandomForest
from cnn.cnn import CNN
from lstm.lstm import LSTM
from data_integ import load_data, evaluate

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15


def main(args):
    data_path = args.data

    print(f"Loading data from {data_path}.")
    dataset = load_data(data_path)
    print(f"Loaded data from {data_path}.")

    models = []
    pipeline = args.pipeline
    if pipeline == "svm" or pipeline == "all":
        models.append(("svm", SVM()))
    if pipeline == "cnn" or pipeline == "all":
        models.append(("cnn", CNN()))
    if pipeline == "lstm" or pipeline == "all":
        models.append(("lstm", LSTM()))
    if pipeline == "randomforest" or pipeline == "all":
        models.append(("randomforest", RandomForest()))
    if not len(models):
        raise ValueError(f"Unknown pipeline type: {args.pipeline}")

    n = len(dataset)

    train_n = int(n * TRAIN_RATIO)
    val_n = int(n * VAL_RATIO)

    if train_n + val_n > n:
        raise ValueError("TRAIN_RATIO + VAL_RATIO must be <= 1.0")

    train_df = dataset.iloc[:train_n].copy()
    val_df = dataset.iloc[train_n : train_n + val_n].copy()
    test_df = dataset.iloc[train_n + val_n :].copy()

    for name, model in models:
        print(f"Starting training for {name}.")
        model.train(train_df, val_df)

        for run in range(args.runs):
            print(f"Starting prediction run {run} for {name}.")
            results = model.predict(test_df)
            evaluate(f"{name}-run{run + 1}", results)


def parse_args():
    parser = argparse.ArgumentParser(description="Run ML pipelines.")

    parser.add_argument(
        "-p",
        "--pipeline",
        required=True,
        choices=["svm", "cnn", "lstm", "randomforest", "all"],
        help="Select which pipeline to run.",
    )

    parser.add_argument(
        "-d",
        "--data",
        required=True,
        help="Path to the training data.",
    )

    parser.add_argument(
        "-r",
        "--runs",
        required=True,
        type=int,
        help="Number of runs to execute for each pipeline.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
