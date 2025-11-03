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

    train_end = len(dataset) * TRAIN_RATIO
    val_end = train_end + len(dataset) * VAL_RATIO
    train_df = dataset.iloc[:train_end].copy()
    val_df = dataset.iloc[train_end:val_end].copy()
    test_df = dataset.iloc[val_end:].copy()

    for name, model in models:
        model.train(train_df, val_df)

        for run in range(args.runs):
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
