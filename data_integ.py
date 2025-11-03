import pandas as pd
import os
from sklearn.metrics import precision_score, recall_score, f1_score


def load_data(data_path):
    """
    Creates a dataframe of data from the datapath.

    Returns a dataframe with the following columns: ["Date", "Open", "Low", "Close", "Adj close", "Summary", "Truth"]
    """

    full_df = pd.DataFrame(
        columns=["Date", "Open", "Low", "Close", "Adj close", "Summary", "Truth"]
    )
    for file in os.listdir(data_path):
        full_path = os.path.join(data_path, file)
        if os.path.isfile(full_path):
            try:
                df = pd.read_csv(full_path)
                df = df.rename(columns={"Lexrank_summary": "Summary"})
                df = df[df["News_flag"] == 1]

                df["Truth"] = (df["Open"] <= df["Close"]).astype(int)

                df = df[
                    ["Date", "Open", "Low", "Close", "Adj close", "Summary", "Truth"]
                ]

                full_df = pd.concat([full_df, df], ignore_index=True)
            except Exception as e:
                print("Exception for path:", full_path)
                print(e)

            print("Read in ", full_path)

    return full_df


def evaluate(exp_name: str, predictions: pd.DataFrame):
    """
    Evaluate binary classification predictions and save results.

    Args:
        exp_name: Name of the experiment. Results are saved in ./output/{exp_name}/metrics.csv
        predictions: DataFrame with columns ['Prediction', 'Truth'] or ['prediction', 'truth'] (case-insensitive).
                     Each column should contain 0/1 values.
    """
    # Normalize column names
    cols = {c.lower(): c for c in predictions.columns}
    if "prediction" not in cols or "truth" not in cols:
        raise KeyError(
            "Predictions DataFrame must contain 'Prediction' and 'Truth' columns."
        )

    y_pred = predictions[cols["prediction"]].astype(int)
    y_true = predictions[cols["truth"]].astype(int)

    # Compute metrics
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    results = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

    # Prepare output path
    output_dir = os.path.join("output", exp_name)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "metrics.csv")

    # Save results
    pd.DataFrame([results]).to_csv(output_path, index=False)

    print(f"✅ Saved evaluation metrics to {output_path}")
    print(results)
    return results
