import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix


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
        exp_name: Name of the experiment. Results are saved in ./output/{exp_name}/
        predictions: DataFrame with columns ['Prediction', 'Truth'] or ['prediction', 'truth'] (case-insensitive).
                     Each column should contain 0/1 values.
    
    Outputs:
        - metrics.csv: Precision, Recall, F1 Score
        - confusion_matrix.png: Confusion matrix visualization
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
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    results = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

    # Prepare output directory
    output_dir = os.path.join("output", exp_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Save metrics CSV
    metrics_path = os.path.join(output_dir, "metrics.csv")
    pd.DataFrame([results]).to_csv(metrics_path, index=False)

    # Save confusion matrix visualization
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    _save_confusion_matrix(cm, y_true, y_pred, cm_path, exp_name)

    print(f"✅ Saved evaluation metrics to {metrics_path}")
    print(f"📊 Saved confusion matrix to {cm_path}")
    print(f"   Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    return results


def _save_confusion_matrix(
    cm: np.ndarray, 
    y_true: pd.Series, 
    y_pred: pd.Series, 
    output_path: str,
    exp_name: str
) -> None:
    """
    Create and save a confusion matrix visualization.
    
    Args:
        cm: Confusion matrix array
        y_true: True labels
        y_pred: Predicted labels
        output_path: Path to save the figure
        exp_name: Experiment name for title
    """
    # Calculate percentages
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create heatmap
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues', 
        cbar=True,
        square=True,
        linewidths=1,
        linecolor='gray',
        ax=ax
    )
    
    # Add percentage annotations
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            percentage = cm_percent[i, j]
            ax.text(
                j + 0.5, i + 0.7, 
                f'({percentage:.1f}%)',
                ha='center', va='center',
                color='darkblue' if cm[i, j] < cm.max() / 2 else 'white',
                fontsize=9
            )
    
    # Labels and title
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax.set_title(f'Confusion Matrix: {exp_name}', fontsize=14, fontweight='bold', pad=20)
    
    # Set tick labels
    ax.set_xticklabels(['Down (0)', 'Up (1)'], fontsize=11)
    ax.set_yticklabels(['Down (0)', 'Up (1)'], fontsize=11, rotation=0)
    
    # Add statistics text box
    accuracy = (cm[0, 0] + cm[1, 1]) / cm.sum()
    stats_text = f'Accuracy: {accuracy:.2%}\nTotal Samples: {cm.sum()}'
    ax.text(
        1.35, 0.5, stats_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='center',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3)
    )
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

