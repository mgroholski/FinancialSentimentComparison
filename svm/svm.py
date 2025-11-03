from typing import Optional
import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC
from sentence_transformers import SentenceTransformer


class SVM:
    """
    Binary (0/1) text classifier using SentenceTransformer embeddings + LinearSVC.
    Expects dataframes with columns: 'Summary' (str), 'Truth' (0 or 1).
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", random_state: int = 42):
        self.encoder = SentenceTransformer(model_name)
        # LinearSVC is fast and works well on dense embeddings.
        self.model = LinearSVC(random_state=random_state)

    @staticmethod
    def _ensure_binary(y: pd.Series) -> np.ndarray:
        y_arr = y.astype(int).to_numpy()
        uniq = set(np.unique(y_arr).tolist())
        if not uniq.issubset({0, 1}):
            raise ValueError(
                f"'Truth' must be binary 0/1. Found labels: {sorted(uniq)}"
            )
        return y_arr

    def _embed(self, texts: pd.Series) -> np.ndarray:
        # normalize_embeddings=True often helps linear models on SBERT vectors
        return self.encoder.encode(
            texts.astype(str).tolist(),
            show_progress_bar=False,
            normalize_embeddings=True,
        )

    def train(
        self, train_df: pd.DataFrame, val_df: Optional[pd.DataFrame] = None
    ) -> None:
        if val_df is not None:
            df = pd.concat([train_df, val_df], ignore_index=True)
        else:
            df = train_df.copy()

        if "Summary" not in df or "Truth" not in df:
            raise KeyError("Dataframes must contain 'Summary' and 'Truth' columns.")

        X = self._embed(df["Summary"])
        y = self._ensure_binary(df["Truth"])

        self.model.fit(X, y)

    def predict(self, eval_df: pd.DataFrame) -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError("Model is not initialized.")
        if "Summary" not in eval_df or "Truth" not in eval_df:
            raise KeyError("eval_df must contain 'Summary' and 'Truth' columns.")

        X = self._embed(eval_df["Summary"])
        preds = self.model.predict(X)

        # Ground truth (validated as 0/1)
        y_true = self._ensure_binary(eval_df["Truth"])

        return pd.DataFrame(
            {
                "Prediction": preds.astype(int),
                "Truth": y_true.astype(int),
            },
            index=eval_df.index,
        )

    def decision_scores(self, eval_df: pd.DataFrame) -> np.ndarray:
        """
        Returns signed distance to the decision boundary for each row
        (useful as a confidence score; larger magnitude = more confident).
        """
        if "Summary" not in eval_df:
            raise KeyError("eval_df must contain 'Summary' column.")
        X = self._embed(eval_df["Summary"])
        return self.model.decision_function(X)
