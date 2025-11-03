from typing import Optional
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
from sentence_transformers import SentenceTransformer


class RandomForest:
    """
    Binary (0/1) text classifier using SentenceTransformer embeddings + RandomForestClassifier.
    Expects dataframes with columns: 'Summary' (str), 'Truth' (0 or 1).
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        random_state: int = 42,
        trees_per_epoch: int = 10,
    ):
        """
        Initialize the RandomForest classifier with SentenceTransformer embeddings.
        
        Args:
            model_name: Name of the SentenceTransformer model for text embeddings
            n_estimators: Number of trees in the forest
            max_depth: Maximum depth of each tree (None means unlimited)
            random_state: Random seed for reproducibility
            trees_per_epoch: Number of trees to add per "epoch" for incremental training
        """
        self.encoder = SentenceTransformer(model_name)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.trees_per_epoch = trees_per_epoch
        self.model = None  # Will be initialized during training
        self._is_fitted = False
        self.training_history = {
            "epoch": [],
            "precision": [],
            "recall": [],
            "f1": [],
        }

    @staticmethod
    def _ensure_binary(y: pd.Series) -> np.ndarray:
        """Validate that labels are binary (0/1)."""
        y_arr = y.astype(int).to_numpy()
        uniq = set(np.unique(y_arr).tolist())
        if not uniq.issubset({0, 1}):
            raise ValueError(
                f"'Truth' must be binary 0/1. Found labels: {sorted(uniq)}"
            )
        return y_arr

    def _embed(self, texts: pd.Series) -> np.ndarray:
        """Convert text summaries to dense embeddings."""
        # normalize_embeddings=True helps with consistent feature scales
        return self.encoder.encode(
            texts.astype(str).tolist(),
            show_progress_bar=False,
            normalize_embeddings=True,
        )

    def train(
        self, train_df: pd.DataFrame, val_df: Optional[pd.DataFrame] = None
    ) -> None:
        """
        Train the RandomForest classifier incrementally and track validation performance.
        
        Args:
            train_df: Training DataFrame with 'Summary' and 'Truth' columns
            val_df: Optional validation DataFrame for monitoring (not used in training)
        """
        if "Summary" not in train_df or "Truth" not in train_df:
            raise KeyError("train_df must contain 'Summary' and 'Truth' columns.")

        if len(train_df) == 0:
            raise ValueError("Training dataframe is empty.")

        print(f"Training RandomForest on {len(train_df)} samples...")
        
        # Embed training data once
        X_train = self._embed(train_df["Summary"])
        y_train = self._ensure_binary(train_df["Truth"])
        
        # Embed validation data if provided
        if val_df is not None:
            if "Summary" not in val_df or "Truth" not in val_df:
                raise KeyError("val_df must contain 'Summary' and 'Truth' columns.")
            X_val = self._embed(val_df["Summary"])
            y_val = self._ensure_binary(val_df["Truth"])
            print(f"Validation set: {len(val_df)} samples")
        
        # Calculate number of epochs
        n_epochs = max(1, self.n_estimators // self.trees_per_epoch)
        
        # Train incrementally
        for epoch in range(1, n_epochs + 1):
            n_trees = min(epoch * self.trees_per_epoch, self.n_estimators)
            
            # Initialize or update model
            self.model = RandomForestClassifier(
                n_estimators=n_trees,
                max_depth=self.max_depth,
                random_state=self.random_state,
                n_jobs=-1,
                class_weight="balanced",
                warm_start=False,  # Train fresh each time for consistency
            )
            
            self.model.fit(X_train, y_train)
            
            # Evaluate on validation set if provided
            if val_df is not None:
                y_pred = self.model.predict(X_val)
                
                precision = precision_score(y_val, y_pred, zero_division=0)
                recall = recall_score(y_val, y_pred, zero_division=0)
                f1 = f1_score(y_val, y_pred, zero_division=0)
                
                self.training_history["epoch"].append(epoch)
                self.training_history["precision"].append(precision)
                self.training_history["recall"].append(recall)
                self.training_history["f1"].append(f1)
                
                print(f"Epoch {epoch}/{n_epochs} ({n_trees} trees) - "
                      f"Val Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        
        self._is_fitted = True
        print("RandomForest training complete.")
        
        # Save training graph if validation was used
        if val_df is not None and len(self.training_history["epoch"]) > 0:
            self._save_training_graph()

    def predict(self, eval_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate predictions for the evaluation dataset.
        
        Args:
            eval_df: DataFrame with 'Summary' and 'Truth' columns
            
        Returns:
            DataFrame with 'Prediction' and 'Truth' columns
        """
        if not self._is_fitted:
            raise RuntimeError("Model is not fitted. Call train() first.")
        
        if "Summary" not in eval_df or "Truth" not in eval_df:
            raise KeyError("eval_df must contain 'Summary' and 'Truth' columns.")

        if len(eval_df) == 0:
            raise ValueError("Evaluation dataframe is empty.")

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

    def predict_proba(self, eval_df: pd.DataFrame) -> np.ndarray:
        """
        Get prediction probabilities for each class.
        
        Args:
            eval_df: DataFrame with 'Summary' column
            
        Returns:
            Array of shape (n_samples, 2) with probabilities for [class_0, class_1]
        """
        if not self._is_fitted:
            raise RuntimeError("Model is not fitted. Call train() first.")
        
        if "Summary" not in eval_df:
            raise KeyError("eval_df must contain 'Summary' column.")

        X = self._embed(eval_df["Summary"])
        return self.model.predict_proba(X)

    def get_feature_importance(self) -> np.ndarray:
        """
        Get feature importance scores from the trained model.
        
        Returns:
            Array of feature importance scores (one per embedding dimension)
        """
        if not self._is_fitted:
            raise RuntimeError("Model is not fitted. Call train() first.")
        
        return self.model.feature_importances_

    def _save_training_graph(self) -> None:
        """
        Save training history graph showing validation metrics per epoch.
        Graph is saved to ./output/randomforest/training_history.png
        """
        output_dir = os.path.join("output", "randomforest")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "training_history.png")
        
        # Create figure with 3 subplots
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        fig.suptitle("RandomForest Training: Validation Performance per Epoch", fontsize=14, fontweight="bold")
        
        epochs = self.training_history["epoch"]
        
        # Precision plot
        axes[0].plot(epochs, self.training_history["precision"], marker='o', linewidth=2, color='#2E86AB')
        axes[0].set_xlabel("Epoch", fontsize=11)
        axes[0].set_ylabel("Precision", fontsize=11)
        axes[0].set_title("Validation Precision", fontsize=12, fontweight="bold")
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim([0, 1.05])
        
        # Recall plot
        axes[1].plot(epochs, self.training_history["recall"], marker='s', linewidth=2, color='#A23B72')
        axes[1].set_xlabel("Epoch", fontsize=11)
        axes[1].set_ylabel("Recall", fontsize=11)
        axes[1].set_title("Validation Recall", fontsize=12, fontweight="bold")
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim([0, 1.05])
        
        # F1 Score plot
        axes[2].plot(epochs, self.training_history["f1"], marker='^', linewidth=2, color='#F18F01')
        axes[2].set_xlabel("Epoch", fontsize=11)
        axes[2].set_ylabel("F1 Score", fontsize=11)
        axes[2].set_title("Validation F1 Score", fontsize=12, fontweight="bold")
        axes[2].grid(True, alpha=0.3)
        axes[2].set_ylim([0, 1.05])
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Saved training history graph to {output_path}")
