import os
import time
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sentence_transformers import SentenceTransformer
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, TensorDataset


class CNNModel(nn.Module):
    """
    1D CNN architecture for text classification on embeddings.
    """

    def __init__(
        self, embedding_dim: int, num_filters: int = 128, dropout: float = 0.5
    ):
        super(CNNModel, self).__init__()

        # Multiple convolutional layers with different kernel sizes
        self.conv1 = nn.Conv1d(
            in_channels=1, out_channels=num_filters, kernel_size=3, padding=1
        )
        self.conv2 = nn.Conv1d(
            in_channels=1, out_channels=num_filters, kernel_size=5, padding=2
        )
        self.conv3 = nn.Conv1d(
            in_channels=1, out_channels=num_filters, kernel_size=7, padding=3
        )

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.pool = nn.AdaptiveMaxPool1d(1)

        # Fully connected layers
        self.fc1 = nn.Linear(num_filters * 3, 64)
        self.fc2 = nn.Linear(64, 2)  # Binary classification (0 or 1)

    def forward(self, x):
        # x shape: (batch_size, embedding_dim)
        # Reshape for 1D convolution: (batch_size, 1, embedding_dim)
        x = x.unsqueeze(1)

        # Apply multiple convolutions
        x1 = self.pool(self.relu(self.conv1(x))).squeeze(2)  # (batch_size, num_filters)
        x2 = self.pool(self.relu(self.conv2(x))).squeeze(2)
        x3 = self.pool(self.relu(self.conv3(x))).squeeze(2)

        # Concatenate features from different kernel sizes
        x = torch.cat([x1, x2, x3], dim=1)  # (batch_size, num_filters * 3)

        # Fully connected layers
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x


class CNN:
    """
    Binary (0/1) text classifier using SentenceTransformer embeddings + CNN.
    Expects dataframes with columns: 'Summary' (str), 'Truth' (0 or 1).
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        num_filters: int = 128,
        dropout: float = 0.5,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        epochs: int = 20,
        random_state: int = 42,
    ):
        """
        Initialize the CNN classifier with SentenceTransformer embeddings.

        Args:
            model_name: Name of the SentenceTransformer model for text embeddings
            num_filters: Number of filters per convolutional layer
            dropout: Dropout rate for regularization
            learning_rate: Learning rate for optimizer
            batch_size: Batch size for training
            epochs: Number of training epochs
            random_state: Random seed for reproducibility
        """
        self.encoder = SentenceTransformer(model_name)
        self.num_filters = num_filters
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.random_state = random_state

        # Set random seeds for reproducibility
        torch.manual_seed(random_state)
        np.random.seed(random_state)

        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = None
        self._is_fitted = False
        self.training_history = {
            "epoch": [],
            "train_loss": [],
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
        return self.encoder.encode(
            texts.astype(str).tolist(),
            show_progress_bar=False,
            normalize_embeddings=True,
        )

    def train(
        self, train_df: pd.DataFrame, val_df: Optional[pd.DataFrame] = None
    ) -> None:
        """
        Train the CNN classifier on the training data.

        Args:
            train_df: Training DataFrame with 'Summary' and 'Truth' columns
            val_df: Optional validation DataFrame for monitoring (not used in training)
        """
        if "Summary" not in train_df or "Truth" not in train_df:
            raise KeyError("train_df must contain 'Summary' and 'Truth' columns.")

        if len(train_df) == 0:
            raise ValueError("Training dataframe is empty.")

        print(f"Training CNN on {len(train_df)} samples...")

        # Embed training data
        X_train = self._embed(train_df["Summary"])
        y_train = self._ensure_binary(train_df["Truth"])

        # Embed validation data if provided
        X_val, y_val = None, None
        if val_df is not None:
            if "Summary" not in val_df or "Truth" not in val_df:
                raise KeyError("val_df must contain 'Summary' and 'Truth' columns.")
            X_val = self._embed(val_df["Summary"])
            y_val = self._ensure_binary(val_df["Truth"])
            print(f"Validation set: {len(val_df)} samples")

        # Initialize model
        embedding_dim = X_train.shape[1]
        self.model = CNNModel(
            embedding_dim=embedding_dim,
            num_filters=self.num_filters,
            dropout=self.dropout,
        ).to(self.device)

        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Create DataLoader
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train), torch.LongTensor(y_train)
        )
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )

        # Training loop
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            epoch_loss = 0.0

            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                # Forward pass
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(train_loader)

            # Evaluate on validation set if provided
            if val_df is not None:
                self.model.eval()
                with torch.no_grad():
                    X_val_tensor = torch.FloatTensor(X_val).to(self.device)
                    outputs = self.model(X_val_tensor)
                    _, y_pred = torch.max(outputs, 1)
                    y_pred = y_pred.cpu().numpy()

                precision = precision_score(y_val, y_pred, zero_division=0)
                recall = recall_score(y_val, y_pred, zero_division=0)
                f1 = f1_score(y_val, y_pred, zero_division=0)

                self.training_history["epoch"].append(epoch)
                self.training_history["train_loss"].append(avg_loss)
                self.training_history["precision"].append(precision)
                self.training_history["recall"].append(recall)
                self.training_history["f1"].append(f1)

                print(
                    f"Epoch {epoch}/{self.epochs} - Loss: {avg_loss:.4f} - "
                    f"Val Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}"
                )
            else:
                print(f"Epoch {epoch}/{self.epochs} - Loss: {avg_loss:.4f}")

        self._is_fitted = True
        print("CNN training complete.")

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

        # Embed evaluation data
        X = self._embed(eval_df["Summary"])
        y_true = self._ensure_binary(eval_df["Truth"])

        # Make predictions
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            outputs = self.model(X_tensor)
            _, preds = torch.max(outputs, 1)
            preds = preds.cpu().numpy()

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

        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            outputs = self.model(X_tensor)
            probs = torch.softmax(outputs, dim=1)
            return probs.cpu().numpy()

    def _save_training_graph(self) -> None:
        """
        Save training history graph showing validation metrics per epoch.
        Graph is saved to ./output/cnn/training_history.png
        """
        output_dir = os.path.join("output", "cnn")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"training_history_{time.time()}.png")

        # Create figure with 4 subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(
            "CNN Training: Validation Performance per Epoch",
            fontsize=14,
            fontweight="bold",
        )

        epochs = self.training_history["epoch"]

        # Training Loss plot
        axes[0, 0].plot(
            epochs,
            self.training_history["train_loss"],
            marker="o",
            linewidth=2,
            color="#E63946",
        )
        axes[0, 0].set_xlabel("Epoch", fontsize=11)
        axes[0, 0].set_ylabel("Loss", fontsize=11)
        axes[0, 0].set_title("Training Loss", fontsize=12, fontweight="bold")
        axes[0, 0].grid(True, alpha=0.3)

        # Precision plot
        axes[0, 1].plot(
            epochs,
            self.training_history["precision"],
            marker="o",
            linewidth=2,
            color="#2E86AB",
        )
        axes[0, 1].set_xlabel("Epoch", fontsize=11)
        axes[0, 1].set_ylabel("Precision", fontsize=11)
        axes[0, 1].set_title("Validation Precision", fontsize=12, fontweight="bold")
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_ylim([0, 1.05])

        # Recall plot
        axes[1, 0].plot(
            epochs,
            self.training_history["recall"],
            marker="s",
            linewidth=2,
            color="#A23B72",
        )
        axes[1, 0].set_xlabel("Epoch", fontsize=11)
        axes[1, 0].set_ylabel("Recall", fontsize=11)
        axes[1, 0].set_title("Validation Recall", fontsize=12, fontweight="bold")
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_ylim([0, 1.05])

        # F1 Score plot
        axes[1, 1].plot(
            epochs,
            self.training_history["f1"],
            marker="^",
            linewidth=2,
            color="#F18F01",
        )
        axes[1, 1].set_xlabel("Epoch", fontsize=11)
        axes[1, 1].set_ylabel("F1 Score", fontsize=11)
        axes[1, 1].set_title("Validation F1 Score", fontsize=12, fontweight="bold")
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_ylim([0, 1.05])

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"📊 Saved training history graph to {output_path}")
