import math
import os
import time
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sentence_transformers import SentenceTransformer
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, TensorDataset


class LSTMModel(nn.Module):
    """
    LSTM-based architecture for text classification on embeddings.
    Treats the embedding as a sequence of small chunks.
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.5,
        chunk_size: int = 8,
    ):
        super().__init__()

        # Split embedding into fixed-size chunks; pad if not divisible.
        self.chunk_size = int(chunk_size)
        self.num_chunks = math.ceil(embedding_dim / self.chunk_size)
        self.padded_dim = self.num_chunks * self.chunk_size

        self.lstm = nn.LSTM(
            input_size=self.chunk_size,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim * 2, 64)  # *2 for BiLSTM
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 2)  # Binary classification

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, embedding_dim)
        bsz, emb_dim = x.shape

        # Pad to multiple of chunk_size if needed
        if emb_dim != self.padded_dim:
            pad_amount = self.padded_dim - emb_dim
            # Pad on the right of last dimension
            x = F.pad(x, (0, pad_amount))

        # Reshape to (batch, seq_len=num_chunks, chunk_size)
        x = x.view(bsz, self.num_chunks, self.chunk_size)

        # LSTM forward
        lstm_out, (h_n, _) = self.lstm(x)  # h_n: (num_layers*2, bsz, hidden_dim)

        # Take last layer forward/backward hidden states
        forward_hidden = h_n[-2, :, :]  # (bsz, hidden_dim)
        backward_hidden = h_n[-1, :, :]  # (bsz, hidden_dim)
        hidden = torch.cat(
            [forward_hidden, backward_hidden], dim=1
        )  # (bsz, 2*hidden_dim)

        x = self.dropout(hidden)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)  # (bsz, 2)
        return x


class LSTM:
    """
    Binary (0/1) text classifier using SentenceTransformer embeddings + LSTM.
    Expects dataframes with columns: 'Summary' (str), 'Truth' (0 or 1).
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.5,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        epochs: int = 20,
        random_state: int = 42,
        chunk_size: int = 8,
        num_workers: Optional[int] = None,
        use_bfloat16: bool = True,  # prefer BF16 on modern GPUs
    ):
        """
        Initialize the LSTM classifier with SentenceTransformer embeddings.

        Args:
            model_name: Name of the SentenceTransformer model for text embeddings
            hidden_dim: Hidden dimension size for LSTM
            num_layers: Number of LSTM layers
            dropout: Dropout rate for regularization
            learning_rate: Learning rate for optimizer
            batch_size: Batch size for training and evaluation
            epochs: Number of training epochs
            random_state: Random seed for reproducibility
            chunk_size: Size of mini-chunks the embedding is split into
            num_workers: DataLoader workers (None -> sensible default)
            use_bfloat16: Use bfloat16 autocast on CUDA if available
        """
        self.encoder = SentenceTransformer(model_name)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.random_state = random_state
        self.chunk_size = chunk_size
        self.use_bfloat16 = use_bfloat16

        torch.manual_seed(random_state)
        np.random.seed(random_state)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[LSTMModel] = None
        self._is_fitted = False

        # choose a conservative default for cross-platform stability
        if num_workers is None:
            self.num_workers = 0
        else:
            self.num_workers = max(0, int(num_workers))

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

    # ---------- Memory-safe helpers ----------

    def _amp_autocast_ctx(self):
        """
        Returns a context manager for mixed precision on CUDA (BF16 preferred),
        otherwise a no-op context on CPU.
        """
        if self.device.type == "cuda":
            # BF16 supported on Ampere+; fall back to FP16 if needed by flipping flag
            dtype = torch.bfloat16 if self.use_bfloat16 else torch.float16
            return torch.amp.autocast(device_type="cuda", dtype=dtype, enabled=True)
        else:
            from contextlib import nullcontext

            return nullcontext()

    def _batched_tensor_iter(self, X_np: np.ndarray, batch_size: int):
        """Yield CUDA/CPU tensors in batches with non_blocking copies."""
        for i in range(0, X_np.shape[0], batch_size):
            batch = torch.from_numpy(X_np[i : i + batch_size]).float()
            yield batch.to(self.device, non_blocking=True)

    # ---------- Training / Evaluation ----------

    def train(
        self, train_df: pd.DataFrame, val_df: Optional[pd.DataFrame] = None
    ) -> None:
        """
        Train the LSTM classifier on the training data.

        Args:
            train_df: Training DataFrame with 'Summary' and 'Truth' columns
            val_df: Optional validation DataFrame for monitoring
        """
        if "Summary" not in train_df or "Truth" not in train_df:
            raise KeyError("train_df must contain 'Summary' and 'Truth' columns.")
        if len(train_df) == 0:
            raise ValueError("Training dataframe is empty.")

        print(f"Training LSTM on {len(train_df)} samples...")

        # Embed training/validation
        X_train = self._embed(train_df["Summary"])
        y_train = self._ensure_binary(train_df["Truth"])

        X_val, y_val = None, None
        if val_df is not None:
            if "Summary" not in val_df or "Truth" not in val_df:
                raise KeyError("val_df must contain 'Summary' and 'Truth' columns.")
            X_val = self._embed(val_df["Summary"])
            y_val = self._ensure_binary(val_df["Truth"])
            print(f"Validation set: {len(val_df)} samples")

        # Initialize model with padding-aware chunking
        embedding_dim = X_train.shape[1]
        self.model = LSTMModel(
            embedding_dim=embedding_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
            chunk_size=self.chunk_size,
        ).to(self.device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # DataLoader (pin_memory for faster H2D copies on CUDA)
        train_dataset = TensorDataset(
            torch.from_numpy(X_train).float(),
            torch.from_numpy(y_train).long(),
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=(self.device.type == "cuda"),
            num_workers=self.num_workers,
            persistent_workers=(self.num_workers > 0),
        )

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            epoch_loss = 0.0

            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device, non_blocking=True)
                batch_y = batch_y.to(self.device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                with self._amp_autocast_ctx():
                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y)

                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / max(1, len(train_loader))

            if val_df is not None:
                # Batched validation to avoid OOM
                self.model.eval()
                y_pred_chunks = []
                with torch.inference_mode():
                    with self._amp_autocast_ctx():
                        for Xb in self._batched_tensor_iter(X_val, self.batch_size):
                            out = self.model(Xb)
                            y_pred_chunks.append(out.argmax(dim=1).cpu())

                y_pred = torch.cat(y_pred_chunks).numpy()
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

            # Optional: help GC/allocator between epochs
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        self._is_fitted = True
        print("LSTM training complete.")

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
        y_true = self._ensure_binary(eval_df["Truth"])

        self.model.eval()
        pred_chunks = []
        with torch.inference_mode():
            with self._amp_autocast_ctx():
                for Xb in self._batched_tensor_iter(X, self.batch_size):
                    out = self.model(Xb)
                    pred_chunks.append(out.argmax(dim=1).cpu())

        preds = torch.cat(pred_chunks).numpy().astype(int)

        return pd.DataFrame(
            {"Prediction": preds, "Truth": y_true.astype(int)},
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
        if len(eval_df) == 0:
            raise ValueError("Evaluation dataframe is empty.")

        X = self._embed(eval_df["Summary"])

        self.model.eval()
        probs_list = []
        with torch.inference_mode():
            with self._amp_autocast_ctx():
                for Xb in self._batched_tensor_iter(X, self.batch_size):
                    out = self.model(Xb)
                    probs_list.append(torch.softmax(out, dim=1).cpu())

        return torch.cat(probs_list, dim=0).numpy()

    def _save_training_graph(self) -> None:
        """
        Save training history graph showing validation metrics per epoch.
        Graph is saved to ./output/lstm/training_history.png
        """
        output_dir = os.path.join("output", "lstm")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"training_history_{time.time()}.png")

        # Create figure with 4 subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(
            "LSTM Training: Validation Performance per Epoch",
            fontsize=14,
            fontweight="bold",
        )

        epochs = self.training_history["epoch"]

        # Training Loss
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

        # Precision
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

        # Recall
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

        # F1
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
        plt.close()

        print(f"📊 Saved training history graph to {output_path}")
