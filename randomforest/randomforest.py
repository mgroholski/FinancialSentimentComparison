import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
from transformers import pipeline
from sentence_transformers import SentenceTransformer


class RandomForest:
    def __init__(
        self, data_folder="../data_processor/price_news_integrate", save_model=True
    ):
        """
        Random Forest model combining FinBERT sentiment + MiniLM semantic embeddings.
        Used for predicting next-day stock trend based on price and news summaries.
        """
        self.data_folder = data_folder
        self.save_model = save_model
        self.model = None
        self.df = None
        self.price_features = [
            "Open",
            "High",
            "Low",
            "Close",
            "Adj_Close",
            "Volume",
            "News_flag",
        ]

        print("Loading FinBERT...")
        self.finbert = pipeline("sentiment-analysis", model="yiyanghkust/finbert-tone")

        print("Loading MiniLM...")
        self.minilm = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

        print("Models loaded successfully")

    # ---------- FinBERT ----------
    def finbert_sentiment(self, text):
        """Convert text into sentiment score (-1 to +1) using FinBERT."""
        if not isinstance(text, str) or not text.strip():
            return 0.0
        try:
            res = self.finbert(text[:512])[0]
            label, score = res["label"].lower(), res["score"]
            if label == "positive":
                return score
            elif label == "negative":
                return -score
            else:
                return 0.0
        except Exception:
            return 0.0

    # ---------- Data Loader ----------
    def load_data(self, symbol):
        """Load and preprocess data."""
        file_path = os.path.join(self.data_folder, f"{symbol.upper()}.csv")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        df = pd.read_csv(file_path)
        df.columns = [c.strip().replace(" ", "_") for c in df.columns]
        df.columns = [c.replace("Adj_close", "Adj_Close") for c in df.columns]
        df = df.sort_values("Date").reset_index(drop=True)

        # Create target label (tomorrow’s price movement)
        df["Return"] = df["Close"].pct_change().shift(-1)
        df["Target"] = (df["Return"] > 0).astype(int)

        # Fill missing summaries
        if "Lexrank_summary" not in df.columns:
            df["Lexrank_summary"] = ""
        df["Lexrank_summary"] = df["Lexrank_summary"].fillna("")

        # Run FinBERT sentiment
        print("→ Running FinBERT sentiment analysis...")
        df["Sentiment_finbert"] = df["Lexrank_summary"].apply(self.finbert_sentiment)
        df["Scaled_sentiment"] = (df["Sentiment_finbert"] + 1) / 2

        df = df.dropna(subset=self.price_features + ["Target"])
        self.df = df
        print(f"Loaded {symbol.upper()} | {len(df)} rows")
        return df

    # ---------- Feature Preparation ----------
    def prepare_features(self):
        """Combine FinBERT sentiment, MiniLM embeddings, and price data."""
        if "Lexrank_summary" not in self.df.columns:
            raise ValueError("Missing Lexrank_summary column for text input")

        print("→ Generating MiniLM embeddings...")
        embeddings = np.vstack(
            self.df["Lexrank_summary"].apply(self.minilm.encode).values
        )
        embed_df = pd.DataFrame(
            embeddings, columns=[f"embed_{i}" for i in range(embeddings.shape[1])]
        )

        # Scale numeric features
        scaler = StandardScaler()
        scaled_numeric = scaler.fit_transform(
            self.df[self.price_features + ["Scaled_sentiment"]]
        )
        numeric_df = pd.DataFrame(
            scaled_numeric, columns=self.price_features + ["Scaled_sentiment"]
        )

        X = pd.concat(
            [numeric_df.reset_index(drop=True), embed_df.reset_index(drop=True)], axis=1
        )
        y = self.df["Target"].reset_index(drop=True)

        self.X, self.y = X, y
        print(f"Feature matrix shape: {X.shape}")
        return X, y

    # ---------- Model Training ----------
    def train(self, test_size=0.2):
        """Train Random Forest with dual sentiment-semantic features."""
        X, y = self.prepare_features()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, shuffle=False
        )

        self.model = RandomForestClassifier(n_estimators=300, random_state=42)
        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        print(f"\nModel accuracy: {acc:.3f}")
        print(classification_report(y_test, y_pred))

        if self.save_model:
            model_name = f"rf_dual_sentiment_{int(acc * 100)}.pkl"
            joblib.dump(self.model, model_name)
            print(f"Model saved as: {model_name}")

        self.X_test, self.y_test, self.y_pred = X_test, y_test, y_pred

    # ---------- Visualization ----------
    def visualize(self, symbol="STOCK"):
        """Plot predictions, confusion matrix, and feature importance."""
        if self.model is None:
            raise ValueError("No trained model to visualize.")

        # Actual vs Predicted
        plt.figure(figsize=(12, 5))
        plt.plot(self.y_test.values, label="Actual", color="blue")
        plt.plot(self.y_pred, label="Predicted", color="orange", alpha=0.7)
        plt.title(f"Actual vs Predicted Trend - {symbol.upper()}")
        plt.xlabel("Samples")
        plt.ylabel("Trend (1=Up, 0=Down)")
        plt.legend()
        plt.grid(True)
        plt.show()

        # Confusion Matrix
        cm = confusion_matrix(self.y_test, self.y_pred)
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
        plt.title(f"Confusion Matrix - {symbol.upper()}")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.show()

        # Feature Importance
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1][:15]
        plt.figure(figsize=(8, 5))
        sns.barplot(
            x=importances[indices],
            y=np.array(self.X.columns)[indices],
            palette="viridis",
        )
        plt.title("Top 15 Feature Importances (FinBERT + MiniLM)")
        plt.xlabel("Importance")
        plt.ylabel("Feature")
        plt.show()

    # ---------- Full Pipeline ----------
    def run(self, symbol=""):
        """Execute the full pipeline."""
        self.load_data(symbol)
        self.train()
        self.visualize(symbol)
