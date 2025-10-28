import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib


class RandomForest:
    def __init__(self, data_folder="../data_processor/gpt_sentiment_price_news_integrate", save_model=True):
        self.data_folder = data_folder
        self.save_model = save_model
        self.model = None
        self.df = None
        self.features = ["Open", "High", "Low", "Close", "Volume", "Scaled_sentiment"]

    def load_data(self, symbol):
        """Load data for the given stock symbol."""
        file_path = os.path.join(self.data_folder, f"{symbol.lower()}.csv")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        df = pd.read_csv(file_path)
        df = df.dropna(subset=["Sentiment_gpt"])
        df["Return"] = df["Close"].pct_change().shift(-1)
        df["Target"] = (df["Return"] > 0).astype(int)
        df = df.dropna(subset=self.features + ["Target"])
        self.df = df
        print(f"Loaded {symbol.upper()} | {len(df)} rows")
        return df

    def train(self, test_size=0.2):
        """Train the Random Forest model."""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        X = self.df[self.features]
        y = self.df["Target"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)

        self.model = RandomForestClassifier(n_estimators=200, random_state=42)
        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"\nModel accuracy: {acc:.3f}")
        print(classification_report(y_test, y_pred))

        if self.save_model:
            model_name = f"random_forest_model_{int(acc*100)}.pkl"
            joblib.dump(self.model, model_name)
            print(f"Model saved to: {model_name}")

        # Save data for visualization
        self.X_test, self.y_test, self.y_pred = X_test, y_test, y_pred

    def visualize(self, symbol="stock"):
        """Generate three key plots: prediction trend, confusion matrix, and feature importance."""
        if self.df is None or self.model is None:
            raise ValueError("No trained model or data to visualize.")

        # (a) Actual vs Predicted Trend
        plt.figure(figsize=(12, 5))
        plt.plot(self.df.index[-len(self.y_test):], self.y_test.values, label="Actual (Up=1 / Down=0)", color="blue")
        plt.plot(self.df.index[-len(self.y_test):], self.y_pred, label="Predicted", color="orange", alpha=0.7)
        plt.title(f"Actual vs Predicted Trend - {symbol.upper()}")
        plt.xlabel("Time index")
        plt.ylabel("Trend (1=Up, 0=Down)")
        plt.legend()
        plt.grid(True)
        plt.show()

        # (b) Confusion Matrix
        cm = confusion_matrix(self.y_test, self.y_pred)
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
        plt.title(f"Confusion Matrix - {symbol.upper()}")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.show()

        # (c) Feature Importance
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1]

        plt.figure(figsize=(8, 5))
        sns.barplot(x=importances[indices], y=np.array(self.features)[indices], palette="viridis")
        plt.title("Feature Importance (Random Forest)")
        plt.xlabel("Importance Score")
        plt.ylabel("Feature")
        plt.show()

    def run(self):
        """Complete workflow: load → train → visualize."""
        symbol = input("Enter stock symbol (e.g., aa): ").strip().lower()
        self.load_data(symbol)
        self.train()
        self.visualize(symbol)
