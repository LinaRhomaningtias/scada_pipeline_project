import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

from model.config import FEATURES, TARGET, HIDDEN_LAYERS, THRESHOLD

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "scada_pipeline.csv"
MODEL_DIR = ROOT / "model"
OUTPUT_DIR = ROOT / "outputs"
MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

class MLP(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 32), torch.nn.ReLU(), torch.nn.Dropout(0.20),
            torch.nn.Linear(32, 16), torch.nn.ReLU(), torch.nn.Dropout(0.10),
            torch.nn.Linear(16, 8), torch.nn.ReLU(),
            torch.nn.Linear(8, 1)
        )
    def forward(self, x):
        return self.net(x)


def main():
    np.random.seed(42)
    torch.manual_seed(42)

    df = pd.read_csv(DATA_PATH, sep=";")
    X = df[FEATURES].astype(float).values
    y = df[TARGET].astype(int).values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.20, random_state=42, stratify=y_train
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    model = MLP(len(FEATURES))
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    xt = torch.tensor(X_train_s, dtype=torch.float32)
    yt = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32)
    xv = torch.tensor(X_val_s, dtype=torch.float32)
    yv = torch.tensor(y_val.reshape(-1, 1), dtype=torch.float32)

    best_state = None
    best_val = float("inf")
    patience, wait = 20, 0
    train_losses, val_losses = [], []

    for epoch in range(250):
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(xt), yt)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            vloss = criterion(model(xv), yv).item()
        train_losses.append(float(loss.item()))
        val_losses.append(float(vloss))

        if vloss < best_val:
            best_val = vloss
            best_state = {k: v.detach().cpu().numpy().copy() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    model.load_state_dict({k: torch.tensor(v) for k, v in best_state.items()})
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(torch.tensor(X_test_s, dtype=torch.float32))).numpy().ravel()
    pred = (probs >= THRESHOLD).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0),
        "threshold": THRESHOLD,
        "epochs": len(train_losses),
        "train_rows": len(y_train),
        "validation_rows": len(y_val),
        "test_rows": len(y_test),
        "normal_count": int((y == 0).sum()),
        "anomaly_count": int((y == 1).sum()),
        "feature_count": len(FEATURES),
    }
    (MODEL_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    joblib.dump(scaler, MODEL_DIR / "scaler.joblib")

    # Deployment-friendly weights: no PyTorch dependency is required by Vercel.
    weights = {}
    for i, (name, value) in enumerate(best_state.items()):
        weights[f"p{i}"] = value
    np.savez(MODEL_DIR / "mlp_weights.npz", **weights)

    # Save compact test examples for the Recent Detections panel.
    recent = df.iloc[:5][["timestamp", "pressure", "flow_rate", "temperature", "target"]].copy()
    recent["target"] = recent["target"].map({0: "Normal", 1: "Anomaly"})
    recent.to_json(OUTPUT_DIR / "recent.json", orient="records")

    plt.figure(figsize=(7, 4))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "loss_curve.png", dpi=150)
    plt.close()

    cm = confusion_matrix(y_test, pred)
    plt.figure(figsize=(5, 4))
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks([0, 1], ["Normal", "Anomaly"])
    plt.yticks([0, 1], ["Normal", "Anomaly"])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha="center", va="center")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=150)
    plt.close()

    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
