from pathlib import Path
import sys
import json

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import joblib
import numpy as np

from model.config import FEATURES, THRESHOLD

MODEL_DIR = ROOT / "model"

_SCALER = joblib.load(MODEL_DIR / "scaler.joblib")
_DATA = np.load(MODEL_DIR / "mlp_weights.npz")
_METRICS = json.loads((MODEL_DIR / "metrics.json").read_text(encoding="utf-8"))

# Saved parameter order from PyTorch Sequential:
# p0 weight, p1 bias, p2 weight, p3 bias, p4 weight, p5 bias, p6 weight, p7 bias.
W1, B1 = _DATA["p0"], _DATA["p1"]
W2, B2 = _DATA["p2"], _DATA["p3"]
W3, B3 = _DATA["p4"], _DATA["p5"]
W4, B4 = _DATA["p6"], _DATA["p7"]


def relu(x):
    return np.maximum(x, 0)


def sigmoid(x):
    x = np.clip(x, -60, 60)
    return 1 / (1 + np.exp(-x))


def predict(values: dict):
    x = np.array([[float(values[f]) for f in FEATURES]], dtype=float)
    xs = _SCALER.transform(x)
    h1 = relu(xs @ W1.T + B1)
    h2 = relu(h1 @ W2.T + B2)
    h3 = relu(h2 @ W3.T + B3)
    logit = h3 @ W4.T + B4
    anomaly_probability = float(sigmoid(logit)[0, 0])
    prediction = int(anomaly_probability >= THRESHOLD)
    return {
        "prediction": prediction,
        "label": "Anomaly" if prediction else "Normal",
        "anomaly_probability": anomaly_probability,
        "normal_probability": 1 - anomaly_probability,
        "threshold": THRESHOLD,
    }
