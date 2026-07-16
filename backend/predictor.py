import torch
import torch.nn as nn
import numpy as np
import joblib
from pathlib import Path

# ── Path setup ───────────────────────────────────────────────────
MODELS_DIR = Path(__file__).parent.parent / "models"


# ── MLP Architecture (must match exactly what was trained) ───────
class ConcreteMLP(nn.Module):
    def __init__(self, input_size=8, hidden_size=16, output_size=1):
        super(ConcreteMLP, self).__init__()
        self.fc1  = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2  = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


# ── Load all models once at startup ──────────────────────────────
def load_models():
    # Load scaler
    scaler = joblib.load(MODELS_DIR / "scaler.pkl")

    # Load Linear Regression
    lr_model = joblib.load(MODELS_DIR / "linear_regression.pkl")

    # Load MLP
    mlp_model = ConcreteMLP()
    mlp_model.load_state_dict(
        torch.load(MODELS_DIR / "mlp_model.pth", map_location="cpu")
    )
    mlp_model.eval()

    return scaler, lr_model, mlp_model


# ── Prediction function ───────────────────────────────────────────
def predict(input_data: list, scaler, lr_model, mlp_model) -> dict:
    # Convert to numpy array, shape (1, 8)
    X = np.array(input_data).reshape(1, -1)

    # Scale using the saved scaler
    X_scaled = scaler.transform(X)

    # Linear Regression prediction
    lr_pred = lr_model.predict(X_scaled)[0]

    # MLP prediction
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    with torch.no_grad():
        mlp_pred = mlp_model(X_tensor).item()

    return {
        "lr":  round(float(lr_pred[0]), 4),
        "mlp": round(float(mlp_pred), 4)
    }