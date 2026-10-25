# Concrete Strength Predictor

Predict concrete compressive strength from a concrete mix using **Linear Regression** and a **PyTorch multilayer perceptron (MLP)**. A Streamlit interface sends the same inputs to a FastAPI backend and displays both predictions in MPa.

## Dataset and inputs

The project uses the [UCI Concrete Compressive Strength dataset](https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength), attributed to I-Cheng Yeh. The committed training notebook contains **1,030 observations**, eight input features, and one target.

| Input | Unit |
| --- | --- |
| Cement | kg/m³ |
| Blast furnace slag | kg/m³ |
| Fly ash | kg/m³ |
| Water | kg/m³ |
| Superplasticizer | kg/m³ |
| Coarse aggregate | kg/m³ |
| Fine aggregate | kg/m³ |
| Concrete age | days |

The target is concrete compressive strength in **MPa**. The local data file is `data/Concrete_Data.csv`.

## Method and recorded results

The [training notebook](notebooks/concrete_strength.ipynb) uses an 80/20 train/test split with `random_state=42` and fits `StandardScaler` on the training features only.

- **Baseline:** scikit-learn Linear Regression.
- **Neural network:** 8 inputs → 16 hidden units → ReLU → 1 output.
- **MLP training:** Adam, learning rate 0.01, MSE loss, 1,000 epochs.

| Model | Recorded test MSE (MPa²) |
| --- | ---: |
| Linear Regression | 95.97 |
| PyTorch MLP | 46.4113 |

These are the outputs saved in the committed notebook, rather than a newly repeated experiment. The split seed is fixed; the notebook does not explicitly fix the PyTorch initialization seed.

**Evaluation note:** the notebook's Linear Regression R² cell passes predictions before ground truth. R² should be recalculated with `r2_score(y_test, lr_pred)` before reporting it. The MSE metric is symmetric in its arguments.

## Run locally

Clone the repository and create an environment:

```bash
git clone https://github.com/mtaimoor1407/Concrete-Strength-Predictor.git
cd Concrete-Strength-Predictor
python -m venv .venv
```

Activate it with `.venv\Scripts\activate` on Windows or `source .venv/bin/activate` on macOS/Linux, then install both sets of dependencies:

```bash
pip install -r backend/requirements.txt -r frontend/requirements.txt
```

The backend loads these files from `models/`:

- `scaler.pkl`
- `linear_regression.pkl`
- `mlp_model.pth`

For local use, change `API_URL` in `frontend/app.py` to:

```python
API_URL = "http://localhost:8000/predict"
```

The committed value, `http://backend:8000/predict`, assumes a network host named `backend`.

Start the backend:

```bash
cd backend
uvicorn main:app --host 127.0.0.1 --port 8000
```

In a second terminal, activate the same environment and run from the repository root:

```bash
streamlit run frontend/app.py
```

Open the Streamlit URL printed in the terminal. API documentation is available at [localhost:8000/docs](http://localhost:8000/docs).

## API example

Send a `POST /predict` request with the eight feature names:

```json
{
  "cement": 350,
  "blast_furnace_slag": 0,
  "fly_ash": 0,
  "water": 180,
  "superplasticizer": 5,
  "coarse_aggregate": 1000,
  "fine_aggregate": 750,
  "age": 28
}
```

The response includes `linear_regression_mpa`, `mlp_mpa`, and the submitted inputs. The existing `stronger_model` field identifies the model that predicts the **higher strength for that input**; it does not identify the more accurate model.

## Repository guide

| Path | Contents |
| --- | --- |
| `backend/` | FastAPI routes, input validation, model loading, inference |
| `frontend/` | Streamlit interface |
| `notebooks/concrete_strength.ipynb` | Data exploration, training, evaluation, model export |
| `data/` | Training data |
| `models/` | Saved scaler and trained model artifacts |

## Limitations

This is a learning and portfolio application. Its predictions are estimates from the training dataset; out-of-distribution mixes may be unreliable. A larger predicted value is not evidence of better model accuracy. Model outputs do not replace physical testing or engineering validation.
