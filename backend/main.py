from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from schemas import ConcreteInput, PredictionResponse
from predictor import load_models, predict


# ── Load models at startup using lifespan ────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Runs ONCE when server starts
    app.state.scaler, app.state.lr_model, app.state.mlp_model = load_models()
    print("✓ Models loaded successfully")
    yield
    # Runs ONCE when server shuts down (cleanup if needed)
    print("Server shutting down")


# ── Create FastAPI app ───────────────────────────────────────────
app = FastAPI(
    title="Concrete Strength Predictor API",
    description="Predicts compressive strength using Linear Regression and MLP",
    version="1.0.0",
    lifespan=lifespan
)


# ── CORS Middleware ──────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Health check route ───────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "API is running"}


# ── Prediction route ─────────────────────────────────────────────
@app.post("/predict", response_model=PredictionResponse)
def predict_strength(payload: ConcreteInput, request: Request):

    # Convert Pydantic model to ordered list of 8 floats
    input_list = [
        payload.cement,
        payload.blast_furnace_slag,
        payload.fly_ash,
        payload.water,
        payload.superplasticizer,
        payload.coarse_aggregate,
        payload.fine_aggregate,
        payload.age
    ]

    # Run prediction
    results = predict(
        input_list,
        request.app.state.scaler,
        request.app.state.lr_model,
        request.app.state.mlp_model
    )

    # Determine stronger model
    stronger = "MLP" if results["mlp"] > results["lr"] else "Linear Regression"

    return PredictionResponse(
        linear_regression_mpa=results["lr"],
        mlp_mpa=results["mlp"],
        stronger_model=stronger,
        input_received=payload
    )


# ── Run directly ─────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="localhost", port=8001, reload=True)