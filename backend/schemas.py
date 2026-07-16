from pydantic import BaseModel, Field


class ConcreteInput(BaseModel):
    cement:            float = Field(..., gt=0, description="Cement (kg/m³)")
    blast_furnace_slag: float = Field(..., ge=0, description="Blast Furnace Slag (kg/m³)")
    fly_ash:           float = Field(..., ge=0, description="Fly Ash (kg/m³)")
    water:             float = Field(..., gt=0, description="Water (kg/m³)")
    superplasticizer:  float = Field(..., ge=0, description="Superplasticizer (kg/m³)")
    coarse_aggregate:  float = Field(..., gt=0, description="Coarse Aggregate (kg/m³)")
    fine_aggregate:    float = Field(..., gt=0, description="Fine Aggregate (kg/m³)")
    age:               float = Field(..., gt=0, description="Age (days)")


class PredictionResponse(BaseModel):
    linear_regression_mpa: float
    mlp_mpa:               float
    stronger_model:        str
    input_received:        ConcreteInput