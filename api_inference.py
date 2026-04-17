import os
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

app = FastAPI(title="AeroMRO Inference API", version="1.0.0")

class TelemetryPayload(BaseModel):
    engine_id: int
    current_cycle: int
    sensors: List[float]      # 20 raw sensors
    environment: List[float]  # 16 environmental variables

class PredictionResponse(BaseModel):
    engine_id: int
    predicted_rul: float
    confidence_90_lower: float
    confidence_90_upper: float
    status: str
    recommendation: str

@app.post("/predict/rul", response_model=PredictionResponse)
async def predict_rul(payload: TelemetryPayload):
    if len(payload.sensors) != 20 or len(payload.environment) != 16:
        raise HTTPException(status_code=400, detail="Invalid telemetry vector dimensions.")
    
    # Mocking the PyTorch inference for the API layer demonstration
    # In production: model.predict(torch.tensor(payload.sensors))
    base_rul = max(0.0, 150.0 - (payload.current_cycle * 0.8))
    
    # Dynamic uncertainty based on cycle life
    uncertainty = 15.0 if base_rul > 50 else 5.0
    
    if base_rul < 20:
        status, rec = "CRITICAL", "Immediate Shop Visit Required"
    elif base_rul < 50:
        status, rec = "WARNING", "Schedule Borescope Inspection"
    else:
        status, rec = "SAFE", "Continue Normal Operations"
        
    return PredictionResponse(
        engine_id=payload.engine_id,
        predicted_rul=round(base_rul, 1),
        confidence_90_lower=round(max(0, base_rul - uncertainty), 1),
        confidence_90_upper=round(base_rul + uncertainty, 1),
        status=status,
        recommendation=rec
    )

if __name__ == "__main__":
    print("\n[Operations] Starting AeroMRO FastAPI Server on port 8000...")
    print("Test with: curl -X POST http://127.0.0.1:8000/predict/rul -H 'Content-Type: application/json' -d '{\"engine_id\":19858,\"current_cycle\":47,\"sensors\":[...],\"environment\":[...]}'")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")