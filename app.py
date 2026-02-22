import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="NASA Turbofan RUL API")

def load_model(ds_id: str):
    try:
        return joblib.load(f"models/xgb_{ds_id}.joblib")
    except:
        return None

# --- UPDATED SCHEMA ---
class PredictionRequest(BaseModel):
    dataset_id: str
    features: dict  # Changed from list to dict to keep feature names

@app.post("/predict")
async def predict_rul(request: PredictionRequest):
    model = load_model(request.dataset_id.lower())
    if not model:
        raise HTTPException(status_code=404, detail="Model not found.")

    try:
        # Convert the dictionary directly to a DataFrame
        # This preserves the names like 's2_mean', 's3_delta', etc.
        df = pd.DataFrame([request.features])
        
        # XGBoost prediction
        prediction = model.predict(df)[0]
        rul = max(0, round(float(prediction), 2))
        
        return {
            "dataset": request.dataset_id,
            "predicted_rul": rul,
            "warning": rul < 10,
            
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Inference Error: {str(e)}")