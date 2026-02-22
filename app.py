import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Turbofan Sequence Predictor")

class SequenceRequest(BaseModel):
    dataset_id: str
    sequence: list[dict] # Accepting a list of row-dictionaries

@app.post("/predict")
async def predict_rul(request: SequenceRequest):
    model = joblib.load(f"models/xgb_{request.dataset_id.lower()}.joblib")
    
    try:
        df_sequence = pd.DataFrame(request.sequence)
        
        # --- THE FIX: Sanitization ---
        # Explicitly drop the columns the model wasn't trained on
        cols_to_drop = ['unit_id', 'time', 'regime_id', 'RUL', 'RUL_clipped']
        latest_state = df_sequence.iloc[[-1]].drop(columns=cols_to_drop, errors='ignore')
        
        prediction = model.predict(latest_state)[0]
        rul = max(0, round(float(prediction), 2))
        
        return {
            "dataset": request.dataset_id,
            "unit_id": int(df_sequence.iloc[-1].get('unit_id', 0)),
            "current_cycle": int(df_sequence.iloc[-1].get('time', 0)),
            "predicted_rul": rul
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))