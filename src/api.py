from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import joblib
import os
from datetime import datetime, timedelta
import numpy as np

app = FastAPI(title="Bitcoin Price Prediction API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictionRequest(BaseModel):
    days_ahead: int = 7

class PredictionResponse(BaseModel):
    dates: list[str]
    predictions: list[float]
    confidence_intervals: list[dict]

def load_models():
    models = {}
    model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
    
    # Load SARIMA model
    try:
        models['sarima'] = joblib.load(os.path.join(model_dir, 'sarima.pkl'))
    except:
        print("Warning: SARIMA model not found")
    
    # Load Prophet model
    try:
        models['prophet'] = joblib.load(os.path.join(model_dir, 'prophet.pkl'))
    except:
        print("Warning: Prophet model not found")
    
    # Load XGBoost model
    try:
        models['xgboost'] = joblib.load(os.path.join(model_dir, 'xgboost.json'))
    except:
        print("Warning: XGBoost model not found")
    
    return models

models = load_models()

@app.get("/")
async def root():
    return {"message": "Bitcoin Price Prediction API"}

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    if not models:
        raise HTTPException(status_code=500, detail="No models loaded")
    
    # Generate future dates
    last_date = datetime.now()
    future_dates = [last_date + timedelta(days=i) for i in range(1, request.days_ahead + 1)]
    
    # Get predictions from each model
    predictions = {}
    for model_name, model in models.items():
        if model_name == 'sarima':
            pred = model.forecast(steps=request.days_ahead)
            predictions[model_name] = pred
        elif model_name == 'prophet':
            future = pd.DataFrame({'ds': future_dates})
            pred = model.predict(future)
            predictions[model_name] = pred['yhat'].values
        elif model_name == 'xgboost':
            # Implement XGBoost prediction logic here
            pass
    
    # Ensemble predictions (simple average for now)
    ensemble_pred = np.mean([pred for pred in predictions.values()], axis=0)
    
    # Calculate confidence intervals (simple implementation)
    std = np.std([pred for pred in predictions.values()], axis=0)
    lower_ci = ensemble_pred - 1.96 * std
    upper_ci = ensemble_pred + 1.96 * std
    
    return PredictionResponse(
        dates=[date.strftime("%Y-%m-%d") for date in future_dates],
        predictions=ensemble_pred.tolist(),
        confidence_intervals=[
            {"lower": float(lower), "upper": float(upper)}
            for lower, upper in zip(lower_ci, upper_ci)
        ]
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 