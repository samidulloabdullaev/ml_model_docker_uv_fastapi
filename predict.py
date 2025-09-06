import pickle
import fastapi
import uvicorn
from typing import Dict, Any


app = fastapi.FastAPI(title="Churn Prediction")

# read the model
with open('model.bin', 'rb') as f_in:
    model = pickle.load(f_in)

def predict_single(customer:Dict[str, Any]) -> float:
    result = model.predict_proba(customer)[0, 1]
    return float(result)

@app.post("/predict")
def predict(customer:Dict[str, Any]) -> dict:
    preds = model.predict_proba(customer)[0, 1]
    return {
        'churn_probability': float(preds),
        'churn': 'Churn' if preds >= 0.5 else 'No Churn'
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)