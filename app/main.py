from fastapi import FastAPI
import pandas as pd
import joblib


app = FastAPI(title="Mental Health Score API")


model = joblib.load("mental_health_model.pkl")
preprocessor = joblib.load("preprocessor.pkl")


@app.get("/")
def home():
    return {"message": "Mental Health Score API is running"}


@app.post("/predict")
def predict(data: dict):

    sample = pd.DataFrame([data])

    sample_processed = preprocessor.transform(sample)

    prediction = model.predict(sample_processed)

    return {
        "mental_health_score": float(prediction[0])
    }
