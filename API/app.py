from fastapi import FastAPI
import joblib
import numpy as np

# Create FastAPI app
app = FastAPI(title="Salary Prediction API")

# Load trained model
model = joblib.load("models/salary_prediction_model.pkl")


@app.get("/")
def home():
    return {"message": "Salary Prediction API is running"}


@app.post("/predict")
def predict_salary(years_experience: float):
    # Convert input into model format
    input_data = np.array([[years_experience]])

    # Make prediction
    prediction = model.predict(input_data)

    return {
        "years_experience": years_experience,
        "predicted_salary": prediction[0]
    }