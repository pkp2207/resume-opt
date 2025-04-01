from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Load the trained model and the TF-IDF vectorizer
model = joblib.load("best_rf_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")

# Initialize FastAPI app
app = FastAPI()

# Define input data model using Pydantic
class PredictionRequest(BaseModel):
    resume_text: str
    job_description: str
    job_skills: str

# Prediction endpoint
@app.post("/predict/")
async def predict(request: PredictionRequest):
    # Combine the input text
    input_text = request.resume_text + " " + request.job_description + " " + request.job_skills
    
    # Transform the input text using the loaded TF-IDF vectorizer
    input_tfidf = tfidf.transform([input_text])
    
    # Get the prediction from the model
    prediction = model.predict(input_tfidf)
    
    # Return the prediction result
    return {"category": prediction[0]}
