from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

# Load and preprocess the data
resume_df = pd.read_csv("UpdatedResumeDataSet.csv")
job_df = pd.read_csv("job_descriptions.csv")

# Combine relevant text fields
resume_df['Combined_Text'] = resume_df['Category'] + " " + resume_df['Resume']
job_df['Combined_Text'] = job_df['Job Title'] + " " + job_df['Job Description'] + " " + job_df['skills']

# Create and fit the TF-IDF vectorizer
vectorizer = TfidfVectorizer(stop_words='english')
resume_tfidf = vectorizer.fit_transform(resume_df['Combined_Text'])
job_tfidf = vectorizer.transform(job_df['Combined_Text'])

class PredictionRequest(BaseModel):
    resume_text: str
    job_description: str
    job_skills: str

@app.post("/predict/")
async def predict(request: PredictionRequest):
    # Combine the input text
    input_text = request.resume_text + " " + request.job_description + " " + request.job_skills
    
    # Transform the input text using the TF-IDF vectorizer
    input_tfidf = vectorizer.transform([input_text])
    
    # Compute cosine similarity
    similarity_scores = cosine_similarity(input_tfidf, resume_tfidf)
    
    # Find the best match
    best_match_index = similarity_scores.argmax()
    
    # Get the predicted category
    predicted_category = resume_df.iloc[best_match_index]['Category']
    
    # Calculate similarity score
    similarity_score = similarity_scores[0][best_match_index]
    
    # Generate suggestions (missing skills)
    input_tokens = set(input_text.lower().split())
    resume_tokens = set(resume_df.iloc[best_match_index]['Combined_Text'].lower().split())
    missing_skills = list(input_tokens - resume_tokens)
    
    return {
        "predicted_category": predicted_category,
        "similarity_score": float(similarity_score),
        "suggestions": missing_skills[:5]  # Return top 5 suggestions
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
