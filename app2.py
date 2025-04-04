import streamlit as st
import joblib
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import re
import unicodedata

# Load pre-trained Sentence-BERT model for text embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')  # Sentence-BERT model

# Function to clean text
def clean_text(text):
    """Clean and preprocess the input text."""
    text = unicodedata.normalize("NFKD", text)
    text = re.sub(r'[^a-zA-Z0-9., ]', '', text)
    text = text.lower().strip()
    return text

# Function to compute similarity and give feedback
def analyze_resume_match(resume_text, job_desc):
    # Clean and preprocess the text
    resume_text_cleaned = clean_text(resume_text)
    job_desc_cleaned = clean_text(job_desc)
    
    # Convert text to embeddings using Sentence-BERT model
    resume_embedding = model.encode([resume_text_cleaned])
    job_embedding = model.encode([job_desc_cleaned])
    
    # Calculate similarity score using cosine similarity
    similarity_score = cosine_similarity(resume_embedding, job_embedding)[0][0]
    
    # Estimate selection probability based on similarity score
    if similarity_score > 0.85:
        selection_prob = "Very High"
    elif similarity_score > 0.70:
        selection_prob = "High"
    elif similarity_score > 0.50:
        selection_prob = "Moderate"
    else:
        selection_prob = "Low"
    
    return similarity_score, selection_prob

# TF-IDF Vectorization for extracting keywords
def extract_keywords(resume_text, job_desc):
    # Initialize TF-IDF vectorizer
    vectorizer = TfidfVectorizer(stop_words='english')
    
    # Combine both the resume and job description for comparison
    combined_text = [resume_text, job_desc]
    
    # Apply TF-IDF transformation
    tfidf_matrix = vectorizer.fit_transform(combined_text)
    
    # Extract feature names (keywords)
    feature_names = vectorizer.get_feature_names_out()
    
    # Convert the TF-IDF matrix to dense array and extract the feature words
    resume_keywords = set([feature_names[i] for i in tfidf_matrix[0].nonzero()[1]])
    job_keywords = set([feature_names[i] for i in tfidf_matrix[1].nonzero()[1]])
    
    # Find missing keywords from the resume
    missing_keywords = job_keywords - resume_keywords
    
    return missing_keywords

# Streamlit interface
def main():
    st.title("AI Resume Match Analyzer")
    st.write("Enter your resume and a job description to analyze the match and get improvement suggestions.")
    
    resume_input = st.text_area("Paste your Resume:", height=200)
    job_input = st.text_area("Paste Job Description:", height=200)
    
    if st.button("Analyze Match"):
        if resume_input and job_input:
            # Step 1: Analyze match between resume and job description
            similarity_score, selection_prob = analyze_resume_match(resume_input, job_input)
            
            # Step 2: Extract missing keywords (areas to improve)
            missing_keywords = extract_keywords(resume_input, job_input)
            
            # Display results
            st.subheader("Results:")
            st.write(f"**Similarity Score:** {similarity_score:.2f}")
            st.write(f"**Selection Probability:** {selection_prob}")
            
            st.subheader("How to Improve Your Resume:")
            if missing_keywords:
                st.write(f"- Add the following relevant skills and keywords from the job description: {', '.join(missing_keywords)}.")
            else:
                st.write("- Your resume already covers the key skills and keywords for this job description!")
                
            st.write("- Ensure your resume is well-structured and concise.")
        else:
            st.write("Please provide both a resume and a job description.")

if __name__ == "__main__":
    main()