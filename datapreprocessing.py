import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load your dataset
df = pd.read_csv("unified_dataset.csv")  # Update with actual dataset path

# Remove duplicate entries
df = df.drop_duplicates()

# Remove missing values
df = df.dropna()

# Standardize text fields (lowercase, remove special characters, and stopwords)
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9 ]', '', text)  # Remove special characters
    text = ' '.join([word for word in text.split() if word not in stop_words])  # Remove stopwords
    return text

df['job_title'] = df['job_title'].apply(clean_text)
df['skills'] = df['skills'].apply(clean_text)
df['experience'] = df['experience'].apply(clean_text)

# Encode categorical variables if necessary
label_encoder = LabelEncoder()
df['job_type_encoded'] = label_encoder.fit_transform(df['job_type'])

# Save cleaned dataset
df.to_csv("cleaned_dataset.csv", index=False)

print("Data preprocessing completed! Cleaned dataset saved.")