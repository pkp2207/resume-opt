import requests

# URL of the deployed FastAPI
url = "https://resume-opt-two.vercel.app/predict/"

# Data to be sent in the POST request
data = {
    "resume_text": "Skills * Programming Languages: Python, SQL, Java, etc.",
    "job_description": "Looking for a Data Scientist with Python, SQL, and Machine Learning experience.",
    "job_skills": "Python, SQL, Machine Learning, Data Science"
}

# Send POST request
response = requests.post(url, json=data)

# Check the status code
print("Status Code:", response.status_code)

# If status code is 200, try printing the JSON response
if response.status_code == 200:
    print(response.json())
else:
    print("Error: Unable to get a valid response from the API.")

print("Response Text:", response.text)
