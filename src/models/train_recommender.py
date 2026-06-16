import json
import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

def train_recommender():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_path = os.path.join(base_dir, "data", "courses.json")
    model_dir = os.path.join(base_dir, "models")
    
    with open(data_path, "r", encoding="utf-8") as f:
        courses = json.load(f)
        
    documents = []
    
    # Combine title, description, and tags into a single "document" for each course
    for course in courses:
        combined_text = f"{course['title']} {course['description']} {' '.join(course['tags'])}"
        documents.append(combined_text)
        
    print(f"Loaded {len(documents)} courses.")
    
    # Create and fit the TF-IDF vectorizer
    # stop_words='english' removes common words like "the", "and", "a"
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(documents)
    
    print("TF-IDF Matrix created.")
    
    os.makedirs(model_dir, exist_ok=True)
    vectorizer_path = os.path.join(model_dir, "recommender_vectorizer.joblib")
    matrix_path = os.path.join(model_dir, "recommender_matrix.joblib")
    
    joblib.dump(vectorizer, vectorizer_path)
    joblib.dump(tfidf_matrix, matrix_path)
    
    print(f"Models saved to {model_dir}")

if __name__ == "__main__":
    train_recommender()
