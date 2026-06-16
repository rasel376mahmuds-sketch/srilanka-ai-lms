import json
import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

def train_intent_model():
    # File paths
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_path = os.path.join(base_dir, "data", "faq_data.json")
    model_dir = os.path.join(base_dir, "models")
    
    # Load dataset
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    documents = []
    labels = []
    
    # Flatten the JSON structure into a dataset of text -> intent
    for intent in data["intents"]:
        tag = intent["tag"]
        for pattern in intent["patterns"]:
            documents.append(pattern)
            labels.append(tag)
            
    print(f"Loaded {len(documents)} training patterns across {len(set(labels))} intents.")
    
    # Create an NLP pipeline: TF-IDF for text vectorization -> Logistic Regression for classification
    pipeline = Pipeline([
        ('vectorizer', TfidfVectorizer(ngram_range=(1, 2))),  # Use unigrams and bigrams
        ('clf', LogisticRegression(random_state=42, max_iter=200))
    ])
    
    print("Training the model...")
    pipeline.fit(documents, labels)
    print("Model training complete!")
    
    # Save the pipeline
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "chatbot_pipeline.joblib")
    joblib.dump(pipeline, model_path)
    print(f"Model successfully saved to {model_path}")

if __name__ == "__main__":
    train_intent_model()
