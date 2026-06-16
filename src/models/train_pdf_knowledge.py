import os
import json
import joblib
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer

def extract_chunks(pdf_path, chunk_size=800):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"
            
    # Clean text
    text = " ".join(text.split())
    
    # Split into chunks roughly by sentences or length
    chunks = []
    sentences = text.split(". ")
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) < chunk_size:
            current_chunk += sentence + ". "
        else:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            current_chunk = sentence + ". "
            
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
        
    return chunks

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    pdf_path = os.path.join(base_dir, "Grade 12 Physics Unit 1 2 en.pdf")
    data_dir = os.path.join(base_dir, "data")
    models_dir = os.path.join(base_dir, "models")
    
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    
    if not os.path.exists(pdf_path):
        print(f"Error: PDF not found at {pdf_path}")
        return
        
    print("Extracting text from PDF...")
    chunks = extract_chunks(pdf_path, chunk_size=800)
    print(f"Extracted {len(chunks)} chunks.")
    
    # Save chunks to JSON
    chunks_path = os.path.join(data_dir, "pdf_chunks.json")
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump({"chunks": chunks}, f, indent=4)
        
    print("Training TF-IDF Vectorizer...")
    vectorizer = TfidfVectorizer(stop_words='english')
    matrix = vectorizer.fit_transform(chunks)
    
    # Save models
    joblib.dump(vectorizer, os.path.join(models_dir, "pdf_vectorizer.joblib"))
    joblib.dump(matrix, os.path.join(models_dir, "pdf_matrix.joblib"))
    print("Models saved successfully!")

if __name__ == "__main__":
    main()
