from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import joblib
import json
import os
from sklearn.metrics.pairwise import cosine_similarity
from .physics_solver import solve_kinematics_s
import google.generativeai as genai
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-2.5-flash')
else:
    gemini_model = None

if GROQ_API_KEY:
    groq_client = Groq(api_key=GROQ_API_KEY)
else:
    groq_client = None

app = FastAPI(
    title="Sri Lankan LMS AI API",
    description="API for the Chatbot and Course Recommendation Engine.",
    version="0.5.0"
)

# Global variables to hold models and data
chatbot_pipeline = None
intent_responses = {}
recommender_vectorizer = None
recommender_matrix = None
courses_data = []
pdf_vectorizer = None
pdf_matrix = None
pdf_chunks = []

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
frontend_dir = os.path.join(base_dir, "src", "frontend")

@app.on_event("startup")
def load_models_and_data():
    global chatbot_pipeline, intent_responses
    global recommender_vectorizer, recommender_matrix, courses_data
    
    # 1. Load Chatbot Model
    chatbot_model_path = os.path.join(base_dir, "models", "chatbot_pipeline.joblib")
    if os.path.exists(chatbot_model_path):
        chatbot_pipeline = joblib.load(chatbot_model_path)
    
    faq_data_path = os.path.join(base_dir, "data", "faq_data.json")
    if os.path.exists(faq_data_path):
        with open(faq_data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            for intent in data["intents"]:
                intent_responses[intent["tag"]] = intent["responses"]
                
    # 2. Load Recommender Model
    rec_vectorizer_path = os.path.join(base_dir, "models", "recommender_vectorizer.joblib")
    rec_matrix_path = os.path.join(base_dir, "models", "recommender_matrix.joblib")
    courses_path = os.path.join(base_dir, "data", "courses.json")
    
    if os.path.exists(rec_vectorizer_path) and os.path.exists(rec_matrix_path):
        recommender_vectorizer = joblib.load(rec_vectorizer_path)
        recommender_matrix = joblib.load(rec_matrix_path)
        
    if os.path.exists(courses_path):
        with open(courses_path, "r", encoding="utf-8") as f:
            courses_data = json.load(f)
            
    # 3. Load PDF RAG Model
    global pdf_vectorizer, pdf_matrix, pdf_chunks
    pdf_vec_path = os.path.join(base_dir, "models", "pdf_vectorizer.joblib")
    pdf_mat_path = os.path.join(base_dir, "models", "pdf_matrix.joblib")
    pdf_chunks_path = os.path.join(base_dir, "data", "pdf_chunks.json")
    
    if os.path.exists(pdf_vec_path) and os.path.exists(pdf_mat_path):
        pdf_vectorizer = joblib.load(pdf_vec_path)
        pdf_matrix = joblib.load(pdf_mat_path)
        
    if os.path.exists(pdf_chunks_path):
        with open(pdf_chunks_path, "r", encoding="utf-8") as f:
            pdf_data = json.load(f)
            pdf_chunks = pdf_data.get("chunks", [])

# --- Chatbot Endpoint ---
class ChatRequest(BaseModel):
    text: str
    language: str = "en"
    model: str = "gemini"

class ChatResponse(BaseModel):
    intent: str
    response: str

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    if chatbot_pipeline is None:
        raise HTTPException(status_code=500, detail="Chatbot model not loaded.")
        
    lang_map = {"en": "English", "si": "Sinhala", "ta": "Tamil"}
    lang_name = lang_map.get(request.language, "English")
    
    # 1. Llama classification: check if query is a physics question
    is_physics = False
    if groq_client:
        try:
            classification_prompt = (
                "You are an AI assistant. Determine if the following user query is a physics question. "
                "Respond with ONLY 'yes' or 'no' in lowercase, without any punctuation, explanations, or additional text.\n\n"
                f"Query: \"{request.text}\""
            )
            response = groq_client.chat.completions.create(
                messages=[{"role": "user", "content": classification_prompt}],
                model="llama-3.3-70b-versatile"
            )
            classification_result = response.choices[0].message.content.strip().lower()
            print(f"DEBUG Llama Physics Classification: '{classification_result}' for query: '{request.text}'")
            if "yes" in classification_result:
                is_physics = True
        except Exception as e:
            print("Llama physics classification API error:", e)
            is_physics = True  # Fallback to True if Llama API fails, to avoid blocking physics questions
    else:
        is_physics = True  # Fallback if Llama is not configured
        
    if not is_physics:
        # Llama replies: "this is not physics question"
        if request.language == "si":
            reply_text = "මෙය භෞතික විද්‍යා ප්‍රශ්නයක් නොවේ"
        elif request.language == "ta":
            reply_text = "இது பௌதிகவியல் கேள்வி அல்ல"
        else:
            reply_text = "this is not physics question"
        response_text = f"<div style='background: rgba(255,255,255,0.05); padding: 10px; border-left: 4px solid #2196f3; margin-bottom: 10px; font-size: 0.9em;'><i>Powered by <b>Llama 3</b></i></div>{reply_text}"
        return ChatResponse(intent="llama_non_physics", response=response_text)
        
    # 2. Redirect to user-selected model
    # Get textbook context if available
    textbook_context = ""
    if pdf_vectorizer is not None and pdf_matrix is not None and len(pdf_chunks) > 0:
        try:
            query_vec = pdf_vectorizer.transform([request.text])
            similarities = cosine_similarity(query_vec, pdf_matrix)[0]
            best_idx = similarities.argmax()
            best_score = similarities[best_idx]
            if best_score > 0.05:
                textbook_context = pdf_chunks[best_idx]
        except Exception as e:
            print("Error retrieving textbook context:", e)
            
    if textbook_context:
        prompt = (
            f"You are an expert A/L Physics Tutor. A student asks: '{request.text}'. "
            f"Explain the answer using ONLY this textbook excerpt as your source of truth: '{textbook_context}'. \n\n"
            f"CRITICAL: First, evaluate if the question is related to Physics. If it is NOT related, you MUST reply ONLY with 'This is not a physics question' (translated into {lang_name} if needed) and nothing else. "
            f"You MUST write your ENTIRE response in {lang_name}. Do not use English unless the selected language is English. If the textbook excerpt does NOT contain the answer, you MUST say 'The textbook does not contain information about this topic' and nothing else. Do NOT use outside knowledge.\n\n"
            "Format nicely with HTML tags like <b> and <br>, do not use markdown."
        )
    else:
        prompt = (
            f"You are an expert A/L Physics Tutor. A student asks: '{request.text}'. Provide a step-by-step mathematical solution or explanation as appropriate. \n\n"
            f"CRITICAL: First, evaluate if the question is related to Physics. If it is NOT related, you MUST reply ONLY with 'This is not a physics question' (translated into {lang_name} if needed) and nothing else. "
            f"You MUST answer the question ENTIRELY in {lang_name}. Do not use English unless the selected language is English.\n\n"
            "Format nicely with HTML tags like <b> and <br>, do not use markdown. Start directly with the solution/explanation."
        )
        
    if request.model == "llama" and groq_client:
        try:
            sys_prompt = f"You are a native {lang_name} speaking Physics tutor. You NEVER output English unless {lang_name} is English. All explanations MUST be translated to and written entirely in {lang_name}."
            response = groq_client.chat.completions.create(
                messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": prompt}],
                model="llama-3.3-70b-versatile"
            )
            ans_text = response.choices[0].message.content.strip()
            response_text = f"<div style='background: rgba(255,255,255,0.05); padding: 10px; border-left: 4px solid #2196f3; margin-bottom: 10px; font-size: 0.9em;'><i>Powered by <b>Llama 3</b></i></div>{ans_text}"
            if textbook_context:
                response_text += f"<br><br><div style='background: rgba(255,255,255,0.05); padding: 10px; border-left: 4px solid #2196f3; margin-top: 15px; font-size: 0.9em;'><b>Textbook Reference:</b><br><i>{textbook_context}</i></div>"
            predicted_intent = "llama_physics_solver"
        except Exception as e:
            response_text = f"Llama API Error: {str(e)}"
            predicted_intent = "llama_error"
    elif gemini_model:
        try:
            response = gemini_model.generate_content(prompt)
            ans_text = response.text.strip()
            response_text = f"<div style='background: rgba(255,255,255,0.05); padding: 10px; border-left: 4px solid #2196f3; margin-bottom: 10px; font-size: 0.9em;'><i>Powered by <b>Gemini AI</b></i></div>{ans_text}"
            if textbook_context:
                response_text += f"<br><br><div style='background: rgba(255,255,255,0.05); padding: 10px; border-left: 4px solid #2196f3; margin-top: 15px; font-size: 0.9em;'><b>Textbook Reference:</b><br><i>{textbook_context}</i></div>"
            predicted_intent = "gemini_physics_solver"
        except Exception as e:
            response_text = f"Gemini API Error: {str(e)}"
            predicted_intent = "gemini_error"
    else:
        response_text = "<b>Error:</b> Neither Gemini nor Llama model could be initialized."
        predicted_intent = "api_error"
        
    return ChatResponse(intent=predicted_intent, response=response_text)

# --- Recommender Endpoint ---
class RecommendRequest(BaseModel):
    interests: str

@app.post("/recommend")
def recommend(request: RecommendRequest):
    if recommender_vectorizer is None or recommender_matrix is None:
        raise HTTPException(status_code=500, detail="Recommender model not loaded.")
        
    user_vector = recommender_vectorizer.transform([request.interests])
    similarities = cosine_similarity(user_vector, recommender_matrix)[0]
    top_indices = similarities.argsort()[-3:][::-1]
    
    recommended_courses = []
    for idx in top_indices:
        if similarities[idx] > 0:
            course = courses_data[idx]
            course['similarity_score'] = round(float(similarities[idx]), 2)
            recommended_courses.append(course)
            
    return {"recommendations": recommended_courses}

# --- Slide Generator Endpoint ---
class SlideRequest(BaseModel):
    topic: str
    language: str = "en"

@app.post("/generate-slides")
def generate_slides(request: SlideRequest):
    topic_lower = request.topic.lower()
    faq_data_path = os.path.join(base_dir, "data", "faq_data.json")
    slides = []
    
    if os.path.exists(faq_data_path):
        with open(faq_data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            for intent in data["intents"]:
                # Filter out math solver intents, we only want facts/definitions
                if "solve" in intent["tag"]:
                    continue
                    
                matches_topic = False
                if topic_lower in intent["tag"].lower():
                    matches_topic = True
                else:
                    for pattern in intent["patterns"]:
                        if topic_lower in pattern.lower():
                            matches_topic = True
                            break
                            
                if matches_topic and request.language in intent["responses"]:
                    # Create a title from the tag (e.g., al_physics_kinematics -> Physics Kinematics)
                    title_words = [word.capitalize() for word in intent["tag"].split("_") if word.lower() not in ["al", "greeting"]]
                    title = " ".join(title_words) if title_words else request.topic.title()
                    content = intent["responses"][request.language]
                    slides.append({"title": title, "content": content})
                        
    # Setup Title Slide
    if len(slides) > 0:
        slides.insert(0, {
            "title": f"{request.topic.title()} Presentation", 
            "content": "Use the arrows below to navigate through the AI generated slides."
        })
    else:
        slides.append({
            "title": "No Content Found", 
            "content": f"I couldn't find any information on '{request.topic}' in my knowledge base."
        })
        
    return {"slides": slides}

# Mount static frontend
app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
