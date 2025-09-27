from fastapi.applications import FastAPI
from fastapi import FastAPI, HTTPException
import os
import uuid
from typing import Dict, List, Optional
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from dotenv import load_dotenv
# LangChain / GeminiAPI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import json
import re


load_dotenv()

app: FastAPI = FastAPI(title="AI Interviewer API ", description="", version="1.0.0", redoc_url=None)

#Prompts
# Generate questions based on role and experience
GENERATE_QUESTIONS_PROMPT = ChatPromptTemplate.from_template(
    """
You are an expert technical interviewer. Based on the job role and experience level, generate exactly 5 relevant technical questions.

Job Role: {job_role}
Experience Level: {experience} years

Generate 5 questions that are:
1. Appropriate for the experience level
2. Technical and role-specific
3. Progressive in difficulty
4. Cover different aspects of the role

Return ONLY a JSON array of 5 questions, no explanations:
[
    "Question 1",
    "Question 2",
    "Question 3",
    "Question 4",
    "Question 5"
]
"""
)


# LLM 
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is missing in environment variables.")


llm = ChatGoogleGenerativeAI(
    model = "gemini-2.0-flash",
    api_key = GEMINI_API_KEY
)
# response = llm.invoke("Who is Mahatma Gandhi")
# print(response.content)

parser = StrOutputParser()

def extract_json(raw):
    # Try vanilla JSON parsing first
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass  # Try regex extraction if vanilla fails

    # Look for the first [ ... ] block in the text (for robustness)
    match = re.search(r'\[\s*(".*?"\s*,?\s*){5}\]', raw, re.DOTALL)
    if match:
        json_text = match.group(0)
        try:
            return json.loads(json_text)
        except Exception as e:
            pass
    raise HTTPException(status_code=502, detail=f"LLM returned invalid JSON: {repr(raw)}")


def generate_questions(job_role: str, experience: int):
    chain = GENERATE_QUESTIONS_PROMPT | llm | parser
    raw = chain.invoke({"job_role": job_role, "experience": experience})
    print("LLM response:", repr(raw))  # Always log for debugging
    if not raw or not raw.strip():
        raise HTTPException(status_code=502, detail="LLM did not return any data. Check API key and network.")
    questions = extract_json(raw)
    questions = [str(q).strip() for q in questions]
    if len(questions) != 5:
        raise HTTPException(status_code=502, detail="LLM did not generate exactly 5 questions.")
    return questions


#Models
class CreateSessionRequest(BaseModel):
    job_role: str = Field(..., example="React Developer")
    experience: int = Field(..., ge=0, le=50, example=2)

class CreateSessionResponse(BaseModel):
    session_id: str
    job_role: str
    experience: int
    questions:List[str]
    current_question_idx : int

class SessionState(BaseModel):
    job_role: str
    experience: int
    data : List[Dict]
    current_question_idx : int


sessions: Dict[str, SessionState] = {}

@app.get("/")
def root():
    return {"message": "Welcome to AI Interviewer"}

@app.get("/health")
def health():
    return {"status": "ok "}

@app.post("/sessions", response_model= CreateSessionResponse, status_code=201)
def create_session(payload: CreateSessionRequest):  
    """Create a new interview session and return the session ID only."""
    sid = uuid.uuid4().hex
    questions = generate_questions(payload.job_role, payload.experience)

    state = SessionState(
        job_role= payload.job_role,
        experience= payload.experience,
        data = [{"question": q, "answer": "", "feedback": ""} for q in questions],
        current_question_idx = 0
    )
    sessions[sid] = state
    return CreateSessionResponse(
        session_id = sid,
        job_role = state.job_role,
        experience = state.experience,
        questions= [q['question'] for q in state.data],
        current_question_idx= state.current_question_idx
    )

@app.get("/sessions/{session_id}")
def get_session(session_id:str):
   state = sessions.get(session_id)
   return state

