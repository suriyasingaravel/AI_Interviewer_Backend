import os
import uuid
from typing import Dict, List, Optional
from fastapi.applications import FastAPI
from fastapi import FastAPI, HTTPException
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

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is missing in environment variables.")

parser = StrOutputParser()


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

# Evaluate each answer
EVALUATE_ANSWER_PROMPT = ChatPromptTemplate.from_template("""
You are an expert technical interviewer evaluating a candidate's answer.

Job Role: {job_role}
Experience Level: {experience} years
Question: {question}
Candidate's Answer: {answer}

Provide a detailed evaluation including:
1. Technical accuracy (1-10)
2. Completeness of answer (1-10)
3. Clarity of explanation (1-10)
4. Specific feedback and suggestions
5. Overall score (1-10)

Format your response as:
Score: X/10
Technical Accuracy: X/10
Completeness: X/10
Clarity: X/10
Feedback: [Your detailed feedback here]
""")

FINAL_REPORT_PROMPT = ChatPromptTemplate.from_template("""
You are an expert technical interviewer creating a comprehensive interview report.

Job Role: {job_role}
Experience Level: {experience} years

Interview Results:
{interview_results}

Create a professional final report including:
1. Overall assessment
2. Strengths identified
3. Areas for improvement
4. Technical competency score (average of all scores)
5. Recommendation (Pass/Fail/Consider with conditions)
6. Detailed breakdown of each question

Format as a professional report.
""")


# LLM 

llm = ChatGoogleGenerativeAI(
    model = "gemini-2.0-flash",
    api_key = GEMINI_API_KEY,
    temperature=0.7
)
# response = llm.invoke("Who is Mahatma Gandhi")
# print(response.content)


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

def evaluate_answer(job_role:str, experience:int, question:str, answer:str):
    chain = EVALUATE_ANSWER_PROMPT | llm | parser
    feedback = chain.invoke({"job_role":job_role, "experience":experience, "question":question, "answer":question})
    return feedback

def build_final_report(job_role: str, experience:int,data:List[dict] ) :
    
    interview_results = []
    for i, row in enumerate(data, start=1):
        interview_results.append(
            f"Question {i}: {row.get('question', "")}\n"
            f"Answer: {row.get('answer', "")}\n"
            f"Feedback : {row.get('feedback', "")}\n"
            + ("-"*40)
        )
    results_blob = "\n".join(interview_results)    
    chain = FINAL_REPORT_PROMPT | llm | parser
    final_report = chain.invoke({"job_role": job_role, "experience": experience, "interview_results": results_blob})
    return final_report


#Models
class CreateSessionRequest(BaseModel):
    job_role: str = Field(..., example="React Developer")
    experience: int = Field(..., ge=0, le=50, example=2)

class SubmitAnswerRequest(BaseModel):
    answer: str 


class CreateSessionResponse(BaseModel):
    session_id: str
    job_role: str
    experience: int
    questions:List[str]
    current_question_idx : int
class SubmitAnswerResponse(BaseModel):
    question_idx: int
    question: str
    feedback: str
    next_question_idx: Optional[int] = None
    next_question: Optional[str] = None


class SessionState(BaseModel):
    job_role: str
    experience: int
    data : List[Dict]
    current_question_idx : int
    final_report: Optional[str] = None



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
        current_question_idx = 0,
        
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


@app.post("/sessions/{session_id}/answers", response_model=SubmitAnswerResponse)
def submit_answer(session_id:str, payload:SubmitAnswerRequest):
    state = sessions.get(session_id)
    idx = state.current_question_idx
    if idx >= 5:
      raise HTTPException(status_code=400, detail="All questions already answered")
    question = state.data[idx]['question']
    # Save the answer
    state.data[idx]['answer']= payload.answer.strip()
    # Evaluate
    feedback = evaluate_answer(state.job_role, state.experience, question, state.data[idx]['answer'])
    state.data[idx]['feedback'] = feedback

    # Move index
    state.current_question_idx +=1

    # Set next question info
    next_q_idx = None
    next_q = None
    if state.current_question_idx < 5 :
        next_q_idx = state.current_question_idx
        next_q = state.data[next_q_idx]['question']
    else:
        state.final_report =  build_final_report(state.job_role, state.experience, state.data)   

    sessions[session_id] = state   
    
    return SubmitAnswerResponse(
        question_idx=idx,
        question=question,
        feedback=feedback,
        next_question_idx=next_q_idx,
        next_question=next_q,
    ) 



@app.get("/sessions/{session_id}/report")
def get_report(session_id:str):
    state = sessions.get(session_id)
    return {
        "job_role": state.job_role,
        "experience":   state.experience,
        "final_report": state.final_report
    }
