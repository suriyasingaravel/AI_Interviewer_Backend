from tkinter.constants import S
from fastapi.applications import FastAPI


from fastapi import FastAPI, HTTPException
import os
import uuid
from typing import Dict
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from dotenv import load_dotenv

load_dotenv()

app: FastAPI = FastAPI(title="AI Interviewer API ", description="", version="1.0.0", redoc_url=None)

class CreateSessionRequest(BaseModel):
    job_role: str = Field(..., example="React Developer")
    experience: int = Field(..., ge=0, le=50, example=3)

class CreateSessionResponse(BaseModel):
    session_id: str
    job_role: str
    experience: int

class SessionState(BaseModel):
    job_role: str
    experience: int


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
    state = SessionState(
        job_role= payload.job_role,
        experience= payload.experience
    )
    sessions[sid] = state
    return CreateSessionResponse(
        session_id = sid,
        job_role = state.job_role,
        experience = state.experience
    )

    return {"message": "Session created successfully"}