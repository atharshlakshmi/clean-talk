# BACKEND FOR STREAMLIT

from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from pathlib import Path
from typing import Optional
import uuid
import sys
import os
from dotenv import load_dotenv
# from slowapi import Limiter, _rate_limit_exceeded_handler
# from slowapi.util import get_remote_address
# from slowapi.errors import RateLimitExceeded

load_dotenv()

# Add src directory to Python path BEFORE imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.safety_rag import policy_check
from core.classifier import classify_prompt
from utils.api_logger import APILogger

# Pinecone
api_key = os.environ.get("PINECONE_API_KEY")
if not api_key:
    raise ValueError("PINECONE_API_KEY environment variable not set")

pc = Pinecone(api_key=api_key)
index_name = "safety-policies"
index = pc.Index(index_name)
model = SentenceTransformer('all-MiniLM-L6-v2')

# FastAPI
logger = APILogger()
app = FastAPI(title="Clean Talk - Guardrail API", version="1.0.0")

# limiter = Limiter(key_func=get_remote_address)
# app.state.limiter = limiter
# app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

class PromptRequest(BaseModel):
    prompt: str

class ClassificationResponse(BaseModel):
    prompt: str
    classification: str
    confidence: float

class PolicyCheckResponse(BaseModel):
    decision: str
    policy: Optional[str] = None
    response_to_user: Optional[str] = None

class Policy(BaseModel):
    policy: str

@app.get("/")
# @limiter.limit("100/minute")
def read_root():
    """Health check endpoint"""
    return {"message": "Prompt Classification API is running"}

@app.post("/classify", response_model=ClassificationResponse)
# @limiter.limit("30/minute")
def classify(request: PromptRequest):
    """
    Args:
        request: PromptRequest containing the prompt text
        
    Returns:
        ClassificationResponse with the classification label
    """
    
    try:
        # Get classification from model
        classification, confidence = classify_prompt(request.prompt)
        
        
        response = ClassificationResponse(
                prompt=request.prompt,
                classification=classification,
                confidence=confidence
            )
        logger.log('classify', request.prompt, response, 'success')
        return response
    
    except Exception as e:
        logger.log('classify', request.prompt, None, 'fail')
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/policy_check", response_model=PolicyCheckResponse)
# @limiter.limit("30/minute")
def check_policy(request: PromptRequest):
    """
    Args:
        request: PromptRequest containing the prompt text
        
    Returns:
        PolicyCheckResponse with the policy check decision
    """
    
    try:
        response = policy_check(request.prompt)
        
        policy_response = PolicyCheckResponse(
            decision=response['decision'],
            policy=response['policy'],
            response_to_user=response['response_to_user']
        )
        
        logger.log('policy_check', request.prompt, response, 'success')
        return policy_response
    
    except Exception as e:
        logger.log('policy_check', request.prompt, None, 'fail')
        raise HTTPException(status_code=500, detail=str(e))

@app.post('/add_policy')
# @limiter.limit("10/minute")
async def add_policy(policy: Policy):
    try: 
        policy_id = str(uuid.uuid4())[:8]
        vector = model.encode(policy.policy).tolist()
        index.upsert(vectors=[(policy_id, vector, {"description": policy.policy})])
        
        return {"id": policy_id, "text": policy.policy, "status": "uploaded"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
