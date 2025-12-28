from fastapi import FastAPI
from pydantic import BaseModel

from pathlib import Path
import sys

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.safety_rag import policy_check
from core.classifier import classify_prompt
from utils.api_logger import APILogger

logger = APILogger()

app = FastAPI(title="Clean Talk - Guardrail API", version="1.0.0")

class PromptRequest(BaseModel):
    prompt: str

class ClassificationResponse(BaseModel):
    prompt: str
    classification: str
    confidence: float

class PolicyCheckResponse(BaseModel):
    decision: str
    policy: str
    response_to_user: str

@app.get("/")
def read_root():
    """Health check endpoint"""
    return {"message": "Prompt Classification API is running"}

@app.post("/classify", response_model=ClassificationResponse)
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
        
        return ClassificationResponse(
                prompt=request.prompt,
                classification=classification,
                confidence=confidence
            )
    except Exception as e:
        return {"error": str(e)}, 500

@app.post("/policy_check", response_model=PolicyCheckResponse)
def classify(request: PromptRequest):
    """
    
    Args:
        request: PromptRequest containing the prompt text
        
    Returns:
        ClassificationResponse with the classification label
    """
    try:
        # Get classification from model
        response = policy_check(request.prompt)
        
        return PolicyCheckResponse(
            decision=response['decision'],
            policy=response['policy'],
            response_to_user=response['response_to_user']
        )
    except Exception as e:
        return {"error": str(e)}, 500

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
