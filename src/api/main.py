from fastapi import FastAPI
from pydantic import BaseModel
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parent.parent))

from core.classifier import classify_prompt

app = FastAPI(title="Prompt Classification API", version="1.0.0")

class PromptRequest(BaseModel):
    prompt: str

class ClassificationResponse(BaseModel):
    prompt: str
    classification: str
    confidence: float

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
