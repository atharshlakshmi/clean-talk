# Clean Talk: Guardrail API

A safety guardrail system that classifies prompts and evaluates policy compliance. Takes a prompt and determines if it's safe or attempting a jailbreak, then retrieves relevant policies and judges whether the content complies with safety or usage guidelines to reduce off-topic prompts.

Possible use cases: 
1. Preventing misuse of AI chatbots by limiting filtering prompts sent to the LLM.
2. Moderating ChatBot behaviour by filtering prompts using chosen policies.

(add video here)

Note: This is an exploratory project. The aim of this project is to learn tools and frameworks that are commonly used in AI.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Setup](#setup)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [API Documentation](#api-documentation)
- [Tools & Technologies](#tools--technologies)

## Overview

Clean Talk is a two-stage safety evaluation system:

1. **Classifier (DistilBERT)** - Classifies prompts into 6 categories:
   - `safe` - Safe and benign prompts
   - `adversarial_harmful` - Adversarial attacks attempting to cause harm
   - `vanilla_harmful` - Directly harmful prompts without adversarial framing
   - `adversarial_benign` - Adversarial attacks on benign topics
   - `unsafe` - Unsafe prompts
   - `vanilla_benign` - Benign prompts without adversarial intent

   Trained on a combination of these 2 datasets: [nvidia/Aegis-AI-Content-Safety-Dataset-2.0](https://huggingface.co/datasets/nvidia/Aegis-AI-Content-Safety-Dataset-2.0) & [allenai/wildjailbreak](https://huggingface.co/datasets/allenai/wildjailbreak)

2. **Judge LLM (RAG + Gemini)** - Uses RAG with Pinecone vector database to retrieve relevant policies and evaluate compliance via Google Gemini.

## Features

- ✅ Real-time prompt classification using DistilBERT
- ✅ Confidence scores for predictions
- ✅ FastAPI backend for easy integration
- ✅ Streamlit web interface with policy management
- ✅ RAG-based policy evaluation using Pinecone & Google Gemini
- ✅ Dynamic custom policy creation and storage
- ✅ Docker support

## Setup

### 1. Clone and Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  
```


### 2. Set up your environment variables.
Copy `.env.example` to `.env` and fill in the given variables.

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Prepare the Model

Train your model by running the notebook:
```bash
jupyter notebook notebooks/02_training.ipynb
```

Then save your trained model to:
```
models/best_model.pt
```

Alternatively, download my pre-trained model from this link and place it in the `models/` directory.

### 5. Update Model Path (if needed)

In `src/core/classifier.py`, update the `final_model_path` if your model is in a different location:
```python
final_model_path = 'models/best_model.pt'
```

### 6. Set up Pinecone RAG
Run the notebook `notebooks/03_setup_rag.ipynb` to initialize Pinecone and seed initial policies (you can also add policies directly through the Streamlit interface).

## Usage

### Prerequisites
Before running the application, ensure:
1. The trained model is saved at `models/best_model.pt`
2. All environment variables are configured in `.env`
3. Pinecone is initialized and the `safety-policies` index is created (see notebook `03_setup_rag.ipynb`)
4. Google Gemini API key is set up and valid

### Running the API

Start the FastAPI backend on `http://localhost:8000`:
```bash
python src/api/main.py
```

The API will be available at: http://localhost:8000/docs (Swagger UI)

Alternatively, using uvicorn directly:
```bash
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

### Running the Streamlit App

In a new terminal (ensure API is running), start the Streamlit frontend:
```bash
streamlit run src/app.py
```

The app will be available at: http://localhost:8501

**Features:**
- **Prompt Classification** - Enter a prompt to get instant safety classification
- **Sample Prompts** - Choose from pre-loaded test prompts to explore the system
- **Policy Management** - Add new safety policies via the sidebar
- **Current Policies** - View all stored policies
- **Policy Compliance Check** - Automatically validates prompts against stored policies using RAG

### Using the API Directly

**Health Check:**
```bash
curl http://localhost:8000/
```

**Classify a Prompt:**
```bash
curl -X POST "http://localhost:8000/classify" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is the capital of France?"}'
```

**Response:**
```json
{
  "prompt": "What is the capital of France?",
  "classification": "safe",
  "confidence": 0.95
}
```

## Project Structure

```
clean-talk/
├── Dockerfile                 # Docker configuration
├── README.md                  # This file
├── requirements.txt           # Python dependencies            
│
├── models/
│   └── best_model.pt          # Trained model checkpoint
│
├── notebooks/
│   ├── 01_data_exploration.ipynb   # EDA and data analysis
│   ├── 02_training.ipynb           # Model training
│   └── 03_setup_rag.ipynb          # Pinecone DB set up
│
├── reports/
│   ├── experiment_logs/       # Training logs and metrics
│   ├── diagrams/              # Training metrics visualisation
│   └── api_log.csv            # Log of all API requests 
│
└── src/
│   ├── api/
│   │   └── main.py            # FastAPI application
│   ├── core/
│   │   ├── classifier.py      # DistilBERT model inference
│   │   ├── features.py        # Feature engineering utilities
│   │   └── safety_rag.py      # RAG pipeline with Pinecone & Gemini
│   ├── app.py                 # Streamlit frontend with policy management
│   └── utils/
│       ├── api_logger.py      # API request/response logging
│       ├── logger.py          # Training logger
│       └── helper.py          # Utility functions
│
└── tests/                     # Test suite
```

## API Documentation

### Endpoints

#### `GET /`
Health check endpoint.

**Response:**
```json
{
  "message": "Prompt Classification API is running"
}
```

#### `POST /classify`
Classify a prompt using the DistilBERT model and return safety classification with confidence score.

**Request:**
```json
{
  "prompt": "string"
}
```

**Response:**
```json
{
  "prompt": "string",
  "classification": "string (one of: safe, adversarial_harmful, vanilla_harmful, adversarial_benign, unsafe, vanilla_benign)",
  "confidence": "float (0.0 to 1.0)"
}
```

**Example:**
```bash
curl -X POST "http://localhost:8000/classify" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is the capital of France?"}'
```

#### `POST /policy_check`
Evaluate prompt compliance against stored policies using RAG (Retrieval-Augmented Generation). Uses Pinecone to retrieve relevant policies and Google Gemini to evaluate compliance.

**Request:**
```json
{
  "prompt": "string"
}
```

**Response:**
```json
{
  "decision": "string (VIOLATION or NOT A VIOLATION)",
  "policy": "string (the relevant policy that was checked)",
  "response_to_user": "string (detailed explanation)"
}
```

**Note:** This endpoint is typically called only for prompts classified as safe by the `/classify` endpoint.

#### `POST /add_policy`
Add a new safety policy to the Pinecone vector database. The policy is encoded using sentence-transformers and stored with semantic search capability.

**Request:**
```json
{
  "policy": "string"
}
```

**Response:**
```json
{
  "id": "string",
  "text": "string",
  "status": "uploaded"
}
```

**Example:**
```bash
curl -X POST "http://localhost:8000/add_policy" \
  -H "Content-Type: application/json" \
  -d '{"policy": "Do not provide medical advice or diagnose medical conditions"}'
```


## Tools & Technologies

### Core Libraries
- **PyTorch** - Deep learning framework
- **Transformers** - Hugging Face model library
- **DistilBERT** - Fast, lightweight BERT model
- **FastAPI** - Modern web framework
- **Streamlit** - Interactive web app framework
- **Uvicorn** - ASGI server for FastAPI

### Data & ML
- **Pandas** - Data manipulation
- **Scikit-learn** - ML utilities
- **Matplotlib & Seaborn** - Visualization
- **Datasets** - Hugging Face datasets library for loading training data


### RAG & LLM
- **Pinecone** - Vector database for policy storage and retrieval
- **Sentence-Transformers** (all-MiniLM-L6-v2) - Embedding generation for semantic search
- **Google Gemini** - LLM for policy evaluation and judgment

### Additional Tools
- **python-dotenv** - Environment variable management
- **slowapi** - Rate limiting for FastAPI (currently disabled)

## Training

To train or fine-tune the model:

1. Open `notebooks/02_training.ipynb`
2. Run the training notebook
3. Save the best model to `models/best_model.pt`

Training metrics and logs are saved to `reports/experiment_logs/`

## Docker

Build and run with Docker:

```bash
docker build -t cleantalk .
docker run -p 8000:8000 -p 8501:8501 cleantalk
```

**Note:** The Dockerfile needs to be created or populated with proper configuration to run the application containerized.

## Testing

A test suite structure exists in the `tests/` folder for unit and integration tests. Currently, the suite is empty and ready for test implementation.

To add tests, create test files in the `tests/` folder following pytest conventions.

## Remarks
Project done by Atharshlakshmi Vijayakumar