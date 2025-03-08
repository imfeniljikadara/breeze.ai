import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
from typing import Optional
import json
from fastapi.responses import StreamingResponse
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure cache directory exists
CACHE_DIR = os.environ.get('TRANSFORMERS_CACHE', '/cache')
os.makedirs(CACHE_DIR, exist_ok=True)

app = FastAPI(
    title="Terra AI GPT Weather API",
    description="GPT-2 powered weather analysis API",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and tokenizer
model = None
tokenizer = None

def load_model():
    global model, tokenizer
    try:
        # Load model - using GPT-2 small for faster responses
        MODEL_PATH = "gpt2"
        logger.info(f"Loading model {MODEL_PATH}...")
        
        # Load tokenizer first
        tokenizer = GPT2Tokenizer.from_pretrained(
            MODEL_PATH,
            cache_dir=CACHE_DIR,
            local_files_only=False
        )
        tokenizer.pad_token = tokenizer.eos_token
        
        # Load model with basic settings
        model = GPT2LMHeadModel.from_pretrained(
            MODEL_PATH,
            cache_dir=CACHE_DIR,
            local_files_only=False
        )
        model.eval()  # Set to evaluation mode
        
        logger.info("Model loaded successfully!")
        return True
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return False

# Load model on startup
load_model()

class ModelRequest(BaseModel):
    prompt: str
    max_length: Optional[int] = 150  # Shorter responses for faster generation
    temperature: Optional[float] = 0.7
    stream: Optional[bool] = False

def generate_response(prompt: str, max_length: int = 150, temperature: float = 0.7):
    if model is None or tokenizer is None:
        if not load_model():
            raise HTTPException(status_code=503, detail="Model not available")
    
    try:
        # Prepare the prompt with weather-specific context
        system_prompt = "You are a weather assistant. Provide brief, accurate responses about weather conditions and recommendations."
        full_prompt = f"{system_prompt}\n\nQuestion: {prompt}\nAnswer:"
        
        # Tokenize input
        inputs = tokenizer.encode(full_prompt, return_tensors="pt")
        
        # Generate with basic parameters
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_length=max_length,
                temperature=temperature,
                num_return_sequences=1,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode and clean response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response.replace(full_prompt, "").strip()
        
        return response
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

async def generate_stream(prompt: str, max_length: int = 150, temperature: float = 0.7):
    try:
        response = generate_response(prompt, max_length, temperature)
        yield json.dumps({"text": response, "domain": "weather"})
    except Exception as e:
        yield json.dumps({"error": str(e)})

@app.get("/")
async def root():
    return {
        "message": "Welcome to Terra AI GPT Weather API",
        "status": "running",
        "model": "gpt2",
        "endpoints": {
            "/generate": "Generate weather response",
            "/health": "Check server health"
        }
    }

@app.post("/generate")
async def generate(request: ModelRequest):
    try:
        if request.stream:
            return StreamingResponse(
                generate_stream(
                    request.prompt,
                    request.max_length,
                    request.temperature
                ),
                media_type="application/json"
            )
        
        response = generate_response(
            request.prompt,
            request.max_length,
            request.temperature
        )
        return {"text": response, "domain": "weather"}
    
    except Exception as e:
        logger.error(f"Error in generate endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    model_status = "healthy" if model is not None and tokenizer is not None else "loading"
    return {
        "status": model_status,
        "model": "gpt2"
    } 