import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
from typing import Optional, Dict
import json
from fastapi.responses import StreamingResponse
import logging
from functools import lru_cache

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure cache directory exists
CACHE_DIR = os.environ.get('TRANSFORMERS_CACHE', '/cache')
os.makedirs(CACHE_DIR, exist_ok=True)

# Weather-related keywords for response validation
WEATHER_KEYWORDS = {
    'weather', 'temperature', 'rain', 'snow', 'wind', 'sunny', 'cloudy', 'storm',
    'forecast', 'humidity', 'precipitation', 'climate', 'cold', 'hot', 'warm', 'cool',
    'degrees', 'celsius', 'fahrenheit', 'outdoor', 'indoor', 'umbrella', 'sunscreen',
    'jacket', 'coat', 'conditions'
}

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
    max_length: Optional[int] = 50  # Even shorter responses
    temperature: Optional[float] = 0.7
    stream: Optional[bool] = False

# Cache for responses
response_cache: Dict[str, str] = {}

@lru_cache(maxsize=100)
def get_cached_response(prompt: str) -> Optional[str]:
    return response_cache.get(prompt)

def cache_response(prompt: str, response: str):
    response_cache[prompt] = response
    if len(response_cache) > 100:  # Limit cache size
        response_cache.pop(next(iter(response_cache)))

def is_weather_related(text: str) -> bool:
    """Check if the response is weather-related."""
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in WEATHER_KEYWORDS)

def get_default_response(prompt: str) -> str:
    """Get a default response for non-weather questions."""
    if "hello" in prompt.lower() or "hi" in prompt.lower():
        return "Hello! I'm Terra AI, a weather assistant. How can I help you with weather-related questions today?"
    elif "how are you" in prompt.lower():
        return "I'm here to help you with weather-related questions. What would you like to know about the weather?"
    else:
        return "I can only assist with weather-related questions. Please ask me about weather conditions, forecasts, or outdoor activity recommendations!"

def generate_response(prompt: str, max_length: int = 50, temperature: float = 0.7):
    # Check if it's a greeting or non-weather question
    if not any(keyword in prompt.lower() for keyword in WEATHER_KEYWORDS):
        return get_default_response(prompt)

    # Check cache first
    cached_response = get_cached_response(prompt)
    if cached_response:
        logger.info("Using cached response")
        return cached_response

    if model is None or tokenizer is None:
        if not load_model():
            raise HTTPException(status_code=503, detail="Model not available")
    
    try:
        # Prepare the prompt with strict weather context and template
        system_prompt = """You are Terra AI, a focused weather assistant. Follow these rules exactly:

1. ONLY discuss current weather conditions
2. Use this EXACT format for location-based questions:
   "Current conditions: [brief description]
   Temperature: [range]
   Recommendation: [1 brief tip]"
3. Keep ALL responses under 50 words
4. NO historical data or long-term trends
5. If unsure, say "I don't have current weather data for [location]"
6. NEVER discuss non-weather topics"""
        
        full_prompt = f"{system_prompt}\n\nQuestion: {prompt}\nWeather-focused answer:"
        
        # Tokenize input
        inputs = tokenizer.encode(full_prompt, return_tensors="pt")
        
        # Generate with very strict parameters
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_length=max_length,
                do_sample=True,
                temperature=0.5,  # Reduced temperature for more focused responses
                top_p=0.8,
                top_k=20,  # More restricted sampling
                num_return_sequences=1,
                pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.3,  # Increased repetition penalty
                length_penalty=1.2,  # Encourage even more concise responses
                early_stopping=True,
                min_length=10,
                no_repeat_ngram_size=3  # Stricter repetition prevention
            )
        
        # Decode and clean response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response.replace(full_prompt, "").strip()
        
        # Validate response is weather-related
        if not is_weather_related(response):
            return "I don't have current weather data for that location. Please ask about specific weather conditions or forecasts."
        
        # Clean up common model artifacts
        response = response.replace("AI:", "").replace("Assistant:", "").strip()
        response = response.split("\n\n")[0]  # Take only the first paragraph
        
        # Cache valid response
        cache_response(prompt, response)
        
        return response
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

async def generate_stream(prompt: str, max_length: int = 50, temperature: float = 0.7):
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
        "model": "gpt2",
        "cache_size": len(response_cache)
    } 