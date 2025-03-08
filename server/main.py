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

def extract_location(prompt: str) -> Optional[str]:
    """Extract location from weather-related query."""
    location_markers = ["in ", "at ", "for ", "about "]
    prompt_lower = prompt.lower()
    
    for marker in location_markers:
        if marker in prompt_lower:
            location = prompt_lower.split(marker)[-1].strip()
            return location.strip("?.,! ")
    return None

def validate_response(response: str, location: Optional[str]) -> str:
    """Validate and fix weather response format."""
    if not response or len(response) < 10:
        return f"I don't have current weather data for {location if location else 'that location'}."
        
    # Force the response into our template
    lines = response.lower().split('\n')
    has_conditions = any('current conditions:' in line for line in lines)
    has_temperature = any('temperature:' in line for line in lines)
    has_recommendation = any('recommendation:' in line for line in lines)
    
    if not (has_conditions and has_temperature and has_recommendation):
        return f"I don't have current weather data for {location if location else 'that location'}."
        
    # Only keep lines matching our template
    valid_lines = []
    for line in lines:
        if any(key in line for key in ['current conditions:', 'temperature:', 'recommendation:']):
            valid_lines.append(line.capitalize())
    
    return '\n'.join(valid_lines) if valid_lines else f"I don't have current weather data for {location if location else 'that location'}."

def generate_response(prompt: str, max_length: int = 50, temperature: float = 0.7):
    # Extract location first
    location = extract_location(prompt)
    
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
        system_prompt = f"""You are Terra AI, a weather assistant. ONLY respond in this EXACT format:

Current conditions: [1-2 words]
Temperature: [number] degrees
Recommendation: [5-10 words]

Location: {location if location else 'unknown'}
Rules:
1. ONLY use the format above
2. NEVER add extra text
3. If unsure, respond: "I don't have current weather data for {location if location else 'that location'}" """
        
        full_prompt = f"{system_prompt}\n\nQuestion: {prompt}\nResponse:"
        
        # Tokenize input
        inputs = tokenizer.encode(full_prompt, return_tensors="pt")
        
        # Generate with very strict parameters
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_length=max_length,
                do_sample=True,
                temperature=0.3,  # Even lower temperature
                top_p=0.7,
                top_k=10,  # Very restricted sampling
                num_return_sequences=1,
                pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.4,
                length_penalty=1.5,
                early_stopping=True,
                min_length=10,
                no_repeat_ngram_size=3
            )
        
        # Decode and clean response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response.replace(full_prompt, "").strip()
        
        # Validate and fix response format
        response = validate_response(response, location)
        
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