import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
from typing import Optional
import json
from fastapi.responses import StreamingResponse

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

# Load model - using GPT-2 small for faster responses
MODEL_PATH = "gpt2"
print(f"Loading model {MODEL_PATH}...")

model = GPT2LMHeadModel.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    device_map="auto"
)
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_PATH)
tokenizer.pad_token = tokenizer.eos_token

print("Model loaded successfully!")

class ModelRequest(BaseModel):
    prompt: str
    max_length: Optional[int] = 150  # Shorter responses for faster generation
    temperature: Optional[float] = 0.7
    stream: Optional[bool] = False

def generate_response(prompt: str, max_length: int = 150, temperature: float = 0.7):
    # Prepare the prompt with weather-specific context
    system_prompt = "You are a weather assistant. Provide brief, accurate responses about weather conditions and recommendations."
    full_prompt = f"{system_prompt}\n\nQuestion: {prompt}\nAnswer:"
    
    inputs = tokenizer(full_prompt, return_tensors="pt", padding=True).to(model.device)
    
    # Generate with optimized parameters
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        temperature=temperature,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        num_return_sequences=1,
        early_stopping=True
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response.replace(full_prompt, "").strip()
    
    return response

async def generate_stream(prompt: str, max_length: int = 150, temperature: float = 0.7):
    response = generate_response(prompt, max_length, temperature)
    yield json.dumps({"text": response, "domain": "weather"})

@app.get("/")
async def root():
    return {
        "message": "Welcome to Terra AI GPT Weather API",
        "status": "running",
        "model": MODEL_PATH,
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
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model": MODEL_PATH
    } 