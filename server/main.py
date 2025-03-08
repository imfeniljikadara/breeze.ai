import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict
import json
from fastapi.responses import StreamingResponse
import logging
from functools import lru_cache
import requests
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Weather API configuration
WEATHER_API_KEY = os.environ.get("WEATHER_API_KEY")
if not WEATHER_API_KEY:
    logger.error("""
OpenWeatherMap API key not configured! Please follow these steps:

1. Sign up for a free account at: https://home.openweathermap.org/users/sign_up
2. Get your API key from: https://home.openweathermap.org/api_keys
3. Set up the API key:
   
   For local development:
   - Add WEATHER_API_KEY=your_api_key to .env file
   
   For Hugging Face Spaces:
   - Go to Space Settings
   - Add Repository Secret: WEATHER_API_KEY=your_api_key
""")
WEATHER_API_URL = "https://api.openweathermap.org/data/2.5/weather"

# Weather-related keywords for query validation
WEATHER_KEYWORDS = {
    'weather', 'temperature', 'rain', 'snow', 'wind', 'sunny', 'cloudy', 'storm',
    'forecast', 'humidity', 'precipitation', 'climate', 'cold', 'hot', 'warm', 'cool',
    'degrees', 'celsius', 'fahrenheit', 'outdoor', 'indoor', 'umbrella', 'sunscreen',
    'jacket', 'coat', 'conditions', 'wear', 'bring', 'pack'
}

app = FastAPI(
    title="Terra AI Weather API",
    description="Weather analysis API with real weather data",
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

class ModelRequest(BaseModel):
    prompt: str
    max_length: Optional[int] = 50
    temperature: Optional[float] = 0.7
    stream: Optional[bool] = False

# Cache for responses (TTL would be better in production)
response_cache: Dict[str, Dict] = {}

@lru_cache(maxsize=100)
def get_cached_response(prompt: str) -> Optional[Dict]:
    # Only return cache if less than 30 minutes old
    if prompt in response_cache:
        timestamp = response_cache[prompt].get("timestamp", 0)
        if datetime.now().timestamp() - timestamp < 1800:  # 30 minutes
            return response_cache[prompt]
    return None

def cache_response(prompt: str, response: Dict):
    response["timestamp"] = datetime.now().timestamp()
    response_cache[prompt] = response
    if len(response_cache) > 100:  # Limit cache size
        response_cache.pop(next(iter(response_cache)))

def is_weather_related(text: str) -> bool:
    """Check if the query is weather-related."""
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in WEATHER_KEYWORDS)

def get_default_response(prompt: str) -> Dict:
    """Get a default response for non-weather questions."""
    if "hello" in prompt.lower() or "hi" in prompt.lower():
        message = "Hello! I'm Terra AI, a weather assistant. How can I help you with weather-related questions today?"
    elif "how are you" in prompt.lower():
        message = "I'm here to help you with weather-related questions. What would you like to know about the weather?"
    else:
        message = "I can only assist with weather-related questions. Please ask me about weather conditions, forecasts, or outdoor activity recommendations!"
    
    return {
        "text": message,
        "domain": "greeting",
        "data": None
    }

def extract_location(prompt: str) -> Optional[str]:
    """Extract location from weather-related query."""
    location_markers = ["in ", "at ", "for ", "about "]
    prompt_lower = prompt.lower()
    
    for marker in location_markers:
        if marker in prompt_lower:
            parts = prompt_lower.split(marker)
            if len(parts) > 1:
                location = parts[-1].strip("?.,! ")
                return location
    
    # Check if location is mentioned without markers
    words = prompt_lower.split()
    potential_locations = []
    for i in range(len(words)):
        if words[i] not in WEATHER_KEYWORDS and len(words[i]) > 3:
            potential_locations.append(words[i])
    
    return potential_locations[-1] if potential_locations else None

def get_weather_data(location: str) -> Optional[Dict]:
    """Fetch weather data from OpenWeatherMap API."""
    if not WEATHER_API_KEY:
        return {
            "error": "API key not configured",
            "message": "Please configure the OpenWeatherMap API key. See logs for instructions."
        }

    try:
        params = {
            "q": location,
            "appid": WEATHER_API_KEY,
            "units": "metric"  # Use metric units
        }
        
        response = requests.get(WEATHER_API_URL, params=params)
        
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 401:
            logger.error("Invalid API key. Please check your OpenWeatherMap API key.")
            return {
                "error": "invalid_key",
                "message": "Invalid API key. Please check your OpenWeatherMap API key configuration."
            }
        elif response.status_code == 404:
            logger.error(f"Location '{location}' not found")
            return {
                "error": "location_not_found",
                "message": f"Could not find weather data for location: {location}"
            }
        else:
            logger.error(f"Weather API error: {response.status_code} - {response.text}")
            return {
                "error": "api_error",
                "message": f"Error fetching weather data: {response.text}"
            }
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Error connecting to weather service: {str(e)}")
        return {
            "error": "connection_error",
            "message": "Could not connect to weather service. Please try again later."
        }
    except Exception as e:
        logger.error(f"Unexpected error fetching weather data: {str(e)}")
        return {
            "error": "unknown_error",
            "message": "An unexpected error occurred while fetching weather data."
        }

def get_clothing_recommendation(weather_data: Dict) -> str:
    """Generate clothing recommendation based on weather conditions."""
    if not weather_data:
        return "No specific recommendations available without weather data."
    
    temp = weather_data.get("main", {}).get("temp", 0)
    weather_condition = weather_data.get("weather", [{}])[0].get("main", "").lower()
    wind_speed = weather_data.get("wind", {}).get("speed", 0)
    
    recommendations = []
    
    # Temperature-based recommendations
    if temp < 0:
        recommendations.append("Wear a heavy winter coat, gloves, and a warm hat")
    elif temp < 10:
        recommendations.append("Wear a warm jacket and consider layering")
    elif temp < 20:
        recommendations.append("A light jacket or sweater should be comfortable")
    elif temp < 30:
        recommendations.append("Light clothing and sun protection recommended")
    else:
        recommendations.append("Wear lightweight, breathable clothing")

    # Condition-based additions
    if "rain" in weather_condition or "drizzle" in weather_condition:
        recommendations.append("bring an umbrella and waterproof jacket")
    elif "snow" in weather_condition:
        recommendations.append("wear waterproof boots")
    elif "thunderstorm" in weather_condition:
        recommendations.append("stay indoors if possible")
    
    # Wind considerations
    if wind_speed > 10:
        recommendations.append("bring a windbreaker")
    
    return " and ".join(recommendations[:2]) + "."

def format_weather_response(weather_data: Dict, location: str) -> Dict:
    """Format weather data into a user-friendly response."""
    if not weather_data:
        return {
            "text": f"I don't have current weather data for {location}.",
            "domain": "weather",
            "data": None
        }
    
    # Check for error responses
    if "error" in weather_data:
        return {
            "text": weather_data["message"],
            "domain": "error",
            "data": weather_data
        }
    
    try:
        temp = weather_data.get("main", {}).get("temp", 0)
        temp_f = (temp * 9/5) + 32  # Convert to Fahrenheit
        
        condition = weather_data.get("weather", [{}])[0].get("main", "Unknown")
        description = weather_data.get("weather", [{}])[0].get("description", "Unknown")
        
        recommendation = get_clothing_recommendation(weather_data)
        
        formatted_response = f"""Current conditions: {description.capitalize()}
Temperature: {temp:.1f}°C ({temp_f:.1f}°F)
Recommendation: {recommendation}"""

        return {
            "text": formatted_response,
            "domain": "weather",
            "data": weather_data
        }
        
    except Exception as e:
        logger.error(f"Error formatting response: {str(e)}")
        return {
            "text": f"I have weather data for {location}, but encountered an error formatting it.",
            "domain": "error",
            "data": weather_data
        }

def generate_response(prompt: str, max_length: int = 50, temperature: float = 0.7) -> Dict:
    """Generate a weather response based on user prompt."""
    # Check if it's a greeting or non-weather question
    if not is_weather_related(prompt):
        return get_default_response(prompt)

    # Check cache first
    cached_response = get_cached_response(prompt)
    if cached_response:
        logger.info("Using cached response")
        return cached_response

    # Extract location
    location = extract_location(prompt)
    if not location:
        return {
            "text": "I need a location to provide weather information. Could you specify where you're asking about?",
            "domain": "weather",
            "data": None
        }

    # Get weather data
    weather_data = get_weather_data(location)
    
    # Format response
    response = format_weather_response(weather_data, location)
    
    # Cache valid response
    if response["data"]:
        cache_response(prompt, response)
    
    return response

async def generate_stream(prompt: str, max_length: int = 50, temperature: float = 0.7):
    """Stream the response back to the client."""
    try:
        response = generate_response(prompt, max_length, temperature)
        yield json.dumps(response)
    except Exception as e:
        yield json.dumps({"error": str(e)})

@app.get("/")
async def root():
    return {
        "message": "Welcome to Terra AI Weather API",
        "status": "running",
        "version": "2.0.0",
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
        return response
    
    except Exception as e:
        logger.error(f"Error in generate endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    # Try to make a test API call to verify weather API is working
    try:
        test_response = requests.get(
            WEATHER_API_URL,
            params={
                "q": "London",
                "appid": WEATHER_API_KEY,
                "units": "metric"
            }
        )
        api_status = "healthy" if test_response.status_code == 200 else "error"
    except:
        api_status = "error"
        
    return {
        "status": "healthy",
        "weather_api_status": api_status,
        "cache_size": len(response_cache)
    } 