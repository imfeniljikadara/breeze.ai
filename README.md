---
title: Terra AI GPT Weather Assistant
emoji: 🌤️
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# Terra AI Weather Assistant

A weather assistant that provides real-time weather information and clothing recommendations using the OpenWeatherMap API.

## Features

- Real-time weather data
- Temperature in both Celsius and Fahrenheit
- Smart clothing recommendations based on weather conditions
- Response caching with 30-minute TTL
- Intelligent location extraction from queries
- Comprehensive weather condition analysis

## Setup

1. Get an API key from [OpenWeatherMap](https://openweathermap.org/api)
2. Create a `.env` file in the project root with:
   ```
   WEATHER_API_KEY=your_api_key_here
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the server:
   ```bash
   uvicorn server.main:app --reload
   ```

## Usage

Send POST requests to `/generate` with a prompt:

```json
{
  "prompt": "What's the weather like in London?"
}
```

The response will include:
- Current conditions
- Temperature (°C and °F)
- Clothing/activity recommendations

## Environment Variables

- `WEATHER_API_KEY`: Your OpenWeatherMap API key (required)

## API Endpoints

- `POST /generate`: Get weather information and recommendations
- `GET /health`: Check API health status
- `GET /`: API information and documentation

## Response Format

```json
{
  "text": "Current conditions: Clear sky\nTemperature: 20.5°C (68.9°F)\nRecommendation: Light clothing and sun protection recommended.",
  "domain": "weather",
  "data": { ... }  // Full weather data from OpenWeatherMap
}
```

## Deployment

This project is designed to be deployed on Hugging Face Spaces. 