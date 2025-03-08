---
title: Terra AI GPT Weather Assistant
emoji: 🌤️
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# Terra AI Weather Assistant 🌤️

A smart weather assistant that provides real-time weather information and personalized clothing recommendations using OpenWeatherMap API.

## Features 🚀

- Real-time weather data from OpenWeatherMap
- Smart clothing recommendations based on weather conditions
- Temperature in both Celsius and Fahrenheit
- Intelligent location extraction from natural language queries
- Response caching with 30-minute TTL
- Comprehensive error handling
- CORS-enabled for web integration

## Tech Stack 💻

- **Backend**: FastAPI
- **Weather Data**: OpenWeatherMap API
- **Caching**: In-memory with TTL
- **Language**: Python 3.8+

## Quick Start 🏃‍♂️

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/terra-ai-weather.git
   cd terra-ai-weather
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up OpenWeatherMap API:
   - Sign up at [OpenWeatherMap](https://home.openweathermap.org/users/sign_up)
   - Get your API key from [API Keys](https://home.openweathermap.org/api_keys)
   - Create `.env` file:
     ```
     WEATHER_API_KEY=your_api_key_here
     ```

5. Run the server:
   ```bash
   uvicorn server.main:app --reload
   ```

6. Visit `http://localhost:8000/docs` for API documentation

## API Endpoints 📡

### POST /generate
Generate weather information and recommendations.

Request:
```json
{
  "prompt": "What's the weather like in London?",
  "max_length": 50,
  "temperature": 0.7,
  "stream": false
}
```

Response:
```json
{
  "text": "Current conditions: Clear sky\nTemperature: 20.5°C (68.9°F)\nRecommendation: Light clothing and sun protection recommended.",
  "domain": "weather",
  "data": { ... }
}
```

### GET /health
Check API health status.

## Environment Variables 🔑

| Variable | Description | Required |
|----------|-------------|----------|
| WEATHER_API_KEY | OpenWeatherMap API key | Yes |

## Error Handling 🚨

The API provides detailed error messages for:
- Missing API key
- Invalid API key
- Location not found
- Connection issues
- Server errors

## Deployment 🚀

### Local Development
```bash
uvicorn server.main:app --reload --port 8000
```

### Production
```bash
uvicorn server.main:app --host 0.0.0.0 --port $PORT
```

## Contributing 🤝

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License 📝

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments 👏

- OpenWeatherMap for providing weather data
- FastAPI for the awesome web framework
- Python community for great packages 