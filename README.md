---
title: Terra AI Weather Assistant
emoji: 🌤️
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# Terra AI Weather Assistant 🌤️

A specialized weather assistant powered by GPT-2, fine-tuned on weather-related conversations.

## Features

- 🌡️ Location-based weather information
- 🏃‍♂️ Activity-specific weather advisories
- 🎯 Focused, template-based responses
- ⚡ Fast response times with caching
- 🔄 Real-time weather recommendations

## API Endpoints

### `/generate`

Generate weather-related responses:

```bash
curl -X POST "https://imfeniljikadara-terra-ai-gpt.hf.space/generate" \
     -H "Content-Type: application/json" \
     -d '{"prompt": "What is the weather like in New York?"}'
```

### Response Format

For locations:
```
Current conditions in [location]:
Temperature: [number range in celsius]
Conditions: [1-2 word description]
Recommendation: [1 specific action]
```

For activities:
```
Weather advisory for [activity]:
Current conditions: [brief description]
Safety level: [Good/Moderate/Poor]
Recommendation: [1 specific action]
```

## Model Details

- Base model: GPT-2
- Fine-tuned on: Weather conversations
- Response format: Structured templates
- Temperature: 0.4 (focused responses)
- Max length: 50 tokens

## Usage

1. Ask about weather in a location:
   ```
   What's the weather like in Tokyo?
   ```

2. Get activity recommendations:
   ```
   Should I go hiking today?
   ```

3. Check conditions for outdoor activities:
   ```
   Is it safe for outdoor photography?
   ```

## Development

Built with:
- FastAPI
- Hugging Face Transformers
- PyTorch
- Docker

## License

MIT License 