---
language:
- en
license: mit
pretty_name: Terra AI Weather Dataset
size_categories:
- 1K<n<10K
task_categories:
- text-generation
---

# Terra AI Weather Dataset

This dataset contains weather-related questions and their corresponding structured responses, designed for fine-tuning language models for weather assistance tasks.

## Dataset Description

### Languages
English

### Dataset Structure

The dataset contains pairs of questions and responses in the following format:

For location queries:
```
Question: "What's the weather like in [location]?"
Response: "Current conditions in [location]:
Temperature: [range in celsius]
Conditions: [brief description]
Recommendation: [specific action]"
```

For activity queries:
```
Question: "Should I [activity] today?"
Response: "Weather advisory for [activity]:
Current conditions: [description]
Safety level: [Good/Moderate/Poor]
Recommendation: [specific action]"
```

### Data Fields

- `question`: Weather-related question
- `response`: Structured weather response

### Data Splits

- Training: 100%
- No validation/test splits (small dataset)

## Dataset Creation

### Curation Rationale

This dataset was created to fine-tune language models specifically for weather-related queries, ensuring:
- Consistent response formatting
- Practical weather recommendations
- Safety-focused advice
- Clear temperature and condition descriptions

### Source Data

Hand-crafted examples covering various weather scenarios and activities.

### Personal and Sensitive Information

This dataset contains no personal information.

## Considerations for Using the Data

### Social Impact of Dataset

The dataset aims to improve weather-related communication and safety recommendations.

### Discussion of Biases

The dataset may have:
- Geographic bias towards well-known cities
- Seasonal bias based on common weather patterns
- Activity bias towards common outdoor activities

### Other Known Limitations

- Limited to general weather patterns
- Does not include real-time weather data
- Responses are template-based

## Additional Information

### Dataset Curators

Created by Terra AI team

### Licensing Information

MIT License

### Citation Information

Please cite this dataset as:
```
@misc{terra-ai-weather-dataset,
  title={Terra AI Weather Dataset},
  year={2024},
  publisher={Hugging Face}
}
``` 