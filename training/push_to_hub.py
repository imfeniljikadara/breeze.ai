from transformers import GPT2LMHeadModel, GPT2Tokenizer
from huggingface_hub import HfApi
import os

def push_to_hub():
    # Load the fine-tuned model and tokenizer
    model_path = "./weather-gpt2"
    model = GPT2LMHeadModel.from_pretrained(model_path)
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    
    # Set your Hugging Face Hub repository name
    repo_name = "imfeniljikadara/terra-ai-weather-gpt2"
    
    # Push to hub
    model.push_to_hub(repo_name)
    tokenizer.push_to_hub(repo_name)
    
    print(f"Model and tokenizer pushed to: https://huggingface.co/{repo_name}")

if __name__ == "__main__":
    # Make sure you're logged in to Hugging Face Hub
    if not os.path.exists("~/.huggingface/token"):
        print("Please login to Hugging Face Hub first using `huggingface-cli login`")
        exit(1)
    
    push_to_hub() 