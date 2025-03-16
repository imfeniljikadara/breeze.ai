import json
import torch
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    TextDataset,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments
)
import os
from datasets import Dataset

def prepare_dataset():
    # Load the dataset
    with open('weather_dataset.json', 'r') as f:
        data = json.load(f)
    
    # Format data for training
    formatted_data = []
    for item in data['data']:
        # Format: question + response + EOS
        text = f"User question: {item['question']}\nWeather response: {item['response']}\n<|endoftext|>"
        formatted_data.append({"text": text})
    
    # Create Hugging Face dataset
    dataset = Dataset.from_list(formatted_data)
    return dataset

def train():
    # Initialize model and tokenizer
    model_name = "gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    
    # Add padding token
    tokenizer.pad_token = tokenizer.eos_token
    model.resize_token_embeddings(len(tokenizer))
    
    # Prepare dataset
    dataset = prepare_dataset()
    
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=128
        )
    
    # Tokenize dataset
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names
    )
    
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        save_steps=100,
        save_total_limit=2,
        learning_rate=5e-5,
        warmup_steps=100,
        logging_dir='./logs',
        logging_steps=10,
        evaluation_strategy="steps",
        eval_steps=100,
        load_best_model_at_end=True,
    )
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False
        )
    )
    
    # Train the model
    trainer.train()
    
    # Save the model
    model_path = "./weather-gpt2"
    trainer.save_model(model_path)
    tokenizer.save_pretrained(model_path)
    
    print("Training completed! Model saved to:", model_path)

if __name__ == "__main__":
    train() 