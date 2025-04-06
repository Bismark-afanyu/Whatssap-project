# Ensure you've run: pip install transformers datasets torch numpy tf-keras
# PyTorch should already be installed (2.4.0 CPU version is fine)

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments, TextClassificationPipeline
from datasets import load_dataset
import numpy as np

# Check device: Use MPS if available (Apple Silicon), else CPU
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Step 1: Load the pre-trained model and tokenizer
model_name = "lxyuan/distilbert-base-multilingual-cased-sentiments-student"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)

# Step 2: Load and prepare the tweet_eval sentiment dataset
dataset = load_dataset("tweet_eval", "sentiment")

# Remap labels: tweet_eval (0=negative, 1=neutral, 2=positive) to our model (0=positive, 1=neutral, 2=negative)
def remap_labels(example):
    label_map = {0: 2, 1: 1, 2: 0}  # Negative->2, Neutral->1, Positive->0
    example["label"] = label_map[example["label"]]
    return example

dataset = dataset.map(remap_labels)

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

tokenized_dataset = dataset.map(tokenize_function, batched=True)
tokenized_dataset = tokenized_dataset.remove_columns(["text"])
tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
tokenized_dataset.set_format("torch")

# Split into train and eval datasets
train_dataset = tokenized_dataset["train"]  # ~45,580 examples
eval_dataset = tokenized_dataset["test"]    # ~12,000 examples

# Step 3: Define a function to compute accuracy
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = (predictions == labels).mean()
    return {"accuracy": accuracy}

# Step 4: Set up training arguments
training_args = TrainingArguments(
    output_dir="./fine-tuned-sentiment-large",
    num_train_epochs=3,
    per_device_train_batch_size=4,  # Reduced for 8GB RAM
    per_device_eval_batch_size=4,   # Reduced for 8GB RAM
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=100,
    eval_strategy="epoch",          # Updated from evaluation_strategy
    save_strategy="epoch",
    learning_rate=2e-5,
    fp16=False,                     # Disabled (not supported on MPS)
    # Use MPS acceleration if available
    no_cuda=True,                   # Force no CUDA since M2 doesn't support it
    # torch.backends.mps.is_available() check is handled by device selection
)

# Step 5: Initialize and train the model
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
)

print("Starting training...")
trainer.train()

# Step 6: Save the fine-tuned model
model.save_pretrained("./fine-tuned-sentiment-large")
tokenizer.save_pretrained("./fine-tuned-sentiment-large")
print("Model saved to ./fine-tuned-sentiment-large")

# Step 7: Evaluate the model on the test set
eval_results = trainer.evaluate()
print(f"Evaluation results: {eval_results}")

# Step 8: Test on your specific examples
classifier = TextClassificationPipeline(
    model=AutoModelForSequenceClassification.from_pretrained("./fine-tuned-sentiment-large").to(device),
    tokenizer=AutoTokenizer.from_pretrained("./fine-tuned-sentiment-large"),
    device=0 if device.type == "mps" else -1,  # 0 for MPS, -1 for CPU
    return_all_scores=False
)

texts = ["Great service!", "It's okay, nothing special.", "Terrible experience."]
results = classifier(texts)

print("\nTesting on custom examples:")
for text, result in zip(texts, results):
    print(f"Text: {text} -> Sentiment: {result['label']}")