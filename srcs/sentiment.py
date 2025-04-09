import pandas as pd
import time
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig

# Use a sentiment-specific model (replace with TinyBERT if fine-tuned)
MODEL = "cardiffnlp/twitter-xlm-roberta-base-sentiment"  # Pre-trained for positive/negative sentiment

print("Loading model and tokenizer...")
start_load = time.time()

# Check for MPS (Metal) availability on M2 chip, fallback to CPU
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

# Load with optimizations (only once, removing redundancy)
tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
model = AutoModelForSequenceClassification.from_pretrained(MODEL).to(device)
config = AutoConfig.from_pretrained(MODEL)

load_time = time.time() - start_load
print(f"Model and tokenizer loaded in {load_time:.2f} seconds\n")

# Optimized preprocessing (unchanged from your code)
def preprocess(text):
    if not isinstance(text, str):
        text = str(text) if not pd.isna(text) else ""
    
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)

# Batch prediction function (optimized for performance)
def predict_sentiment_batch(texts: list, batch_size: int = 16) -> list:
    if not isinstance(texts, list):
        raise TypeError(f"Expected list of texts, got {type(texts)}")
    
    # Validate and clean inputs
    valid_texts = [str(text) for text in texts if isinstance(text, str) and text.strip()]
    if not valid_texts:
        return []  # Return empty list if no valid texts
    
    print(f"Processing {len(valid_texts)} valid samples...")
    processed_texts = [preprocess(text) for text in valid_texts]
    
    predictions = []
    for i in range(0, len(processed_texts), batch_size):
        batch = processed_texts[i:i + batch_size]
        try:
            inputs = tokenizer(
                batch,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=64  # Reduced for speed on short texts like tweets
            ).to(device)
            
            with torch.no_grad():
                outputs = model(**inputs)
            
            batch_preds = outputs.logits.argmax(dim=1).cpu().numpy()
            predictions.extend([config.id2label[p] for p in batch_preds])
        except Exception as e:
            print(f"Error processing batch {i // batch_size}: {str(e)}")
            predictions.extend(["neutral"] * len(batch))  # Consider logging instead
        
    print(f"Predictions for {len(valid_texts)} samples generated in {time.time() - start_load:.2f} seconds")
    predictions = [prediction.lower() for prediction in predictions]

    print(predictions)
    
    return predictions

# # Example usage with your dataset (uncomment and adjust paths)
# test_data = pd.read_csv("/Users/caasidev/development/AI/AI_Spectrum/2024/Multi-lingual_sentiment_analysis/tweets.csv")
# print(f"Processing {len(test_data)} samples...")
# start_prediction = time.time()

# text_samples = test_data['text'].tolist()
# test_data['predicted_sentiment'] = predict_sentiment_batch(text_samples)

# prediction_time = time.time() - start_prediction
# time_per_sample = prediction_time / len(test_data)

# # Print runtime statistics
# print("\nRuntime Statistics:")
# print(f"- Model loading time: {load_time:.2f} seconds")
# print(f"- Total prediction time for {len(test_data)} samples: {prediction_time:.2f} seconds")
# print(f"- Average time per sample: {time_per_sample:.4f} seconds")
# print(f"- Estimated time for 1000 samples: {(time_per_sample * 1000):.2f} seconds")
# print(f"- Estimated time for 20000 samples: {(time_per_sample * 20000 / 60):.2f} minutes")

# # Print a sample of predictions
# print("\nPredicted Sentiments (first 5 samples):")
# print(test_data[['text', 'predicted_sentiment']].head())