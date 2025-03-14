from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
import torch

MODEL = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
ROBERTA_SUPPORTED_LANGUAGES = ('ar', 'en', 'fr', 'de', 'hi', 'it', 'es', 'pt')

# Load the model, tokenizer, and config
model = AutoModelForSequenceClassification.from_pretrained(MODEL)
tokenizer = AutoTokenizer.from_pretrained(MODEL)
config = AutoConfig.from_pretrained(MODEL)

# Save the model locally (optional)
model.save_pretrained(MODEL)
tokenizer.save_pretrained(MODEL)

def preprocess(text):
    remove_words = ["<Media", "<Mediaomitted>"]
    # Remove unwanted words
    for word in remove_words:
        text = text.replace(word, "")
    
    # Process each word in the text
    new_text = [
        '@user' if t.startswith('@') and len(t) > 1 else
        'http' if t.startswith('http') else
        t
        for t in text.split(" ")
    ]
    return " ".join(new_text)

def predict_sentiment(text: str) -> str:
    # Preprocess the text
    processed_text = preprocess(text)
    
    # Tokenize the input
    encoded_input = tokenizer(processed_text, return_tensors='pt', padding=True, truncation=True)
    
    # Get model predictions
    with torch.no_grad():
        output = model(**encoded_input)
    
    # Get the predicted sentiment
    index_of_sentiment = output.logits.argmax().item()
    sentiment = config.id2label[index_of_sentiment]
    
    return sentiment

# Example usage
# text = "la pizza da @michele è veramente buona https://www.youtube.com"
# print(predict_sentiment(text))