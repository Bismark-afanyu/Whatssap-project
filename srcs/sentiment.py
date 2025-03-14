from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
import torch

MODEL = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
ROBERTA_SUPPORTED_LANGUAGES = ('ar', 'en', 'fr', 'de', 'hi', 'it', 'es', 'pt')

# Load model, tokenizer, and config
model = AutoModelForSequenceClassification.from_pretrained(MODEL)
tokenizer = AutoTokenizer.from_pretrained(MODEL)
config = AutoConfig.from_pretrained(MODEL)

# Save the model locally
model.save_pretrained("./model")
tokenizer.save_pretrained("./model")

def preprocess(text):
    """Preprocess text by removing unwanted words and formatting mentions/URLs."""
    remove_words = ["<Media", "<Mediaomitted>"]
    for word in remove_words:
        text = text.replace(word, "")
    
    new_text = [
        '@user' if t.startswith('@') and len(t) > 1 else
        'http' if t.startswith('http') else
        t
        for t in text.split(" ")
    ]
    return " ".join(new_text)


def predict_sentiment(text: str) -> str:
    """Predict sentiment of the input text."""
    try:
        if not text or not isinstance(text, str):
            return "neutral"  # Default sentiment for invalid input
        
        # Preprocess the text
        processed_text = preprocess(text)
        
        # Tokenize the text with truncation
        encoded_input = tokenizer(
            processed_text,
            return_tensors='pt',
            max_length=512,  # Truncate to the model's max length
            truncation=True,
            padding=True
        )
        
        # Get model output
        with torch.no_grad():
            output = model(**encoded_input)
        
        # Get predicted sentiment
        index_of_sentiment = output.logits.argmax().item()
        sentiment = config.id2label[index_of_sentiment]
        
        return sentiment
    except Exception as e:
        print(f"Error predicting sentiment: {e}")
        return "neutral"  # Fallback sentiment