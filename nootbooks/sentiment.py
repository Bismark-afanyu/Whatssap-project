from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
import torch

MODEL = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
ROBERTA_SUPPORTED_LANGUAGES = ('ar', 'en', 'fr', 'de', 'hi', 'it', 'es', 'pt')

# Load model, tokenizer, and config
model = AutoModelForSequenceClassification.from_pretrained(MODEL)
tokenizer = AutoTokenizer.from_pretrained(MODEL)
config = AutoConfig.from_pretrained(MODEL)
# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

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


def predict_sentiment(texts, batch_size=32):
    """Predict sentiment for a batch of texts."""
    sentiments = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        processed_texts = [preprocess(text) for text in batch_texts]

        # Tokenize batch
        encoded_inputs = tokenizer(
            processed_texts,
            return_tensors='pt',
            max_length=512,
            truncation=True,
            padding=True
        ).to(device)  # Move to GPU if available
        
        # Get model predictions
        with torch.no_grad():
            outputs = model(**encoded_inputs)
        
        # Convert predictions to labels
        batch_sentiments = [config.id2label[idx] for idx in outputs.logits.argmax(dim=1).tolist()]
        sentiments.extend(batch_sentiments)
    
    return sentiments
    