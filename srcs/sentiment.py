import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig

MODEL = "cardiffnlp/twitter-xlm-roberta-base-sentiment"

# Load the model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained(MODEL)
tokenizer = AutoTokenizer.from_pretrained(MODEL)
config = AutoConfig.from_pretrained(MODEL)

# Load the test data and labels
test_data = pd.read_csv("/Users/caasidev/development/AI/AI_Spectrum/2024/Multi-lingual_sentiment_analysis/tweets.csv")
test_labels = pd.read_csv("/Users/caasidev/development/AI/AI_Spectrum/2024/Multi-lingual_sentiment_analysis/test_labels.csv")

# Ensure the test data has the same length as the labels
assert len(test_data) == len(test_labels), "Test data and labels must have the same length."

# Preprocess text (username and link placeholders)
def preprocess(text):
    if not isinstance(text, str):  # Ensure text is a string
        text = str(text) if not pd.isna(text) else ""  # Convert to string or replace NaN with an empty string
    
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    
    return " ".join(new_text)

def predict_sentiment_bert(text: str) -> str:
    processed_text = preprocess(text)
    encoded_input = tokenizer(processed_text, return_tensors='pt')
    output = model(**encoded_input)
    index_of_sentiment = output.logits.argmax().item()
    sentiment = config.id2label[index_of_sentiment]
    return sentiment

# Predict sentiments for the test data
test_data['predicted_sentiment'] = test_data['text'].apply(predict_sentiment_bert)

# Calculate accuracy
accuracy = accuracy_score(test_labels['label'], test_data['predicted_sentiment'])
print(f"Accuracy: {accuracy:.2f}")

# Generate a classification report
report = classification_report(test_labels['label'], test_data['predicted_sentiment'])
print("Classification Report:\n", report)
