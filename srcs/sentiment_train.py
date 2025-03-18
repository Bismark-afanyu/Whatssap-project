import joblib
import string
import re
import nltk
from nltk.corpus import stopwords

# Load saved model and vectorizer
model = joblib.load("naive_bayes_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Load stopwords
nltk.download("stopwords")
stop_words = set(stopwords.words("english") + stopwords.words("french"))

# Function to clean text (must match preprocessing in training script)
def clean_text(text):
    if isinstance(text, float):
        return ""
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", "", text)
    text = " ".join([word for word in text.split() if word not in stop_words])
    return text

# Function to predict sentiment
def predict_sentiment(text):
    cleaned_text = clean_text(text)
    vectorized_text = vectorizer.transform([cleaned_text])
    prediction = model.predict(vectorized_text)
    return prediction[0]
