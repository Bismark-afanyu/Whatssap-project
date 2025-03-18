import nltk
import string
import re
import pandas as pd
import numpy as np
import joblib
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from googletrans import Translator
from imblearn.over_sampling import SMOTE

nltk.download('stopwords')
nltk.download('punkt')

translator = Translator()

# Load dataset
data = pd.read_csv('/Users/caasidev/development/AI/datasets/train.csv', encoding='ISO-8859-1')

# Drop missing values
data = data.dropna(subset=['text', 'sentiment'])

stop_words = set(stopwords.words('english') + stopwords.words('french'))

# Function to clean text
def clean_text(text):
    if isinstance(text, float):
        return ""
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", "", text)
    text = " ".join([word for word in text.split() if word not in stop_words])
    return text

# Apply text cleaning
data['Cleaned_Text'] = data['text'].apply(clean_text)

# **Vectorization BEFORE SMOTE**
vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_df=0.85, min_df=2, max_features=10000)
X_tfidf = vectorizer.fit_transform(data['Cleaned_Text'])
y = data['sentiment']

# Apply SMOTE **after** vectorization
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_tfidf, y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Train Naive Bayes
model = MultinomialNB(alpha=0.5)
model.fit(X_train, y_train)

# Save model and vectorizer
joblib.dump(model, "naive_bayes_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
print("Model and vectorizer saved successfully!")

# Predictions
y_pred = model.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
print(f"Improved Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:\n", classification_report(y_test, y_pred))
