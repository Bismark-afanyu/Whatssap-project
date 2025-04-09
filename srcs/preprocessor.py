import re
import pandas as pd
import spacy
from langdetect import detect_langs
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from spacy.lang.fr.stop_words import STOP_WORDS as FRENCH_STOP_WORDS
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
import streamlit as st

# Lighter model
MODEL ="cardiffnlp/twitter-xlm-roberta-base-sentiment"

# Cache model loading with fallback for quantization
@st.cache_resource
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL).to(device)
    
    # Attempt quantization with fallback
    try:
        # Set quantization engine explicitly (fbgemm for x86, qnnpack for ARM)
        torch.backends.quantized.engine = 'fbgemm' if torch.cuda.is_available() else 'qnnpack'
        model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
        print("Model quantized successfully.")
    except RuntimeError as e:
        print(f"Quantization failed: {e}. Using non-quantized model.")
    
    config = AutoConfig.from_pretrained(MODEL)
    return tokenizer, model, config, device

tokenizer, model, config, device = load_model()

nlp_fr = spacy.load("fr_core_news_sm")   
nlp_en = spacy.load("en_core_web_sm")
custom_stop_words = list(ENGLISH_STOP_WORDS.union(FRENCH_STOP_WORDS))

def preprocess(text):
    if text is None:
        return ""
    if not isinstance(text, str):
        try:
            text = str(text)
        except:
            return ""
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)

def clean_message(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = text.replace("<media omitted>", "").replace("this message was deleted", "").replace("null", "")
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r"[^a-zA-ZÀ-ÿ0-9\s]", "", text)
    return text.strip()

def lemmatize_text(text, lang):
    if lang == 'fr':
        doc = nlp_fr(text)
    else:
        doc = nlp_en(text)
    return " ".join([token.lemma_ for token in doc if not token.is_punct])

def preprocess(data):
    pattern = r"^(?P<Date>\d{1,2}/\d{1,2}/\d{2,4}),\s+(?P<Time>[\d:]+(?:\S*\s?[AP]M)?)\s+-\s+(?:(?P<Sender>.*?):\s+)?(?P<Message>.*)$"
    filtered_messages, valid_dates = [], []
    
    for line in data.strip().split("\n"):
        match = re.match(pattern, line)
        if match:
            entry = match.groupdict()
            sender = entry.get("Sender")
            if sender and sender.strip().lower() != "system":
                filtered_messages.append(f"{sender.strip()}: {entry['Message']}")
                valid_dates.append(f"{entry['Date']}, {entry['Time'].replace('â€¯', ' ')}")

    df = pd.DataFrame({'user_message': filtered_messages, 'message_date': valid_dates})
    df['message_date'] = pd.to_datetime(df['message_date'], format='%m/%d/%y, %I:%M %p', errors='coerce')
    df.rename(columns={'message_date': 'date'}, inplace=True)

    users, messages = [], []
    msg_pattern = r"^(.*?):\s(.*)$"
    for message in df["user_message"]:
        match = re.match(msg_pattern, message)
        if match:
            users.append(match.group(1))
            messages.append(match.group(2))
        else:
            users.append("group_notification")
            messages.append(message)

    df["user"] = users
    df["message"] = messages
    df = df[df["user"] != "group_notification"].reset_index(drop=True)
    df["unfiltered_messages"] = df["message"]
    df["message"] = df["message"].apply(clean_message)
    
    # Extract time-based features
    df['year'] = pd.to_numeric(df['date'].dt.year, downcast='integer')
    df['month'] = df['date'].dt.month_name()
    df['day'] = pd.to_numeric(df['date'].dt.day, downcast='integer')
    df['hour'] = pd.to_numeric(df['date'].dt.hour, downcast='integer')
    df['day_of_week'] = df['date'].dt.day_name()
    
    # Lemmatize messages for topic modeling
    lemmatized_messages = []
    for message in df["message"]:
        try:
            lang = detect_langs(message)
            lemmatized_messages.append(lemmatize_text(message, lang))
        except:
            lemmatized_messages.append("")
    df["lemmatized_message"] = lemmatized_messages
    
    df = df[df["message"].notnull() & (df["message"] != "")].copy()
    df.drop(columns=["user_message"], inplace=True)

    # Perform topic modeling
    vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words=custom_stop_words)
    dtm = vectorizer.fit_transform(df['lemmatized_message'])

    # Apply LDA
    lda = LatentDirichletAllocation(n_components=5, random_state=42)
    lda.fit(dtm)

    # Assign topics to messages
    topic_results = lda.transform(dtm)
    df = df.iloc[:topic_results.shape[0]].copy()
    df['topic'] = topic_results.argmax(axis=1)

    # Store topics for visualization
    topics = []
    for topic in lda.components_:
        topics.append([vectorizer.get_feature_names_out()[i] for i in topic.argsort()[-10:]])
    print("Top words for each topic-----------------------------------------------------:")
    print(topics)
    
    return df, topics

def preprocess_for_clustering(df, n_clusters=5):
    df = df[df["lemmatized_message"].notnull() & (df["lemmatized_message"].str.strip() != "")]
    df = df.reset_index(drop=True)

    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(df['lemmatized_message'])
    
    if tfidf_matrix.shape[0] < 2:
        raise ValueError("Not enough messages for clustering.")

    df = df.iloc[:tfidf_matrix.shape[0]].copy()

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(tfidf_matrix)
    
    df['cluster'] = clusters
    tsne = TSNE(n_components=2, random_state=42)
    reduced_features = tsne.fit_transform(tfidf_matrix.toarray())
    
    return df, reduced_features, kmeans.cluster_centers_


def predict_sentiment_batch(texts: list, batch_size: int = 32) -> list:
    """Predict sentiment for a batch of texts"""
    if not isinstance(texts, list):
        raise TypeError(f"Expected list of texts, got {type(texts)}")
    
    processed_texts = [preprocess(text) for text in texts]
    
    predictions = []
    for i in range(0, len(processed_texts), batch_size):
        batch = processed_texts[i:i+batch_size]
        
        inputs = tokenizer(
            batch, 
            padding=True, 
            truncation=True, 
            return_tensors="pt", 
            max_length=128
        ).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        batch_preds = outputs.logits.argmax(dim=1).cpu().numpy()
        predictions.extend([config.id2label[p] for p in batch_preds])
   
    return predictions