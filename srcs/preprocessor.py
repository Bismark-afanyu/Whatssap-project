import re
import pandas as pd
from sentiment import predict_sentiment
import spacy
from langdetect import detect, LangDetectException
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from spacy.lang.fr.stop_words import STOP_WORDS as FRENCH_STOP_WORDS

# Load language models
nlp_fr = spacy.load("fr_core_news_sm")
nlp_en = spacy.load("en_core_web_sm")

# Merge English and French stop words
custom_stop_words = list(ENGLISH_STOP_WORDS.union(FRENCH_STOP_WORDS))

def lemmatize_text(text, lang):
    if lang == 'fr':
        doc = nlp_fr(text)
    else:
        doc = nlp_en(text)
    return " ".join([token.lemma_ for token in doc if not token.is_punct])

def clean_message(text):
    """ Remove media notifications, special characters, and unwanted symbols. """
    if not isinstance(text, str):
        return ""
    text = text.lower()  # Convert to lowercase
    text = re.sub(r"<media omitted>", "", text)  # Remove media notifications
    text = re.sub(r"this message was deleted", "", text)
    text = re.sub(r"[^a-zA-ZÀ-ÿ0-9\s]", "", text)  # Remove special characters
    return text

def preprocess(data):
    pattern = r"^(?P<Date>\d{1,2}/\d{1,2}/\d{2,4}),\s+(?P<Time>[\d:]+(?:\S*\s?[AP]M)?)\s+-\s+(?:(?P<Sender>.*?):\s+)?(?P<Message>.*)$"

    filtered_messages = []
    valid_dates = []

    for line in data.strip().split("\n"):
        match = re.match(pattern, line)
        if match:
            entry = match.groupdict()
            sender = entry.get("Sender")
            if sender and sender.strip().lower() != "system":  # Remove system messages
                filtered_messages.append(f"{sender.strip()}: {entry['Message']}")
                valid_dates.append(f"{entry['Date']}, {entry['Time'].replace('â€¯', ' ')}")

    # Create DataFrame
    df = pd.DataFrame({'user_message': filtered_messages, 'message_date': valid_dates})
    df['message_date'] = pd.to_datetime(df['message_date'], format='%m/%d/%y, %I:%M %p', errors='coerce')
    df.rename(columns={'message_date': 'date'}, inplace=True)

    # Separate Users and Messages
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
    df = df[df["user"] != "group_notification"]
    df.reset_index(drop=True, inplace=True)

    # Clean messages
    df["message"] = df["message"].apply(clean_message)

    # Filter and lemmatize messages
    lemmatized_messages = []
    for message in df["message"]:
        try:
            lang = detect(message)
            lemmatized_messages.append(lemmatize_text(message, lang))
        except LangDetectException:
            lemmatized_messages.append("")

    df["lemmatized_message"] = lemmatized_messages

    # Drop original column
    df.drop(columns=["user_message"], inplace=True)

    # Extract time-based features
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month_name()
    df['day'] = df['date'].dt.day
    df['hour'] = df['date'].dt.hour
    df['day_of_week'] = df['date'].dt.day_name()
    df['minute'] = df['date'].dt.minute

    # Apply sentiment analysis
    df['sentiment'] = df["message"].apply(predict_sentiment)

    # Filter out rows with null lemmatized_message
    df = df.dropna(subset=['lemmatized_message'])

    # **Fix: Use a custom stop word list**
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

    return df, topics
