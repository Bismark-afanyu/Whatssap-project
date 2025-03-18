from urlextract import URLExtract
from wordcloud import WordCloud
import pandas as pd
from collections import Counter
import emoji
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import plotly.express as px


extract = URLExtract()

def fetch_stats(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    # fetch the number of messages
    num_messages = df.shape[0]

    # fetch the total number of words
    words = []
    for message in df['message']:
        words.extend(message.split())

    # fetch number of media messages
    num_media_messages = df[df['unfiltered_messages'] == '<media omitted>\n'].shape[0]

    # fetch number of links shared
    links = []
    for message in df['unfiltered_messages']:
        links.extend(extract.find_urls(message))

    return num_messages,len(words),num_media_messages,len(links)

def most_busy_users(df):
    x = df['user'].value_counts().head()
    df = round((df['user'].value_counts() / df.shape[0]) * 100, 2).reset_index().rename(
        columns={'index': 'percentage', 'user': 'Name'})
    return x,df

def create_wordcloud(selected_user, df):
    # f = open('stop_hinglish.txt', 'r')
    stop_words = df

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    temp = df[df['user'] != 'group_notification']
    temp = temp[~temp['message'].str.lower().str.contains('<media omitted>')]

    def remove_stop_words(message):
        y = []
        for word in message.lower().split():
            if word not in stop_words:
                y.append(word)
        return " ".join(y)

    wc = WordCloud(width=500, height=500, min_font_size=10, background_color='white')
    temp['message'] = temp['message'].apply(remove_stop_words)
    df_wc = wc.generate(temp['message'].str.cat(sep=" "))
    return df_wc

def most_common_words(selected_user, df):
    # f = open('stop_hinglish.txt','r')
    stop_words = df

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    temp = df[df['user'] != 'group_notification']
    temp = temp[~temp['message'].str.lower().str.contains('<media omitted>')]

    words = []

    for message in temp['message']:
        for word in message.lower().split():
            if word not in stop_words:
                words.append(word)

    most_common_df = pd.DataFrame(Counter(words).most_common(20))
    return most_common_df

def emoji_helper(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    emojis = []
    for message in df['unfiltered_messages']:
        emojis.extend([c for c in message if c in emoji.EMOJI_DATA])

    emoji_df = pd.DataFrame(Counter(emojis).most_common(len(Counter(emojis))))

    return emoji_df


def monthly_timeline(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    timeline = df.groupby(['year','month']).count()['message'].reset_index()

    time = []
    for i in range(timeline.shape[0]):
        time.append(timeline['month'][i] + "-" + str(timeline['year'][i]))

    timeline['time'] = time

    return timeline

def daily_timeline(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    daily_timeline = df.groupby('date').count()['message'].reset_index()

    return daily_timeline

def week_activity_map(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    return df['day'].value_counts()

def month_activity_map(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    return df['month'].value_counts()

def activity_heatmap(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    user_heatmap = df.pivot_table(index='day', columns='period', values='message', aggfunc='count').fillna(0)

    return user_heatmap
def generate_wordcloud(text, color):
    wordcloud = WordCloud(width=400, height=300, background_color=color, colormap="viridis").generate(text)
    return wordcloud

# def plot_topics(topics):
#     """
#     Plots a bar chart for the top words in each topic.
#     """
#     if not topics or not isinstance(topics[0], list):
#         raise ValueError("topics must be a list of lists of words.")

#     print("Topics received:", topics)  # Debugging
#     fig, axes = plt.subplots(1, len(topics), figsize=(20, 10))
#     if len(topics) == 1:
#         axes = [axes]  # Ensure axes is iterable for single topic

#     for idx, topic in enumerate(topics):
#         if not isinstance(topic, list):
#             raise ValueError(f"Topic {idx} is not a list of words.")

#         top_words = topic
#         print(f"Top words for Topic {idx}: {top_words}")  # Debugging
#         axes[idx].barh(top_words, range(len(top_words)))
#         axes[idx].set_title(f"Topic {idx}")
#         axes[idx].set_xlabel("Word Importance")
#         axes[idx].set_ylabel("Top Words")

#     plt.tight_layout()
#     return fig
def plot_topic_distribution(df):
    """
    Plots the distribution of topics in the chat data.
    """
    topic_counts = df['topic'].value_counts().sort_index()  
    fig, ax = plt.subplots()
    sns.barplot(x=topic_counts.index, y=topic_counts.values, ax=ax, palette="viridis")
    ax.set_title("Topic Distribution")
    ax.set_xlabel("Topic")
    ax.set_ylabel("Number of Messages")
    return fig

def most_frequent_keywords(messages, top_n=10):
    """
    Extracts the most frequent keywords from a list of messages.
    """
    words = [word for msg in messages for word in msg.split()]
    word_freq = Counter(words)
    return word_freq.most_common(top_n)
def plot_topic_distribution_over_time(topic_distribution):
    """
    Plots the distribution of topics over time using a line chart.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot each topic as a separate line
    for topic in topic_distribution.columns:
        ax.plot(topic_distribution.index.to_timestamp(), topic_distribution[topic], label=f"Topic {topic}")
    
    ax.set_title("Topic Distribution Over Time")
    ax.set_xlabel("Time Period")
    ax.set_ylabel("Number of Messages")
    ax.legend(title="Topics", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig

def plot_most_frequent_keywords(keywords):
    """
    Plots the most frequent keywords.
    """
    words, counts = zip(*keywords)
    fig, ax = plt.subplots()
    sns.barplot(x=list(counts), y=list(words), ax=ax, palette="viridis")
    ax.set_title("Most Frequent Keywords")
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Keyword")
    return fig
def topic_distribution_over_time(df, time_freq='M'):
    """
    Analyzes the distribution of topics over time.
    """
    # Group by time interval and topic
    df['time_period'] = df['date'].dt.to_period(time_freq)
    topic_distribution = df.groupby(['time_period', 'topic']).size().unstack(fill_value=0)
    return topic_distribution

def plot_topic_distribution_over_time(topic_distribution):
    """
    Plots the distribution of topics over time using a line chart.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot each topic as a separate line
    for topic in topic_distribution.columns:
        ax.plot(topic_distribution.index.to_timestamp(), topic_distribution[topic], label=f"Topic {topic}")
    
    ax.set_title("Topic Distribution Over Time")
    ax.set_xlabel("Time Period")
    ax.set_ylabel("Number of Messages")
    ax.legend(title="Topics", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig

def plot_topic_distribution_over_time_plotly(topic_distribution):
    """
    Plots the distribution of topics over time using Plotly.
    """
    topic_distribution = topic_distribution.reset_index()
    topic_distribution['time_period'] = topic_distribution['time_period'].dt.to_timestamp()
    topic_distribution = topic_distribution.melt(id_vars='time_period', var_name='topic', value_name='count')
    
    fig = px.line(topic_distribution, x='time_period', y='count', color='topic', 
                  title="Topic Distribution Over Time", labels={'time_period': 'Time Period', 'count': 'Number of Messages'})
    fig.update_layout(legend_title_text='Topics', xaxis_tickangle=-45)
    return fig
def plot_clusters(reduced_features, clusters):
    """
    Visualize clusters using t-SNE.
    Args:
        reduced_features (np.array): 2D array of reduced features.
        clusters (np.array): Cluster labels.
    Returns:
        fig (plt.Figure): Matplotlib figure object.
    """
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x=reduced_features[:, 0],
        y=reduced_features[:, 1],
        hue=clusters,
        palette="viridis",
        legend="full"
    )
    plt.title("Message Clusters (t-SNE Visualization)")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.tight_layout()
    return plt.gcf()
def get_cluster_labels(df, n_clusters):
    """
    Generate descriptive labels for each cluster based on top keywords.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    import numpy as np

    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(df['lemmatized_message'])

    cluster_labels = {}
    for cluster_id in range(n_clusters):
        cluster_indices = df[df['cluster'] == cluster_id].index
        if len(cluster_indices) > 0:
            cluster_tfidf = tfidf_matrix[cluster_indices]
            top_keywords = np.argsort(cluster_tfidf.sum(axis=0).A1)[-3:][::-1]
            cluster_labels[cluster_id] = ", ".join(vectorizer.get_feature_names_out()[top_keywords])
        else:
            cluster_labels[cluster_id] = "No dominant theme"
    return cluster_labels

def get_temporal_trends(df):
    """
    Analyze temporal trends for each cluster (peak day and time).
    """
    temporal_trends = {}
    for cluster_id in df['cluster'].unique():
        cluster_data = df[df['cluster'] == cluster_id]
        if not cluster_data.empty:
            peak_day = cluster_data['day_of_week'].mode()[0]
            peak_time = cluster_data['hour'].mode()[0]
            temporal_trends[cluster_id] = {"peak_day": peak_day, "peak_time": f"{peak_time}:00"}
    return temporal_trends

def get_user_contributions(df):
    """
    Identify top contributors for each cluster.
    """
    user_contributions = {}
    for cluster_id in df['cluster'].unique():
        cluster_data = df[df['cluster'] == cluster_id]
        if not cluster_data.empty:
            top_users = cluster_data['user'].value_counts().head(3).index.tolist()
            user_contributions[cluster_id] = top_users
    return user_contributions

def get_sentiment_by_cluster(df):
    """
    Analyze sentiment distribution for each cluster.
    """
    sentiment_by_cluster = {}
    for cluster_id in df['cluster'].unique():
        cluster_data = df[df['cluster'] == cluster_id]
        if not cluster_data.empty:
            sentiment_counts = cluster_data['sentiment'].value_counts(normalize=True) * 100
            sentiment_by_cluster[cluster_id] = {
                "positive": round(sentiment_counts.get('positive', 0)),
                "neutral": round(sentiment_counts.get('neutral', 0)),
                "negative": round(sentiment_counts.get('negative', 0))
            }
    return sentiment_by_cluster

def detect_anomalies(df):
    """
    Detect anomalies in each cluster (e.g., high link or media share).
    """
    anomalies = {}
    for cluster_id in df['cluster'].unique():
        cluster_data = df[df['cluster'] == cluster_id]
        if not cluster_data.empty:
            link_share = (cluster_data['message'].str.contains('http').mean()) * 100
            media_share = (cluster_data['message'].str.contains('<media omitted>').mean()) * 100
            if link_share > 50:
                anomalies[cluster_id] = f"{round(link_share)}% of messages contain links."
            elif media_share > 50:
                anomalies[cluster_id] = f"{round(media_share)}% of messages are media files."
    return anomalies

def generate_recommendations(df):
    """
    Generate actionable recommendations based on cluster insights.
    """
    recommendations = []
    for cluster_id in df['cluster'].unique():
        cluster_data = df[df['cluster'] == cluster_id]
        if not cluster_data.empty:
            sentiment_counts = cluster_data['sentiment'].value_counts(normalize=True) * 100
            if sentiment_counts.get('negative', 0) > 50:
                recommendations.append(f"Address negative sentiment in Cluster {cluster_id} by revisiting feedback processes.")
            if cluster_data['message'].str.contains('http').mean() > 0.5:
                recommendations.append(f"Pin resources from Cluster {cluster_id} (most-shared links) for easy access.")
    return recommendations