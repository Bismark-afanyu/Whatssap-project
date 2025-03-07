from urlextract import URLExtract
from wordcloud import WordCloud
import pandas as pd
from collections import Counter
import emoji
import matplotlib.pyplot as plt
import seaborn as sns

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
    num_media_messages = df[df['message'] == '<Media omitted>\n'].shape[0]

    # fetch number of links shared
    links = []
    for message in df['message']:
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
    for message in df['message']:
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

def plot_topics(topics):
    """
    Plots a bar chart for the top words in each topic.
    """
    fig, axes = plt.subplots(1, len(topics), figsize=(20, 10))
    if len(topics) == 1:
        axes = [axes]  # Ensure axes is iterable for single topic

    for idx, topic in enumerate(topics):
        top_words = topic
        axes[idx].barh(top_words, range(len(top_words)))
        axes[idx].set_title(f"Topic {idx}")
        axes[idx].set_xlabel("Word Importance")
        axes[idx].set_ylabel("Top Words")

    plt.tight_layout()
    return fig
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













