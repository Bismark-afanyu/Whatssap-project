import streamlit as st
import  preprocessor, helper
import matplotlib.pyplot as plt
import seaborn as sns

# Theme customization
st.set_page_config(page_title="Whatsapp Chat Analyzer", layout="wide")
st.markdown(
    """
    <style>
    .main {background-color: #f0f2f6;}
    </style>
    """,
    unsafe_allow_html=True
)

st.sidebar.title("Whatsapp Chat Analyzer")

uploaded_file = st.sidebar.file_uploader("Choose a file")
if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    data = bytes_data.decode("utf-8")
    df = preprocessor.preprocess(data)

    # fetch unique users
    user_list = df['user'].unique().tolist()
    user_list.sort()
    user_list.insert(0, "Overall")

    selected_user = st.sidebar.selectbox("Show analysis wrt", user_list)

    if st.sidebar.button("Show Analysis"):

        # Stats Area
        num_messages, words, num_media_messages, num_links = helper.fetch_stats(selected_user, df)
        st.title("Top Statistics")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.header("Total Messages")
            st.title(num_messages)
        with col2:
            st.header("Total Words")
            st.title(words)
        with col3:
            st.header("Media Shared")
            st.title(num_media_messages)
        with col4:
            st.header("Links Shared")
            st.title(num_links)

        # Monthly timeline
        st.title("Monthly Timeline")
        timeline = helper.monthly_timeline(selected_user, df)
        fig, ax = plt.subplots()
        sns.lineplot(data=timeline, x='time', y='message', ax=ax, color='green')
        ax.set_title("Monthly Timeline")
        ax.set_xlabel("Time")
        ax.set_ylabel("Messages")
        plt.xticks(rotation='vertical')
        st.pyplot(fig)

        # Daily timeline
        st.title("Daily Timeline")
        daily_timeline = helper.daily_timeline(selected_user, df)
        fig, ax = plt.subplots()
        sns.lineplot(data=daily_timeline, x='date', y='message', ax=ax, color='black')
        ax.set_title("Daily Timeline")
        ax.set_xlabel("Date")
        ax.set_ylabel("Messages")
        plt.xticks(rotation='vertical')
        st.pyplot(fig)

        # Activity map
        st.title('Activity Map')
        col1, col2 = st.columns(2)

        with col1:
            st.header("Most busy day")
            busy_day = helper.week_activity_map(selected_user, df)
            fig, ax = plt.subplots()
            sns.barplot(x=busy_day.index, y=busy_day.values, ax=ax, color='purple')
            ax.set_title("Most Busy Day")
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        with col2:
            st.header("Most busy month")
            busy_month = helper.month_activity_map(selected_user, df)
            fig, ax = plt.subplots()
            sns.barplot(x=busy_month.index, y=busy_month.values, ax=ax, color='orange')
            ax.set_title("Most Busy Month")
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        # Finding the busiest users in the group (Group level)
        if selected_user == 'Overall':
            st.title('Most Busy Users')
            x, new_df = helper.most_busy_users(df)
            fig, ax = plt.subplots()

            col1, col2 = st.columns(2)

            with col1:
                sns.barplot(x=x.index, y=x.values, ax=ax, color='red')
                ax.set_title("Most Busy Users")
                plt.xticks(rotation='vertical')
                st.pyplot(fig)
            with col2:
                st.dataframe(new_df)

        # WordCloud
        show_wordcloud = st.checkbox("Show Wordcloud")
        if show_wordcloud:
            st.title("Wordcloud")
            df_wc = helper.create_wordcloud(selected_user, df)
            fig, ax = plt.subplots()
            ax.imshow(df_wc)
            st.pyplot(fig)

        # Most common words
        most_common_df = helper.most_common_words(selected_user, df)
        fig, ax = plt.subplots()
        sns.barplot(y=most_common_df[0], x=most_common_df[1], ax=ax)
        ax.set_title("Most Common Words")
        plt.xticks(rotation='vertical')
        st.pyplot(fig)

        # Emoji analysis
        emoji_df = helper.emoji_helper(selected_user, df)
        st.title("Emoji Analysis")

        col1, col2 = st.columns(2)

        with col1:
            st.dataframe(emoji_df)
        with col2:
            fig, ax = plt.subplots()
            ax.pie(emoji_df[1].head(), labels=emoji_df[0].head(), autopct="%0.2f")
            ax.set_title("Emoji Distribution")
            st.pyplot(fig)
