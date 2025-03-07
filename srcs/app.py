import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import preprocessor, helper 
import calendar


# Theme customization
st.set_page_config(page_title="WhatsApp Chat Analyzer", layout="wide")
st.markdown(
    """
    <style>
    .main {background-color: #f0f2f6;}
    </style>
    """,
    unsafe_allow_html=True
)

st.title("📊 WhatsApp Chat Sentiment Analysis Dashboard")
st.sidebar.title("Whatsapp Chat Analyzer")

uploaded_file = st.sidebar.file_uploader("Upload your chat file (.txt)", type="txt")
if uploaded_file is not None:
    raw_data = uploaded_file.read().decode("utf-8")
    df = preprocessor.preprocess(raw_data)

    # Sidebar filters
    st.sidebar.header("🔍 Filters")
    user_list = ["Overall"] + sorted(df["user"].unique().tolist())
    selected_user = st.sidebar.selectbox("Select User", user_list)

    # Filter data
    if selected_user != "Overall":
        df = df[df["user"] == selected_user]

    if st.sidebar.button("Show Analysis"):
        # Check if the filtered DataFrame is empty
        if df.empty:
            st.warning(f"No data found for user: {selected_user}")
        else:
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
            if not timeline.empty:
                fig, ax = plt.subplots()
                sns.lineplot(data=timeline, x='time', y='message', ax=ax, color='green')
                ax.set_title("Monthly Timeline")
                ax.set_xlabel("Time")
                ax.set_ylabel("Messages")
                plt.xticks(rotation='vertical')
                st.pyplot(fig)
            else:
                st.warning("No data available for the monthly timeline.")

            # Daily timeline
            st.title("Daily Timeline")
            daily_timeline = helper.daily_timeline(selected_user, df)
            if not daily_timeline.empty:
                fig, ax = plt.subplots()
                sns.lineplot(data=daily_timeline, x='date', y='message', ax=ax, color='black')
                ax.set_title("Daily Timeline")
                ax.set_xlabel("Date")
                ax.set_ylabel("Messages")
                plt.xticks(rotation='vertical')
                st.pyplot(fig)
            else:
                st.warning("No data available for the daily timeline.")

            # Activity map
            st.title('Activity Map')
            col1, col2 = st.columns(2)

            with col1:
                st.header("Most busy day")
                busy_day = helper.week_activity_map(selected_user, df)
                if not busy_day.empty:
                    fig, ax = plt.subplots()
                    sns.barplot(x=busy_day.index, y=busy_day.values, ax=ax, color='purple')
                    ax.set_title("Most Busy Day")
                    plt.xticks(rotation='vertical')
                    st.pyplot(fig)
                else:
                    st.warning("No data available for the most busy day.")

            with col2:
                st.header("Most busy month")
                busy_month = helper.month_activity_map(selected_user, df)
                if not busy_month.empty:
                    fig, ax = plt.subplots()
                    sns.barplot(x=busy_month.index, y=busy_month.values, ax=ax, color='orange')
                    ax.set_title("Most Busy Month")
                    plt.xticks(rotation='vertical')
                    st.pyplot(fig)
                else:
                    st.warning("No data available for the most busy month.")

            # Finding the busiest users in the group (Group level)
            if selected_user == 'Overall':
                st.title('Most Busy Users')
                x, new_df = helper.most_busy_users(df)
                if not x.empty:
                    fig, ax = plt.subplots()
                    col1, col2 = st.columns(2)

                    with col1:
                        sns.barplot(x=x.index, y=x.values, ax=ax, color='red')
                        ax.set_title("Most Busy Users")
                        plt.xticks(rotation='vertical')
                        st.pyplot(fig)
                    with col2:
                        st.dataframe(new_df)
                else:
                    st.warning("No data available for most busy users.")

            # WordCloud
            show_wordcloud = st.checkbox("Show Wordcloud")
            if show_wordcloud:
                st.title("Wordcloud")
                df_wc = helper.create_wordcloud(selected_user, df)
                if df_wc is not None:
                    fig, ax = plt.subplots()
                    ax.imshow(df_wc)
                    ax.axis("off")
                    st.pyplot(fig)
                else:
                    st.warning("No data available for the word cloud.")

            # Most common words
            most_common_df = helper.most_common_words(selected_user, df)
            if not most_common_df.empty:
                fig, ax = plt.subplots()
                sns.barplot(y=most_common_df[0], x=most_common_df[1], ax=ax)
                ax.set_title("Most Common Words")
                plt.xticks(rotation='vertical')
                st.pyplot(fig)
            else:
                st.warning("No data available for most common words.")

            # Emoji analysis
            emoji_df = helper.emoji_helper(selected_user, df)
            if not emoji_df.empty:
                st.title("Emoji Analysis")
                col1, col2 = st.columns(2)

                with col1:
                    st.dataframe(emoji_df)
                with col2:
                    fig, ax = plt.subplots()
                    ax.pie(emoji_df[1].head(), labels=emoji_df[0].head(), autopct="%0.2f")
                    ax.set_title("Emoji Distribution")
                    st.pyplot(fig)
            else:
                st.warning("No data available for emoji analysis.")

            # Convert month names to abbreviated format (e.g., "June" -> "Jun")
            month_map = {
                'January': 'Jan', 'February': 'Feb', 'March': 'Mar', 'April': 'Apr',
                'May': 'May', 'June': 'Jun', 'July': 'Jul', 'August': 'Aug',
                'September': 'Sep', 'October': 'Oct', 'November': 'Nov', 'December': 'Dec'
            }
            df['month'] = df['month'].map(month_map)

            # Group by month and sentiment
            monthly_sentiment = df.groupby(['month', 'sentiment']).size().unstack(fill_value=0)

            # Plotting: Histogram (Bar Chart) for each sentiment
            st.write("### Sentiment Count by Month (Histogram)")

            # Create a figure with subplots for each sentiment
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))

            # Plot Positive Sentiment
            axes[0].bar(monthly_sentiment.index, monthly_sentiment['positive'], color='green')
            axes[0].set_title('Positive Sentiment')
            axes[0].set_xlabel('Month')
            axes[0].set_ylabel('Count')

            # Plot Neutral Sentiment
            axes[1].bar(monthly_sentiment.index, monthly_sentiment['neutral'], color='blue')
            axes[1].set_title('Neutral Sentiment')
            axes[1].set_xlabel('Month')
            axes[1].set_ylabel('Count')

            # Plot Negative Sentiment
            axes[2].bar(monthly_sentiment.index, monthly_sentiment['negative'], color='red')
            axes[2].set_title('Negative Sentiment')
            axes[2].set_xlabel('Month')
            axes[2].set_ylabel('Count')

            # Display the plots in Streamlit
            st.pyplot(fig)

            # Count sentiments per day of the week
            sentiment_counts = df.groupby(['day_of_week', 'sentiment']).size().unstack(fill_value=0)

            # Sort days correctly
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            sentiment_counts = sentiment_counts.reindex(day_order)

            # Streamlit App
            st.title("Daily Sentiment Analysis")

            # Create a Matplotlib figure
            fig, ax = plt.subplots()
            sentiment_counts.plot(kind='bar', stacked=False, ax=ax)

            # Customize the plot
            ax.set_xlabel("Day of the Week")
            ax.set_ylabel("Count")
            ax.set_title("Sentiment Distribution per Day of the Week")
            ax.legend(title="Sentiment")

            # Display the plot in Streamlit
            st.pyplot(fig)

            # Count messages per user per sentiment
            sentiment_counts = df.groupby(['user', 'sentiment']).size().reset_index(name='Count')

            # Calculate total messages per sentiment
            total_per_sentiment = df['sentiment'].value_counts().to_dict()

            # Add percentage column
            sentiment_counts['Percentage'] = sentiment_counts.apply(
                lambda row: (row['Count'] / total_per_sentiment[row['sentiment']]) * 100, axis=1
            )

            # Separate tables for each sentiment
            positive_df = sentiment_counts[sentiment_counts['sentiment'] == 'positive'].sort_values(by='Count', ascending=False).head(10)
            neutral_df = sentiment_counts[sentiment_counts['sentiment'] == 'neutral'].sort_values(by='Count', ascending=False).head(10)
            negative_df = sentiment_counts[sentiment_counts['sentiment'] == 'negative'].sort_values(by='Count', ascending=False).head(10)

            # Streamlit App
            st.title("Sentiment Contribution Analysis")

            # Create three columns for side-by-side display
            col1, col2, col3 = st.columns(3)

            # Display Positive Table
            with col1:
                st.subheader("Top Positive Contributors")
                st.table(positive_df[['user', 'Count', 'Percentage']])

            # Display Neutral Table
            with col2:
                st.subheader("Top Neutral Contributors")
                st.table(neutral_df[['user', 'Count', 'Percentage']])

            # Display Negative Table
            with col3:
                st.subheader("Top Negative Contributors")
                st.table(negative_df[['user', 'Count', 'Percentage']])

            # Get text for each sentiment
            positive_text = " ".join(df[df['sentiment'] == 'positive']['message'])
            neutral_text = " ".join(df[df['sentiment'] == 'neutral']['message'])
            negative_text = " ".join(df[df['sentiment'] == 'negative']['message'])

            # # Create three columns for side-by-side display
            # col1, col2, col3 = st.columns(3)

            # # Display Positive Word Cloud
            # with col1:
            #     st.subheader("Positive Word Cloud")
            #     fig, ax = plt.subplots()
            #     ax.imshow(helper.generate_wordcloud(positive_text, "white"), interpolation='bilinear')
            #     ax.axis("off")
            #     st.pyplot(fig)

            # # Display Neutral Word Cloud
            # with col2:
            #     st.subheader("Neutral Word Cloud")
            #     fig, ax = plt.subplots()
            #     ax.imshow(helper.generate_wordcloud(neutral_text, "white"), interpolation='bilinear')
            #     ax.axis("off")
            #     st.pyplot(fig)

            # # Display Negative Word Cloud
            # with col3:
            #     st.subheader("Negative Word Cloud")
            #     fig, ax = plt.subplots()
            #     ax.imshow(helper.generate_wordcloud(negative_text, "black"), interpolation='bilinear')
            #     ax.axis("off")
            #     st.pyplot(fig) 

