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
st.subheader('Instructions')
st.markdown("1. Open the side bar and upload your WhatsApp chat file in .txt format.")
st.markdown("2. Wait for it to load")
st.markdown("3. Once the data is loaded, you can customize the analysis by selecting specific users or filtering the data.")
st.markdown("4. Click on the 'Show Analysis' button to update the analysis with your selected filters.")

st.sidebar.title("Whatsapp Chat Analyzer")

uploaded_file = st.sidebar.file_uploader("Upload your chat file (.txt)", type="txt")
if uploaded_file is not None:
    raw_data = uploaded_file.read().decode("utf-8")
    df, topics = preprocessor.preprocess(raw_data)

    # Sidebar filters
    st.sidebar.header("🔍 Filters")
    user_list = ["Overall"] + sorted(df["user"].unique().tolist())
    selected_user = st.sidebar.selectbox("Select User", user_list)

    # Filter data based on selected user
    df_filtered = df if selected_user == "Overall" else df[df["user"] == selected_user]

    # Clustering Section
    if st.sidebar.checkbox("Show Message Clustering"):
        st.title("Message Clustering Analysis")

    # Show Analysis button
    if st.sidebar.button("Show Analysis"):
        if df_filtered.empty:
            st.warning(f"No data found for user: {selected_user}")
        else:
            # Stats Area
            num_messages, words, num_media_messages, num_links = helper.fetch_stats(selected_user, df_filtered)
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
            timeline = helper.monthly_timeline(selected_user, df_filtered)
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
            daily_timeline = helper.daily_timeline(selected_user, df_filtered)
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
                st.header("Most Busy Day")
                busy_day = helper.week_activity_map(selected_user, df_filtered)
                if not busy_day.empty:
                    fig, ax = plt.subplots()
                    sns.barplot(x=busy_day.index, y=busy_day.values, ax=ax, color='purple')
                    ax.set_title("Most Busy Day")
                    plt.xticks(rotation='vertical')
                    st.pyplot(fig)
                else:
                    st.warning("No data available for the most busy day.")

            with col2:
                st.header("Most Busy Month")
                busy_month = helper.month_activity_map(selected_user, df_filtered)
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
                x, new_df = helper.most_busy_users(df_filtered)
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
                df_wc = helper.create_wordcloud(selected_user, df_filtered)
                if df_wc is not None:
                    fig, ax = plt.subplots()
                    ax.imshow(df_wc)
                    ax.axis("off")
                    st.pyplot(fig)
                else:
                    st.warning("No data available for the word cloud.")

            # Most common words
            most_common_df = helper.most_common_words(selected_user, df_filtered)
            if not most_common_df.empty:
                fig, ax = plt.subplots()
                sns.barplot(y=most_common_df[0], x=most_common_df[1], ax=ax)
                ax.set_title("Most Common Words")
                plt.xticks(rotation='vertical')
                st.pyplot(fig)
            else:
                st.warning("No data available for most common words.")

            # Emoji analysis
            emoji_df = helper.emoji_helper(selected_user, df_filtered)
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

        # if len(topics) > 0:  # Check if topics is not empty
        #     st.title("Topic Analysis")
        #     fig = helper.plot_topics(topics)
        #     st.pyplot(fig)
        # else:
        #     st.warning("No topics found for visualization.")

        # Area of Focus: Topic Analysis
        st.title("Area of Focus: Topic Analysis")

        # Plot Topic Distribution
        st.header("Topic Distribution")
        fig = helper.plot_topic_distribution(df)
        st.pyplot(fig)

        # Display Top Words for Each Topic
        # st.header("Top Words for Each Topic")
        # for idx, topic in enumerate(topics):
        #     st.subheader(f"Topic {idx}")
        #     st.write(", ".join(topic))  # Display top 10 words for the topic

        # Display Sample Messages for Each Topic
        st.header("Sample Messages for Each Topic")
        for topic_id in df['topic'].unique():
            st.subheader(f"Topic {topic_id}")

            # Get messages for the current topic
            filtered_messages = df[df['topic'] == topic_id]['message']

            # Determine sample size (no more than available messages)
            sample_size = min(5, len(filtered_messages))

            if sample_size > 0:
                sample_messages = filtered_messages.sample(sample_size, replace=False).tolist()
                for msg in sample_messages:
                    st.write(f"- {msg}")
            else:
                st.write("No messages available for this topic.")

        # Topic Distribution Over Time
        st.title("What They Most Talk About")
        st.header("Topic Distribution Over Time")

        # Add a dropdown to select time frequency
        time_freq = st.selectbox("Select Time Frequency", ["Daily", "Weekly", "Monthly"])

        # Map selection to pandas frequency
        freq_map = {"Daily": "D", "Weekly": "W", "Monthly": "M"}
        topic_distribution = helper.topic_distribution_over_time(df, time_freq=freq_map[time_freq])

        # Choose between Matplotlib and Plotly
        use_plotly = st.checkbox("Use Interactive Plot (Plotly)")

        if use_plotly:
            fig = helper.plot_topic_distribution_over_time_plotly(topic_distribution)
            st.plotly_chart(fig, use_container_width=True)
        else:
            fig = helper.plot_topic_distribution_over_time(topic_distribution)
            st.pyplot(fig)

        # Number of clusters input
        n_clusters = st.slider("Select Number of Clusters", min_value=2, max_value=10, value=5)

        # Perform clustering
        df, reduced_features, _ = preprocessor.preprocess_for_clustering(df, n_clusters=n_clusters)

        # Plot clusters
        st.header("Cluster Visualization")
        fig = helper.plot_clusters(reduced_features, df['cluster'])
        st.pyplot(fig)

        # Show insights for each cluster
        st.header("Insights from Clusters")

        # 1. Dominant Conversation Themes
        st.subheader("1. Dominant Conversation Themes")
        cluster_labels = helper.get_cluster_labels(df, n_clusters)  # Function to generate cluster labels
        for cluster_id, label in cluster_labels.items():
            st.write(f"**Cluster {cluster_id}**: {label}")
        st.write("**Why it matters**: Helps users quickly identify the main interests or priorities of the group without reading thousands of messages.")

        # 2. Temporal Topic Trends
        st.subheader("2. Temporal Topic Trends")
        temporal_trends = helper.get_temporal_trends(df)  # Function to analyze temporal trends
        for cluster_id, trend in temporal_trends.items():
            st.write(f"**Cluster {cluster_id}**: Most messages occur on {trend['peak_day']} at {trend['peak_time']}.")
        st.write("**Why it matters**: Reveals when specific topics trend, helping users optimize posting times or understand group rhythms.")

        # 3. User-Specific Contributions
        st.subheader("3. User-Specific Contributions")
        user_contributions = helper.get_user_contributions(df)  # Function to analyze user contributions
        for cluster_id, users in user_contributions.items():
            st.write(f"**Cluster {cluster_id}**: Top contributors are {', '.join(users)}.")
        st.write("**Why it matters**: Highlights individual roles and expertise within the group, useful for team management or moderation.")

        # 4. Sentiment by Topic
        st.subheader("4. Sentiment by Topic")
        sentiment_by_cluster = helper.get_sentiment_by_cluster(df)  # Function to analyze sentiment
        for cluster_id, sentiment in sentiment_by_cluster.items():
            st.write(f"**Cluster {cluster_id}**: {sentiment['positive']}% positive, {sentiment['neutral']}% neutral, {sentiment['negative']}% negative.")
        st.write("**Why it matters**: Flags problem areas or success stories tied to specific topics for targeted action.")

        # 5. Anomaly Detection
        st.subheader("5. Anomaly Detection")
        anomalies = helper.detect_anomalies(df)  # Function to detect anomalies
        for cluster_id, anomaly in anomalies.items():
            st.write(f"**Cluster {cluster_id}**: {anomaly}.")
        st.write("**Why it matters**: Identifies unusual patterns like spam, off-topic discussions, or critical resource sharing.")

        # 6. Actionable Recommendations
        st.subheader("6. Actionable Recommendations")
        recommendations = helper.generate_recommendations(df)  # Function to generate recommendations
        for recommendation in recommendations:
            st.write(f"- {recommendation}")

        # Show sample messages from each cluster
        st.header("Sample Messages from Each Cluster")
        for cluster_id in sorted(df['cluster'].unique()):
            st.subheader(f"Cluster {cluster_id}")
            cluster_messages = df[df['cluster'] == cluster_id]['message']
            sample_size = min(3, len(cluster_messages))  # Ensure sample size <= available messages
            if sample_size > 0:
                sample_messages = cluster_messages.sample(sample_size, replace=False).tolist()
                for msg in sample_messages:
                    st.write(f"- {msg}")
            else:
                st.write("No messages available for this cluster.")
