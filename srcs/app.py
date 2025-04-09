import streamlit as st
st.set_page_config(page_title="WhatsApp Chat Analyzer", layout="wide")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import preprocessor, helper
from sentiment import predict_sentiment_batch
import os
os.environ["STREAMLIT_SERVER_RUN_ON_SAVE"] = "false"

# Theme customization
st.markdown(
    """
    <style>
    .main {background-color: #f0f2f6;}
    </style>
    """,
    unsafe_allow_html=True
)

# Set seaborn style
sns.set_theme(style="whitegrid")

st.title("📊 WhatsApp Chat Sentiment Analysis Dashboard")
st.subheader('Instructions')
st.markdown("1. Open the sidebar and upload your WhatsApp chat file in .txt format.")
st.markdown("2. Wait for the initial processing (minimal delay).")
st.markdown("3. Customize the analysis by selecting users or filters.")
st.markdown("4. Click 'Show Analysis' for detailed results.")

st.sidebar.title("Whatsapp Chat Analyzer")
uploaded_file = st.sidebar.file_uploader("Upload your chat file (.txt)", type="txt")

@st.cache_data
def load_and_preprocess(file_content):
    return preprocessor.preprocess(file_content)

if uploaded_file is not None:
    raw_data = uploaded_file.read().decode("utf-8")
    with st.spinner("Loading chat data..."):
        df, _ = load_and_preprocess(raw_data)
    st.session_state.df = df

    st.sidebar.header("🔍 Filters")
    user_list = ["Overall"] + sorted(df["user"].unique().tolist())
    selected_user = st.sidebar.selectbox("Select User", user_list)

    df_filtered = df if selected_user == "Overall" else df[df["user"] == selected_user]

    if st.sidebar.button("Show Analysis"):
        if df_filtered.empty:
            st.warning(f"No data found for user: {selected_user}")
        else:
            with st.spinner("Analyzing..."):
                if 'sentiment' not in df_filtered.columns:
                    try:
                        print("Starting sentiment analysis...")
                        # Get messages as clean strings
                        message_list = df_filtered["message"].astype(str).tolist()
                        message_list = [msg for msg in message_list if msg.strip()]
                        
                        print(f"Processing {len(message_list)} messages")
                        print(f"Sample messages: {message_list[:5]}")
                        
                        # Directly call the sentiment analysis function
                        df_filtered['sentiment'] = predict_sentiment_batch(message_list)
                        print("Sentiment analysis completed successfully")
                        
                    except Exception as e:
                        st.error(f"Sentiment analysis failed: {str(e)}")
                        print(f"Full error: {str(e)}")
                    
                    st.session_state.df_filtered = df_filtered
                else:
                    st.session_state.df_filtered = df_filtered

                # Display statistics and visualizations
                num_messages, words, num_media, num_links = helper.fetch_stats(selected_user, df_filtered)
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
                    st.title(num_media)
                with col4:
                    st.header("Links Shared")
                    st.title(num_links)

                st.title("Monthly Timeline")
                timeline = helper.monthly_timeline(selected_user, df_filtered.sample(min(5000, len(df_filtered))))
                if not timeline.empty:
                    plt.figure(figsize=(10, 5))
                    sns.lineplot(data=timeline, x='time', y='message', color='green')
                    plt.title("Monthly Timeline")
                    plt.xlabel("Date")
                    plt.ylabel("Messages")
                    st.pyplot(plt)
                    plt.clf()

                st.title("Daily Timeline")
                daily_timeline = helper.daily_timeline(selected_user, df_filtered.sample(min(5000, len(df_filtered))))
                if not daily_timeline.empty:
                    plt.figure(figsize=(10, 5))
                    sns.lineplot(data=daily_timeline, x='date', y='message', color='black')
                    plt.title("Daily Timeline")
                    plt.xlabel("Date")
                    plt.ylabel("Messages")
                    st.pyplot(plt)
                    plt.clf()

                st.title("Activity Map")
                col1, col2 = st.columns(2)
                with col1:
                    st.header("Most Busy Day")
                    busy_day = helper.week_activity_map(selected_user, df_filtered)
                    if not busy_day.empty:
                        plt.figure(figsize=(10, 5))
                        sns.barplot(x=busy_day.index, y=busy_day.values, palette="Purples_r")
                        plt.title("Most Busy Day")
                        plt.xlabel("Day of Week")
                        plt.ylabel("Message Count")
                        st.pyplot(plt)
                        plt.clf()
                with col2:
                    st.header("Most Busy Month")
                    busy_month = helper.month_activity_map(selected_user, df_filtered)
                    if not busy_month.empty:
                        plt.figure(figsize=(10, 5))
                        sns.barplot(x=busy_month.index, y=busy_month.values, palette="Oranges_r")
                        plt.title("Most Busy Month")
                        plt.xlabel("Month")
                        plt.ylabel("Message Count")
                        st.pyplot(plt)
                        plt.clf()

                if selected_user == 'Overall':
                    st.title("Most Busy Users")
                    x, new_df = helper.most_busy_users(df_filtered)
                    if not x.empty:
                        plt.figure(figsize=(10, 5))
                        sns.barplot(x=x.index, y=x.values, palette="Reds_r")
                        plt.title("Most Busy Users")
                        plt.xlabel("User")
                        plt.ylabel("Message Count")
                        plt.xticks(rotation=45)
                        st.pyplot(plt)
                        st.title("Word Count by User")
                        plt.clf()
                        st.dataframe(new_df)
                
                # Most common words analysis
                st.title("Most Common Words")
                most_common_df = helper.most_common_words(selected_user, df_filtered)
                if not most_common_df.empty:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.barplot(y=most_common_df[0], x=most_common_df[1], ax=ax, palette="Blues_r")
                    ax.set_title("Top 20 Most Common Words")
                    ax.set_xlabel("Frequency")
                    ax.set_ylabel("Words")
                    plt.xticks(rotation='vertical')
                    st.pyplot(fig)
                    plt.clf()
                else:
                    st.warning("No data available for most common words.")

                # Emoji analysis
                st.title("Emoji Analysis")
                emoji_df = helper.emoji_helper(selected_user, df_filtered)
                if not emoji_df.empty:
                    col1, col2 = st.columns(2)

                    with col1:
                        st.subheader("Top Emojis Used")
                        st.dataframe(emoji_df)
                    
                    with col2:
                        fig, ax = plt.subplots(figsize=(8, 8))
                        ax.pie(emoji_df[1].head(), labels=emoji_df[0].head(), 
                              autopct="%0.2f%%", startangle=90,
                              colors=sns.color_palette("pastel"))
                        ax.set_title("Top Emoji Distribution")
                        st.pyplot(fig)
                        plt.clf()
                else:
                    st.warning("No data available for emoji analysis.")
                
                # Sentiment Analysis Visualizations
                st.title("📈 Sentiment Analysis")
                
                # Convert month names to abbreviated format
                month_map = {
                    'January': 'Jan', 'February': 'Feb', 'March': 'Mar', 'April': 'Apr',
                    'May': 'May', 'June': 'Jun', 'July': 'Jul', 'August': 'Aug',
                    'September': 'Sep', 'October': 'Oct', 'November': 'Nov', 'December': 'Dec'
                }
                df_filtered['month'] = df_filtered['month'].map(month_map)

                # Group by month and sentiment
                monthly_sentiment = df_filtered.groupby(['month', 'sentiment']).size().unstack(fill_value=0)

                # Plotting: Histogram (Bar Chart) for each sentiment
                st.write("### Sentiment Count by Month (Histogram)")

                # Create a figure with subplots for each sentiment
                fig, axes = plt.subplots(1, 3, figsize=(18, 5))

                # Plot Positive Sentiment
                if 'positive' in monthly_sentiment:
                    axes[0].bar(monthly_sentiment.index, monthly_sentiment['positive'], color='green')
                axes[0].set_title('Positive Sentiment')
                axes[0].set_xlabel('Month')
                axes[0].set_ylabel('Count')

                # Plot Neutral Sentiment
                if 'neutral' in monthly_sentiment:
                    axes[1].bar(monthly_sentiment.index, monthly_sentiment['neutral'], color='blue')
                axes[1].set_title('Neutral Sentiment')
                axes[1].set_xlabel('Month')
                axes[1].set_ylabel('Count')

                # Plot Negative Sentiment
                if 'negative' in monthly_sentiment:
                    axes[2].bar(monthly_sentiment.index, monthly_sentiment['negative'], color='red')
                axes[2].set_title('Negative Sentiment')
                axes[2].set_xlabel('Month')
                axes[2].set_ylabel('Count')

                # Display the plots in Streamlit
                st.pyplot(fig)
                plt.clf()

                # Count sentiments per day of the week
                sentiment_counts = df_filtered.groupby(['day_of_week', 'sentiment']).size().unstack(fill_value=0)

                # Sort days correctly
                day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                sentiment_counts = sentiment_counts.reindex(day_order)

                # Daily Sentiment Analysis
                st.write("### Daily Sentiment Analysis")

                # Create a Matplotlib figure
                fig, ax = plt.subplots(figsize=(10, 5))
                sentiment_counts.plot(kind='bar', stacked=False, ax=ax, color=['red', 'blue', 'green'])

                # Customize the plot
                ax.set_xlabel("Day of the Week")
                ax.set_ylabel("Count")
                ax.set_title("Sentiment Distribution per Day of the Week")
                ax.legend(title="Sentiment")

                # Display the plot in Streamlit
                st.pyplot(fig)
                plt.clf()

                # Count messages per user per sentiment (only for Overall view)
                if selected_user == 'Overall':
                    sentiment_counts = df_filtered.groupby(['user', 'sentiment']).size().reset_index(name='Count')

                    # Calculate total messages per sentiment
                    total_per_sentiment = df_filtered['sentiment'].value_counts().to_dict()

                    # Add percentage column
                    sentiment_counts['Percentage'] = sentiment_counts.apply(
                        lambda row: (row['Count'] / total_per_sentiment[row['sentiment']]) * 100, axis=1
                    )

                    # Separate tables for each sentiment
                    positive_df = sentiment_counts[sentiment_counts['sentiment'] == 'positive'].sort_values(by='Count', ascending=False).head(10)
                    neutral_df = sentiment_counts[sentiment_counts['sentiment'] == 'neutral'].sort_values(by='Count', ascending=False).head(10)
                    negative_df = sentiment_counts[sentiment_counts['sentiment'] == 'negative'].sort_values(by='Count', ascending=False).head(10)

                    # Sentiment Contribution Analysis
                    st.write("### Sentiment Contribution by User")

                    # Create three columns for side-by-side display
                    col1, col2, col3 = st.columns(3)

                    # Display Positive Table
                    with col1:
                        st.subheader("Top Positive Contributors")
                        if not positive_df.empty:
                            st.dataframe(positive_df[['user', 'Count', 'Percentage']])
                        else:
                            st.warning("No positive sentiment data")

                    # Display Neutral Table
                    with col2:
                        st.subheader("Top Neutral Contributors")
                        if not neutral_df.empty:
                            st.dataframe(neutral_df[['user', 'Count', 'Percentage']])
                        else:
                            st.warning("No neutral sentiment data")

                    # Display Negative Table
                    with col3:
                        st.subheader("Top Negative Contributors")
                        if not negative_df.empty:
                            st.dataframe(negative_df[['user', 'Count', 'Percentage']])
                        else:
                            st.warning("No negative sentiment data")

                             # Topic Analysis Section
                st.title("🔍 Area of Focus: Topic Analysis")
                
                # Check if topic column exists, otherwise perform topic modeling
                # if 'topic' not in df_filtered.columns:
                #     with st.spinner("Performing topic modeling..."):
                #         try:
                #             # Add topic modeling here or ensure your helper functions handle it
                #             df_filtered = helper.perform_topic_modeling(df_filtered)
                #         except Exception as e:
                #             st.error(f"Topic modeling failed: {str(e)}")
                #             st.stop()
                
                # Plot Topic Distribution
                st.header("Topic Distribution")
                try:
                    fig = helper.plot_topic_distribution(df_filtered)
                    st.pyplot(fig)
                    plt.clf()
                except Exception as e:
                    st.warning(f"Could not display topic distribution: {str(e)}")

                # Display Sample Messages for Each Topic
                st.header("Sample Messages for Each Topic")
                if 'topic' in df_filtered.columns:
                    for topic_id in sorted(df_filtered['topic'].unique()):
                        st.subheader(f"Topic {topic_id}")
                        
                        # Get messages for the current topic
                        filtered_messages = df_filtered[df_filtered['topic'] == topic_id]['message']
                        
                        # Determine sample size
                        sample_size = min(5, len(filtered_messages))
                        
                        if sample_size > 0:
                            sample_messages = filtered_messages.sample(sample_size, replace=False).tolist()
                            for msg in sample_messages:
                                st.write(f"- {msg}")
                        else:
                            st.write("No messages available for this topic.")
                else:
                    st.warning("Topic information not available")

                # Topic Distribution Over Time
                st.header("📅 Topic Trends Over Time")
                
                # Add time frequency selector
                time_freq = st.selectbox("Select Time Frequency", ["Daily", "Weekly", "Monthly"], key='time_freq')
                
                # Plot topic trends
                try:
                    freq_map = {"Daily": "D", "Weekly": "W", "Monthly": "M"}
                    topic_distribution = helper.topic_distribution_over_time(df_filtered, time_freq=freq_map[time_freq])
                    
                    # Choose between static and interactive plot
                    use_plotly = st.checkbox("Use interactive visualization", value=True, key='use_plotly')
                    
                    if use_plotly:
                        fig = helper.plot_topic_distribution_over_time_plotly(topic_distribution)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        fig = helper.plot_topic_distribution_over_time(topic_distribution)
                        st.pyplot(fig)
                        plt.clf()
                except Exception as e:
                    st.warning(f"Could not display topic trends: {str(e)}")

                # Clustering Analysis Section
                st.title("🧩 Conversation Clusters")
                
                # Number of clusters input
                n_clusters = st.slider("Select number of clusters", 
                                       min_value=2, 
                                       max_value=10, 
                                       value=5,
                                       key='n_clusters')
                
                # Perform clustering
                with st.spinner("Analyzing conversation clusters..."):
                    try:
                        df_clustered, reduced_features, _ = preprocessor.preprocess_for_clustering(df_filtered, n_clusters=n_clusters)
                        
                        # Plot clusters
                        st.header("Cluster Visualization")
                        fig = helper.plot_clusters(reduced_features, df_clustered['cluster'])
                        st.pyplot(fig)
                        plt.clf()
                        
                        # Cluster Insights
                        st.header("📌 Cluster Insights")
                        
                        # 1. Dominant Conversation Themes
                        st.subheader("1. Dominant Themes")
                        cluster_labels = helper.get_cluster_labels(df_clustered, n_clusters)
                        for cluster_id, label in cluster_labels.items():
                            st.write(f"**Cluster {cluster_id}**: {label}")
                        
                        # 2. Temporal Patterns
                        st.subheader("2. Temporal Patterns")
                        temporal_trends = helper.get_temporal_trends(df_clustered)
                        for cluster_id, trend in temporal_trends.items():
                            st.write(f"**Cluster {cluster_id}**: Peaks on {trend['peak_day']} around {trend['peak_time']}")
                        
                        # 3. User Contributions
                        if selected_user == 'Overall':
                            st.subheader("3. Top Contributors")
                            user_contributions = helper.get_user_contributions(df_clustered)
                            for cluster_id, users in user_contributions.items():
                                st.write(f"**Cluster {cluster_id}**: {', '.join(users[:3])}...")
                        
                        # 4. Sentiment by Cluster
                        st.subheader("4. Sentiment Analysis")
                        sentiment_by_cluster = helper.get_sentiment_by_cluster(df_clustered)
                        for cluster_id, sentiment in sentiment_by_cluster.items():
                            st.write(f"**Cluster {cluster_id}**: {sentiment['positive']}% positive, {sentiment['neutral']}% neutral, {sentiment['negative']}% negative")
                        
                        # Sample messages from each cluster
                        st.subheader("Sample Messages")
                        for cluster_id in sorted(df_clustered['cluster'].unique()):
                            with st.expander(f"Cluster {cluster_id} Messages"):
                                cluster_msgs = df_clustered[df_clustered['cluster'] == cluster_id]['message']
                                sample_size = min(3, len(cluster_msgs))
                                if sample_size > 0:
                                    for msg in cluster_msgs.sample(sample_size, replace=False):
                                        st.write(f"- {msg}")
                                else:
                                    st.write("No messages available")
                        
                    except Exception as e:
                        st.error(f"Clustering failed: {str(e)}")