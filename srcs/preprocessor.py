import re
import pandas as pd
from sentiment import predict_sentiment


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
    df['message_date'] = pd.to_datetime(df['message_date'], format='%m/%d/%y, %I:%M %p')
    df.rename(columns={'message_date': 'date'}, inplace=True)

    # Separate Users and Messages
    users, messages = [], []

    # Define the pattern to match user messages
    msg_pattern = r"^(.*?):\s(.*)$"

    # Iterate through each message in the "user_message" column
    for message in df["user_message"]:
        match = re.match(msg_pattern, message)  # Match the pattern
        if match:
            users.append(match.group(1))  # Extract the user
            messages.append(match.group(2))  # Extract the message
        else:
            users.append("group_notification")  # Assign to group_notification if no match
            messages.append(message)  # Keep the original message

    # Add the extracted users and messages as new columns in the DataFrame
    df["user"] = users
    df["message"] = messages
    # Remove rows where the user is "group_notification"
    df = df[df["user"] != "group_notification"]
    # Reset the index (for clean indexing)
    df.reset_index(drop=True, inplace=True)

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

    for meg in df["message"]:
        print(meg)
    
    df['sentiment'] =  df["message"].apply(predict_sentiment)

    return df
