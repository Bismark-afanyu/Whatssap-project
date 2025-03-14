# import pandas as pd
# from openai import OpenAI

# def get_chat_completion(user_message, base_url="https://openrouter.ai/api/v1"):
#     client = OpenAI(
#         base_url=base_url,
#         api_key="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"
#     )

#     prompt_template = f"""Translate the following text to English and format the response in JSON:
#     "{user_message}". Do not send original_text back in the response."""

#     completion = client.chat.completions.create(
#         extra_headers={
#             "HTTP-Referer": "https://www.example.com",  # Optional. Site URL for rankings on openrouter.ai.
#             "X-Title": "Example Site",  # Optional. Site title for rankings on openrouter.ai.
#         },
#         extra_body={},
#         model="deepseek/deepseek-r1-zero:free",
#         messages=[
#             {
#                 "role": "user",
#                 "content": prompt_template
#             }
#         ]
#     )
#     return completion.choices[0].message.content

# # Sample DataFrame with text data
# data = {
#     'text': [
#         "What is the meaning of life?",
#         "این غذا خیلی شوره!",
#         "Bonjour tout le monde!"
#     ]
# }

# df = pd.DataFrame(data)

# # Process each text entry in the DataFrame
# df['translated_text'] = df['text'].apply(get_chat_completion)

# print(df)
