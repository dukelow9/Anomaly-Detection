import openai
import pandas as pd
from scipy.stats import zscore
import time

openai.api_key = "Insert API key here"

def detect_anomalies_and_describe(query, file_path, threshold=3.0):
    # Load time series data from the CSV file
    data = pd.read_csv(file_path, skiprows=range(0, 0), nrows=100)

    # Assuming the time series data has a 'timestamp' and a 'value' column
    # Adjust the column names accordingly based on your actual data
    timestamp_col = 'Date'
    value_col = 'Spot Rate'

    # Preprocess the data if needed (e.g., convert timestamps to datetime objects)
    data[timestamp_col] = pd.to_datetime(data[timestamp_col])

    # Calculate the Z-score for the 'value' column
    data['z_score'] = zscore(data[value_col])

    # Identify anomalies based on the Z-score
    anomalies = data[data['z_score'].abs() > threshold]

    # Extract information about anomalies for ChatGPT
    anomaly_info = f"Anomalies were detected in the time series data. There are {len(anomalies)} anomalies with a Z-score exceeding {threshold}."

    # Generate a description for each anomaly using ChatGPT
    anomaly_descriptions = []
    for index, anomaly_row in anomalies.iterrows():
        prompt = f"Describe the anomaly in the time series data at timestamp {anomaly_row[timestamp_col]} with a value of {anomaly_row[value_col]}. The context is {query}."

        # Retry logic for handling rate limit errors
        retries = 3
        while retries > 0:
            try:
                response = openai.Completion.create(
                    engine="text-davinci-002",
                    prompt=prompt,
                    max_tokens=150,
                    n=1,
                    stop=None,
                )
                anomaly_description = response.choices[0].text.strip()
                anomaly_descriptions.append(anomaly_description)
                break  # Break the loop if the API call is successful
            except openai.error.RateLimitError as e:
                print(f"Rate limit exceeded. Retrying in 60 seconds. {e}")
                time.sleep(60)  # Wait for 60 seconds before retrying
                retries -= 1

    return anomaly_info, anomalies, anomaly_descriptions

def explore_anomalies(anomalies, anomaly_descriptions, query):
    for i, (index, anomaly_row) in enumerate(anomalies.iterrows(), 1):
        print(f"\nAnomaly {i}:")
        print(f"Timestamp: {anomaly_row['Date']}")
        print(f"Value: {anomaly_row['Spot Rate']}")
        print(f"Description: {anomaly_descriptions[i-1]}")

        # Allow the user to interactively ask questions
        user_question = input("\nEnter your question about this anomaly (or type 'exit' to end): ")
        if user_question.lower() == 'exit':
            break

        # Use ChatGPT to generate responses to user questions
        prompt = f"Explain the anomaly in the time series data at timestamp {anomaly_row['Date']} with a value of {anomaly_row['Spot Rate']}. The context is {query}. User's question: {user_question}"
        response = openai.Completion.create(
            engine="text-davinci-002",
            prompt=prompt,
            max_tokens=150,  # Adjust for the desired output length
            n=1,             # Number of responses to generate
            stop=None,       # Stop criteria
        )
        chatgpt_response = response.choices[0].text.strip()
        print(f"ChatGPT's response: {chatgpt_response}")

# Example usage
query = "increasing with a sudden jump"
file_path = '1128.csv'
anomaly_info, anomalies, anomaly_descriptions = detect_anomalies_and_describe(query, file_path)

# Print overall anomaly information
print(anomaly_info)

# Explore anomalies interactively
explore_anomalies(anomalies, anomaly_descriptions, query)
