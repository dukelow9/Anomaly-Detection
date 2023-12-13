
import openai
import pandas as pd

# Replace "YOUR_API_KEY" with your actual OpenAI API key
openai.api_key = "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"

def detect_anomalies(query, file_path):
    # Load the first 100 rows of time series data from the CSV file
    data = pd.read_csv(file_path, skiprows=range(0,0), nrows=1653)

    # Assuming your time series data has a 'timestamp' and a 'value' column
    # Adjust the column names accordingly based on your actual data
    timestamp_col = 'Date'
    value_col = 'Spot Rate'

    # Preprocess the data if needed (e.g., convert timestamps to datetime objects)
    data[timestamp_col] = pd.to_datetime(data[timestamp_col])

    # Use f-strings for better readability
    prompt = f"Determine if there are any anomalies in the time series: '{file_path}'. The trend appears to be {query}."

    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        max_tokens=200,  # Adjust for desired output length
        n=1,             # Number of responses to generate
        stop=None,       # Stop criteria
    )

    # Extract the text from the response
    anomaly_detection_result = response.choices[0].text.strip()

    return anomaly_detection_result

# Example usage
query = "increasing with a sudden jump"
file_path = '1128.csv'
anomaly_detection_result = detect_anomalies(query, file_path)
print(f"Anomaly detection result: {anomaly_detection_result}")
