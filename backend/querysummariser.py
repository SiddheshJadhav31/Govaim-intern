
import pandas as pd
import json
import requests
from transformers import pipeline
import numpy as np

def compress_dataframe(df):
    summary = {
        "columns": [
            {
                "name": col,
                "type": str(df[col].dtype),
                "unique_values": df[col].nunique() if df[col].nunique() < 5 else "Many",
            }
            for col in df.columns
        ],
        "sample_data": df.sample(min(1, len(df))).to_dict(orient="records") 
    }

    summary_text = f"Dataset contains {df.shape[0]} rows and {df.shape[1]} columns. " \
                f"Numeric columns: {list(df.select_dtypes(include=['int64', 'float64']).columns)}. " \
                f"Categorical columns: {list(df.select_dtypes(include=['object']).columns)}."

    summarized_text = summarizer(summary_text, max_length=50, min_length=20, do_sample=False)[0]['summary_text']
    
    return {"summary": summarized_text, "json": summary}


def remove_nan_values(data):
    if isinstance(data, dict):
        return {k: remove_nan_values(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [remove_nan_values(v) for v in data]
    elif isinstance(data, float) and np.isnan(data):
        return None
    else:
        return data



def send_to_backend(dataset_summary):
    url = "http://localhost:8000/analyze/"

    dataset_summary = remove_nan_values(dataset_summary)
    response = requests.post(url, json=dataset_summary)
    response_data = response.json()

    if response.status_code == 200:
        print("Response from backend:", response.json()) 
        with open("response.json", "w") as file:
            json.dump(response_data, file, indent=4)
    else:
        print("Error:", response.text)
        with open("response.json", "w") as file:
            json.dump({}, file, indent=4)

if __name__ == "__main__":
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", device=-1)
    df = pd.read_csv('titanic.csv')
    compressed_data = compress_dataframe(df)
    send_to_backend(compressed_data)
