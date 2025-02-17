import pandas as pd
import json
from transformers import pipeline

if __name__ == "__main__":

    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", device=-1)

    def compress_dataframe(df):
        """Summarizes dataset and converts it into a compact JSON format."""
        summary = {
            "columns": [
                {
                    "name": col,
                    "type": str(df[col].dtype),
                    "unique_values": df[col].nunique() if df[col].nunique() < 5  else "Many",
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


    df = pd.read_csv('titanic.csv')
    compressed_data = compress_dataframe(df)
    print(json.dumps(compressed_data, indent=4))
