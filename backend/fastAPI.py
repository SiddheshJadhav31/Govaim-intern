from fastapi import FastAPI
import json
import os
import re
from dotenv import load_dotenv
from groq import Groq

# Load API key from .env
load_dotenv()
Api_key = os.getenv("GROQ_API_KEY")

# Initialize FastAPI app
app = FastAPI()
client = Groq(api_key=Api_key)

@app.post("/analyze/")
async def analyze_data(data: dict):
    print("Received data:", json.dumps(data, indent=4))  # Debugging

    dataset_description = json.dumps(data, indent=4)  # Convert dataset to JSON string

    prompt = f"""
    Given the following dataset description:
    {dataset_description}

    Suggest 5 best visualizations.  Return ONLY the JSON output in this format:
    {{
      "visualizations": [
        {{
          "type": "Visualization Type",
          "x_column": "column name",
          "y_column": "column name"
        }}
      ]
    }}
    Do not include any explanations, just return valid JSON.
    """

    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
        )

        print("Groq API raw response:", chat_completion)  # Debugging

        if not chat_completion or not chat_completion.choices:
            return {"error": "Groq API returned an empty response."}

        response = chat_completion.choices[0].message.content.strip()

        if not response:
            return {"error": "Groq API returned an empty message."}

        # Extract JSON using regex
        json_match = re.search(r"\{[\s\S]*\}", response)  # Finds the first JSON block
        if not json_match:
            return {"error": f"Groq response does not contain valid JSON: {response}"}

        extracted_json = json_match.group(0)  # Get matched JSON string

        # Convert JSON string to Python dictionary
        visualization_response = json.loads(extracted_json)

        return {"visualizations": visualization_response["visualizations"]}

    except json.JSONDecodeError as json_err:
        return {"error": f"Groq response is not valid JSON: {json_err}, Extracted JSON: {extracted_json}"}

    except Exception as e:
        return {"error": f"Error communicating with Groq: {e}"}
