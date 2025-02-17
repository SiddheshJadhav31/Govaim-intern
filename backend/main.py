from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import pandas as pd
from typing import List, Dict, Any, Optional
import json
import numpy as np
import re
from groq import Groq
from dotenv import load_dotenv
import os

class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if pd.isna(obj):
            return None
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return super().default(obj)

app = FastAPI(
    title="Govaim Dataset Analysis API",
    description="API for analyzing datasets and generating visualization suggestions",
    version="1.0.0"
)
load_dotenv()
Api_key = os.getenv("GROQ_API_KEY")

client = Groq(api_key=Api_key)

class ColumnInfo(BaseModel):
    name: str
    dtype: str
    unique_values: Optional[List[Any]] = None
    is_numeric: bool
    is_categorical: bool

    class Config:
        json_encoders = {
            np.integer: lambda x: int(x),
            np.floating: lambda x: float(x),
            pd.NA: lambda x: None
        }

class DatasetAnalysis(BaseModel):
    numerical_columns: List[str]
    categorical_columns: List[str]
    column_details: List[ColumnInfo]
    sample_data: List[Dict[str, Any]]

    class Config:
        json_encoders = {
            np.integer: lambda x: int(x),
            np.floating: lambda x: float(x),
            pd.NA: lambda x: None
        }

def clean_data_for_json(data):
    if isinstance(data, dict):
        return {k: clean_data_for_json(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [clean_data_for_json(x) for x in data]
    elif isinstance(data, (np.integer, np.floating)):
        return float(data) if isinstance(data, np.floating) else int(data)
    elif pd.isna(data):
        return None
    return data

def analyze_dataset(df: pd.DataFrame) -> DatasetAnalysis:
    numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_columns = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    
    column_details = []
    for column in df.columns:
        is_numeric = column in numerical_columns
        is_categorical = column in categorical_columns
        
        column_info = {
            "name": column,
            "dtype": str(df[column].dtype),
            "is_numeric": is_numeric,
            "is_categorical": is_categorical
        }
        
        if is_categorical:
            unique_values = df[column].unique().tolist()
            unique_values = [x if not pd.isna(x) else None for x in unique_values]
            if len(unique_values) <= 5:
                column_info["unique_values"] = unique_values
            else:
                column_info["unique_values"] = ["many"]
        
        column_details.append(ColumnInfo(**column_info))
    

    sample_data = df.head(2).to_dict('records')
    sample_data = clean_data_for_json(sample_data)
    
    return DatasetAnalysis(
        numerical_columns=numerical_columns,
        categorical_columns=categorical_columns,
        column_details=column_details,
        sample_data=sample_data
    )

async def get_visualization_suggestions(data: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Get visualization suggestions from Groq based on dataset analysis.
    """
    # Clean data before JSON serialization
    clean_data = clean_data_for_json(data)
    print("Received data:", json.dumps(clean_data, indent=4))
    
    dataset_description = json.dumps(clean_data, indent=4)
    
    prompt = f"""
    Given the following dataset description:
    {dataset_description}

    Suggest 5 best visualizations. Return ONLY the JSON output in this format:
    {{
      "visualizations": [
        {{
          "type": "Visualization Type",
          "x_column": "column name",
          "y_column": "column name"
        }}
      ]
    }}
    Do not include any explanations, just return valid JSON. Also you can provide the same type twice with different columns.
    """

    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
        )

        if not chat_completion or not chat_completion.choices:
            raise HTTPException(status_code=500, detail="Groq API returned an empty response.")

        response = chat_completion.choices[0].message.content.strip()

        if not response:
            raise HTTPException(status_code=500, detail="Groq API returned an empty message.")

        json_match = re.search(r"\{[\s\S]*\}", response)
        if not json_match:
            raise HTTPException(
                status_code=500, 
                detail=f"Groq response does not contain valid JSON: {response}"
            )

        extracted_json = json_match.group(0)
        visualization_response = json.loads(extracted_json)

        return {"visualizations": visualization_response["visualizations"]}

    except json.JSONDecodeError as json_err:
        raise HTTPException(
            status_code=500,
            detail=f"Groq response is not valid JSON: {json_err}, Extracted JSON: {extracted_json}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error communicating with Groq: {str(e)}"
        )

@app.post("/analyze-dataset")
async def analyze_uploaded_dataset(file: UploadFile = File(...)):
    """
    Analyze an uploaded dataset and generate visualization suggestions.
    """
    try:
        if file.filename.endswith('.csv'):
            df = pd.read_csv(file.file)
        elif file.filename.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(file.file)
        else:
            raise HTTPException(
                status_code=400,
                detail="Unsupported file format. Please upload CSV or Excel file."
            )
        
        analysis = analyze_dataset(df)
        
        analysis_dict = clean_data_for_json(analysis.dict())
        
        visualization_suggestions = await get_visualization_suggestions(analysis_dict)
        
        response = {
            "visualizations": visualization_suggestions["visualizations"]
        }
    
        clean_response = clean_data_for_json(response)
        
        return clean_response
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing dataset: {str(e)}"
        )

@app.get("/")
async def root():
    """
    Root endpoint providing API information and available endpoints.
    """
    return {
        "name": "Govaim Dataset Analysis API",
        "version": "1.0.0",
        "endpoints": {
            "/analyze-dataset": "POST endpoint for dataset analysis and visualization suggestions",
        }
    }