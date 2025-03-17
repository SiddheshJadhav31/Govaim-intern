from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from pydantic import BaseModel
import pandas as pd
from typing import List, Dict, Any, Optional
import json
import numpy as np
import re
from groq import Groq
from dotenv import load_dotenv
import os
from fastapi.middleware.cors import CORSMiddleware
import io
import networkx as nx
import base64
import matplotlib.pyplot as plt
from starlette.responses import JSONResponse
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
import uvicorn

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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this to the specific frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

models = {
    'Random Forest': RandomForestClassifier(),
    'Logistic Regression': LogisticRegression(),
    'Support Vector Machine': SVC(),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Decision Tree': DecisionTreeClassifier(),
    'Naive Bayes': GaussianNB()
}

# Define hyperparameters for tuning each model
param_grids = {
    'Random Forest': {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20]},
    'Logistic Regression': {'C': [0.1, 1, 10], 'solver': ['lbfgs', 'liblinear']},
    'Support Vector Machine': {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']},
    'K-Nearest Neighbors': {'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance']},
    'Decision Tree': {'max_depth': [None, 10, 20], 'min_samples_split': [2, 5, 10]},
    'Naive Bayes': {}
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
    
@app.post("/upload/")
async def upload_csv(file: UploadFile = File(...)):

    df = pd.read_csv(file.file)
    return {"columns": df.columns.tolist()}


@app.post("/generate-graph/")
async def generate_cashflow_graph(
    file: UploadFile = File(...),
    from_col: str = Form(...),
    to_col: str = Form(...),
    amount_col: str = Form(...),
):


    df = pd.read_csv(file.file)

    if not {from_col, to_col, amount_col}.issubset(df.columns):
        return JSONResponse(content={"error": "Invalid column mappings"}, status_code=400)

    G = nx.DiGraph()

    for _, row in df.iterrows():
        G.add_edge(row[from_col], row[to_col], weight=row[amount_col])

    pos1 = nx.spring_layout(G, seed=42)
    pos2 = nx.shell_layout(G)

    node_colors = {}
    for node in G.nodes():
        if "Politician" in node:
            node_colors[node] = "red"
        elif "Corporate" in node:
            node_colors[node] = "blue"
        elif "Lobbying" in node:
            node_colors[node] = "green"
        else:
            node_colors[node] = "orange"

    node_color_list = [node_colors.get(node, "gray") for node in G.nodes()]

    edge_widths = [G[u][v]["weight"] / 20000 for u, v in G.edges()]

    def generate_graph(pos):
        plt.figure(figsize=(12, 8))
        nx.draw(
            G, pos, with_labels=True, node_color=node_color_list, node_size=2000,
            font_size=10, font_weight="bold", arrows=True, width=edge_widths, edge_color="gray"
        )
        edge_labels = nx.get_edge_attributes(G, "weight")
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        plt.close()

        buf.seek(0)
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    img1_base64 = generate_graph(pos1)
    img2_base64 = generate_graph(pos2)

    return {
        "spring_layout": img1_base64,
        "shell_layout": img2_base64,
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...), model_name: str = 'Random Forest'):
    if model_name not in models:
        raise HTTPException(status_code=400, detail="Invalid model name. Choose from: " + ", ".join(models.keys()))

    # Load the uploaded CSV file
    try:
        data = pd.read_csv(file.file)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading CSV file: {str(e)}")

    if 'Passed' not in data.columns:
        raise HTTPException(status_code=400, detail="CSV must contain 'Passed' column as the target variable")

    # Split data into features and target
    X = data.drop(columns=['Passed'])
    y = data['Passed']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Select and fine-tune the model
    model = models[model_name]
    param_grid = param_grids[model_name]
    if param_grid:
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
        grid_search.fit(X_train, y_train)
        model = grid_search.best_estimator_
    else:
        model.fit(X_train, y_train)

    # Make predictions and evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    return {
        "model": model_name,
        "accuracy": accuracy,
        "classification_report": report
    }

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