import requests
import json

def send_to_backend(dataset_summary):
    """Send the dataset summary to FastAPI backend for analysis."""
    url = "http://localhost:8000/analyze/"
    
    response = requests.post(url, json=dataset_summary)
    
    if response.status_code == 200:
        return response.json()  # Return the analysis response
    else:
        return {"error": f"Failed to analyze dataset: {response.text}"}


data = {
    "summary": "Dataset contains 891 rows and 12 columns. Numeric columns: ['PassengerId', 'Survived', 'Pclass', 'Age', 'SibSp', and 'Parch']. Categorical columns:",
    "json": {
        "columns": [
            {"name": "PassengerId", "type": "int64", "unique_values": "Many"},
            {"name": "Survived", "type": "int64", "unique_values": 2},
            {"name": "Pclass", "type": "int64", "unique_values": 3},
            {"name": "Name", "type": "object", "unique_values": "Many"},
            {"name": "Sex", "type": "object", "unique_values": 2},
            {"name": "Age", "type": "float64", "unique_values": "Many"},
            {"name": "SibSp", "type": "int64", "unique_values": "Many"},
            {"name": "Parch", "type": "int64", "unique_values": "Many"},
            {"name": "Ticket", "type": "object", "unique_values": "Many"},
            {"name": "Fare", "type": "float64", "unique_values": "Many"},
            {"name": "Cabin", "type": "object", "unique_values": "Many"},
            {"name": "Embarked", "type": "object", "unique_values": 3}
        ],
        "sample_data": [
            {
                "PassengerId": 712,
                "Survived": 0,
                "Pclass": 1,
                "Name": "Klaber, Mr. Herman",
                "Sex": "male",
                "Age": "Nan", 
                "SibSp": 0,
                "Parch": 0,
                "Ticket": "113028",
                "Fare": 26.55,
                "Cabin": "C124",
                "Embarked": "S"
            }
        ]
    }
}


result = send_to_backend(data)
print("-----------> ANSWER ->")
print(result)
