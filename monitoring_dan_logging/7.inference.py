import requests
import time
import random
import pandas as pd
import mlflow.sklearn
import os
from dotenv import load_dotenv

load_dotenv()

token = os.environ.get("DAGSHUB_TOKEN")
os.environ["MLFLOW_TRACKING_URI"] = "https://dagshub.com/NajwaKurnia/Ekperimen_SML_Najwa-Kurnia.mlflow"
os.environ["MLFLOW_TRACKING_USERNAME"] = "NajwaKurnia"
os.environ["MLFLOW_TRACKING_PASSWORD"] = token

url = "http://localhost:5000/predict"

# Kirim 50 request untuk generate metrics
for i in range(50):
    sample_data = {
        "fixed acidity": random.uniform(0.1, 0.9),
        "volatile acidity": random.uniform(0.1, 0.9),
        "citric acid": random.uniform(0.1, 0.9),
        "residual sugar": random.uniform(0.1, 0.9),
        "chlorides": random.uniform(0.1, 0.9),
        "free sulfur dioxide": random.uniform(0.1, 0.9),
        "total sulfur dioxide": random.uniform(0.1, 0.9),
        "density": random.uniform(0.1, 0.9),
        "pH": random.uniform(0.1, 0.9),
        "sulphates": random.uniform(0.1, 0.9),
        "alcohol": random.uniform(0.1, 0.9)
    }
    response = requests.post(url, json=sample_data)
    print(f"Request {i+1}: {response.json()}")
    time.sleep(0.5)