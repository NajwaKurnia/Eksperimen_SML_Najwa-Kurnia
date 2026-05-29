import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

mlflow.set_tracking_uri("mlruns")
mlflow.set_experiment("WineQuality_Baseline")

df = pd.read_csv("winequality_preprocessing/winequality_clean.csv")
X = df.drop("quality", axis=1)
y = df["quality"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

with mlflow.start_run(run_name="RandomForest_WithModel") as run:
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    
    # Log model secara eksplisit
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="random_forest_model"
    )
    
    run_id = run.info.run_id
    print(f"Run ID: {run_id}")
    print("Model tersimpan!")