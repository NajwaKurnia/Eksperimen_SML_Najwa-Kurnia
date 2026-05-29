import pandas as pd
import mlflow
import mlflow.sklearn
import dagshub
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Setup DagsHub 
dagshub.init(
    repo_owner="NajwaKurnia",
    repo_name="Ekperimen_SML_Najwa-Kurnia",
    mlflow=True
)

# Load Data 
df = pd.read_csv("winequality_preprocessing/winequality_clean.csv")
X = df.drop("quality", axis=1)
y = df["quality"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# MLflow Autolog 
mlflow.set_tracking_uri("mlruns")
mlflow.set_experiment("WineQuality_Baseline")
mlflow.sklearn.autolog(log_models=True)

with mlflow.start_run(run_name="RandomForest_Baseline"):
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    model.fit(X_train, y_train)
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="random_forest_model",
        input_example=X_train.head(1)
    )
    y_pred = model.predict(X_test)
    print("Run selesai! Cek DagsHub untuk melihat hasilnya.")