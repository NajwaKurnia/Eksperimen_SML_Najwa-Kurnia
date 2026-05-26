import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn
import dagshub
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score)
import os

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

#  MLflow Manual Logging 
mlflow.set_experiment("WineQuality_Baseline")

with mlflow.start_run(run_name="RandomForest_Baseline"):

    # Parameter
    n_estimators = 100
    max_depth = 10
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("test_size", 0.2)
    mlflow.log_param("random_state", 42)

    # Train
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    roc = roc_auc_score(y_test, model.predict_proba(X_test),
                        multi_class='ovr', average='weighted')
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("roc_auc", roc)
    print(f"Accuracy: {acc:.4f} | ROC-AUC: {roc:.4f}")

    # Artefak 1: Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    mlflow.log_artifact("confusion_matrix.png")
    plt.close()

    # Artefak 2: Feature Importance
    importances = model.feature_importances_
    feat_df = pd.DataFrame({
        "feature": X.columns,
        "importance": importances
    }).sort_values("importance", ascending=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=feat_df, x="importance", y="feature", ax=ax)
    ax.set_title("Feature Importance")
    plt.tight_layout()
    plt.savefig("feature_importance.png")
    mlflow.log_artifact("feature_importance.png")
    plt.close()

    # Log Model
    mlflow.sklearn.log_model(model, "random_forest_model")

    print("Run selesai! Cek DagsHub untuk melihat hasilnya.")