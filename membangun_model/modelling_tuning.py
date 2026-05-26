import pandas as pd
import mlflow
import mlflow.sklearn
import dagshub
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score

dagshub.init(
    repo_owner="NajwaKurnia",  
    repo_name="Ekperimen_SML_Najwa-Kurnia",
    mlflow=True
)

df = pd.read_csv("winequality_preprocessing/winequality_clean.csv")
X = df.drop("quality", axis=1)
y = df["quality"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

mlflow.set_experiment("WineQuality_Tuning")

param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [5, 10, 15]
}

with mlflow.start_run(run_name="RandomForest_GridSearch"):

    grid = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid, cv=3, scoring="accuracy", n_jobs=-1
    )
    grid.fit(X_train, y_train)

    best = grid.best_estimator_
    y_pred = best.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    roc = roc_auc_score(y_test, best.predict_proba(X_test),
                        multi_class='ovr', average='weighted')

    # Log best params & metrics
    mlflow.log_params(grid.best_params_)
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("roc_auc", roc)
    mlflow.sklearn.log_model(best, "best_model")

    print(f"Best Params: {grid.best_params_}")
    print(f"Accuracy: {acc:.4f} | ROC-AUC: {roc:.4f}")