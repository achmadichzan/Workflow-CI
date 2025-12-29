import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report

DAGSHUB_USERNAME = "AchmadIchzan"
DAGSHUB_REPO_NAME = "Eksperimen_SML_AchmadIchzan"

mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI"))
mlflow.set_experiment("Breast_Cancer_CI_Pipeline")

def main():
    print("Memulai proses training...")

    train_path = 'breast_cancer_preprocessing/train_data_clean.csv'
    test_path = 'breast_cancer_preprocessing/test_data_clean.csv'

    try:
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
    except FileNotFoundError:
        print(f"Error: File data tidak ditemukan di {train_path}. Pastikan folder preprocessing sudah disalin.")
        return

    X_train = train_df.drop('diagnosis', axis=1)
    y_train = train_df['diagnosis']
    X_test = test_df.drop('diagnosis', axis=1)
    y_test = test_df['diagnosis']

    rf = RandomForestClassifier(random_state=42)
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5]
    }

    print("Sedang melakukan Hyperparameter Tuning...")
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    print(f"Best Params: {best_params}")

    y_pred = best_model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)

    print(f"Accuracy: {acc}")

    print("Mulai Logging ke MLflow...")
        
    mlflow.log_params(best_params)
    
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall", rec)

    mlflow.sklearn.log_model(best_model, "model")

    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Benign', 'Malignant'], yticklabels=['Benign', 'Malignant'])
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    
    cm_path = "confusion_matrix.png"
    plt.savefig(cm_path)
    plt.close()
    
    mlflow.log_artifact(cm_path)
    print("Artefak 1 (Confusion Matrix) terupload.")

    report = classification_report(y_test, y_pred)
    report_path = "classification_report.txt"
    with open(report_path, "w") as f:
        f.write(report)
    
    mlflow.log_artifact(report_path)
    print("Artefak 2 (Classification Report) terupload.")

    if os.path.exists(cm_path): os.remove(cm_path)
    if os.path.exists(report_path): os.remove(report_path)

    print("Proses Training & Logging ke DagsHub Selesai!")

if __name__ == "__main__":
    main()