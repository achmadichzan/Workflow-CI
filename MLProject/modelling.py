import os
import argparse
import pandas as pd
import mlflow
import mlflow.sklearn
import dagshub
from sklearn.ensemble import RandomForestClassifier

def train_ci():
    # Parsing argumen dari MLproject
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_estimators', type=int, default=100)
    parser.add_argument('--max_depth', type=int, default=5)
    args = parser.parse_args()

    # Hubungkan ke DagsHub agar hasil re-train tercatat otomatis secara online
    # Jika berjalan di GitHub Actions, kita set tracking URI secara langsung tanpa lewat browser OAuth
    if os.environ.get('GITHUB_ACTIONS') == 'true':
        print("Berjalan di GitHub Actions. Mengonfigurasi MLflow tracking secara otomatis...")
        mlflow.set_tracking_uri("https://dagshub.com/achmadichzan/Eksperimen_SML_Achmadichzan.mlflow")
    else:
        # Jika berjalan di lokal komputer Achmad, tetap gunakan login interaktif biasa
        print("Berjalan di lingkungan lokal. Menginisialisasi DagsHub...")
        dagshub.init(repo_owner='achmadichzan', repo_name='Eksperimen_SML_Achmadichzan', mlflow=True)
        
    mlflow.set_experiment("Breast_Cancer_Workflow_CI")

    # Load data
    data_dir = 'breast_cancer_preprocessing'
    train_df = pd.read_csv(os.path.join(data_dir, 'train_data_clean.csv'))
    X_train = train_df.drop('diagnosis', axis=1)
    y_train = train_df['diagnosis']

    with mlflow.start_run(run_name="CI_Automated_Retrain"):
        # Log parameter yang dikirim oleh CI
        mlflow.log_param("n_estimators", args.n_estimators)
        mlflow.log_param("max_depth", args.max_depth)

        # Training
        model = RandomForestClassifier(n_estimators=args.n_estimators, max_depth=args.max_depth, random_state=42)
        model.fit(X_train, y_train)

        # Log & Register Model
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name="BreastCancer_CI_Model"
        )
        print("Re-training via MLProject sukses dan model berhasil diregistrasi!")

if __name__ == "__main__":
    train_ci()