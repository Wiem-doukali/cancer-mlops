MLOps Mini-Projet : Cancer du Sein (UCI Breast Cancer)
Objectifs

Ce projet démontre un workflow MLOps complet :

Classification binaire : malin vs bénin

Baseline : RandomForest + SVM

Versioning (Git + DVC)

Tracking (MLflow)

Pipeline (ZenML)

Optimisation (Optuna)

API d'inférence + déploiement

CI/CD (GitHub Actions)

Structure du projet
cancer-mlops/
├── data/
│   ├── raw/              # Données brutes (DVC tracked)
│   └── processed/        # Données traitées
├── src/
│   ├── data/             # Chargement & préprocessing
│   ├── models/           # Entraînement & inférence
│   ├── utils/            # Utilitaires
│   └── api/              # FastAPI pour serving
├── configs/              # YAML : paramètres, hyperparamètres
├── notebooks/            # Exploration EDA
├── mlruns/               # MLflow artifacts
├── .dvc/                 # DVC config
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .github/workflows/    # CI/CD GitHub Actions
└── README.md

Démarrage rapide
1. Cloner et installer
git clone <repo-url>
cd cancer-mlops
python -m venv venv
# Windows
venv\Scripts\activate
# Linux / macOS
source venv/bin/activate
pip install -r requirements.txt

2. Initialiser DVC
dvc init
dvc remote add -d myremote s3://my-bucket/dvc-store  # ou autre remote

3. Baseline training
python src/models/train.py --config configs/baseline.yaml

4. Lancer MLflow UI
mlflow ui --host 0.0.0.0 --port 5000

5. Pipeline ZenML
python src/pipeline/run_pipeline.py

6. Optimisation Optuna
python src/optimization/optimize.py --n-trials 10

7. API d'inférence
python src/api/main.py
# Accès : http://localhost:8000/docs

8. Docker Compose
docker-compose up -d

Dataset

UCI Breast Cancer (Diagnostic) : 569 samples, 30 features

Bénin : 357 (63%)

Malin : 212 (37%)

Baseline

RandomForest : 30 estimators

SVM : kernel='rbf'

Métriques : Accuracy, Precision, Recall, F1

Outils utilisés
Outil	Usage
Git / GitHub	Version control
DVC	Data versioning
MLflow	Experiment tracking
ZenML	Pipeline orchestration
Optuna	Hyperparameter optimization
FastAPI	API d'inférence
Docker	Conteneurisation
GitHub Actions	CI/CD
Déploiement

Voir section "Déploiement" ci-dessous pour v1 → v2 + rollback.

Documentation supplémentaire

Consulter les fichiers dans configs/ et les notebooks pour détails.

Auteur : Wiam Doukali
Date : 2026
