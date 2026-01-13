# MLOps Mini-Projet : Cancer du Sein (UCI Breast Cancer)

## Objectifs

Ce projet démontre un **workflow MLOps complet** :
- Classification binaire : malin vs bénin
- Baseline : RandomForest + SVM
- Versioning (Git + DVC)
- Tracking (MLflow)
- Pipeline (ZenML)
- Optimisation (Optuna)
- API d'inférence + déploiement
- CI/CD (Github CI)

## Démarrage 

### 1. Cloner et installer

```bash
git clone <repo-url>
cd cancer-mlops
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Initialiser DVC

```bash
dvc init
dvc remote add -d myremote s3://my-bucket/dvc-store  # ou autre remote
```

### 3. Baseline training

```bash
python src/models/train.py --config configs/baseline.yaml
```

### 4. Lancer MLflow UI

```bash
mlflow ui --host 0.0.0.0 --port 5000
```

### 5. Pipeline ZenML

```bash
python src/pipeline/run_pipeline.py
```

### 6. Optimisation Optuna

```bash
python src/optimization/optimize.py --n-trials 10
```

### 7. API d'inférence

```bash
python src/api/main.py
# Accès : http://localhost:8000/docs
```

### 8. Docker Compose

```bash
docker-compose up -d
```

## Dataset

**UCI Breast Cancer (Diagnostic)** : 569 samples, 30 features
- Bénin : 357 (63%)
- Malin : 212 (37%)

## Baseline

- **RandomForest** : 30 estimators
- **SVM** : kernel='rbf'

Métrique : **Accuracy, Precision, Recall, F1**

## Outils utilisés

| Outil | Usage |
|-------|-------|
| Git/GitLab | Version control |
| DVC | Data versioning |
| MLflow | Experiment tracking |
| ZenML | Pipeline orchestration |
| Optuna | Hyperparameter optimization |
| FastAPI | API d'inférence |
| Docker | Conteneurisation |
| Github CI | CI/CD |


## Déploiement

Voir section "Déploiement" ci-dessous pour v1 → v2 + rollback.


---

**Auteur** : Wiem Doukali 
**Date** : 13/01/2026
