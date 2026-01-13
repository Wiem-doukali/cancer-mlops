"""
Entraînement Baseline : RandomForest + SVM avec MLflow tracking

EXÉCUTION:
    python src/models/train.py

RÉSULTAT ATTENDU:
    - 2 modèles entraînés (RF + SVM)
    - Métriques loggées dans MLflow
    - Modèles sauvegardés dans dossier 'models/'
"""

import os
import logging
import joblib
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import mlflow
import mlflow.sklearn
import sys

# Configuration logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Ajouter le chemin du projet
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.load_data import DataLoader


class BaselineTrainer:
    """Entraîneur pour les modèles baseline"""
    
    def __init__(self, mlflow_uri="./mlruns"):
        """
        Initialiser le trainer
        
        Args:
            mlflow_uri: URI de MLflow (par défaut: local)
        """
        self.mlflow_uri = mlflow_uri
        
        # Configurer MLflow
        mlflow.set_tracking_uri(mlflow_uri)
        mlflow.set_experiment("cancer_baseline")
        
        logger.info(f"MLflow configuré: {mlflow_uri}")
    
    def load_data(self):
        """
        Charger et préprocesser les données
        
        Returns:
            X_train, X_test, y_train, y_test, scaler
        """
        logger.info("Chargement des données...")
        
        loader = DataLoader()
        X_train, X_test, y_train, y_test, scaler = loader.create_train_test_sets()
        
        logger.info(f"✓ Données chargées:")
        logger.info(f"  - Train: {X_train.shape}")
        logger.info(f"  - Test: {X_test.shape}")
        logger.info(f"  - Classes: 0={sum(y_train==0)}, 1={sum(y_train==1)}")
        
        return X_train, X_test, y_train, y_test, scaler
    
    def train_random_forest(self, X_train, y_train, n_estimators=30, max_depth=10):
        """
        Entraîner RandomForest
        
        Args:
            X_train: Features d'entraînement
            y_train: Labels d'entraînement
            n_estimators: Nombre d'arbres
            max_depth: Profondeur maximale
            
        Returns:
            Modèle RandomForest entraîné
        """
        logger.info(f"Entraînement RandomForest (n_est={n_estimators}, depth={max_depth})...")
        
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        
        logger.info("✓ RandomForest entraîné")
        return model
    
    def train_svm(self, X_train, y_train, kernel='rbf', C=1.0):
        """
        Entraîner SVM
        
        Args:
            X_train: Features d'entraînement
            y_train: Labels d'entraînement
            kernel: Type de kernel
            C: Paramètre de régularisation
            
        Returns:
            Modèle SVM entraîné
        """
        logger.info(f"Entraînement SVM (kernel={kernel}, C={C})...")
        
        model = SVC(kernel=kernel, C=C, probability=True, random_state=42)
        model.fit(X_train, y_train)
        
        logger.info("✓ SVM entraîné")
        return model
    
    def evaluate(self, model, X_test, y_test, model_name="model"):
        """
        Évaluer un modèle
        
        Args:
            model: Modèle à évaluer
            X_test: Features de test
            y_test: Labels de test
            model_name: Nom du modèle (pour logging)
            
        Returns:
            metrics (dict), y_pred
        """
        logger.info(f"Évaluation {model_name}...")
        
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
        }
        
        if y_proba is not None:
            metrics['auc'] = roc_auc_score(y_test, y_proba)
        
        # Afficher les métriques
        logger.info(f"\n{model_name} Metrics:")
        for key, val in metrics.items():
            logger.info(f"  {key:12s}: {val:.4f}")
        
        return metrics, y_pred
    
    def log_to_mlflow(self, model, metrics, params, model_name, X_test, y_test, y_pred):
        """
        Logger le run dans MLflow
        
        Args:
            model: Modèle entraîné
            metrics: Dictionnaire des métriques
            params: Dictionnaire des paramètres
            model_name: Nom du run
            X_test: Features de test
            y_test: Labels de test
            y_pred: Prédictions
        """
        with mlflow.start_run(run_name=model_name):
            # Logger les paramètres
            for key, val in params.items():
                mlflow.log_param(key, val)
            
            # Logger les métriques
            for key, val in metrics.items():
                mlflow.log_metric(key, val)
            
            # Logger le modèle
            mlflow.sklearn.log_model(model, "model")
            
            logger.info(f"✓ Run '{model_name}' loggé dans MLflow")
    
    def run_baseline(self):
        """Pipeline complet d'entraînement baseline"""
        logger.info("\n" + "="*60)
        logger.info("🚀 BASELINE TRAINING")
        logger.info("="*60 + "\n")
        
        # Charger les données
        X_train, X_test, y_train, y_test, scaler = self.load_data()
        
        # ===== RandomForest Baseline =====
        logger.info("\n1️⃣  RandomForest Baseline")
        logger.info("-" * 60)
        
        rf_model = self.train_random_forest(X_train, y_train, n_estimators=30, max_depth=10)
        rf_metrics, rf_pred = self.evaluate(rf_model, X_test, y_test, "RandomForest")
        rf_params = {
            'model': 'RandomForest',
            'n_estimators': 30,
            'max_depth': 10,
            'random_state': 42
        }
        
        self.log_to_mlflow(rf_model, rf_metrics, rf_params, "baseline_rf", X_test, y_test, rf_pred)
        
        # Sauvegarder
        Path("models").mkdir(exist_ok=True)
        joblib.dump(rf_model, "models/rf_baseline.pkl")
        logger.info("✓ Modèle RandomForest sauvegardé: models/rf_baseline.pkl")
        
        # ===== SVM Baseline =====
        logger.info("\n2️⃣  SVM Baseline")
        logger.info("-" * 60)
        
        svm_model = self.train_svm(X_train, y_train, kernel='rbf', C=1.0)
        svm_metrics, svm_pred = self.evaluate(svm_model, X_test, y_test, "SVM")
        svm_params = {
            'model': 'SVM',
            'kernel': 'rbf',
            'C': 1.0,
            'random_state': 42
        }
        
        self.log_to_mlflow(svm_model, svm_metrics, svm_params, "baseline_svm", X_test, y_test, svm_pred)
        
        # Sauvegarder
        joblib.dump(svm_model, "models/svm_baseline.pkl")
        logger.info("✓ Modèle SVM sauvegardé: models/svm_baseline.pkl")
        
        # ===== Résumé =====
        logger.info("\n" + "="*60)
        logger.info("✅ BASELINE TRAINING COMPLETED")
        logger.info("="*60)
        logger.info("\nRésumé:")
        logger.info(f"  RandomForest F1: {rf_metrics['f1']:.4f}")
        logger.info(f"  SVM F1:          {svm_metrics['f1']:.4f}")
        logger.info(f"\nModèles sauvegardés:")
        logger.info(f"  - models/rf_baseline.pkl")
        logger.info(f"  - models/svm_baseline.pkl")
        logger.info(f"\nVisualiser les résultats:")
        logger.info(f"  mlflow ui")
        logger.info("="*60 + "\n")
        
        return rf_model, svm_model


if __name__ == "__main__":
    trainer = BaselineTrainer()
    trainer.run_baseline()