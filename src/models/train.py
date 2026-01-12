"""
Entraînement baseline avec MLflow tracking
"""

import os
import logging
import joblib
import yaml
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, classification_report
)
import mlflow
import mlflow.sklearn

from src.data.load_data import DataLoader

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class BaselineTrainer:
    def __init__(self, mlflow_uri="./mlruns"):
        self.mlflow_uri = mlflow_uri
        mlflow.set_tracking_uri(mlflow_uri)
        mlflow.set_experiment("cancer_baseline")
    
    def load_data(self):
        """Charger les données"""
        loader = DataLoader()
        X_train, X_test, y_train, y_test, scaler = loader.create_train_test_sets()
        return X_train, X_test, y_train, y_test, scaler
    
    def train_random_forest(self, X_train, y_train, n_estimators=30, max_depth=10, random_state=42):
        """Entraîner RandomForest"""
        logger.info(f"Entraînement RandomForest (n_estimators={n_estimators})...")
        
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        return model
    
    def train_svm(self, X_train, y_train, kernel='rbf', C=1.0, gamma='scale'):
        """Entraîner SVM"""
        logger.info(f"Entraînement SVM (kernel={kernel}, C={C})...")
        
        model = SVC(kernel=kernel, C=C, gamma=gamma, probability=True)
        model.fit(X_train, y_train)
        return model
    
    def evaluate(self, model, X_test, y_test, model_name="model"):
        """Évaluer le modèle"""
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
        
        logger.info(f"\n{model_name} Metrics:")
        for key, val in metrics.items():
            logger.info(f"  {key}: {val:.4f}")
        
        return metrics, y_pred
    
    def log_to_mlflow(self, model, metrics, params, model_name, X_test, y_test, y_pred):
        """Logger le run dans MLflow"""
        with mlflow.start_run(run_name=model_name):
            # Paramètres
            for key, val in params.items():
                mlflow.log_param(key, val)
            
            # Métriques
            for key, val in metrics.items():
                mlflow.log_metric(key, val)
            
            # Modèle
            mlflow.sklearn.log_model(model, "model")
            
            # Artefacts
            cm = confusion_matrix(y_test, y_pred)
            logger.info(f"Confusion Matrix:\n{cm}")
            
            # Log du rapport de classification
            report = classification_report(y_test, y_pred, output_dict=True)
            
            logger.info(f"Run '{model_name}' loggé dans MLflow")
    
    def run_baseline(self):
        """Pipeline complet baseline"""
        logger.info("=" * 50)
        logger.info("BASELINE TRAINING")
        logger.info("=" * 50)
        
        # Charger données
        X_train, X_test, y_train, y_test, scaler = self.load_data()
        
        # 1. RandomForest
        rf_model = self.train_random_forest(X_train, y_train, n_estimators=30, max_depth=10)
        rf_metrics, rf_pred = self.evaluate(rf_model, X_test, y_test, "RandomForest")
        rf_params = {'model': 'RandomForest', 'n_estimators': 30, 'max_depth': 10}
        
        self.log_to_mlflow(rf_model, rf_metrics, rf_params, "baseline_rf", X_test, y_test, rf_pred)
        
        # Sauvegarder
        Path("models").mkdir(exist_ok=True)
        joblib.dump(rf_model, "models/rf_baseline.pkl")
        logger.info("Modèle RandomForest sauvegardé")
        
        # 2. SVM
        svm_model = self.train_svm(X_train, y_train, kernel='rbf', C=1.0)
        svm_metrics, svm_pred = self.evaluate(svm_model, X_test, y_test, "SVM")
        svm_params = {'model': 'SVM', 'kernel': 'rbf', 'C': 1.0}
        
        self.log_to_mlflow(svm_model, svm_metrics, svm_params, "baseline_svm", X_test, y_test, svm_pred)
        
        # Sauvegarder
        joblib.dump(svm_model, "models/svm_baseline.pkl")
        logger.info("Modèle SVM sauvegardé")
        
        logger.info("=" * 50)
        logger.info("Baseline training terminé !")
        logger.info("=" * 50)
        
        return rf_model, svm_model

if __name__ == "__main__":
    trainer = BaselineTrainer()
    trainer.run_baseline()