"""
Entraînement avec variations pour comparaison MLflow

EXÉCUTION:
    python -m src.models.train_variations

RÉSULTAT ATTENDU:
    - 4 variations supplémentaires
    - Total 6 runs dans MLflow
    - Comparaison dans MLflow UI
"""

import logging
import joblib
import json
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
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


class VariationTrainer:
    """Entraîneur pour les variations"""
    
    def __init__(self, mlflow_uri="./mlruns"):
        self.mlflow_uri = mlflow_uri
        mlflow.set_tracking_uri(mlflow_uri)
        mlflow.set_experiment("cancer_baseline")  # Même experiment que baseline
        self.metrics_history = []
    
    def load_data(self):
        """Charger les données"""
        loader = DataLoader()
        X_train, X_test, y_train, y_test, scaler = loader.create_train_test_sets()
        return X_train, X_test, y_train, y_test, scaler
    
    def evaluate(self, model, X_test, y_test):
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
        
        return metrics
    
    def train_and_log(self, model, params, X_train, y_train, X_test, y_test, run_name):
        """Entraîner et logger le run"""
        logger.info(f"\nEntraînement: {run_name}")
        logger.info(f"Paramètres: {params}")
        
        with mlflow.start_run(run_name=run_name):
            # Fit
            model.fit(X_train, y_train)
            
            # Éval
            metrics = self.evaluate(model, X_test, y_test)
            
            # Log params
            for key, val in params.items():
                mlflow.log_param(key, val)
            
            # Log metrics
            for key, val in metrics.items():
                mlflow.log_metric(key, val)
            
            # Log model
            mlflow.sklearn.log_model(model, "model")
            
            logger.info(f"Métriques:")
            for key, val in metrics.items():
                logger.info(f"  {key}: {val:.4f}")
            
            self.metrics_history.append({
                "run_name": run_name,
                "params": params,
                "metrics": metrics
            })
            
            return model, metrics
    
    def run_variations(self):
        """Lancer les variations"""
        logger.info("\n" + "="*70)
        logger.info("🔬 EXPERIMENT SUITE: Variations")
        logger.info("="*70 + "\n")
        
        X_train, X_test, y_train, y_test, scaler = self.load_data()
        
        # ===== VARIATION 1 : RF avec plus d'arbres =====
        logger.info("\n1️⃣  VARIATION 1 : RandomForest avec plus d'arbres")
        logger.info("-" * 70)
        
        rf_params_v1 = {
            'model': 'RandomForest',
            'n_estimators': 100,
            'max_depth': 15,
            'min_samples_split': 5,
            'min_samples_leaf': 2
        }
        rf_model_v1 = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        rf_v1, rf_metrics_v1 = self.train_and_log(
            rf_model_v1, rf_params_v1, X_train, y_train, X_test, y_test,
            "var1_rf_more_trees"
        )
        
        # ===== VARIATION 2 : RF plus profond =====
        logger.info("\n2️⃣  VARIATION 2 : RandomForest plus profond")
        logger.info("-" * 70)
        
        rf_params_v2 = {
            'model': 'RandomForest',
            'n_estimators': 50,
            'max_depth': 20,
            'min_samples_split': 3,
            'min_samples_leaf': 1
        }
        rf_model_v2 = RandomForestClassifier(
            n_estimators=50,
            max_depth=20,
            min_samples_split=3,
            min_samples_leaf=1,
            random_state=42,
            n_jobs=-1
        )
        rf_v2, rf_metrics_v2 = self.train_and_log(
            rf_model_v2, rf_params_v2, X_train, y_train, X_test, y_test,
            "var2_rf_deeper"
        )
        
        # ===== VARIATION 3 : SVM avec kernel linéaire =====
        logger.info("\n3️⃣  VARIATION 3 : SVM avec kernel linéaire")
        logger.info("-" * 70)
        
        svm_params_v1 = {
            'model': 'SVM',
            'kernel': 'linear',
            'C': 1.0
        }
        svm_model_v1 = SVC(kernel='linear', C=1.0, probability=True, random_state=42)
        svm_v1, svm_metrics_v1 = self.train_and_log(
            svm_model_v1, svm_params_v1, X_train, y_train, X_test, y_test,
            "var3_svm_linear"
        )
        
        # ===== VARIATION 4 : SVM avec C élevé =====
        logger.info("\n4️⃣  VARIATION 4 : SVM avec C élevé")
        logger.info("-" * 70)
        
        svm_params_v2 = {
            'model': 'SVM',
            'kernel': 'rbf',
            'C': 10.0
        }
        svm_model_v2 = SVC(kernel='rbf', C=10.0, probability=True, random_state=42)
        svm_v2, svm_metrics_v2 = self.train_and_log(
            svm_model_v2, svm_params_v2, X_train, y_train, X_test, y_test,
            "var4_svm_high_c"
        )
        
        # ===== RÉSUMÉ =====
        logger.info("\n" + "="*70)
        logger.info("✅ VARIATIONS COMPLETED")
        logger.info("="*70)
        
        self.print_comparison()
        self.save_summary()
    
    def print_comparison(self):
        """Afficher comparaison de tous les runs"""
        logger.info("\nComparaison par F1-Score (du meilleur au pire):\n")
        
        sorted_runs = sorted(
            self.metrics_history,
            key=lambda x: x['metrics']['f1'],
            reverse=True
        )
        
        for i, run in enumerate(sorted_runs, 1):
            logger.info(
                f"{i}. {run['run_name']:25} → "
                f"F1: {run['metrics']['f1']:.4f} | "
                f"Acc: {run['metrics']['accuracy']:.4f} | "
                f"AUC: {run['metrics']['auc']:.4f}"
            )
    
    def save_summary(self):
        """Sauvegarder le résumé en JSON"""
        Path("metrics").mkdir(exist_ok=True)
        
        summary = {
            "experiment": "cancer_baseline",
            "num_variations": len(self.metrics_history),
            "runs": self.metrics_history
        }
        
        with open("metrics/variations_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"\n✓ Résumé sauvegardé dans metrics/variations_summary.json")
        logger.info("\nVisualiser les résultats:")
        logger.info("  mlflow ui")
        logger.info("="*70 + "\n")


if __name__ == "__main__":
    trainer = VariationTrainer()
    trainer.run_variations()