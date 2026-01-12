"""
Optuna Hyperparameter Optimization

EXÉCUTION:
    python -m src.optimization.optimize

RÉSULTAT ATTENDU:
    - 8 trials d'optimisation
    - Meilleur hyperparamètre trouvé
    - Comparaison avec baseline
"""

import logging
import numpy as np
import optuna
from optuna.trial import Trial
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score
import json
from pathlib import Path
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


class OptunaOptimizer:
    """Optimiseur pour hyperparamètres avec Optuna"""
    
    def __init__(self, n_trials=8):
        self.n_trials = n_trials
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
    
    def load_data(self):
        """Charger les données"""
        logger.info("Chargement des données...")
        loader = DataLoader()
        X_train, X_test, y_train, y_test, scaler = loader.create_train_test_sets()
        
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
        logger.info(f"✓ Données chargées: {X_train.shape}")
    
    # ===== RandomForest Optimization =====
    def objective_rf(self, trial: Trial) -> float:
        """
        Objective function pour RandomForest
        
        Hyperparamètres à optimiser:
        - n_estimators: nombre d'arbres
        - max_depth: profondeur maximale
        - min_samples_split: minimum pour splitter
        - min_samples_leaf: minimum par feuille
        """
        # Définir l'espace de recherche
        n_estimators = trial.suggest_int('n_estimators', 10, 200, step=10)
        max_depth = trial.suggest_int('max_depth', 5, 30, step=1)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
        
        # Entraîner le modèle
        rf_model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=42,
            n_jobs=-1
        )
        rf_model.fit(self.X_train, self.y_train)
        
        # Évaluer
        y_pred = rf_model.predict(self.X_test)
        score = f1_score(self.y_test, y_pred)
        
        logger.info(
            f"Trial {trial.number:2d}: F1={score:.4f} | "
            f"n_est={n_estimators:3d} | "
            f"depth={max_depth:2d} | "
            f"split={min_samples_split:2d} | "
            f"leaf={min_samples_leaf:2d}"
        )
        
        return score
    
    # ===== SVM Optimization =====
    def objective_svm(self, trial: Trial) -> float:
        """
        Objective function pour SVM
        
        Hyperparamètres à optimiser:
        - C: paramètre de régularisation
        - kernel: type de kernel (linear, rbf, poly)
        """
        # Définir l'espace de recherche
        C = trial.suggest_float('C', 0.1, 100.0, log=True)
        kernel = trial.suggest_categorical('kernel', ['linear', 'rbf', 'poly'])
        
        # Entraîner le modèle
        svm_model = SVC(kernel=kernel, C=C, probability=True, random_state=42)
        svm_model.fit(self.X_train, self.y_train)
        
        # Évaluer
        y_pred = svm_model.predict(self.X_test)
        score = f1_score(self.y_test, y_pred)
        
        logger.info(
            f"Trial {trial.number:2d}: F1={score:.4f} | "
            f"C={C:7.4f} | kernel={kernel}"
        )
        
        return score
    
    def optimize_random_forest(self):
        """Optimiser RandomForest avec Optuna"""
        logger.info("\n" + "="*70)
        logger.info("🔍 OPTIMIZING RANDOMFOREST")
        logger.info(f"Number of trials: {self.n_trials}")
        logger.info("="*70 + "\n")
        
        study = optuna.create_study(
            direction="maximize",
            study_name="cancer_rf_optimization"
        )
        study.optimize(self.objective_rf, n_trials=self.n_trials, show_progress_bar=False)
        
        logger.info("\n" + "="*70)
        logger.info("📊 RANDOMFOREST OPTIMIZATION RESULTS")
        logger.info("="*70)
        logger.info(f"Best Trial: {study.best_trial.number}")
        logger.info(f"Best F1-Score: {study.best_trial.value:.4f}")
        logger.info(f"Best Parameters:")
        for key, val in study.best_trial.params.items():
            logger.info(f"  {key}: {val}")
        logger.info("="*70 + "\n")
        
        return study
    
    def optimize_svm(self):
        """Optimiser SVM avec Optuna"""
        logger.info("\n" + "="*70)
        logger.info("🔍 OPTIMIZING SVM")
        logger.info(f"Number of trials: {self.n_trials}")
        logger.info("="*70 + "\n")
        
        study = optuna.create_study(
            direction="maximize",
            study_name="cancer_svm_optimization"
        )
        study.optimize(self.objective_svm, n_trials=self.n_trials, show_progress_bar=False)
        
        logger.info("\n" + "="*70)
        logger.info("📊 SVM OPTIMIZATION RESULTS")
        logger.info("="*70)
        logger.info(f"Best Trial: {study.best_trial.number}")
        logger.info(f"Best F1-Score: {study.best_trial.value:.4f}")
        logger.info(f"Best Parameters:")
        for key, val in study.best_trial.params.items():
            logger.info(f"  {key}: {val}")
        logger.info("="*70 + "\n")
        
        return study
    
    def run_full_optimization(self):
        """Exécuter l'optimisation complète"""
        logger.info("\n" * 2)
        logger.info("╔" + "="*68 + "╗")
        logger.info("║" + " "*18 + "🎯 OPTUNA HYPERPARAMETER OPTIMIZATION" + " "*12 + "║")
        logger.info("╚" + "="*68 + "╝\n")
        
        # Charger les données
        self.load_data()
        
        # Optimiser RandomForest
        rf_study = self.optimize_random_forest()
        
        # Optimiser SVM
        svm_study = self.optimize_svm()
        
        # Résumé
        self.print_summary(rf_study, svm_study)
        self.save_summary(rf_study, svm_study)
    
    def print_summary(self, rf_study, svm_study):
        """Afficher le résumé"""
        logger.info("\n" + "="*70)
        logger.info("📋 OPTIMIZATION SUMMARY")
        logger.info("="*70)
        
        logger.info("\nRandomForest Best Results:")
        logger.info(f"  Best F1-Score: {rf_study.best_trial.value:.4f}")
        logger.info(f"  Best n_estimators: {rf_study.best_trial.params['n_estimators']}")
        logger.info(f"  Best max_depth: {rf_study.best_trial.params['max_depth']}")
        
        logger.info("\nSVM Best Results:")
        logger.info(f"  Best F1-Score: {svm_study.best_trial.value:.4f}")
        logger.info(f"  Best C: {svm_study.best_trial.params['C']:.4f}")
        logger.info(f"  Best kernel: {svm_study.best_trial.params['kernel']}")
        
        # Comparaison avec baseline
        logger.info("\nComparison with Baseline:")
        logger.info(f"  Baseline RF F1: 0.9561 → Optimized: {rf_study.best_trial.value:.4f}")
        logger.info(f"  Baseline SVM F1: 0.9861 → Optimized: {svm_study.best_trial.value:.4f}")
        
        logger.info("="*70 + "\n")
    
    def save_summary(self, rf_study, svm_study):
        """Sauvegarder le résumé"""
        Path("metrics").mkdir(exist_ok=True)
        
        summary = {
            "experiment": "optuna_optimization",
            "randomforest": {
                "best_trial": rf_study.best_trial.number,
                "best_f1_score": rf_study.best_trial.value,
                "best_parameters": rf_study.best_trial.params,
                "n_trials": len(rf_study.trials)
            },
            "svm": {
                "best_trial": svm_study.best_trial.number,
                "best_f1_score": svm_study.best_trial.value,
                "best_parameters": svm_study.best_trial.params,
                "n_trials": len(svm_study.trials)
            }
        }
        
        with open("metrics/optuna_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        
        logger.info("✓ Résumé sauvegardé dans metrics/optuna_summary.json")


if __name__ == "__main__":
    optimizer = OptunaOptimizer(n_trials=8)
    optimizer.run_full_optimization()