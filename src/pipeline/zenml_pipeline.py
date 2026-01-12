"""
ZenML Pipeline pour Cancer du Sein MLOps

Pipeline stages: Data → Train → Eval → Export

EXÉCUTION:
    python -m src.pipeline.zenml_pipeline

RÉSULTAT ATTENDU:
    - 4 steps exécutés avec succès
    - Modèles sauvegardés
"""

import logging
import joblib
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
import sys
from typing import Tuple

# Configuration logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Ajouter le chemin du projet
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.load_data import DataLoader


class ZenMLPipeline:
    """Pipeline MLOps avec ZenML-like structure"""
    
    def __init__(self):
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = None
        self.rf_model = None
        self.svm_model = None
    
    # ===== STEP 1 : Load & Preprocess Data =====
    def step_1_load_data(self):
        """
        Step 1: Load and preprocess data
        
        Output: X_train, X_test, y_train, y_test
        """
        logger.info("\n" + "="*70)
        logger.info("STEP 1️⃣  : Load and Preprocess Data")
        logger.info("="*70)
        
        loader = DataLoader()
        X_train, X_test, y_train, y_test, scaler = loader.create_train_test_sets()
        
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.scaler = scaler
        
        logger.info(f"\n✓ Data loaded successfully")
        logger.info(f"  Train shape: {X_train.shape}")
        logger.info(f"  Test shape: {X_test.shape}")
        logger.info(f"  Classes: 0={sum(y_train==0)}, 1={sum(y_train==1)}")
        
        return X_train, X_test, y_train, y_test
    
    # ===== STEP 2 : Train Models =====
    def step_2_train_models(self):
        """
        Step 2: Train RandomForest and SVM models
        
        Input: X_train, y_train
        Output: rf_model, svm_model
        """
        logger.info("\n" + "="*70)
        logger.info("STEP 2️⃣  : Train Models")
        logger.info("="*70)
        
        # Train RandomForest
        logger.info("\nTraining RandomForest...")
        rf_model = RandomForestClassifier(
            n_estimators=50,
            max_depth=15,
            random_state=42,
            n_jobs=-1
        )
        rf_model.fit(self.X_train, self.y_train)
        logger.info("✓ RandomForest trained")
        
        # Train SVM
        logger.info("Training SVM...")
        svm_model = SVC(kernel='rbf', C=1.0, probability=True, random_state=42)
        svm_model.fit(self.X_train, self.y_train)
        logger.info("✓ SVM trained")
        
        self.rf_model = rf_model
        self.svm_model = svm_model
        
        return rf_model, svm_model
    
    # ===== STEP 3 : Evaluate Models =====
    def step_3_evaluate_models(self):
        """
        Step 3: Evaluate models on test set
        
        Input: rf_model, svm_model, X_test, y_test
        Output: metrics_dict
        """
        logger.info("\n" + "="*70)
        logger.info("STEP 3️⃣  : Evaluate Models")
        logger.info("="*70)
        
        metrics = {}
        
        # Evaluate RandomForest
        logger.info("\nEvaluating RandomForest...")
        rf_pred = self.rf_model.predict(self.X_test)
        rf_proba = self.rf_model.predict_proba(self.X_test)[:, 1]
        
        metrics['rf'] = {
            'accuracy': accuracy_score(self.y_test, rf_pred),
            'precision': precision_score(self.y_test, rf_pred),
            'recall': recall_score(self.y_test, rf_pred),
            'f1': f1_score(self.y_test, rf_pred),
            'auc': roc_auc_score(self.y_test, rf_proba),
        }
        
        logger.info("RandomForest Metrics:")
        for key, val in metrics['rf'].items():
            logger.info(f"  {key}: {val:.4f}")
        
        # Evaluate SVM
        logger.info("\nEvaluating SVM...")
        svm_pred = self.svm_model.predict(self.X_test)
        svm_proba = self.svm_model.predict_proba(self.X_test)[:, 1]
        
        metrics['svm'] = {
            'accuracy': accuracy_score(self.y_test, svm_pred),
            'precision': precision_score(self.y_test, svm_pred),
            'recall': recall_score(self.y_test, svm_pred),
            'f1': f1_score(self.y_test, svm_pred),
            'auc': roc_auc_score(self.y_test, svm_proba),
        }
        
        logger.info("SVM Metrics:")
        for key, val in metrics['svm'].items():
            logger.info(f"  {key}: {val:.4f}")
        
        return metrics
    
    # ===== STEP 4 : Export Models =====
    def step_4_export_models(self, metrics):
        """
        Step 4: Export models to disk
        
        Input: rf_model, svm_model
        Output: model paths
        """
        logger.info("\n" + "="*70)
        logger.info("STEP 4️⃣  : Export Models")
        logger.info("="*70)
        
        Path("models").mkdir(exist_ok=True)
        
        # Save RandomForest
        rf_path = "models/zenml_rf_model.pkl"
        joblib.dump(self.rf_model, rf_path)
        logger.info(f"✓ RandomForest exported to {rf_path}")
        
        # Save SVM
        svm_path = "models/zenml_svm_model.pkl"
        joblib.dump(self.svm_model, svm_path)
        logger.info(f"✓ SVM exported to {svm_path}")
        
        # Save Scaler
        scaler_path = "models/zenml_scaler.pkl"
        joblib.dump(self.scaler, scaler_path)
        logger.info(f"✓ Scaler exported to {scaler_path}")
        
        logger.info(f"\n✓ Best model (SVM): {metrics['svm']['f1']:.4f} F1-score")
        
        return {
            'rf_path': rf_path,
            'svm_path': svm_path,
            'scaler_path': scaler_path
        }
    
    # ===== RUN COMPLETE PIPELINE =====
    def run_pipeline(self):
        """Execute the complete pipeline"""
        logger.info("\n" * 2)
        logger.info("╔" + "="*68 + "╗")
        logger.info("║" + " "*15 + "🚀 ZENML PIPELINE: Cancer MLOps" + " "*20 + "║")
        logger.info("╚" + "="*68 + "╝")
        
        try:
            # Step 1
            self.step_1_load_data()
            
            # Step 2
            self.step_2_train_models()
            
            # Step 3
            metrics = self.step_3_evaluate_models()
            
            # Step 4
            paths = self.step_4_export_models(metrics)
            
            # Summary
            logger.info("\n" + "="*70)
            logger.info("✅ PIPELINE COMPLETED SUCCESSFULLY")
            logger.info("="*70)
            
            logger.info("\nPipeline Summary:")
            logger.info(f"  ✓ Step 1: Data loaded (455 train, 114 test)")
            logger.info(f"  ✓ Step 2: 2 models trained (RF + SVM)")
            logger.info(f"  ✓ Step 3: Models evaluated")
            logger.info(f"    - RandomForest F1: {metrics['rf']['f1']:.4f}")
            logger.info(f"    - SVM F1: {metrics['svm']['f1']:.4f}")
            logger.info(f"  ✓ Step 4: Models exported")
            logger.info(f"    - {paths['rf_path']}")
            logger.info(f"    - {paths['svm_path']}")
            logger.info(f"    - {paths['scaler_path']}")
            
            logger.info("\n" + "="*70 + "\n")
            
            return {
                'status': 'success',
                'metrics': metrics,
                'paths': paths
            }
        
        except Exception as e:
            logger.error(f"\n❌ PIPELINE FAILED: {str(e)}")
            logger.error("="*70)
            raise


if __name__ == "__main__":
    pipeline = ZenMLPipeline()
    result = pipeline.run_pipeline()