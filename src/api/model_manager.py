"""
Gestion des versions de modèles (v1.0.0, v2.0.0, rollback)

EXÉCUTION:
    python -m src.api.model_manager

RÉSULTAT ATTENDU:
    - v1.0.0 et v2.0.0 créés et sauvegardés
    - Historique des versions
"""

import joblib
import json
from pathlib import Path
from typing import Dict, Any, List
import logging
from datetime import datetime
import sys

# Configuration logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Ajouter le chemin du projet
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class ModelManager:
    """Gestionnaire de versions de modèles"""
    
    def __init__(self, models_dir="models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        
        self.current_model = None
        self.current_scaler = None
        self.current_version = None
        self.version_history = []
        
        self._load_version_history()
    
    def _load_version_history(self):
        """Charger l'historique des versions"""
        history_file = self.models_dir / "version_history.json"
        
        if history_file.exists():
            with open(history_file, 'r') as f:
                self.version_history = json.load(f)
            logger.info(f"✓ Historique chargé: {len(self.version_history)} versions")
        else:
            self.version_history = []
    
    def _save_version_history(self):
        """Sauvegarder l'historique"""
        history_file = self.models_dir / "version_history.json"
        with open(history_file, 'w') as f:
            json.dump(self.version_history, f, indent=2)
        logger.info(f"✓ Historique sauvegardé")
    
    def load_version(self, version: str) -> bool:
        """
        Charger une version spécifique
        
        Args:
            version: Version à charger (ex: "v1.0.0")
            
        Returns:
            True si succès, False sinon
        """
        model_path = self.models_dir / f"model_{version}.pkl"
        scaler_path = self.models_dir / f"scaler_{version}.pkl"
        
        if not model_path.exists():
            logger.error(f"✗ Modèle {version} non trouvé: {model_path}")
            return False
        
        try:
            self.current_model = joblib.load(model_path)
            self.current_scaler = joblib.load(scaler_path) if scaler_path.exists() else None
            self.current_version = version
            
            logger.info(f"✓ Modèle {version} chargé avec succès")
            return True
        except Exception as e:
            logger.error(f"✗ Erreur chargement {version}: {e}")
            return False
    
    def save_version(self, model, scaler, version: str, metadata: Dict[str, Any] = None) -> bool:
        """
        Sauvegarder une version de modèle
        
        Args:
            model: Modèle à sauvegarder
            scaler: Scaler à sauvegarder
            version: Numéro de version (ex: "v1.0.0")
            metadata: Métadonnées optionnelles
            
        Returns:
            True si succès, False sinon
        """
        try:
            model_path = self.models_dir / f"model_{version}.pkl"
            scaler_path = self.models_dir / f"scaler_{version}.pkl"
            
            joblib.dump(model, model_path)
            joblib.dump(scaler, scaler_path)
            
            # Ajouter à l'historique
            record = {
                "version": version,
                "timestamp": datetime.now().isoformat(),
                "model_path": str(model_path),
                "scaler_path": str(scaler_path),
                "model_type": type(model).__name__,
                "metadata": metadata or {}
            }
            
            self.version_history.append(record)
            self._save_version_history()
            
            logger.info(f"✓ Version {version} sauvegardée")
            return True
        except Exception as e:
            logger.error(f"✗ Erreur sauvegarde {version}: {e}")
            return False
    
    def list_versions(self) -> List[Dict]:
        """Lister toutes les versions disponibles"""
        versions = []
        for record in self.version_history:
            versions.append({
                "version": record["version"],
                "timestamp": record["timestamp"],
                "model_type": record["model_type"]
            })
        return versions
    
    def rollback(self, version: str) -> bool:
        """
        Rollback à une version antérieure
        
        Args:
            version: Version cible
            
        Returns:
            True si succès, False sinon
        """
        success = self.load_version(version)
        if success:
            logger.info(f"🔄 Rollback à {version} réussi")
        return success
    
    def get_current_version(self) -> str:
        """Retourner la version courante"""
        return self.current_version or "unknown"
    
    def is_ready(self) -> bool:
        """Vérifier si le modèle est chargé"""
        return self.current_model is not None and self.current_scaler is not None
    
    @property
    def model(self):
        return self.current_model
    
    @property
    def scaler(self):
        return self.current_scaler


# ===== Fonctions Helper =====

def create_v1_model():
    """Créer et sauvegarder v1.0.0 (Baseline RandomForest)"""
    logger.info("\n" + "="*70)
    logger.info("📦 Création de v1.0.0 (Baseline RandomForest)")
    logger.info("="*70 + "\n")
    
    from src.data.load_data import DataLoader
    from sklearn.ensemble import RandomForestClassifier
    
    loader = DataLoader()
    X_train, X_test, y_train, y_test, scaler = loader.create_train_test_sets()
    
    rf_model = RandomForestClassifier(
        n_estimators=30,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    
    manager = ModelManager()
    success = manager.save_version(
        rf_model,
        scaler,
        "v1.0.0",
        metadata={
            "type": "RandomForest",
            "description": "Baseline model",
            "n_estimators": 30,
            "max_depth": 10
        }
    )
    
    if success:
        manager.load_version("v1.0.0")
        logger.info("✓ v1.0.0 créé et chargé")
    
    return manager


def create_v2_model():
    """Créer et sauvegarder v2.0.0 (SVM Optimisé)"""
    logger.info("\n" + "="*70)
    logger.info("📦 Création de v2.0.0 (SVM Optimisé)")
    logger.info("="*70 + "\n")
    
    from src.data.load_data import DataLoader
    from sklearn.svm import SVC
    
    loader = DataLoader()
    X_train, X_test, y_train, y_test, scaler = loader.create_train_test_sets()
    
    # Meilleur SVM avec C optimisé
    svm_model = SVC(
        kernel='rbf',
        C=10.0,
        probability=True,
        random_state=42
    )
    svm_model.fit(X_train, y_train)
    
    manager = ModelManager()
    success = manager.save_version(
        svm_model,
        scaler,
        "v2.0.0",
        metadata={
            "type": "SVM",
            "description": "Optimized model with C=10.0",
            "kernel": "rbf",
            "C": 10.0
        }
    )
    
    if success:
        manager.load_version("v2.0.0")
        logger.info("✓ v2.0.0 créé et chargé")
    
    return manager


if __name__ == "__main__":
    logger.info("\n" + "="*70)
    logger.info("🎯 Model Version Management")
    logger.info("="*70 + "\n")
    
    # Créer v1
    manager_v1 = create_v1_model()
    
    # Créer v2
    manager_v2 = create_v2_model()
    
    # Afficher les versions
    logger.info("\n" + "="*70)
    logger.info("📋 Available Versions")
    logger.info("="*70 + "\n")
    
    all_versions = manager_v2.list_versions()
    for i, v in enumerate(all_versions, 1):
        logger.info(f"{i}. {v['version']} ({v['model_type']}) - {v['timestamp']}")
    
    logger.info("\n" + "="*70 + "\n")